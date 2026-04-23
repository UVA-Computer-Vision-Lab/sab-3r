#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# SAB3R gradio demo functions
#
# Builds on the MASt3R sparse gradio demo and adds SAB3R's
# CLIP feature head + open-vocabulary query overlay.
# --------------------------------------------------------
import io
import math
import gradio
import os
import numpy as np
import functools
import trimesh
import copy
from scipy.spatial.transform import Rotation
import tempfile
import shutil
import torch
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess
from mast3r.model import AsymmetricMASt3R
import mast3r.utils.path_to_dust3r  # noqa
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.utils.image import load_images, rgb
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.demo import get_args_parser as dust3r_get_args_parser
from featup.featurizers.maskclip.clip import tokenize
import matplotlib.pyplot as pl

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ZeroGPU / Hugging Face Spaces support.
# `spaces.GPU` allocates a GPU for the duration of a decorated call; locally it's a no-op.
try:
    import spaces  # type: ignore
    GPU = spaces.GPU
except ImportError:
    def GPU(*args, **kwargs):
        """No-op fallback for non-Spaces environments."""
        if args and callable(args[0]):
            return args[0]
        def _decorator(fn):
            return fn
        return _decorator

inf = float("inf")

# `img_update` (text-query overlay) runs from `get_3D_model_from_scene`, which
# is NOT @GPU-decorated — i.e. the main process. On a HF Space with ZeroGPU,
# the main process has no GPU (GPU is only visible inside @spaces.GPU calls)
# even though torch.cuda.is_available() can return True at module init. To
# avoid a torch._cuda_init() crash later, keep the upsampler on CPU on Spaces.
_ON_HF_SPACE = bool(os.environ.get("SPACE_ID"))
DEMO_DEVICE = "cpu" if _ON_HF_SPACE else ("cuda" if torch.cuda.is_available() else "cpu")
upsampler = torch.hub.load("mhamilton723/FeatUp", "maskclip", use_norm=False).to(DEMO_DEVICE)

# Module-level model singleton. ZeroGPU keeps globally-reachable CUDA tensors
# alive across @spaces.GPU worker invocations; passing the model via argument
# pickles its parameters as CPU tensors and breaks on first inference call.
_SAB3R_MODEL = None


def _get_sab3r_model():
    if _SAB3R_MODEL is None:
        raise RuntimeError(
            "SAB3R model not initialized. main_demo() must be called before "
            "any @GPU-decorated handler fires."
        )
    return _SAB3R_MODEL


class SparseGAState:
    def __init__(
        self, sparse_ga, should_delete=False, cache_dir=None, outfile_name=None
    ):
        self.sparse_ga = sparse_ga
        self.cache_dir = cache_dir
        self.outfile_name = outfile_name
        self.should_delete = should_delete

    def __del__(self):
        # `getattr` with a default: when unpickled on the main process (e.g. on
        # return from a @spaces.GPU worker), __init__ isn't re-run, and the
        # instance may briefly lack `should_delete` during teardown.
        if not getattr(self, "should_delete", False):
            return
        if self.cache_dir is not None and os.path.isdir(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        self.cache_dir = None
        if self.outfile_name is not None and os.path.isfile(self.outfile_name):
            os.remove(self.outfile_name)
        self.outfile_name = None


def get_args_parser():
    parser = dust3r_get_args_parser()
    parser.add_argument("--share", action="store_true")
    parser.add_argument(
        "--gradio_delete_cache",
        default=None,
        type=int,
        help="age/frequency at which gradio removes the file. If >0, matching cache is purged",
    )

    actions = parser._actions
    for action in actions:
        if action.dest == "model_name":
            action.choices = ["MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"]
    # change defaults
    parser.prog = "sab3r demo"
    return parser


def _convert_scene_output_to_glb(
    outfile,
    imgs,
    pts3d,
    mask,
    focals,
    cams2world,
    cam_size=0.05,
    cam_color=None,
    as_pointcloud=False,
    transparent_cams=False,
    silent=False,
    query=False,
):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    # full pointcloud
    if as_pointcloud:
        pts = np.concatenate([p[m.ravel()] for p, m in zip(pts3d, mask)]).reshape(-1, 3)
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)]).reshape(-1, 3)
        valid_msk = np.isfinite(pts.sum(axis=1))
        pct = trimesh.PointCloud(pts[valid_msk], colors=col[valid_msk])
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            pts3d_i = pts3d[i].reshape(imgs[i].shape)
            msk_i = mask[i] & np.isfinite(pts3d_i.sum(axis=-1))
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d_i, msk_i))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    # add each camera
    for i, pose_c2w in enumerate(cams2world):
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(
            scene,
            pose_c2w,
            camera_edge_color,
            None if transparent_cams else imgs[i],
            focals[i],
            imsize=imgs[i].shape[1::-1],
            screen_width=cam_size,
        )

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler("y", np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    if query:
        outfile = os.path.join(os.path.dirname(outfile), "scene_query.glb")
    if not silent:
        print("(exporting 3D scene to", outfile, ")")
    scene.export(file_obj=outfile)
    return outfile


def img_update(images, image_feats, text, thd_per, thd_val):
    sims = []
    sim_min = 1e09
    sim_max = -1e09
    print(text)
    text = tokenize(text).to(DEMO_DEVICE)
    text_feat = (
        upsampler.model.model.encode_text(text)
        .squeeze()
        .to(torch.float32)
        .detach()
        .cpu()
    )
    for image, image_feat in zip(images, image_feats):
        sim = torch.einsum(
            "chw,c->hw",
            F.normalize(image_feat.permute(2, 0, 1).to(torch.float32), dim=0),
            F.normalize(text_feat, dim=0),
        )
        sim_min = min(sim_min, sim.min())
        sim_max = max(sim_max, sim.max())
        sims.append(sim)
    threshold = max(np.percentile(np.vstack(sims), thd_per), thd_val)
    combined_images = []
    for image, sim in zip(images, sims):
        combined_image = image.copy()
        sim_norm = (sim - sim_min) / (sim_max - sim_min)
        mask = sim_norm < threshold
        sim_norm = (sim_norm - threshold) / (sim_norm.max() - threshold)
        colored_matrix = plt.cm.viridis(sim_norm)[:, :, :3]
        colored_matrix[mask, :3] = [0, 0, 0]
        weight1 = 0.6
        weight2 = 0.4
        combined_image[~mask] = weight1 * image[~mask] + weight2 * colored_matrix[~mask]
        combined_images.append(combined_image)
    return combined_images


def get_3D_model_from_scene(
    silent,
    scene_state,
    min_conf_thr=2,
    as_pointcloud=False,
    mask_sky=False,
    clean_depth=False,
    transparent_cams=False,
    cam_size=0.05,
    TSDF_thresh=0,
    feats=None,
    text=None,
    query_only=False,
    thd_per=90,
    thd_val=0.75,
):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    if scene_state is None:
        return None
    outfile = scene_state.outfile_name
    if outfile is None:
        return None

    # get optimized values from scene
    sparse_ga = scene_state.sparse_ga
    rgbimg = sparse_ga.imgs
    focals = sparse_ga.get_focals().cpu()
    cams2world = sparse_ga.get_im_poses().cpu()

    # 3D pointcloud from depthmap, poses and intrinsics
    if TSDF_thresh > 0:
        tsdf = TSDFPostProcess(sparse_ga, TSDF_thresh=TSDF_thresh)
        pts3d, _, confs = to_numpy(tsdf.get_dense_pts3d(clean_depth=clean_depth))
    else:
        pts3d, _, confs = to_numpy(sparse_ga.get_dense_pts3d(clean_depth=clean_depth))
    msk = to_numpy([c > min_conf_thr for c in confs])
    if text:
        queryimg = img_update(rgbimg, feats, text, thd_per, thd_val)

        if query_only:
            imgs = []
            for i in range(len(rgbimg)):
                imgs.append(rgbimg[i])
                imgs.append(queryimg[i])
            return (
                _convert_scene_output_to_glb(
                    outfile,
                    queryimg,
                    pts3d,
                    msk,
                    focals,
                    cams2world,
                    as_pointcloud=as_pointcloud,
                    transparent_cams=transparent_cams,
                    cam_size=cam_size,
                    silent=silent,
                    query=True,
                ),
                imgs,
            )
        else:
            return _convert_scene_output_to_glb(
                outfile,
                rgbimg,
                pts3d,
                msk,
                focals,
                cams2world,
                as_pointcloud=as_pointcloud,
                transparent_cams=transparent_cams,
                cam_size=cam_size,
                silent=silent,
            ), _convert_scene_output_to_glb(
                outfile,
                queryimg,
                pts3d,
                msk,
                focals,
                cams2world,
                as_pointcloud=as_pointcloud,
                transparent_cams=transparent_cams,
                cam_size=cam_size,
                silent=silent,
                query=True,
            )
    else:
        return _convert_scene_output_to_glb(
            outfile,
            rgbimg,
            pts3d,
            msk,
            focals,
            cams2world,
            as_pointcloud=as_pointcloud,
            transparent_cams=transparent_cams,
            cam_size=cam_size,
            silent=silent,
        )


@GPU(duration=120)
def get_reconstructed_scene(
    outdir,
    gradio_delete_cache,
    device,
    silent,
    image_size,
    current_scene_state,
    filelist,
    optim_level,
    lr1,
    niter1,
    lr2,
    niter2,
    min_conf_thr,
    matching_conf_thr,
    as_pointcloud,
    mask_sky,
    clean_depth,
    transparent_cams,
    cam_size,
    scenegraph_type,
    winsize,
    win_cyclic,
    refid,
    TSDF_thresh,
    shared_intrinsics,
    **kw,
):
    """
    from a list of images, run SAB3R inference, sparse global aligner.
    then run get_3D_model_from_scene
    """
    model = _get_sab3r_model()
    imgs = load_images(filelist, size=image_size, verbose=not silent)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]["idx"] = 1
        filelist = [filelist[0], filelist[0] + "_2"]

    scene_graph_params = [scenegraph_type]
    if scenegraph_type in ["swin", "logwin"]:
        scene_graph_params.append(str(winsize))
    elif scenegraph_type == "oneref":
        scene_graph_params.append(str(refid))
    if scenegraph_type in ["swin", "logwin"] and not win_cyclic:
        scene_graph_params.append("noncyclic")
    scene_graph = "-".join(scene_graph_params)
    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=1, verbose=not silent)
    feats = [None] * len(imgs)
    for i, idx in enumerate(output["view1"]["idx"]):
        feats[idx] = output["pred1"]["clip"][i]

    if optim_level == "coarse":
        niter2 = 0
    # Sparse GA (forward SAB3R -> matching -> 3D optim -> 2D refinement -> triangulation)
    if (
        current_scene_state is not None
        and not current_scene_state.should_delete
        and current_scene_state.cache_dir is not None
    ):
        cache_dir = current_scene_state.cache_dir
    elif gradio_delete_cache:
        cache_dir = tempfile.mkdtemp(suffix="_cache", dir=outdir)
    else:
        cache_dir = os.path.join(outdir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    scene_ga = sparse_global_alignment(
        filelist,
        pairs,
        cache_dir,
        model,
        lr1=lr1,
        niter1=niter1,
        lr2=lr2,
        niter2=niter2,
        device=device,
        opt_depth="depth" in optim_level,
        shared_intrinsics=shared_intrinsics,
        matching_conf_thr=matching_conf_thr,
        **kw,
    )
    if (
        current_scene_state is not None
        and not current_scene_state.should_delete
        and current_scene_state.outfile_name is not None
    ):
        outfile_name = current_scene_state.outfile_name
    else:
        outfile_name = tempfile.mktemp(suffix="_scene.glb", dir=outdir)

    scene = SparseGAState(scene_ga, gradio_delete_cache, cache_dir, outfile_name)
    outfile = get_3D_model_from_scene(
        silent,
        scene,
        min_conf_thr,
        as_pointcloud,
        mask_sky,
        clean_depth,
        transparent_cams,
        cam_size,
        TSDF_thresh,
    )
    rgbimg = scene_ga.imgs
    depths = to_numpy(scene_ga.get_depthmaps())
    confs = to_numpy(scene_ga.confs)
    cmap = pl.get_cmap("jet")
    depth_cmap = pl.get_cmap("plasma")  # Plasma colormap for depth
    
    # Process depths more carefully
    depth_images = []
    for i, depth in enumerate(depths):
        # Handle invalid values
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Create mask for valid depth values
        valid_mask = (depth > 0) & np.isfinite(depth)
        
        if np.any(valid_mask):
            # Get valid depth values for statistics
            valid_depths = depth[valid_mask]
            
            # Use more conservative percentiles for better visualization
            depth_min = np.percentile(valid_depths, 2)   # 2nd percentile as min
            depth_max = np.percentile(valid_depths, 98)  # 98th percentile as max
            
            # Normalize to [0, 1] - closer objects get higher values (brighter)
            depth_norm = np.zeros_like(depth)
            if depth_max > depth_min:
                depth_norm[valid_mask] = 1.0 - (depth[valid_mask] - depth_min) / (depth_max - depth_min)
                depth_norm = np.clip(depth_norm, 0, 1)
            
            # Apply colormap
            depth_colored = depth_cmap(depth_norm)
            
            # Handle colormap output dimensions
            if len(depth_colored.shape) == 3 and depth_colored.shape[2] >= 3:
                depth_colored = depth_colored[:, :, :3]  # Take RGB channels
            elif len(depth_colored.shape) == 3 and depth_colored.shape[2] == 1:
                # Single channel, convert to RGB
                depth_colored = np.repeat(depth_colored, 3, axis=2)
            else:
                # If something went wrong, create a grayscale version
                # Ensure depth_norm is 2D for proper stacking
                if len(depth_norm.shape) != 2:
                    depth_norm = np.squeeze(depth_norm)
                depth_colored = np.stack([depth_norm, depth_norm, depth_norm], axis=-1)
            
            # Set invalid areas to black
            depth_colored[~valid_mask] = [0, 0, 0]
            
        else:
            # No valid depth, create black image
            depth_colored = np.zeros((depth.shape[0], depth.shape[1], 3))
        
        depth_images.append(depth_colored)
    
    # Process confidence maps
    confs_max = max([d.max() for d in confs]) if confs else 1.0
    confs = [cmap(d / max(confs_max, 1e-8)) for d in confs]

    imgs = []
    for i in range(len(rgbimg)):
        imgs.append(rgbimg[i])
        # Remove depth images - skip: imgs.append(depth_images[i])
        imgs.append(rgb(confs[i]))

    # Move EVERY tensor in the return to CPU before handing back to the main
    # process. ZeroGPU only attaches GPU inside this @GPU call; on return,
    # torch.multiprocessing tries to rebuild any CUDA tensor via CUDA IPC,
    # which triggers torch._C._cuda_init() in the main process (no GPU) and
    # raises "PyTorch CUDA emulation mode did not intercept a CUDA operation".
    #
    # nn.Module.cpu() only covers registered parameters/buffers; SparseGAState
    # (and its sparse_ga) hold raw tensor attributes that aren't registered.
    # torch.save/load(map_location="cpu") walks the full object graph and
    # guarantees everything lands on CPU.
    feats = _tensors_to_cpu(feats)
    scene = _tensors_to_cpu(scene)

    return scene, outfile, imgs, feats


def _tensors_to_cpu(obj):
    """Deep-copy `obj` while forcing every torch.Tensor AND every cached
    device descriptor (e.g. `sparse_ga.working_device`) onto CPU.

    `torch.load(map_location="cpu")` remaps tensor storage but does NOT rewrite
    Python attributes like `self.working_device = torch.device('cuda:0')`.
    Downstream methods (`sparse_ga.get_focals().to(self.working_device)`) then
    re-target CUDA from the main process, which has no GPU under ZeroGPU.
    """
    buf = io.BytesIO()
    torch.save(obj, buf)
    buf.seek(0)
    obj = torch.load(buf, map_location="cpu", weights_only=False)
    _patch_device_attrs_to_cpu(obj)
    return obj


def _patch_device_attrs_to_cpu(root):
    """Walk the object graph and replace any cached `cuda` device descriptor
    (torch.device or device-string) with a CPU equivalent."""
    seen = set()
    cpu_device = torch.device("cpu")
    stack = [root]
    while stack:
        obj = stack.pop()
        oid = id(obj)
        if oid in seen or obj is None:
            continue
        seen.add(oid)
        # Descend into container types
        if isinstance(obj, (list, tuple, set)):
            stack.extend(obj)
            continue
        if isinstance(obj, dict):
            stack.extend(obj.values())
            continue
        # Patch attributes on regular objects (skip torch tensors themselves)
        if torch.is_tensor(obj):
            continue
        if not hasattr(obj, "__dict__"):
            continue
        for k, v in list(obj.__dict__.items()):
            if isinstance(v, torch.device) and v.type == "cuda":
                setattr(obj, k, cpu_device)
            elif isinstance(v, str) and v.startswith("cuda"):
                # Only rewrite names that look like device fields
                if "device" in k.lower():
                    setattr(obj, k, "cpu")
            else:
                stack.append(v)


def set_scenegraph_options(inputfiles, win_cyclic, refid, scenegraph_type):
    num_files = len(inputfiles) if inputfiles is not None else 1
    show_win_controls = scenegraph_type in ["swin", "logwin"]
    show_winsize = scenegraph_type in ["swin", "logwin"]
    show_cyclic = scenegraph_type in ["swin", "logwin"]
    max_winsize, min_winsize = 1, 1
    if scenegraph_type == "swin":
        if win_cyclic:
            max_winsize = max(1, math.ceil((num_files - 1) / 2))
        else:
            max_winsize = num_files - 1
    elif scenegraph_type == "logwin":
        if win_cyclic:
            half_size = math.ceil((num_files - 1) / 2)
            max_winsize = max(1, math.ceil(math.log(half_size, 2)))
        else:
            max_winsize = max(1, math.ceil(math.log(num_files, 2)))
    winsize = gradio.Slider(
        label="Scene Graph: Window Size",
        value=max_winsize,
        minimum=min_winsize,
        maximum=max_winsize,
        step=1,
        visible=show_winsize,
    )
    win_cyclic = gradio.Checkbox(
        value=win_cyclic, label="Cyclic sequence", visible=show_cyclic
    )
    win_col = gradio.Column(visible=show_win_controls)
    refid = gradio.Slider(
        label="Scene Graph: Id",
        value=0,
        minimum=0,
        maximum=num_files - 1,
        step=1,
        visible=scenegraph_type == "oneref",
    )
    return win_col, winsize, win_cyclic, refid


def load_model(model, ckp_path, device):
    ckp = torch.load(ckp_path, map_location="cpu")
    if ckp_path.endswith(".pth"):
        model.load_state_dict(ckp["model"], strict=False)
    elif ckp_path.endswith(".pt"):
        model.load_state_dict(ckp["module"])
    else:
        raise ValueError(f"Unknown checkpoint format: {ckp_path}")
    model = model.to(device)


def enable_mast3r(args):
    args = args.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if "landscape_only" not in args:
        args = args[:-1] + ", landscape_only=False)"
    else:
        args = args.replace(" ", "").replace(
            "landscape_only=True", "landscape_only=False"
        )
    return args


def main_demo(
    tmpdirname,
    model,
    device,
    image_size,
    server_name,
    server_port,
    silent=False,
    share=False,
    gradio_delete_cache=False,
    checkpoint_dir=None,
):
    """
    Launch the SAB3R gradio demo.

    If `checkpoint_dir` is provided, the UI exposes a dropdown to switch between
    all `.pt` checkpoints found under its sub-directories (local dev workflow).
    Otherwise the single pre-loaded `model` is used (HF Spaces / single-checkpoint
    deployment).
    """
    if not silent:
        print("Outputing stuff in", tmpdirname)

    def load_checkpoint(ckpt_name, model):
        # Clear previous model from VRAM
        del model  # Remove current model instance to free memory
        torch.cuda.empty_cache()  # Clear unused VRAM

        if "+" in ckpt_name:
            model_config = "AsymmetricMASt3R(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='catmlp+dpt', output_mode='pts3d+desc24', clip_head_type='dpt', dino_head_type='dpt', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, two_confs=True)"
        else:
            model_config = "AsymmetricMASt3R(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='catmlp+dpt', output_mode='pts3d+desc24', clip_head_type='dpt', dino_head_type=None, depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, two_confs=True)"
        model_config = enable_mast3r(model_config)

        model = eval(model_config)

        if ckpt_name:
            ckp_path = os.path.join(checkpoint_dir, ckpt_name, ckpt_name + ".pt")
            load_model(model, ckp_path, device)
            return f"Checkpoint '{ckpt_name}' loaded successfully."
        else:
            return f"Checkpoint '{ckpt_name}' not found."

    def find_checkpoints(checkpoint_dir):
        checkpoints = []
        for subdir in os.listdir(checkpoint_dir):
            full_path = os.path.join(checkpoint_dir, subdir)
            if os.path.isdir(full_path) and any(
                fname.endswith(".pt") for fname in os.listdir(full_path)
            ):
                checkpoints.append(subdir)
        return checkpoints

    # Publish the model as a module-level singleton so @GPU handlers
    # can fetch it without it being pickled across ZeroGPU's worker boundary
    # (see _SAB3R_MODEL above).
    global _SAB3R_MODEL
    _SAB3R_MODEL = model

    recon_fun = functools.partial(
        get_reconstructed_scene,
        tmpdirname,
        gradio_delete_cache,
        device,
        silent,
        image_size,
    )
    model_from_scene_fun = functools.partial(get_3D_model_from_scene, silent)

    def get_context(delete_cache):
        css = """.gradio-container {margin: 0 !important; min-width: 100%};"""
        title = "SAB3R: Semantic-Augmented Backbone in 3D Reconstruction"
        if delete_cache:
            return gradio.Blocks(
                css=css, title=title, delete_cache=(delete_cache, delete_cache)
            )
        else:
            return gradio.Blocks(
                css=css, title="SAB3R: Semantic-Augmented Backbone in 3D Reconstruction"
            )  # for compatibility with older versions

    with get_context(gradio_delete_cache) as demo:
        # scene state is save so that you can change conf_thr, cam_size... without rerunning the inference
        ckpt_options = find_checkpoints(checkpoint_dir) if checkpoint_dir else []
        scene = gradio.State(None)
        feats = gradio.State(None)
        query_only = gradio.State(True)
        gradio.HTML('<h2 style="text-align: center;">SAB3R: Semantic-Augmented Backbone in 3D Reconstruction</h2>')
        gradio.HTML('''
        <div style="text-align: center; margin-bottom: 10px;">
            <div style="display: inline-flex; gap: 5px; flex-wrap: wrap; justify-content: center;">
                <a href="https://uva-computer-vision-lab.github.io/sab3r/" target="_blank" 
                   style="display: inline-flex; align-items: center; padding: 8px 16px; background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px; text-decoration: none; color: #495057; font-size: 14px;">
                    <span style="margin-right: 8px;">🌐</span>
                    <span>Website</span>
                </a>
                <a href="https://github.com/UVA-Computer-Vision-Lab/sab-3r" target="_blank"
                   style="display: inline-flex; align-items: center; padding: 8px 16px; background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px; text-decoration: none; color: #495057; font-size: 14px;">
                    <span style="margin-right: 8px;">⚡</span>
                    <span>GitHub</span>
                </a>
                <a href="https://www.arxiv.org/abs/2506.02112" target="_blank"
                   style="display: inline-flex; align-items: center; padding: 8px 16px; background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px; text-decoration: none; color: #495057; font-size: 14px;">
                    <span style="margin-right: 8px;">📑</span>
                    <span>Paper</span>
                </a>
            </div>
        </div>
        ''')
        
        gradio.HTML('''
        <div style="background-color: #e8f4fd; border: 1px solid #bee5eb; border-radius: 8px; padding: 16px; margin-bottom: 20px;">
            <h3 style="margin: 0 0 12px 0; color: #0c5460; font-size: 16px;">📋 How to Use</h3>
            <ol style="margin: 0; padding-left: 20px; color: #0c5460; line-height: 1.6;">
                <li><strong>Load Images:</strong> Upload multiple images using the file selector below</li>
                <li><strong>Reconstruct:</strong> Click "Run" to perform 3D reconstruction and generate semantic features</li>
                <li><strong>Query:</strong> Enter a text description (e.g., "office chair") and click "Query" to segment objects</li>
            </ol>
            <p style="margin: 12px 0 0 0; font-size: 14px; color: #6c757d; font-style: italic;">
                💡 Note: Reconstruction generates reusable semantic features, so querying with different text prompts doesn't require reconstruction again.
            </p>
        </div>
        ''')
        with gradio.Column():
            if ckpt_options:
                ckpt_dropdown = gradio.Dropdown(
                    choices=ckpt_options, label="Select Checkpoint"
                )
                ckpt_dropdown.change(load_checkpoint, inputs=ckpt_dropdown, outputs=None)

            inputfiles = gradio.File(file_count="multiple")
            with gradio.Row():
                with gradio.Column():
                    with gradio.Row():
                        lr1 = gradio.Slider(
                            label="Coarse LR",
                            value=0.07,
                            minimum=0.01,
                            maximum=0.2,
                            step=0.01,
                        )
                        niter1 = gradio.Number(
                            value=500,
                            precision=0,
                            minimum=0,
                            maximum=10_000,
                            label="num_iterations",
                            info="For coarse alignment!",
                        )
                        lr2 = gradio.Slider(
                            label="Fine LR",
                            value=0.014,
                            minimum=0.005,
                            maximum=0.05,
                            step=0.001,
                        )
                        niter2 = gradio.Number(
                            value=200,
                            precision=0,
                            minimum=0,
                            maximum=100_000,
                            label="num_iterations",
                            info="For refinement!",
                        )
                        optim_level = gradio.Dropdown(
                            ["coarse", "refine", "refine+depth"],
                            value="refine",
                            label="OptLevel",
                            info="Optimization level",
                        )
                    with gradio.Row():
                        matching_conf_thr = gradio.Slider(
                            label="Matching Confidence Thr",
                            value=5.0,
                            minimum=0.0,
                            maximum=30.0,
                            step=0.1,
                            info="Before Fallback to Regr3D!",
                        )
                        shared_intrinsics = gradio.Checkbox(
                            value=False,
                            label="Shared intrinsics",
                            info="Only optimize one set of intrinsics for all views",
                        )
                        scenegraph_type = gradio.Dropdown(
                            [
                                ("complete: all possible image pairs", "complete"),
                                ("swin: sliding window", "swin"),
                                ("logwin: sliding window with long range", "logwin"),
                                ("oneref: match one image with all", "oneref"),
                            ],
                            value="complete",
                            label="Scenegraph",
                            info="Define how to make pairs",
                            interactive=True,
                        )
                        with gradio.Column(visible=False) as win_col:
                            winsize = gradio.Slider(
                                label="Scene Graph: Window Size",
                                value=1,
                                minimum=1,
                                maximum=1,
                                step=1,
                            )
                            win_cyclic = gradio.Checkbox(
                                value=False, label="Cyclic sequence"
                            )
                        refid = gradio.Slider(
                            label="Scene Graph: Id",
                            value=0,
                            minimum=0,
                            maximum=0,
                            step=1,
                            visible=False,
                        )
            run_btn = gradio.Button("Run")

            with gradio.Row():
                # adjust the confidence threshold
                min_conf_thr = gradio.Slider(
                    label="min_conf_thr", value=1.5, minimum=0.0, maximum=10, step=0.1
                )
                # adjust the camera size in the output pointcloud
                cam_size = gradio.Slider(
                    label="cam_size", value=0.2, minimum=0.001, maximum=1.0, step=0.001
                )
                TSDF_thresh = gradio.Slider(
                    label="TSDF Threshold",
                    value=0.0,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                )
            with gradio.Row():
                as_pointcloud = gradio.Checkbox(value=True, label="As pointcloud")
                # two post process implemented
                mask_sky = gradio.Checkbox(value=False, label="Mask sky")
                clean_depth = gradio.Checkbox(value=True, label="Clean-up depthmaps")
                transparent_cams = gradio.Checkbox(
                    value=False, label="Transparent cameras"
                )

            with gradio.Row():
                text = gradio.Textbox(label="Text query:", value="")
                with gradio.Column():
                    query_btn = gradio.Button("Query")
                    clear_btn = gradio.Button("Clear Query")

            with gradio.Row():
                # adjust the query percentage threshold
                thd_per = gradio.Slider(
                    label="percentage_threshold",
                    value=0.90,
                    minimum=0,
                    maximum=1,
                    step=0.01,
                )
                # adjust the query value threshold
                thd_val = gradio.Slider(
                    label="value_threshold", value=0.75, minimum=0, maximum=1, step=0.01
                )

            with gradio.Row():
                outmodel = gradio.Model3D()
                outmodel_query = gradio.Model3D()

            with gradio.Row():
                outgallery = gradio.Gallery(label="rgb,confidence", columns=2)
                outgalleryquery = gradio.Gallery(label="rgb,semantics", columns=2)

            # events
            scenegraph_type.change(
                set_scenegraph_options,
                inputs=[inputfiles, win_cyclic, refid, scenegraph_type],
                outputs=[win_col, winsize, win_cyclic, refid],
            )
            inputfiles.change(
                set_scenegraph_options,
                inputs=[inputfiles, win_cyclic, refid, scenegraph_type],
                outputs=[win_col, winsize, win_cyclic, refid],
            )
            win_cyclic.change(
                set_scenegraph_options,
                inputs=[inputfiles, win_cyclic, refid, scenegraph_type],
                outputs=[win_col, winsize, win_cyclic, refid],
            )
            run_btn.click(
                fn=recon_fun,
                inputs=[
                    scene,
                    inputfiles,
                    optim_level,
                    lr1,
                    niter1,
                    lr2,
                    niter2,
                    min_conf_thr,
                    matching_conf_thr,
                    as_pointcloud,
                    mask_sky,
                    clean_depth,
                    transparent_cams,
                    cam_size,
                    scenegraph_type,
                    winsize,
                    win_cyclic,
                    refid,
                    TSDF_thresh,
                    shared_intrinsics,
                ],
                outputs=[scene, outmodel, outgallery, feats],
            )
            query_btn.click(
                fn=model_from_scene_fun,
                inputs=[
                    scene,
                    min_conf_thr,
                    as_pointcloud,
                    mask_sky,
                    clean_depth,
                    transparent_cams,
                    cam_size,
                    TSDF_thresh,
                    feats,
                    text,
                    query_only,
                    thd_per,
                    thd_val,
                ],
                outputs=[outmodel_query, outgalleryquery],
            )
            
            # Clear query functionality
            def clear_query():
                return "", None, None
            
            clear_btn.click(
                fn=clear_query,
                inputs=[],
                outputs=[text, outmodel_query, outgalleryquery],
            )
            min_conf_thr.release(
                fn=model_from_scene_fun,
                inputs=[
                    scene,
                    min_conf_thr,
                    as_pointcloud,
                    mask_sky,
                    clean_depth,
                    transparent_cams,
                    cam_size,
                    TSDF_thresh,
                ],
                outputs=outmodel,
            )
            cam_size.change(
                fn=model_from_scene_fun,
                inputs=[
                    scene,
                    min_conf_thr,
                    as_pointcloud,
                    mask_sky,
                    clean_depth,
                    transparent_cams,
                    cam_size,
                    TSDF_thresh,
                ],
                outputs=outmodel,
            )
            TSDF_thresh.change(
                fn=model_from_scene_fun,
                inputs=[
                    scene,
                    min_conf_thr,
                    as_pointcloud,
                    mask_sky,
                    clean_depth,
                    transparent_cams,
                    cam_size,
                    TSDF_thresh,
                ],
                outputs=outmodel,
            )
            as_pointcloud.change(
                fn=model_from_scene_fun,
                inputs=[
                    scene,
                    min_conf_thr,
                    as_pointcloud,
                    mask_sky,
                    clean_depth,
                    transparent_cams,
                    cam_size,
                    TSDF_thresh,
                ],
                outputs=outmodel,
            )
            mask_sky.change(
                fn=model_from_scene_fun,
                inputs=[
                    scene,
                    min_conf_thr,
                    as_pointcloud,
                    mask_sky,
                    clean_depth,
                    transparent_cams,
                    cam_size,
                    TSDF_thresh,
                ],
                outputs=outmodel,
            )
            clean_depth.change(
                fn=model_from_scene_fun,
                inputs=[
                    scene,
                    min_conf_thr,
                    as_pointcloud,
                    mask_sky,
                    clean_depth,
                    transparent_cams,
                    cam_size,
                    TSDF_thresh,
                ],
                outputs=outmodel,
            )
            transparent_cams.change(
                model_from_scene_fun,
                inputs=[
                    scene,
                    min_conf_thr,
                    as_pointcloud,
                    mask_sky,
                    clean_depth,
                    transparent_cams,
                    cam_size,
                    TSDF_thresh,
                ],
                outputs=outmodel,
            )
    demo.launch(share=share, server_name=server_name, server_port=server_port)
