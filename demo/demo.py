#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# SAB3R gradio demo executable
# --------------------------------------------------------
import os
import sys
import argparse
import tempfile
from contextlib import nullcontext

# Allow `python demo/demo.py` from the repo root to find the sibling
# mast3r/ and dust3r/ packages.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
from huggingface_hub import hf_hub_download

from mast3r.demo import get_args_parser as sab3r_get_args_parser, main_demo
from mast3r.model import AsymmetricMASt3R  # noqa: F401  (referenced via eval() below)
from mast3r.utils.misc import hash_md5

import mast3r.utils.path_to_dust3r  # noqa: F401  (side-effect: puts vendored dust3r on sys.path)
from dust3r.demo import set_print_with_timestamp

import matplotlib.pyplot as pl
pl.ion()

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

inf = float("inf")

DEFAULT_MODEL_REPO = "uva-cv-lab/SAB3R"
DEFAULT_CKPT_FILENAME = "demo_ckpt/base/base.pt"


def get_args_parser():
    parser = sab3r_get_args_parser()
    parser.add_argument(
        "--model_repo",
        default=os.environ.get("SAB3R_MODEL_REPO", DEFAULT_MODEL_REPO),
        help="Hugging Face Hub repo id hosting the SAB3R checkpoint "
             "(used only when --weights is not provided).",
    )
    parser.add_argument(
        "--ckpt_filename",
        default=os.environ.get("SAB3R_CKPT_FILENAME", DEFAULT_CKPT_FILENAME),
        help="Checkpoint filename inside --model_repo.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default=os.environ.get("SAB3R_CHECKPOINT_DIR", None),
        help="Optional local directory containing one sub-directory per "
             "checkpoint (each sub-dir must hold `<name>.pt`). When provided, "
             "the UI exposes a dropdown to switch between them. Useful for "
             "local dev; leave unset for single-checkpoint HF Spaces deployments.",
    )
    return parser


def load_weights(model, ckp_path, device):
    ckp = torch.load(ckp_path, map_location='cpu')
    if ckp_path.endswith('.pth'):
        model.load_state_dict(ckp['model'], strict=False)
    elif ckp_path.endswith('.pt'):
        model.load_state_dict(ckp['module'])
    else:
        raise ValueError(f"Unknown checkpoint format: {ckp_path}")
    model.to(device)


def build_model_config():
    return (
        "AsymmetricMASt3R(pos_embed='RoPE100', patch_embed_cls='PatchEmbedDust3R', "
        "img_size=(512, 512), head_type='catmlp+dpt', output_mode='pts3d+desc24', "
        "clip_head_type='dpt', dino_head_type='dpt', "
        "depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), "
        "enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, "
        "dec_embed_dim=768, dec_depth=12, dec_num_heads=12, "
        "two_confs=True, landscape_only=False)"
    )


def resolve_weights_path(args: argparse.Namespace) -> str:
    if args.weights:
        return args.weights
    print(f"[sab3r] Downloading checkpoint from HF Hub: {args.model_repo}/{args.ckpt_filename}")
    return hf_hub_download(repo_id=args.model_repo, filename=args.ckpt_filename)


def main(argv=None):
    parser = get_args_parser()
    args = parser.parse_args(argv)
    set_print_with_timestamp()

    if args.server_name is not None:
        server_name = args.server_name
    else:
        server_name = '0.0.0.0' if args.local_network else '127.0.0.1'

    model = eval(build_model_config())
    ckp_path = resolve_weights_path(args)
    load_weights(model, ckp_path, args.device)
    chkpt_tag = hash_md5(ckp_path)

    def get_context(tmp_dir):
        return (tempfile.TemporaryDirectory(suffix='_sab3r_gradio_demo') if tmp_dir is None
                else nullcontext(tmp_dir))

    with get_context(args.tmp_dir) as tmpdirname:
        cache_path = os.path.join(tmpdirname, chkpt_tag)
        os.makedirs(cache_path, exist_ok=True)
        main_demo(
            cache_path, model, args.device, args.image_size, server_name, args.server_port,
            silent=args.silent, share=args.share, gradio_delete_cache=args.gradio_delete_cache,
            checkpoint_dir=args.checkpoint_dir,
        )


if __name__ == '__main__':
    main()
