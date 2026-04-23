# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# training code for DUSt3R
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import sys
import time
import math
from collections import defaultdict
from pathlib import Path
from typing import Sized

import torch
import torch.backends.cudnn as cudnn
# from torch.utils.tensorboard import SummaryWrfiter

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

from dust3r.model import (
    AsymmetricCroCo3DStereo,
    inf,
)  # noqa: F401, needed when loading the model
from dust3r.datasets import get_data_loader  # noqa
from dust3r.losses import *  # noqa: F401, needed when loading the model
from dust3r.inference import loss_of_one_batch  # noqa

import dust3r.utils.path_to_croco  # noqa: F401
import croco.utils.misc as misc  # noqa
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler  # noqa

import matplotlib
matplotlib.use('Agg') # weird bug in 4+ GPU
import matplotlib.pyplot as plt
from featup.util import norm, unnorm
from featup.util import pca, remove_axes

import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import deepspeed

@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="training_config",
)
def train(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    misc.init_distributed_mode(cfg)
    global_rank = misc.get_rank()
    world_size = misc.get_world_size()

    print("output_dir: " + cfg.output_dir)

    if cfg.output_dir:
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    print("global_rank: ", global_rank)
    print("world_size: ", world_size)
    if global_rank == 0:

        time_suffix = time.strftime("%Y%m%d-%H%M%S")
        wandb.init(
            project=cfg.wandb.project_name,
            entity=cfg.wandb.entity,
            config=OmegaConf.to_container(cfg),
            group=cfg.wandb.group,
            dir=cfg.output_dir
        )
        wandb.run.name = os.path.basename(cfg.output_dir.strip("/")) + "_" + time_suffix
        os.environ["WANDB_RUN_GROUP"] = "experiment-" + wandb.util.generate_id()

    # auto resume
    last_ckpt_fname = os.path.join(cfg.output_dir, f"checkpoint-last.pth")
    cfg.resume = last_ckpt_fname if os.path.isfile(last_ckpt_fname) else None

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # fix the seed
    seed = cfg.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = not cfg.disable_cudnn_benchmark

    with open(cfg.deepspeed_config) as json_data:
        ds_config = json.loads(json_data.read())
    batch_size = ds_config["train_micro_batch_size_per_gpu"]
    accum_iter = ds_config["gradient_accumulation_steps"]
    # training dataset and loader
    print("Building train dataset {:s}".format(cfg.dataset.train))
    #  dataset and loader
    data_loader_train = build_dataset(
        cfg.dataset.train, batch_size, cfg.num_workers, test=False
    )
    # print("Building test dataset {:s}".format(cfg.dataset.train))
    # data_loader_test = {
    #     dataset.split("(")[0]: build_dataset(
    #         dataset, batch_size, cfg.num_workers, test=True
    #     )
    #     for dataset in cfg.dataset.test.split("+")
    # }

    # model
    print("Loading model: {:s}".format(cfg.model.name))
    model = eval(cfg.model.name)
    print(f">> Creating train criterion = {cfg.train_criterion}")
    train_criterion = eval(cfg.train_criterion).to(device)
    # print(f">> Creating test criterion = {cfg.test_criterion or cfg.train_criterion}")
    # test_criterion = eval(cfg.test_criterion or cfg.criterion).to(device)

    model.to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    if cfg.model.pretrained and not cfg.resume:
        print("Loading pretrained: ", cfg.model.pretrained)
        ckpt = torch.load(cfg.model.pretrained, map_location=device)
        
        # Filter out mismatched keys
        model_state_dict = model.state_dict()
        loaded_state_dict = ckpt["model"]
        filtered_state_dict = {k: v for k, v in loaded_state_dict.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}
        # Update the model state dict
        model_state_dict.update(filtered_state_dict)
        # Load the updated state dict
        print(model.load_state_dict(model_state_dict, strict=False))
        # for name, param in model.named_parameters():
        #     if not (name.startswith('downstream_head1') or name.startswith('downstream_head2') or name.startswith('dec_blocks')):
        #         param.requires_grad = False
        
        del ckpt  # in case it occupies memory

    eff_batch_size = (
        batch_size * accum_iter * misc.get_world_size()
    )
    # if cfg.training.lr is None:  # only base_lr is specified
    #     cfg.training.lr = cfg.training.blr * eff_batch_size / 256
    
    # print("base lr: %.2e" % (cfg.training.lr * 256 / eff_batch_size))
    # print("actual lr: %.2e" % cfg.training.lr)
    print("accumulate grad iterations: %d" % accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # if cfg.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(
    #         model, device_ids=[cfg.gpu], find_unused_parameters=True, static_graph=True
    #     )
    #     model_without_ddp = model.module
    model, optimizer, _, _ = deepspeed.initialize(
        model=model, model_parameters=model.parameters(), config=cfg.deepspeed_config
    )

    # following timm: set wd as 0 for bias and norm layers
    # param_groups = misc.get_parameter_groups(
    #     model_without_ddp, cfg.training.weight_decay
    # )
    # optimizer = torch.optim.AdamW(param_groups, lr=cfg.training.lr, betas=(0.9, 0.95))
    # print(optimizer)
    #loss_scaler = NativeScaler()

    def write_log_stats(epoch, train_stats, test_stats):
        if misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()

            log_stats = dict(
                epoch=epoch, **{f"train_{k}": v for k, v in train_stats.items()}
            )
            # for test_name in data_loader_test:
            #     if test_name not in test_stats:
            #         continue
            #     log_stats.update(
            #         {test_name + "_" + k: v for k, v in test_stats[test_name].items()}
            #     )

            with open(
                os.path.join(cfg.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

    def save_model(model, epoch, fname, best_so_far):
        output_dir = Path(cfg.output_dir)
        if fname is None: fname = str(epoch)
        checkpoint_path = output_dir / ('checkpoint-epoch:%s.pth' % fname)
        model.save_checkpoint(checkpoint_path)
        # misc.save_model(
        #     args=cfg,
        #     model_without_ddp=model_without_ddp,
        #     optimizer=optimizer,
        #     epoch=epoch,
        #     fname=fname,
        #     best_so_far=best_so_far,
        # )

    def save_model_torch(model, epoch, fname, best_so_far):
        modelfs = model.module if hasattr(model, 'module') else model  # Handle wrapped models
        output_dir = Path(cfg.output_dir)
        if fname is None: fname = str(epoch)
        checkpoint_path = output_dir / ('checkpoint-%s.pth' % fname)
        torch.save(modelfs.state_dict(), checkpoint_path)

    best_so_far = misc.load_model(
        args=cfg,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
    )
    if best_so_far is None:
        best_so_far = float("inf")
    # if global_rank == 0 and cfg.output_dir is not None:
    #     log_writer = SummaryWriter(log_dir=cfg.output_dir)
    # else:
    log_writer = None

    print(f"Start training for {cfg.training.epochs} epochs")
    start_time = time.time()
    train_stats = test_stats = {}
    for epoch in range(cfg.start_epoch, cfg.training.epochs + 1):

        # Save immediately the last checkpoint
        if epoch > cfg.start_epoch:
            if (
                cfg.saving.save_freq
                and epoch % cfg.saving.save_freq == 0
                or epoch == cfg.training.epochs
            ):
                save_model(model, epoch - 1, "last", best_so_far)

        # Test on multiple datasets
        new_best = False
        # if epoch > 0 and cfg.saving.eval_freq > 0 and epoch % cfg.saving.eval_freq == 0:
        #     test_stats = {}
        #     for test_name, testset in data_loader_test.items():
        #         stats = test_one_epoch(
        #             model,
        #             test_criterion,
        #             testset,
        #             device,
        #             epoch,
        #             log_writer=log_writer,
        #             args=cfg,
        #             prefix=test_name,
        #         )
        #         test_stats[test_name] = stats

        #         # Save best of all
        #         if stats["loss_med"] < best_so_far:
        #             best_so_far = stats["loss_med"]
        #             new_best = True

        # Save more stuff
        write_log_stats(epoch, train_stats, test_stats)

        if epoch > cfg.start_epoch:
            if cfg.saving.keep_freq and epoch % cfg.saving.keep_freq == 0:
                save_model(model, epoch - 1, str(epoch), best_so_far)
            if new_best:
                #save_model(model, epoch - 1, "best", best_so_far)
                if misc.is_main_process():
                    save_model_torch(model, epoch - 1, "best", best_so_far)
        if epoch >= cfg.training.epochs:
            break  # exit after writing last test to disk

        # Train
        train_stats = train_one_epoch(
            model,
            train_criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            log_writer=log_writer,
            args=cfg,
            accum_iter = accum_iter,
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))

    save_final_model(
        model, cfg, cfg.training.epochs, model_without_ddp, best_so_far=best_so_far
    )
    wandb.finish()


def save_final_model(model, args, epoch, model_without_ddp, best_so_far=None):
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / "checkpoint-final.pth"
    model.save_checkpoint(checkpoint_path)
    # to_save = {
    #     "args": OmegaConf.to_container(args),
    #     "model": (
    #         model_without_ddp
    #         if isinstance(model_without_ddp, dict)
    #         else model_without_ddp.cpu().state_dict()
    #     ),
    #     "epoch": epoch,
    # }
    # if best_so_far is not None:
    #     to_save["best_so_far"] = best_so_far
    # print(f">> Saving model to {checkpoint_path} ...")
    # misc.save_on_master(to_save, checkpoint_path)


def build_dataset(dataset, batch_size, num_workers, test=False):
    split = ["Train", "Test"][test]
    print(f"Building {split} Data loader for dataset: ", dataset)
    loader = get_data_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_mem=True,
        shuffle=not (test),
        drop_last=not (test),
    )

    print(f"{split} dataset length: ", len(loader))
    return loader

def make_depth_image(
    depths: torch.Tensor,
    masks: torch.Tensor,
    max_quantile: float = 0.98,
    min_quantile: float = 0.02,
    min_out_depth: float = 0.1,
    max_out_depth: float = 0.9,
) -> torch.Tensor:
    """
    Convert a batch of depth maps to a grayscale image.

    Args:
        depths: A tensor of shape `(B, 1, H, W)` containing a batch of depth maps.
        masks: A tensor of shape `(B, 1, H, W)` containing a batch of foreground masks.
        max_quantile: The quantile of the input depth values which will
            be mapped to `max_out_depth`.
        min_quantile: The quantile of the input depth values which will
            be mapped to `min_out_depth`.
        min_out_depth: The minimal value in each depth map will be assigned this color.
        max_out_depth: The maximal value in each depth map will be assigned this color.

    Returns:
        depth_image: A tensor of shape `(B, 1, H, W)` a batch of grayscale
            depth images.
    """
    depths = depths.detach().cpu().unsqueeze(0).unsqueeze(0).contiguous()
    masks = masks.detach().cpu().unsqueeze(0).unsqueeze(0).contiguous()
    normfacs = []
    for d, m in zip(depths, masks):
        ok = (d.view(-1) > 1e-6) * (m.view(-1) > 0.5)
        if ok.sum() <= 1:
            normfacs.append(torch.zeros(2).type_as(depths))
            continue
        dok = d.view(-1)[ok].view(-1)
        _maxk = max(int(round((1 - max_quantile) * (dok.numel()))), 1)
        _mink = max(int(round(min_quantile * (dok.numel()))), 1)
        normfac_max = dok.topk(k=_maxk, dim=-1).values[-1]
        normfac_min = dok.topk(k=_mink, dim=-1, largest=False).values[-1]
        normfacs.append(torch.stack([normfac_min, normfac_max]))
    normfacs = torch.stack(normfacs)
    _min, _max = (normfacs[:, 0].view(-1, 1, 1, 1), normfacs[:, 1].view(-1, 1, 1, 1))
    depths = (depths - _min) / (_max - _min).clamp(1e-4)
    depths = (
        (depths * (max_out_depth - min_out_depth) + min_out_depth) * masks.float()
    ).clamp(0.0, 1.0)
    return depths

def visualize_depth(pred, target, colormap="inferno"):
    # Convert tensors to numpy arrays
    pred = pred.squeeze().squeeze().detach().cpu().numpy()
    target = target.squeeze().squeeze().detach().cpu().numpy()

    # Apply a color map (e.g., 'viridis') to the predictions and targets
    pred_colored = plt.get_cmap(colormap)(pred / np.max(pred))[
        :, :, :3
    ]  # Only take RGB values
    target_colored = plt.get_cmap(colormap)(target / np.max(target))[:, :, :3]

    return pred_colored, target_colored

def log_image_origin_feats_distilled_feats_to_wandb(
    loss_tuple, loss_details, train=True
):
    if misc.get_rank() == 0:
        image = loss_tuple["view1"]["img_orig"]
        has_clip = "gt1_clip_orig" in loss_details
        has_dino = "gt1_dino_orig" in loss_details
        if has_clip:
            lr_clip = loss_details["gt1_clip_orig"][0]
            hr_clip = loss_tuple["pred1"]["clip"][0]
            hr_clip = hr_clip.permute(2, 0, 1).unsqueeze(0)
        if has_dino:
            lr_dino = loss_details["gt1_dino_orig"][0]
            hr_dino = loss_tuple["pred1"]["dino"][0]
            hr_dino = hr_dino.permute(2, 0, 1).unsqueeze(0)
        if image[0].shape[-1] != 3:
            image = image[0].permute(0, 2, 1).detach().cpu()
        else:
            image = image[0].detach().cpu()
        if has_clip: [lr_clip_feats_pca, hr_clip_feats_pca], _ = pca([lr_clip.unsqueeze(0), hr_clip])
        if has_dino: [lr_dino_feats_pca, hr_dino_feats_pca], _ = pca([lr_dino.unsqueeze(0), hr_dino])
        fig, ax = plt.subplots(1, 7, figsize=(35, 5))
        ax[0].imshow(image)
        ax[0].set_title("Original Unscaled Image (Might be bit different)")
        
        depth_vis_gt = make_depth_image(loss_tuple['view1']['depthmap'][0], loss_tuple['view1']['valid_mask'][0])
        depth_vis_pred = make_depth_image(loss_tuple['pred1']['pts3d'][0, : , : , 2], loss_tuple['view1']['valid_mask'][0])
        depth_vis_gt, depth_vis_pred =  visualize_depth(depth_vis_pred, depth_vis_gt)
        ax[1].imshow(depth_vis_gt)
        ax[1].set_title("Original Trained Depth Features")
        ax[2].imshow(depth_vis_pred)
        ax[2].set_title("Predicted Depth Features")
        if has_clip: 
            ax[3].imshow(lr_clip_feats_pca[0].permute(1, 2, 0))
            ax[3].set_title("Original Trained Clip Features")
            ax[4].imshow(hr_clip_feats_pca[0].permute(1, 2, 0))
            ax[4].set_title("Predicted Clip Features")
        if has_dino:
            ax[5].imshow(lr_dino_feats_pca[0].permute(1, 2, 0))
            ax[5].set_title("Original Trained Dino Features")
            ax[6].imshow(hr_dino_feats_pca[0].permute(1, 2, 0))
            ax[6].set_title("Predicted Dino Features")
        remove_axes(ax)

        if train:
            wandb.log({"training visualization": wandb.Image(fig)})
        else:
            wandb.log({"test visualization": wandb.Image(fig)})

        plt.close(fig)

def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Sized,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args,
    log_writer=None,
    accum_iter = 1,
):
    assert torch.backends.cuda.matmul.allow_tf32 == True

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, "sampler") and hasattr(data_loader.sampler, "set_epoch"):
        data_loader.sampler.set_epoch(epoch)

    optimizer.zero_grad()

    for data_iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, args.training.print_freq, header)
    ):
        epoch_f = epoch + data_iter_step / len(data_loader)

        # we use a per iteration (instead of per epoch) lr scheduler
        # change to deepspeed
        # if data_iter_step % accum_iter == 0:
        #     misc.adjust_learning_rate(optimizer, epoch_f, args)

        loss_tuple = loss_of_one_batch(
            batch,
            model,
            criterion,
            device,
            symmetrize_batch=True,
            use_amp=bool(args.amp),
        )
        loss, loss_details = loss_tuple["loss"]  # criterion returns two values
        loss_value = float(loss)
        if (data_iter_step + 1) % args.saving.plot_steps == 0:
            log_image_origin_feats_distilled_feats_to_wandb(
                loss_tuple=loss_tuple, loss_details=loss_details, train=True
            )

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), force=True)
            sys.exit(1)

        # loss /= accum_iter # Looks not good
        # loss_scaler(
        #     loss,
        #     optimizer,
        #     parameters=model.parameters(),
        #     update_grad=(data_iter_step + 1) % accum_iter == 0,
        # )
        # if (data_iter_step + 1) % accum_iter == 0:
        #     optimizer.zero_grad()
        model.backward(loss)
        grad_norm = model.get_global_grad_norm()
        model.step()

        del loss
        del batch
        torch.cuda.empty_cache()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(epoch=epoch_f)
        metric_logger.update(lr=lr)
        if "gt1_clip_orig" in loss_details:
            del loss_details["gt1_clip_orig"]
            del loss_details["gt2_clip_orig"]
            loss_details.pop("gt1_clip_orig", None)
            loss_details.pop("gt2_clip_orig", None)
        if "gt1_dino_orig" in loss_details:
            del loss_details["gt1_dino_orig"]
            del loss_details["gt2_dino_orig"]
            loss_details.pop("gt1_dino_orig", None)
            loss_details.pop("gt2_dino_orig", None)

        metric_logger.update(loss=loss_value, **loss_details)

        if misc.get_rank() == 0:
            wandb.log(
                {"data_iter_step": data_iter_step, "grad_norm": grad_norm}
            )
        if (data_iter_step + 1) % accum_iter == 0 and (
            (data_iter_step + 1) % (accum_iter * args.training.print_freq)
        ) == 0:
            loss_value_reduce = misc.all_reduce_mean(
                loss_value
            )  # MUST BE EXECUTED BY ALL NODES
            if misc.get_rank() == 0:
                wandb.log(
                {"train_loss": loss_value_reduce, "train_lr": lr, "epoch": epoch_f}
                )
            if log_writer is None:
                continue
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int(epoch_f * 1000)
            log_writer.add_scalar("train_loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("train_lr", lr, epoch_1000x)
            log_writer.add_scalar("train_iter", epoch_1000x, epoch_1000x)
            for name, val in loss_details.items():
                log_writer.add_scalar("train_" + name, val, epoch_1000x)
        
        def save_model(model, epoch, fname, best_so_far):
            output_dir = Path(args.output_dir)
            if fname is None: fname = str(epoch)
            checkpoint_path = output_dir / ('checkpoint-%s.pth' % fname)
            model.save_checkpoint(checkpoint_path)
        
        if (data_iter_step + 1) % args.saving.save_steps == 0:
            save_model(model, epoch - 1, "epoch:" + str(epoch) + "step:" + str(data_iter_step), None)

        del loss_tuple, loss_details
        torch.cuda.empty_cache()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def test_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Sized,
    device: torch.device,
    epoch: int,
    args,
    log_writer=None,
    prefix="test",
):

    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))
    header = "Test Epoch: [{}]".format(epoch)

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, "sampler") and hasattr(data_loader.sampler, "set_epoch"):
        data_loader.sampler.set_epoch(epoch)

    for data_iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, args.training.print_freq, header)
    ):
        loss_tuple = loss_of_one_batch(
            batch,
            model,
            criterion,
            device,
            symmetrize_batch=True,
            use_amp=bool(args.amp),
        )
        loss_value, loss_details = loss_tuple["loss"]  # criterion returns two values
        if (data_iter_step + 1) % 10 == 0:
            log_image_origin_feats_distilled_feats_to_wandb(
                loss_tuple=loss_tuple, loss_details=loss_details, train=False
            )
        if "gt1_clip_orig" in loss_details:
            loss_details.pop("gt1_clip_orig", None)
            loss_details.pop("gt2_clip_orig", None)
        if "gt1_dino_orig" in loss_details:
            loss_details.pop("gt1_dino_orig", None)
            loss_details.pop("gt2_dino_orig", None)
        metric_logger.update(loss=float(loss_value), **loss_details)

        del batch, loss_value, loss_details, loss_tuple
        torch.cuda.empty_cache()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    aggs = [("avg", "global_avg"), ("med", "median")]
    results = {
        f"{k}_{tag}": getattr(meter, attr)
        for k, meter in metric_logger.meters.items()
        for tag, attr in aggs
    }

    if log_writer is not None:
        for name, val in results.items():
            log_writer.add_scalar(prefix + "_" + name, val, 1000 * epoch)

    return results
