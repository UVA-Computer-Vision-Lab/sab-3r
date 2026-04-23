# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# post process function for all heads: extract 3D points/confidence from output
# --------------------------------------------------------
import torch


def postprocess(out, depth_mode, conf_mode):
    """
    extract 3D points/confidence from prediction head output
    """
    fmap = out.permute(0, 2, 3, 1)  # B,H,W,3
    res = dict(pts3d=reg_dense_depth(fmap[:, :, :, 0:3], mode=depth_mode))

    if conf_mode is not None:
        res['conf'] = reg_dense_conf(fmap[:, :, :, 3], mode=conf_mode)
    return res


def reg_dense_depth(xyz, mode):
    """
    extract 3D points from prediction head output
    """
    mode, vmin, vmax = mode

    no_bounds = (vmin == -float('inf')) and (vmax == float('inf'))
    assert no_bounds

    if mode == 'linear':
        if no_bounds:
            return xyz  # [-inf, +inf]
        return xyz.clip(min=vmin, max=vmax)

    # distance to origin
    d = xyz.norm(dim=-1, keepdim=True)
    xyz = xyz / d.clip(min=1e-8)

    if mode == 'square':
        return xyz * d.square()

    if mode == 'exp':
        return xyz * torch.expm1(d)

    raise ValueError(f'bad {mode=}')


def reg_dense_conf(x, mode):
    """
    extract confidence from prediction head output
    """
    mode, vmin, vmax = mode
    if mode == 'exp':
        return vmin + x.exp().clip(max=vmax-vmin)
    if mode == 'sigmoid':
        return (vmax - vmin) * torch.sigmoid(x) + vmin
    raise ValueError(f'bad {mode=}')

def reg_dense_feat(clip_features, mode):
    """
    Apply transformations to CLIP features based on the mode.
    """
    mode, vmin, vmax = mode

    # Check for bounds
    no_bounds = (vmin == -float('inf')) and (vmax == float('inf'))

    if mode == 'sigmoid':
        return clip_features * torch.sigmoid(clip_features)
    
    if mode == 'linear':
        if no_bounds:
            return clip_features  # [-inf, +inf]
        return clip_features.clamp(min=vmin, max=vmax)

    # Compute norm of the clip features
    d = clip_features.norm(dim=-1, keepdim=True)

    # Normalize clip_features by their norm
    clip_features = clip_features / d.clamp(min=1e-8)

    if mode == 'square':
        return clip_features * d.square()

    if mode == 'exp':
        return clip_features * torch.expm1(d)

    raise ValueError(f'Unsupported mode: {mode}')