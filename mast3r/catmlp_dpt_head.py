# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# MASt3R heads
# --------------------------------------------------------
import torch
import torch.nn.functional as F

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.heads.postprocess import reg_dense_depth, reg_dense_conf, reg_dense_feat  # noqa
from dust3r.heads.dpt_head import PixelwiseTaskWithDPT, PixelwiseTaskWithDoubleDPT, DPTOutputAdapter_fix  # noqa
import dust3r.utils.path_to_croco  # noqa
from models.blocks import Mlp  # noqa

from dust3r.model import inf


def reg_desc(desc, mode):
    if 'norm' in mode:
        desc = desc / desc.norm(dim=-1, keepdim=True)
    else:
        raise ValueError(f"Unknown desc mode {mode}")
    return desc


def postprocess(out, depth_mode, conf_mode, desc_dim=None, desc_mode='norm', clip_mode = None, dino_mode = None, two_confs=False, desc_conf_mode=None):
    if desc_conf_mode is None:
        desc_conf_mode = conf_mode
    fmap = out.permute(0, 2, 3, 1)  # B,H,W,D
    res = dict(pts3d=reg_dense_depth(fmap[..., 0:3], mode=depth_mode))
    dimtotal = 3
    if conf_mode is not None:
        res['conf'] = reg_dense_conf(fmap[..., 3], mode=conf_mode)
        dimtotal += 1
    if desc_dim is not None:
        start = 3 + int(conf_mode is not None)
        res['desc'] = reg_desc(fmap[..., start:start + desc_dim], mode=desc_mode)
        dimtotal += desc_dim
        if two_confs:
            res['desc_conf'] = reg_dense_conf(fmap[..., start + desc_dim], mode=desc_conf_mode)
            dimtotal += 1
        else:
            res['desc_conf'] = res['conf'].clone()
    if clip_mode is not None:
        res['clip'] = reg_dense_feat(fmap[..., dimtotal:dimtotal+512], mode=clip_mode)
        dimtotal += 512
    if dino_mode is not None:
        res['dino'] = reg_dense_feat(fmap[..., dimtotal:dimtotal+384], mode=dino_mode)
    return res


class Cat_MLP_LocalFeatures_DPT_Pts3d(PixelwiseTaskWithDPT):
    """ Mixture between MLP and DPT head that outputs 3d points and local features (with MLP).
    The input for both heads is a concatenation of Encoder and Decoder outputs
    """

    def __init__(self, net, has_conf=False, local_feat_dim=16, hidden_dim_factor=4., hooks_idx=None, dim_tokens=None,
                 num_channels=1, postprocess=None, feature_dim=256, last_dim=32, depth_mode=None, conf_mode=None, head_type="regression", **kwargs):
        super().__init__(num_channels=num_channels, feature_dim=feature_dim, last_dim=last_dim, hooks_idx=hooks_idx,
                         dim_tokens=dim_tokens, depth_mode=depth_mode, postprocess=postprocess, conf_mode=conf_mode, head_type=head_type)
        self.local_feat_dim = local_feat_dim

        patch_size = net.patch_embed.patch_size
        if isinstance(patch_size, tuple):
            assert len(patch_size) == 2 and isinstance(patch_size[0], int) and isinstance(
                patch_size[1], int), "What is your patchsize format? Expected a single int or a tuple of two ints."
            assert patch_size[0] == patch_size[1], "Error, non square patches not managed"
            patch_size = patch_size[0]
        self.patch_size = patch_size

        self.desc_mode = net.desc_mode
        self.has_conf = has_conf
        self.two_confs = net.two_confs  # independent confs for 3D regr and descs
        self.desc_conf_mode = net.desc_conf_mode
        idim = net.enc_embed_dim + net.dec_embed_dim

        self.head_local_features = Mlp(in_features=idim,
                                       hidden_features=int(hidden_dim_factor * idim),
                                       out_features=(self.local_feat_dim + self.two_confs) * self.patch_size**2)

        #clip head
        hidden_clip_dim_factor = 1
        clip_feat_dim = 512
        self.head_clip_features = Mlp(in_features=idim,
                                      hidden_features=int(hidden_clip_dim_factor * idim),
                                      out_features=(clip_feat_dim + 1) * self.patch_size**2)

    def forward(self, decout, img_shape):
        # pass through the heads
        pts3d = self.dpt(decout, image_size=(img_shape[0], img_shape[1]))

        # recover encoder and decoder outputs
        enc_output, dec_output = decout[0], decout[-1]
        cat_output = torch.cat([enc_output, dec_output], dim=-1)  # concatenate
        H, W = img_shape
        B, S, D = cat_output.shape

        # extract local_features
        local_features = self.head_local_features(cat_output)  # B,S,D
        local_features = local_features.transpose(-1, -2).view(B, -1, H // self.patch_size, W // self.patch_size)
        local_features = F.pixel_shuffle(local_features, self.patch_size)  # B,d,H,W

        # extract clip_features
        clip_features = self.head_clip_features(cat_output)  # B,S,D
        clip_features = clip_features.transpose(-1, -2).view(B, -1, H // self.patch_size, W // self.patch_size)
        clip_features = F.pixel_shuffle(clip_features, self.patch_size)  # B,d,H,W

        # post process 3D pts, descriptors and confidences
        out = torch.cat([pts3d, local_features, clip_features], dim=1)
        if self.postprocess:
            out = self.postprocess(out,
                                   depth_mode=self.depth_mode,
                                   conf_mode=self.conf_mode,
                                   desc_dim=self.local_feat_dim,
                                   desc_mode=self.desc_mode,
                                   two_confs=self.two_confs,
                                   desc_conf_mode=self.desc_conf_mode)
        return out

class MultiFeatures_DPT(PixelwiseTaskWithDPT):
    """ Mixture between MLP and DPT head that outputs 3d points and local features (with MLP).
    The input for both heads is a concatenation of Encoder and Decoder outputs
    """
    def create_head(self, head_class, feat_dim, feature_dim, last_dim, hooks_idx, dim_tokens, head_type, idim):
        if head_class == 'dpt':
            output_width_ratio=1
            dpt_args = dict(output_width_ratio=output_width_ratio,
                    num_channels=feat_dim, feature_dim=feature_dim, 
                    last_dim=last_dim, head_type=head_type)
            if hooks_idx is not None:
                dpt_args.update(hooks=hooks_idx)
            dpt = DPTOutputAdapter_fix(**dpt_args)
            dpt_init_args = {} if dim_tokens is None else {'dim_tokens_enc': dim_tokens}
            dpt.init(**dpt_init_args)
            return dpt

        elif head_class == 'linear':
            hidden_dim_factor_head = 1

            return Mlp(in_features=idim,
                       hidden_features=int(hidden_dim_factor_head * idim),
                       out_features=feat_dim * self.patch_size**2)
        else:
            return None



    def __init__(self, net, has_conf=False, clip_head_type=None, dino_head_type=None, local_feat_dim=16, hidden_dim_factor=4., hooks_idx=None, dim_tokens=None,
                 num_channels=1, num_channels_clip=1, num_channels_dino=1, postprocess=None, feature_dim=256, last_dim=32, depth_mode=None, conf_mode=None, head_type="regression", **kwargs):
        super().__init__(num_channels=num_channels, feature_dim=feature_dim, last_dim=last_dim, hooks_idx=hooks_idx,
                         dim_tokens=dim_tokens, depth_mode=depth_mode, postprocess=postprocess, conf_mode=conf_mode, head_type=head_type)
        self.local_feat_dim = local_feat_dim
        self.clip_head_type = clip_head_type
        self.dino_head_type = dino_head_type

        patch_size = net.patch_embed.patch_size
        if isinstance(patch_size, tuple):
            assert len(patch_size) == 2 and isinstance(patch_size[0], int) and isinstance(
                patch_size[1], int), "What is your patchsize format? Expected a single int or a tuple of two ints."
            assert patch_size[0] == patch_size[1], "Error, non square patches not managed"
            patch_size = patch_size[0]
        self.patch_size = patch_size

        self.desc_mode = net.desc_mode
        self.has_conf = has_conf
        self.two_confs = net.two_confs  # independent confs for 3D regr and descs
        self.desc_conf_mode = net.desc_conf_mode
        idim = net.enc_embed_dim + net.dec_embed_dim

        self.head_local_features = Mlp(in_features=idim,
                                       hidden_features=int(hidden_dim_factor * idim),
                                       out_features=(self.local_feat_dim + self.two_confs) * self.patch_size**2)

        self.clip_head = self.create_head(head_class = clip_head_type, feat_dim = num_channels_clip, feature_dim = feature_dim, last_dim = last_dim, 
                                          hooks_idx = hooks_idx, dim_tokens = dim_tokens, head_type = head_type, idim = idim)
        self.dino_head = self.create_head(head_class = dino_head_type, feat_dim = num_channels_dino, feature_dim = feature_dim, last_dim = last_dim, 
                                          hooks_idx = hooks_idx, dim_tokens = dim_tokens, head_type = head_type, idim = idim)

    def extract_feat(self, head, head_type, decout, cat_output, img_shape):
        if head_type == 'dpt':
            return head(decout, image_size=(img_shape[0], img_shape[1]))
        elif head_type == 'linear':
            features = head(cat_output)  # B,S,D
            features = features.transpose(-1, -2).view(cat_output.shape[0], -1, img_shape[0] // self.patch_size, img_shape[1] // self.patch_size)
            features = F.pixel_shuffle(features, self.patch_size)  # B,d,H,W
            return features
        else :
            return None


    def forward(self, decout, img_shape):
        # pass through the heads
        pts3d = self.dpt(decout, image_size=(img_shape[0], img_shape[1]))

        # recover encoder and decoder outputs
        enc_output, dec_output = decout[0], decout[-1]
        cat_output = torch.cat([enc_output, dec_output], dim=-1)  # concatenate
        H, W = img_shape
        B, S, D = cat_output.shape

        # extract local_features
        local_features = self.head_local_features(cat_output)  # B,S,D
        local_features = local_features.transpose(-1, -2).view(B, -1, H // self.patch_size, W // self.patch_size)
        local_features = F.pixel_shuffle(local_features, self.patch_size)  # B,d,H,W

        cat_list = [pts3d, local_features]
        clip_feat = self.extract_feat(self.clip_head, self.clip_head_type, decout, cat_output, img_shape)
        if clip_feat is not None:
            cat_list.append(clip_feat)
        dino_feat = self.extract_feat(self.dino_head, self.dino_head_type, decout, cat_output, img_shape)
        if dino_feat is not None:
            cat_list.append(dino_feat)
        # post process 3D pts, descriptors and confidences
        out = torch.cat(cat_list, dim=1)
        if self.postprocess:
            out = self.postprocess(out,
                                   depth_mode=self.depth_mode,
                                   conf_mode=self.conf_mode,
                                   desc_dim=self.local_feat_dim,
                                   desc_mode=self.desc_mode,
                                   two_confs=self.two_confs,
                                   desc_conf_mode=self.desc_conf_mode,
                                   clip_mode=('exp', -inf, inf) if self.clip_head_type is not None else None,
                                   dino_mode=('exp', -inf, inf) if self.dino_head_type is not None else None)
        return out

def mast3r_head_factory(head_type, output_mode, clip_head_type, dino_head_type, net, has_conf=False):
    """" build a prediction head for the decoder 
    """
    if head_type == 'catmlp+dpt' and output_mode.startswith('pts3d+desc'):
        local_feat_dim = int(output_mode[10:])
        assert net.dec_depth > 9
        l2 = net.dec_depth
        feature_dim = 256
        last_dim = feature_dim // 2
        out_nchan = 3
        clip_out_nchan = 512
        dino_out_nchan = 384
        ed = net.enc_embed_dim
        dd = net.dec_embed_dim
        # return Cat_MLP_LocalFeatures_DPT_Pts3d(net, local_feat_dim=local_feat_dim, has_conf=has_conf,
        #                                        num_channels=out_nchan + has_conf,
        #                                        num_channels_clip=clip_out_nchan + has_clip_conf,
        #                                        feature_dim=feature_dim,
        #                                        last_dim=last_dim,
        #                                        hooks_idx=[0, l2 * 2 // 4, l2 * 3 // 4, l2],
        #                                        dim_tokens=[ed, dd, dd, dd],
        #                                        postprocess=postprocess,
        #                                        depth_mode=net.depth_mode,
        #                                        conf_mode=net.conf_mode,
        #                                        head_type='regression')
        return MultiFeatures_DPT(net, local_feat_dim=local_feat_dim, has_conf=has_conf,
                                               clip_head_type=clip_head_type, dino_head_type=dino_head_type,
                                               num_channels=out_nchan + has_conf,
                                               num_channels_clip=clip_out_nchan,
                                               num_channels_dino=dino_out_nchan,
                                               feature_dim=feature_dim,
                                               last_dim=last_dim,
                                               hooks_idx=[0, l2 * 2 // 4, l2 * 3 // 4, l2],
                                               dim_tokens=[ed, dd, dd, dd],
                                               postprocess=postprocess,
                                               depth_mode=net.depth_mode,
                                               conf_mode=net.conf_mode,
                                               head_type='regression')
    else:
        raise NotImplementedError(
            f"unexpected {head_type=} and {output_mode=}")
