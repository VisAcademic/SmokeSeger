"""
@File  : mix_trans_fuse.py
@Author: tao.jing
@Date  : 2022/2/13
@Desc  :
"""
from scipy.linalg import _flapack

import math
from functools import partial

import paddle
from paddle import nn
import paddle.nn.initializer as paddle_init
from paddleseg.models.backbones.mix_transformer import MixVisionTransformer

from paddleseg.cvlibs import manager
from paddleseg.models.backbones.transformer_utils import *

__all__ = [
    'SMixTransFuse',
    'SMixTransFuse_B2'
]


class SMixTransFuse(MixVisionTransformer):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8],
                 mlp_ratios=[4, 4, 4, 4],
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1],
                 pretrained=None,
                 fuse_type='embed',
                 hd_chans=None,
                 need_out_attn=False):
        super(SMixTransFuse, self).__init__(img_size=img_size,
                                           patch_size=patch_size,
                                           in_chans=in_chans,
                                           num_classes=num_classes,
                                           embed_dims=embed_dims,
                                           num_heads=num_heads,
                                           mlp_ratios=mlp_ratios,
                                           qkv_bias=qkv_bias,
                                           qk_scale=qk_scale,
                                           drop_rate=drop_rate,
                                           attn_drop_rate=attn_drop_rate,
                                           drop_path_rate=drop_path_rate,
                                           norm_layer=norm_layer,
                                           depths=depths,
                                           sr_ratios=sr_ratios,
                                           pretrained=pretrained)
        if hd_chans is None:
            hd_chans = {
                's2_chans': 32,
                's4_chans': 64,
                's8_chans': 96,
                's16_chans': 160,
                's32_chans': 224,
            }
        if fuse_type not in ['embed', 'attn']:
            raise ValueError(f'Invalid fuse_type {fuse_type}')
        self.fuse_type = fuse_type

        self.embed_dims = embed_dims
        sf_s4_chans, sf_s8_chans, sf_s16_chans, sf_s32_chans = embed_dims

        self.fuse_list = nn.LayerList([
            nn.Conv2D(in_channels=sf_s4_chans + hd_chans['s4_chans'],
                                      out_channels=sf_s4_chans,
                                      kernel_size=1),
            nn.Conv2D(in_channels=sf_s8_chans + hd_chans['s8_chans'],
                                      out_channels=sf_s8_chans,
                                      kernel_size=1),
            nn.Conv2D(in_channels=sf_s16_chans + hd_chans['s16_chans'],
                                      out_channels=sf_s16_chans,
                                      kernel_size=1),
            nn.Conv2D(in_channels=sf_s32_chans + hd_chans['s32_chans'],
                                      out_channels=sf_s32_chans,
                                      kernel_size=1)
        ])
        self.init_fuse_list()

    def init_fuse_list(self):
        for m in self.fuse_list:
            fan_out = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
            fan_out //= m._groups
            paddle_init.Normal(0, math.sqrt(2.0 / fan_out))(m.weight)
            if m.bias is not None:
                zeros_(m.bias)

    def fuse_x_hd(self, mode, x, hd, H, W):
        mode_list = ['s4', 's8', 's16', 's32']
        idx = mode_list.index(mode)
        fuse_conv = self.fuse_list[idx]
        # x: (B, H*W, embed_dim)
        # hd: (B, hd_chans, H, W)
        x = x.transpose([0, 2, 1])
        x = x.reshape((-1, self.embed_dims[idx], H, W))
        x = paddle.concat([x, hd], axis=1)
        x = fuse_conv(x)
        x = x.reshape((-1, self.embed_dims[idx], H * W))
        x = x.transpose([0, 2, 1])
        return x

    def forward(self, x_fuse):
        x, fuse = x_fuse
        return self.forward_fuse(x, fuse)

    def forward_fuse(self, x, fuse):
        # hd_s2: (B, 32, 256, 256) - 48
        # hd_s4: (B, 64, 128, 128) - 78
        # hd_s8: (B, 96, 64, 64) - 160
        # hd_s16: (B, 160, 32, 32) - 214
        # hd_s32: (B, 224, 16, 16) - 320
        hd_s2, hd_s4, hd_s8, hd_s16, hd_s32 = fuse

        B = paddle.shape(x)[0]
        outs = []

        # stage 1
        # patch_embed1: (B, 3, 512, 512) --> (1, 128*128, 64) --> s4
        x, H, W = self.patch_embed1(x)
        x = self.fuse_x_hd('s4', x, hd_s4, H, W)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)

        x = self.norm1(x)
        x = x.reshape([B, H, W, self.feat_channels[0]]).transpose([0, 3, 1, 2])
        outs.append(x)

        # stage 2
        # patch_embed2: (B, 64, 128, 128) --> (1, 64*64, 128) --> s8
        x, H, W = self.patch_embed2(x)
        x = self.fuse_x_hd('s8', x, hd_s8, H, W)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape([B, H, W, self.feat_channels[1]]).transpose([0, 3, 1, 2])
        outs.append(x)

        # stage 3
        # patch_embed3: (B, 128, 64, 64) --> (1, 32*32, 320) --> s16
        x, H, W = self.patch_embed3(x)
        x = self.fuse_x_hd('s16', x, hd_s16, H, W)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape([B, H, W, self.feat_channels[2]]).transpose([0, 3, 1, 2])
        outs.append(x)

        # stage 4
        # patch_embed1: (B, 320, 32, 32) --> (1, 16*16, 512)  --> s32
        x, H, W = self.patch_embed4(x)
        x = self.fuse_x_hd('s32', x, hd_s32, H, W)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape([B, H, W, self.feat_channels[3]]).transpose([0, 3, 1, 2])
        outs.append(x)

        return outs


@manager.BACKBONES.add_component
def SMixTransFuse_B2(**kwargs):
    return SMixTransFuse(
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
        drop_rate=0.0,
        drop_path_rate=0.1,
        pretrained=None,
        fuse_type='embed',
        **kwargs)


if __name__ == '__main__':
    network = SMixTransFuse()
