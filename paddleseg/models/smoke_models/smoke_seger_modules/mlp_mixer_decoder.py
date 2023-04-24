"""
@File  : mlp_mixer_decoder.py
@Author: tao.jing
@Date  : 2022/1/11
@Desc  :
"""
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.models.smoke_models.smoke_seger_modules.droppath import DropPath
from paddleseg.models import layers
from paddleseg.cvlibs import manager


__all__ = [
    'MlpMixerDecoder',
    'MlpMixerConvDecoder',
]


class Identity(nn.Layer):
    """ Identity layer

    The output of this layer is the input without any change.
    Use this layer to avoid if condition in some forward methods

    """
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x


class Mlp(nn.Layer):
    """ MLP module

    Impl using nn.Linear and activation is GELU, dropout is applied.
    Ops: fc -> act -> dropout -> fc -> dropout

    Attributes:
        fc1: nn.Linear
        fc2: nn.Linear
        act: GELU
        dropout1: dropout after fc1
        dropout2: dropout after fc2
    """

    def __init__(self, in_features, hidden_features, dropout):
        super(Mlp, self).__init__()
        w_attr_1, b_attr_1 = self._init_weights()
        self.fc1 = nn.Linear(in_features,
                             hidden_features,
                             weight_attr=w_attr_1,
                             bias_attr=b_attr_1)

        w_attr_2, b_attr_2 = self._init_weights()
        self.fc2 = nn.Linear(hidden_features,
                             in_features,
                             weight_attr=w_attr_2,
                             bias_attr=b_attr_2)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.XavierUniform())
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Normal(std=1e-6))
        return weight_attr, bias_attr

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MixerBlock(nn.Layer):
    """Mixer Block

    This block implements Mixer layer which contains 2 MLP blocks and residuals.
    The 1st is token-mixing MLP, the 2nd is channel-mixing MLP.

    Attributes:
        mlp_tokens: Mlp layer for token mixing
        mlp_channels: Mlp layer for channel mixing
        tokens_dim: mlp hidden dim for mlp_tokens
        channels_dim: mlp hidden dim for mlp_channels
        norm1: nn.LayerNorm, apply before mlp_tokens
        norm2: nn.LayerNorm, apply before mlp_channels
    """

    def __init__(self, dim, seq_len, mlp_ratio=(0.5, 4.0), dropout=0., droppath=0.):
        super(MixerBlock, self).__init__()
        tokens_dim = int(mlp_ratio[0] * dim)
        channels_dim = int(mlp_ratio[1] * dim)
        self.norm1 = nn.LayerNorm(dim, epsilon=1e-6)
        self.mlp_tokens = Mlp(seq_len, tokens_dim, dropout=dropout)
        self.drop_path = DropPath(droppath)
        self.norm2 = nn.LayerNorm(dim, epsilon=1e-6)
        self.mlp_channels = Mlp(dim, channels_dim, dropout=dropout)

    def forward(self, x):
        h = x
        x = self.norm1(x)
        x = x.transpose([0, 2, 1])
        x = self.mlp_tokens(x)
        x = x.transpose([0, 2, 1])
        x = self.drop_path(x)
        x = x + h

        h = x
        x = self.norm2(x)
        x = self.mlp_channels(x)
        x = self.drop_path(x)
        x = x + h

        return x


class MixerBlockList(nn.Layer):
    """Mixer Block List
    """
    def __init__(self,
                 dim,
                 seq_len,
                 mlp_ratio=(0.5, 4.0),
                 dropout=0.,
                 droppath=0.,
                 repeat=1):
        super(MixerBlockList, self).__init__()

        self.mixers = nn.LayerList([
            MixerBlock(dim,
                       seq_len,
                       mlp_ratio,
                       dropout,
                       droppath) for i in range(repeat)
        ])

    def forward(self, x):
        for mixer in self.mixers:
            x = mixer(x)
        return x


@manager.MODELS.add_component
class MlpMixerDecoder(nn.Layer):
    def __init__(self,
                 num_classes=2,
                 img_size=(512, 512),
                 mlp_channels=768,
                 mlp_ratio=(0.5, 2.0),
                 dropout=0.,
                 droppath=0.,
                 repeat=1,
                 align_corners=False,
                 encoder_channels=None):
        super(MlpMixerDecoder, self).__init__()

        if encoder_channels is None:
            encoder_channels = dict()
            encoder_channels['s4'] = 142
            encoder_channels['s8'] = 288
            encoder_channels['s16'] = 534
            encoder_channels['s32'] = 832

        self.num_classes = num_classes
        self.mlp_channels = mlp_channels
        self.align_corners = align_corners
        self.repeat = repeat

        norm_layer = nn.LayerNorm(mlp_channels, epsilon=1e-6)

        seq_len_s4  = int((img_size[0] / 4) * (img_size[1] / 4))
        seq_len_s8  = int((img_size[0] / 8) * (img_size[1] / 8))
        seq_len_s16 = int((img_size[0] / 16) * (img_size[1] / 16))
        seq_len_s32 = int((img_size[0] / 32) * (img_size[1] / 32))
        self.mixer4 = MixerBlockList(encoder_channels['s4'],
                                     seq_len_s4,
                                     mlp_ratio,
                                     dropout,
                                     droppath, repeat=repeat)
        self.mixer8 = MixerBlockList(encoder_channels['s8'],
                                     seq_len_s8,
                                     mlp_ratio,
                                     dropout,
                                     droppath, repeat=repeat)
        self.mixer16 = MixerBlockList(encoder_channels['s16'],
                                      seq_len_s16,
                                      mlp_ratio,
                                      dropout,
                                      droppath, repeat=repeat)
        self.mixer32 = MixerBlockList(encoder_channels['s32'],
                                      seq_len_s32,
                                      mlp_ratio,
                                      dropout,
                                      droppath, repeat=repeat)

        self.dropout = nn.Dropout2D(0.1)

        fuse_in_channels = \
            encoder_channels['s4'] + \
            encoder_channels['s8'] + \
            encoder_channels['s16'] + \
            encoder_channels['s32']
        self.linear_fuse = layers.ConvBNReLU(
            in_channels=fuse_in_channels,
            out_channels=mlp_channels,
            kernel_size=1,
            bias_attr=False)

        self.linear_pred = nn.Conv2D(
            mlp_channels, self.num_classes, kernel_size=1)

    def forward(self, x, encoder_outs):
        ec_s4, ec_s8, ec_s16, ec_s32 = encoder_outs
        ec_s4_shape = paddle.shape(ec_s4)
        ec_s8_shape = paddle.shape(ec_s8)
        ec_s16_shape = paddle.shape(ec_s16)
        ec_s32_shape = paddle.shape(ec_s32)

        ec_s4 = ec_s4.flatten(2).transpose([0, 2, 1])
        dc_s4 = self.mixer4(ec_s4).transpose([0, 2, 1]).reshape(
            [0, 0, ec_s4_shape[2], ec_s4_shape[3]])

        ec_s8 = ec_s8.flatten(2).transpose([0, 2, 1])
        dc_s8 = self.mixer8(ec_s8).transpose([0, 2, 1]).reshape(
            [0, 0, ec_s8_shape[2], ec_s8_shape[3]])
        dc_s8 = F.interpolate(
            dc_s8,
            size=ec_s4_shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        ec_s16 = ec_s16.flatten(2).transpose([0, 2, 1])
        dc_s16 = self.mixer16(ec_s16).transpose([0, 2, 1]).reshape(
            [0, 0, ec_s16_shape[2], ec_s16_shape[3]])
        dc_s16 = F.interpolate(
            dc_s16,
            size=ec_s4_shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        ec_s32 = ec_s32.flatten(2).transpose([0, 2, 1])
        dc_s32 = self.mixer32(ec_s32).transpose([0, 2, 1]).reshape(
            [0, 0, ec_s32_shape[2], ec_s32_shape[3]])
        dc_s32 = F.interpolate(
            dc_s32,
            size=ec_s4_shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        dc = self.linear_fuse(paddle.concat([dc_s4, dc_s8, dc_s16, dc_s32],
                                            axis=1))
        logit = self.dropout(dc)
        logit = self.linear_pred(logit)

        return [
            F.interpolate(
                logit,
                size=paddle.shape(x)[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]



@manager.MODELS.add_component
class MlpMixerConvDecoder(nn.Layer):
    def __init__(self,
                 num_classes=2,
                 img_size=(512, 512),
                 mlp_channels=768,
                 conv_channels=256,
                 conv_kernel=3,
                 mlp_ratio=(0.5, 2.0),
                 dropout=0.,
                 droppath=0.,
                 align_corners=False,
                 encoder_channels=None):
        super(MlpMixerConvDecoder, self).__init__()

        if encoder_channels is None:
            encoder_channels = dict()
            encoder_channels['s4'] = 142
            encoder_channels['s8'] = 288
            encoder_channels['s16'] = 534
            encoder_channels['s32'] = 832

        self.num_classes = num_classes
        self.mlp_channels = mlp_channels
        self.conv_channels = conv_channels
        self.conv_kernel = conv_kernel
        self.align_corners = align_corners

        norm_layer = nn.LayerNorm(mlp_channels, epsilon=1e-6)

        seq_len_s4  = int((img_size[0] / 4) * (img_size[1] / 4))
        seq_len_s8  = int((img_size[0] / 8) * (img_size[1] / 8))
        seq_len_s16 = int((img_size[0] / 16) * (img_size[1] / 16))
        seq_len_s32 = int((img_size[0] / 32) * (img_size[1] / 32))
        self.mixer_s4 = MixerBlock(encoder_channels['s4'],
                                   seq_len_s4,
                                   mlp_ratio,
                                   dropout,
                                   droppath)
        self.mixer_s8 = MixerBlock(encoder_channels['s8'],
                                   seq_len_s8,
                                   mlp_ratio,
                                   dropout,
                                   droppath)
        self.mixer_s16 = MixerBlock(encoder_channels['s16'],
                                    seq_len_s16,
                                    mlp_ratio,
                                    dropout,
                                    droppath)
        self.mixer_s32 = MixerBlock(encoder_channels['s32'],
                                    seq_len_s32,
                                    mlp_ratio,
                                    dropout,
                                    droppath)

        self.conv_s32 = layers.ConvBNPReLU(
            in_channels=encoder_channels['s32'],
            out_channels=conv_channels,
            kernel_size=conv_kernel)
        self.conv_s16 = layers.ConvBNPReLU(
            in_channels=encoder_channels['s16'] + conv_channels,
            out_channels=conv_channels * 2,
            kernel_size=conv_kernel)
        self.conv_s8 = layers.ConvBNPReLU(
            in_channels=encoder_channels['s8'] + conv_channels * 2,
            out_channels=conv_channels * 3,
            kernel_size=conv_kernel)

        self.dropout = nn.Dropout2D(0.1)

        fuse_in_channels = \
            encoder_channels['s4'] + conv_channels * 3

        self.linear_fuse = layers.ConvBNReLU(
            in_channels=fuse_in_channels,
            out_channels=mlp_channels,
            kernel_size=1,
            bias_attr=False)

        self.linear_pred = nn.Conv2D(
            mlp_channels, self.num_classes, kernel_size=1)

    def forward(self, x, encoder_outs):
        ec_s4, ec_s8, ec_s16, ec_s32 = encoder_outs
        ec_s4_shape = paddle.shape(ec_s4)
        ec_s8_shape = paddle.shape(ec_s8)
        ec_s16_shape = paddle.shape(ec_s16)
        ec_s32_shape = paddle.shape(ec_s32)

        ec_s32 = ec_s32.flatten(2).transpose([0, 2, 1])
        dc_s32 = self.mixer_s32(ec_s32).transpose([0, 2, 1]).reshape(
            [0, 0, ec_s32_shape[2], ec_s32_shape[3]])
        dc_s32 = self.conv_s32(dc_s32)
        dc_s32 = F.interpolate(
            dc_s32,
            size=ec_s16_shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        ec_s16 = ec_s16.flatten(2).transpose([0, 2, 1])
        dc_s16 = self.mixer_s16(ec_s16).transpose([0, 2, 1]).reshape(
            [0, 0, ec_s16_shape[2], ec_s16_shape[3]])
        dc_s16 = self.conv_s16(paddle.concat([dc_s16, dc_s32], axis=1))
        dc_s16 = F.interpolate(
            dc_s16,
            size=ec_s8_shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        ec_s8 = ec_s8.flatten(2).transpose([0, 2, 1])
        dc_s8 = self.mixer_s8(ec_s8).transpose([0, 2, 1]).reshape(
            [0, 0, ec_s8_shape[2], ec_s8_shape[3]])
        dc_s8 = self.conv_s8(paddle.concat([dc_s8, dc_s16], axis=1))
        dc_s8 = F.interpolate(
            dc_s8,
            size=ec_s4_shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        ec_s4 = ec_s4.flatten(2).transpose([0, 2, 1])
        dc_s4 = self.mixer_s4(ec_s4).transpose([0, 2, 1]).reshape(
            [0, 0, ec_s4_shape[2], ec_s4_shape[3]])

        dc = self.linear_fuse(paddle.concat([dc_s4, dc_s8],
                                            axis=1))
        logit = self.dropout(dc)
        logit = self.linear_pred(logit)

        return [
            F.interpolate(
                logit,
                size=paddle.shape(x)[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]


if __name__ == '__main__':
    pass