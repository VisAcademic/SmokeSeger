"""
@File  : mlp_decoder.py
@Author: tao.jing
@Date  : 2022/1/11
@Desc  :
"""
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.models import layers
from paddleseg.cvlibs import manager


__all__ = [
    'MLPDecoder'
]


class SFMLP(nn.Layer):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose([0, 2, 1])
        x = self.proj(x)
        return x


@manager.MODELS.add_component
class MLPDecoder(nn.Layer):
    def __init__(self,
                 num_classes=2,
                 img_size=(512, 512),
                 mlp_channels=1024,
                 repeat=1,
                 align_corners=False,
                 encoder_channels=None):
        super(MLPDecoder, self).__init__()

        if encoder_channels is None:
            encoder_channels = dict()
            encoder_channels['s4'] = 142
            encoder_channels['s8'] = 288
            encoder_channels['s16'] = 534
            encoder_channels['s32'] = 832
        elif encoder_channels:
            pass

        self.num_classes = num_classes
        self.mlp_channels = mlp_channels
        self.align_corners = align_corners
        self.repeat = repeat

        self.linear_s4 = SFMLP(input_dim=encoder_channels['s4'],
                               embed_dim=mlp_channels)
        self.linear_s8 = SFMLP(input_dim=encoder_channels['s8'],
                               embed_dim=mlp_channels)
        self.linear_s16 = SFMLP(input_dim=encoder_channels['s16'],
                                embed_dim=mlp_channels)
        self.linear_s32 = SFMLP(input_dim=encoder_channels['s32'],
                                embed_dim=mlp_channels)

        self.dropout = nn.Dropout2D(0.1)
        self.linear_fuse = layers.ConvBNReLU(
            in_channels=mlp_channels * 4,
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

        dc_s4 = self.linear_s4(ec_s4).transpose([0, 2, 1]).reshape(
            [0, 0, ec_s4_shape[2], ec_s4_shape[3]])

        dc_s8 = self.linear_s8(ec_s8).transpose([0, 2, 1]).reshape(
            [0, 0, ec_s8_shape[2], ec_s8_shape[3]])
        dc_s8 = F.interpolate(
            dc_s8,
            size=ec_s4_shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        dc_s16 = self.linear_s16(ec_s16).transpose([0, 2, 1]).reshape(
            [0, 0, ec_s16_shape[2], ec_s16_shape[3]])
        dc_s16 = F.interpolate(
            dc_s16,
            size=ec_s4_shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        dc_s32 = self.linear_s32(ec_s32).transpose([0, 2, 1]).reshape(
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


if __name__ == '__main__':
    decoder = MLPDecoder(num_classes=2, mlp_channels=512)
    decoder2 = type(decoder)(num_classes=decoder.num_classes, mlp_channels=1024)
    print('Decoder')
