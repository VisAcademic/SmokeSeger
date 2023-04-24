"""
@File  : smoke_seger.py
@Author: tao.jing
@Date  : 2022/1/11
@Desc  :
"""

import paddle
import paddle.nn as nn

from paddleseg.models.smoke_models import HarDBackbone
from paddleseg.models.smoke_models import SFBackbone
from paddleseg.models.smoke_models import MLPDecoder
from paddleseg.models.smoke_models import MlpMixerDecoder
from paddleseg.models.smoke_models import MlpMixerConvDecoder

from paddleseg.models import layers
from paddleseg.cvlibs import manager
from paddleseg.utils import utils


__all__ = [
    'SmokeSeger'
]


@manager.MODELS.add_component
class SmokeSeger(nn.Layer):
    def __init__(self,
                 num_classes=2,
                 decoder=MLPDecoder,
                 img_size=(512, 512),
                 cnn_pretrain=None,
                 trans_pretrain=None,
                 pretrain=None,
                 lr_coeff=(1.0, 1.0, 1.0),
                 need_out_attn=False):
        super(SmokeSeger, self).__init__()

        self.num_classes = num_classes
        # trans_lr, cnn_lr, decoder_lr
        self.lr_coeff = lr_coeff
        self.need_out_attn = need_out_attn
        print(f' ---------- [SmokeSeger] lr_coeff {lr_coeff} ---------- ')

        if pretrain is not None:
            cnn_pretrain = None
            trans_pretrain = None

        # Encoder
        # SegFormer branch
        self.sf = SFBackbone(
            img_size=img_size,
            pretrain=trans_pretrain,
            need_out_attn=need_out_attn
        )
        self.sf_backbone = self.sf.backbone
        # HarDNet branch
        self.hard_backbone = HarDBackbone(pretrain=cnn_pretrain)

        sf_s4_chans, sf_s8_chans, sf_s16_chans, sf_s32_chans = \
            self.sf_backbone.feat_channels

        hd_s2_chans, hd_s4_chans, hd_s8_chans, hd_s16_chans, _ = \
            self.hard_backbone.encoder.get_skip_channels()
        hd_s32_chans = \
            self.hard_backbone.encoder.get_out_channels()

        # Encoder output channels
        s4_chans = hd_s4_chans + sf_s4_chans
        s8_chans = hd_s8_chans + sf_s8_chans
        s16_chans = hd_s16_chans + sf_s16_chans
        s32_chans = hd_s32_chans + sf_s32_chans

        # Decoder
        encoder_chans = dict()
        encoder_chans['s4'] = s4_chans
        encoder_chans['s8'] = s8_chans
        encoder_chans['s16'] = s16_chans
        encoder_chans['s32'] = s32_chans

        assert isinstance(decoder, MlpMixerConvDecoder) or \
               isinstance(decoder, MlpMixerDecoder) or \
               isinstance(decoder, MLPDecoder) or \
            f'Invalid decoder type.'
        self.decoder = type(decoder)(num_classes=decoder.num_classes,
                                     img_size=img_size,
                                     mlp_channels=decoder.mlp_channels,
                                     align_corners=decoder.align_corners,
                                     encoder_channels=encoder_chans)

        if pretrain is not None:
            utils.load_entire_model(self, pretrain)

        self.set_lr_coeff()

    def set_lr_coeff(self):
        trans_lr, cnn_lr, decoder_lr = self.lr_coeff
        for parameter in self.sf_backbone.parameters():
            parameter.optimize_attr['learning_rate'] = trans_lr
        for parameter in self.hard_backbone.parameters():
            parameter.optimize_attr['learning_rate'] = cnn_lr
        for parameter in self.decoder.parameters():
            parameter.optimize_attr['learning_rate'] = decoder_lr

    def get_attn_depths(self):
        return self.sf_backbone.depths

    def forward(self, x):
        # SegFormer branch
        if self.need_out_attn:
            out, attn_weights = self.sf_backbone(x)
        else:
            out = self.sf_backbone(x)
        sf_s4, sf_s8, sf_s16, sf_s32 = out

        # HarDNet branch
        hd_s2, hd_s4, hd_s8, hd_s16, hd_s32 = self.hard_backbone(x)

        # Fuse conv and transformer
        ec_s4  = paddle.concat([sf_s4, hd_s4], axis=1)
        ec_s8  = paddle.concat([sf_s8, hd_s8], axis=1)
        ec_s16 = paddle.concat([sf_s16, hd_s16], axis=1)
        ec_s32 = paddle.concat([sf_s32, hd_s32], axis=1)

        out = self.decoder(x, [ec_s4, ec_s8, ec_s16, ec_s32])
        if self.need_out_attn:
            return out, attn_weights
        return out
