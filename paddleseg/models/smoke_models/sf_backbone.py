"""
@File  : sf_backbone.py
@Author: tao.jing
@Date  : 2022/1/11
@Desc  :
"""
import os

import paddle
import paddle.nn as nn

from paddleseg.models.smoke_models.smoke_seger_modules import SMixVisionTransformer_B2
from paddleseg.models.smoke_models.smoke_seger_modules import SMixTransFuse_B2

from paddleseg.utils import logger

__all__ = [
    'SFBackbone',
    'SFBackboneFuse'
]

class SFBackbone(nn.Layer):
    def __init__(self,
                 img_size=(512, 512),
                 pretrain=None,
                 need_out_attn=False):
        super(SFBackbone, self).__init__()

        self.backbone = SMixVisionTransformer_B2(
            img_size=img_size,
            need_out_attn=need_out_attn
        )

        if pretrain is not None:
            self.init_model_from_segformer(pretrain)

    def init_model_from_segformer(self, segformer_pretrain):
        assert os.path.exists(segformer_pretrain), \
            f'SegFormer pretrain { segformer_pretrain} not exist.'
        para_state_dict = paddle.load(segformer_pretrain)

        model_state_dict = self.state_dict()
        keys = model_state_dict.keys()
        num_params_loaded = 0
        for k in keys:
            if k.startswith('backbone'):
                if k not in para_state_dict:
                    logger.warning("{} is not in pretrained model".format(k))
                elif list(para_state_dict[k].shape) != list(
                        model_state_dict[k].shape):
                    logger.warning(
                        "[SKIP] Shape of pretrained params {} doesn't match.(Pretrained: {}, Actual: {})"
                        .format(k, para_state_dict[k].shape,
                                model_state_dict[k].shape))
                else:
                    model_state_dict[k] = para_state_dict[k]
                    num_params_loaded += 1
        self.set_dict(model_state_dict)
        logger.info("There are {}/{} variables loaded into {}.".format(
            num_params_loaded, len(model_state_dict),
            self.__class__.__name__))

    def forward(self, x):
        return self.backbone(x)


class SFBackboneFuse(nn.Layer):
    def __init__(self,
                 hd_chans=None,
                 img_size=(512, 512),
                 pretrain=None,                 
                 need_out_attn=False):
        super(SFBackboneFuse, self).__init__()

        self.backbone = SMixTransFuse_B2(
            hd_chans=hd_chans,
            img_size=img_size,
            need_out_attn=need_out_attn,
        )

        if pretrain is not None:
            self.init_model_from_segformer(pretrain)

    def init_model_from_segformer(self, segformer_pretrain):
        assert os.path.exists(segformer_pretrain), \
            f'SegFormer pretrain { segformer_pretrain} not exist.'
        para_state_dict = paddle.load(segformer_pretrain)

        model_state_dict = self.state_dict()
        keys = model_state_dict.keys()
        num_params_loaded = 0
        for k in keys:
            if k.startswith('backbone'):
                if k not in para_state_dict:
                    logger.warning("{} is not in pretrained model".format(k))
                elif list(para_state_dict[k].shape) != list(
                        model_state_dict[k].shape):
                    logger.warning(
                        "[SKIP] Shape of pretrained params {} doesn't match.(Pretrained: {}, Actual: {})"
                        .format(k, para_state_dict[k].shape,
                                model_state_dict[k].shape))
                else:
                    model_state_dict[k] = para_state_dict[k]
                    num_params_loaded += 1
        self.set_dict(model_state_dict)
        logger.info("There are {}/{} variables loaded into {}.".format(
            num_params_loaded, len(model_state_dict),
            self.__class__.__name__))

    def forward(self, x):
        return self.backbone(x)


if __name__ == '__main__':
    network = SFBackbone()
    keys = network.state_dict().keys()
    for key in keys:
        print(key)
