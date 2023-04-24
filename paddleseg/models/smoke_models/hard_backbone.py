"""
@File  : hard_encoder.py
@Author: tao.jing
@Date  : 2022/1/11
@Desc  :
"""
from scipy.linalg import _flapack

import os

import paddle
import paddle.nn as nn

from paddleseg.models.hardnet import Encoder as HarDEncoder
from paddleseg.models import layers

from paddleseg.utils import utils
from paddleseg.utils import logger

__all__ = [
    'HarDBackbone'
]

class HarDBackbone(nn.Layer):
    def __init__(self,
                 stem_channels=(16, 24, 32, 48),
                 ch_list=(64, 96, 160, 224, 320),
                 grmul=1.7,
                 gr=(10, 16, 18, 24, 32),
                 n_layers=(4, 4, 8, 8, 8),
                 pretrain=None):
        super(HarDBackbone, self).__init__()

        encoder_blks_num = len(n_layers)
        encoder_in_channels = stem_channels[3]

        self.stem = nn.Sequential(
            layers.ConvBNReLU(
                3, stem_channels[0], kernel_size=3, bias_attr=False),
            layers.ConvBNReLU(
                stem_channels[0],
                stem_channels[1],
                kernel_size=3,
                bias_attr=False),
            layers.ConvBNReLU(
                stem_channels[1],
                stem_channels[2],
                kernel_size=3,
                stride=2,
                bias_attr=False),
            layers.ConvBNReLU(
                stem_channels[2],
                stem_channels[3],
                kernel_size=3,
                bias_attr=False))

        self.encoder = HarDEncoder(encoder_blks_num,
                                   encoder_in_channels,
                                   ch_list, gr, grmul, n_layers)

        if pretrain is not None:
            self.init_model_from_hardnet(pretrain)

    def init_model_from_hardnet(self, hardnet_pretrain):
        assert os.path.exists(hardnet_pretrain), \
            f'HarDnet pretrain { hardnet_pretrain} not exist.'
        para_state_dict = paddle.load(hardnet_pretrain)

        model_state_dict = self.state_dict()
        keys = model_state_dict.keys()
        num_params_loaded = 0
        for k in keys:
            if k.startswith('stem') or k.startswith('encoder'):
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
        x = self.stem(x)
        x, skip_connections = self.encoder(x)
        hd_s32 = x
        hd_s2, hd_s4, hd_s8, hd_s16 = skip_connections
        return hd_s2, hd_s4, hd_s8, hd_s16, hd_s32


if __name__ == '__main__':
    network = HarDBackbone()
    keys = network.state_dict().keys()
    for key in keys:
        print(key)