"""
@File  : rand_augment_config.py
@Author: tao.jing
@Date  : 2022/1/20
@Desc  :
"""

from paddleseg.transforms.seg_aug_ops import *


ImageNetRandAug = {
    'op_list': [
        'ShearX',
        'ShearY',
        #'TranslateX',
        #'TranslateY',
        'Rotate',
        'AutoContrast',
        'Equalize',
        'Contrast',
        'Color',
        # 'Brightness',
        'Sharpness',
        #'Posterize',
        #'Solarize',
        # 'Invert',
    ],
    'op_ranges': {
        'ShearX': (-0.3, 0.3),
        'ShearY': (-0.3, 0.3),
        'TranslateX': (-0.45, 0.45),
        'TranslateY': (-0.45, 0.45),
        'Rotate': (-30, 30),
        'Color': (0.1, 1.9),
        'Contrast': (0.1, 1.9),
        'AutoContrast': (0, 1),  # no range
        'Posterize': (4, 8),
        'Solarize': (0, 256),
        'Sharpness': (0.1, 1.9),
        'Brightness': (0.1, 1.9),
        'Equalize': (0, 1),  # no range
        'Invert': (0, 1),  # no range
    },
    'op_funcs': {
        'ShearX': seg_shear_x,
        'ShearY': seg_shear_y,
        'TranslateX': seg_translate_x_relative,
        'TranslateY': seg_translate_y_relative,
        'Rotate': seg_rotate,
        'AutoContrast': seg_auto_contrast,
        'Equalize': seg_equalize, # Less
        'Contrast': seg_contrast_no_rand_neg, # Less
        'Color': seg_color_no_rand_neg,
        'Brightness': seg_brightness_no_rand_neg, # Less
        'Sharpness': seg_sharpness_no_rand_neg,
        'Posterize': seg_posterize,
        'Solarize': seg_solarize,
        'Invert': seg_invert,
    },
}
