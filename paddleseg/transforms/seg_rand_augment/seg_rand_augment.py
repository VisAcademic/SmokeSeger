"""
@File  : seg_rand_augment.py
@Author: tao.jing
@Date  : 2022/1/20
@Desc  :
"""
import random
import numpy as np
from PIL import Image

from paddleseg.transforms.seg_rand_augment import rand_augment_config
from paddleseg.cvlibs import manager


@manager.TRANSFORMS.add_component
class SegRandAugment:
    def __init__(self, n, m, policy_type):
        self.n = n
        self.m = m  # [0, 30]

        if policy_type == 'ImageNet':
            self.policy = rand_augment_config.ImageNetRandAug
        else:
            raise ValueError(f'Invalid policy type {policy_type}')

        self.op_list = self.policy['op_list']
        self.op_ranges = self.policy['op_ranges']
        self.op_funcs = self.policy['op_funcs']
        self.fail_cnt = 0

    def __call__(self, img, mask):
        img_ori = img.copy()
        mask_ori = mask.copy()
        smoke_pixels = get_smoke_pixels(mask, target_id=1)

        aug_valid = False
        aug_cnt = 0
        aug_max_cnt = 3
        ops_record = list()
        m = self.m
        mag_factor = 1.0
        while not aug_valid and aug_cnt < aug_max_cnt:
            if not self.policy:
                return img, mask

            ops = random.choices(self.op_list, k=self.n)
            for op_idx, op_name in enumerate(ops):
                min_val, max_val = self.op_ranges[op_name]
                op = self.op_funcs[op_name]
                val = (((float(m) / 30)
                      * float(max_val - min_val)) * mag_factor
                      + min_val)

                ops_record.append((op_idx, op_name, val))
                img, mask = op(img, mask, val)

            aug_smoke_pixels = get_smoke_pixels(mask, target_id=1)
            if aug_smoke_pixels < smoke_pixels * 0.8:
                mag_factor *= 0.8
                aug_cnt += 1
            else:
                aug_valid = True

        if aug_valid == False:
            '''
            record_str1 = f'-----------' \
                          f'RandAug fail to keep 0.8 smoke region ' \
                          f'after aug {aug_cnt} times.' \
                          f'-----------'
            record_str2 = '; '.join(ops_record)
            print(record_str1)
            print(record_str2)
            '''
            self.fail_cnt += 1
            return img_ori, mask_ori
        return img, mask


def get_smoke_pixels(mask, target_id=1):
    if isinstance(mask, Image.Image):
        mask = np.asarray(mask)
    assert isinstance(mask, np.ndarray), \
        f'Invalid mask type {type(mask)}'
    pixel_values, pixel_counts = np.unique(mask, return_counts=True)
    if target_id in pixel_values:
        smoke_pixels = pixel_counts[pixel_values == target_id]
        assert len(smoke_pixels) == 1, \
            f'Invalid smoke pixel count {smoke_pixels}'
        return smoke_pixels[0]
    else:
        return 0