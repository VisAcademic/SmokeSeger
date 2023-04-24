"""
@File  : seg_auto_image_ops.py
@Author: tao.jing
@Date  : 2022/1/13
@Desc  :
"""
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

__all__ = [
    'get_color_map_list',
    'seg_shear_x',
    'seg_shear_y',
    'seg_translate_x_relative',
    'seg_translate_y_relative',
    'seg_rotate',
    'seg_auto_contrast',
    'seg_invert',
    'seg_equalize',
    'seg_solarize',
    'seg_posterize',
    'seg_contrast',
    'seg_contrast_no_rand_neg',
    'seg_color',
    'seg_color_no_rand_neg',
    'seg_brightness',
    'seg_brightness_no_rand_neg',
    'seg_sharpness',
    'seg_sharpness_no_rand_neg'
]


def get_color_map_list(num_classes):
    """
    Returns the color map for visualizing the segmentation mask,
    which can support arbitrary number of classes.
    Args:
        num_classes (int): Number of classes.
    Returns:
        (list). The color map.
    """
    num_classes += 1
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = color_map[3:]
    return color_map


def np_input_decorator(func):
    def wrapper(image, mask, magnitude, **kwargs):
        input_ndarray = False
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
            mask = Image.fromarray(mask, mode='P')
            mask.putpalette(get_color_map_list(256))
            input_ndarray = True
        image, mask = func(image, mask, magnitude, **kwargs)
        if input_ndarray:
            return np.asarray(image), np.asarray(mask)
        return image, mask
    return wrapper


@np_input_decorator
def seg_shear_x(image, mask,
                magnitude,
                fillcolor=(128, 128, 128),
                mask_fillcolor=0):
    factor = magnitude * random.choice([-1, 1]) # random negative
    image = image.transform(image.size, Image.AFFINE,
                           (1, factor, 0, 0, 1, 0), fillcolor=fillcolor)
    mask = mask.transform(mask.size, Image.AFFINE,
                          (1, factor, 0, 0, 1, 0), fillcolor=mask_fillcolor)
    return image, mask


@np_input_decorator
def seg_shear_y(image, mask,
                magnitude,
                fillcolor=(128, 128, 128),
                mask_fillcolor=0):
    factor = magnitude * random.choice([-1, 1]) # random negative
    image = image.transform(image.size, Image.AFFINE,
                            (1, 0, 0, factor, 1, 0), fillcolor=fillcolor)
    mask = mask.transform(mask.size, Image.AFFINE,
                          (1, 0, 0, factor, 1, 0), fillcolor=mask_fillcolor)
    return image, mask


@np_input_decorator
def seg_translate_x_relative(image, mask,
                             magnitude,
                             fillcolor=(128, 128, 128),
                             mask_fillcolor=0):
    pixels = magnitude * image.size[0]
    pixels = pixels * random.choice([-1, 1]) # random negative
    image = image.transform(image.size, Image.AFFINE,
                            (1, 0, pixels, 0, 1, 0), fillcolor=fillcolor)
    mask = mask.transform(mask.size, Image.AFFINE,
                          (1, 0, pixels, 0, 1, 0), fillcolor=mask_fillcolor)
    return image, mask


@np_input_decorator
def seg_translate_y_relative(image, mask,
                             magnitude,
                             fillcolor=(128, 128, 128),
                             mask_fillcolor=0):
    pixels = magnitude * image.size[0]
    pixels = pixels * random.choice([-1, 1]) # random negative
    image = image.transform(image.size, Image.AFFINE,
                    (1, 0, 0, 0, 1, pixels), fillcolor=fillcolor)
    mask = mask.transform(mask.size, Image.AFFINE,
                          (1, 0, 0, 0, 1, pixels), fillcolor=mask_fillcolor)
    return image, mask


'''
# (a, b, c, d, e, f)
# (x, y) -> (ax + by + c, dx + ey + f)
@np_input_decorator
def seg_translate_x_absolute(image, mask,
                             magnitude,
                             fillcolor=(128, 128, 128),
                             mask_fillcolor=0):
    magnitude = magnitude * random.choice([-1, 1]) # random negative
    image = image.transform(image.size, Image.AFFINE,
                            (1, 0, magnitude, 0, 1, 0), fillcolor=fillcolor)
    mask = mask.transform(mask.size, Image.AFFINE,
                            (1, 0, magnitude, 0, 1, 0), fillcolor=mask_fillcolor)
    return image, mask


@np_input_decorator
def seg_translate_y_absolute(image, mask,
                             magnitude,
                             fillcolor=(128, 128, 128),
                             mask_fillcolor=0):
    magnitude = magnitude * random.choice([-1, 1]) # random negative
    image = image.transform(image.size, Image.AFFINE,
                            (1, 0, 0, 0, 1, magnitude), fillcolor=fillcolor)
    mask = mask.transform(mask.size, Image.AFFINE,
                            (1, 0, 0, 0, 1, magnitude), fillcolor=mask_fillcolor)
    return image, mask
'''


@np_input_decorator
def seg_rotate(image, mask, magnitude):
    _palette = mask.getpalette()
    image_rot = image.convert("RGBA").rotate(magnitude)
    mask_rot = mask.convert("L").rotate(magnitude)
    image_ret = Image.composite(image_rot,
                           Image.new('RGBA', image_rot.size, (128, ) * 4),
                           image_rot).convert(image.mode)
    mask_ret = Image.composite(mask_rot,
                          Image.new('L', mask_rot.size, 0),
                          mask_rot)#.convert(mode='L')
    pixel_eles = (np.unique(mask_ret))
    if len(pixel_eles) == 3:
        mask_ret = mask_ret.point(lambda x: 1 if x > pixel_eles[1] else 0)
    elif len(pixel_eles) == 2:
        mask_ret = mask_ret.point(lambda x: 1 if x > pixel_eles[0] else 0)
    else:
        assert False, \
            f'Invalid mask content {pixel_eles}'

    mask_ret = Image.fromarray(np.asarray(mask_ret).astype(np.uint8), mode='P')
    mask_ret.putpalette(_palette)
    return image_ret, mask_ret


@np_input_decorator
def seg_auto_contrast(image, mask, magnitude=None):
    image = ImageOps.autocontrast(image)
    return image, mask


# Not involved!
@np_input_decorator
def seg_invert(image, mask, magnitude=None):
    image = ImageOps.invert(image)
    return image, mask


@np_input_decorator
def seg_equalize(image, mask, magnitude=None):
    image = ImageOps.equalize(image)
    return image, mask


# Not involved!
@np_input_decorator
def seg_solarize(image, mask, magnitude):
    image = ImageOps.solarize(image, magnitude)
    return image, mask


# Not involved!
@np_input_decorator
def seg_posterize(image, mask, magnitude):
    magnitude = int(magnitude)
    image = ImageOps.posterize(image, magnitude)
    return image, mask


@np_input_decorator
def seg_contrast(image, mask, magnitude):
    magnitude = magnitude * random.choice([-1, 1]) # random negative
    image = ImageEnhance.Contrast(image).enhance(1 + magnitude)
    return image, mask


@np_input_decorator
def seg_contrast_no_rand_neg(image, mask, magnitude):
    image = ImageEnhance.Contrast(image).enhance(magnitude)
    return image, mask


@np_input_decorator
def seg_color(image, mask, magnitude):
    magnitude = magnitude * random.choice([-1, 1]) # random negative
    image = ImageEnhance.Color(image).enhance(1 + magnitude)
    return image, mask


@np_input_decorator
def seg_color_no_rand_neg(image, mask, magnitude):
    image = ImageEnhance.Color(image).enhance(magnitude)
    return image, mask


@np_input_decorator
def seg_brightness(image, mask, magnitude):
    magnitude = magnitude * random.choice([-1, 1]) # random negative
    image = ImageEnhance.Brightness(image).enhance(1 + magnitude)
    return image, mask


@np_input_decorator
def seg_brightness_no_rand_neg(image, mask, magnitude):
    image = ImageEnhance.Brightness(image).enhance(magnitude)
    return image, mask


@np_input_decorator
def seg_sharpness(image, mask, magnitude):
    magnitude = magnitude * random.choice([-1, 1]) # random negative
    image = ImageEnhance.Sharpness(image).enhance(1 + magnitude)
    return image, mask


@np_input_decorator
def seg_sharpness_no_rand_neg(image, mask, magnitude):
    image = ImageEnhance.Sharpness(image).enhance(magnitude)
    return image, mask
