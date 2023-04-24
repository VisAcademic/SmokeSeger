"""
@File  : h5_dataset.py
@Author: tao.jing
@Date  : 2022/11/7
@Desc  :
"""
import io
from pathlib import Path
import h5py
import numpy as np
from PIL import Image
from paddleseg.datasets.dataset import Dataset
from paddleseg.cvlibs import manager
import paddleseg.transforms.functional as F


@manager.DATASETS.add_component
class ImgH5Dataset(Dataset):
    def __init__(self,
                 h5_file_path: str,
                 transforms,
                 dataset_root,
                 num_classes,
                 mode='eval',
                 train_path=None,
                 val_path=None,
                 test_path=None,
                 separator=' ',
                 ignore_index=255,
                 edge=False):
        super(ImgH5Dataset, self).__init__(transforms,
                                           dataset_root,
                                           num_classes,
                                           mode=mode,
                                           train_path=train_path,
                                           val_path=val_path,
                                           test_path=test_path,
                                           separator=separator,
                                           ignore_index=ignore_index,
                                           edge=edge)
        self.h5_file_path = h5_file_path
        self.img_dataset = dict()
        self.mask_dataset = dict()
        self.load_h5_file()

    def __getitem__(self, idx):
        image_path, label_path = self.file_list[idx]
        image_name = Path(image_path).stem
        # Get image ndarray
        img_bin = np.asarray(self.img_dataset[image_name])
        img_pil = Image.open(io.BytesIO(img_bin))
        img_pil_nd = np.asarray(img_pil.convert('RGB'))
        img_pil_nd = img_pil_nd[:, :, ::-1].astype(np.float32)

        # Get mask ndarray
        mask_bin = np.asarray(self.mask_dataset[image_name])
        mask_pil = Image.open(io.BytesIO(mask_bin))
        mask_pil_nd = np.asarray(mask_pil).astype(np.uint8)

        if self.mode == 'test':
            im, _ = self.transforms(im=img_pil_nd)
            im = im[np.newaxis, ...]
            return im, image_path
        elif self.mode == 'val':
            im, _ = self.transforms(im=img_pil_nd)
            label = mask_pil_nd
            label = label[np.newaxis, :, :]
            return im, label
        else:
            im, label = self.transforms(im=img_pil_nd, label=mask_pil_nd)
            if self.edge:
                edge_mask = F.mask_to_binary_edge(
                    label, radius=2, num_classes=self.num_classes)
                return im, label, edge_mask
            else:
                return im, label

    def load_h5_file(self):
        img_name_list = list()
        for image_path, label_path in self.file_list:
            img_name = Path(image_path).stem
            img_name_list.append(img_name)

        self.img_dataset.clear()
        self.mask_dataset.clear()
        with h5py.File(self.h5_file_path, 'r') as hf:
            img_grp = hf['image']
            mask_grp = hf['mask']
            for key in list(img_grp.keys()):
                if key not in img_name_list:
                    continue
                img_bin = np.asarray(img_grp[key])
                mask_bin = np.asarray(mask_grp[key])
                self.img_dataset[key] = img_bin
                self.mask_dataset[key] = mask_bin
        print(f'[{self.mode}] Load image num: {len(self.img_dataset)}.')

