import cv2
import torch
import numpy as np
import random
from torch.utils import data

from config import config
from furnace.utils.img_utils import random_scale, random_mirror, normalize, \
    generate_random_crop_pos, random_crop_pad_to_shape


class TrainPre(object):
    def __init__(self, img_mean, img_std, ignore_label=255):
        self.img_mean = img_mean
        self.img_std = img_std
        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

    def id2trainId(self, label, reverse=False):
        label_copy = label.copy()
        if reverse:
            for v, k in self.id_to_trainid.items():
                label_copy[label == k] = v
        else:
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v
        return label_copy

    def __call__(self, img, gt):
        # change to trainID first
        gt = self.id2trainId(gt)

        img, gt = random_mirror(img, gt)

        if config.train_scale_array is not None:
            img, gt, scale = random_scale(img, gt, config.train_scale_array)

        crop_size = (config.image_height, config.image_width)
        crop_pos = generate_random_crop_pos(img.shape[:2], crop_size)

        p_img, _ = random_crop_pad_to_shape(img, crop_pos, crop_size, 0)
        p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 255)
        # why gt bound with eval_width??
        # p_gt = cv2.resize(p_gt, (config.eval_width, config.eval_height),
        #                 interpolation=cv2.INTER_NEAREST)
        p_gt = cv2.resize(p_gt, (config.image_width//config.gt_down_sampling, config.image_height//config.gt_down_sampling),
                          interpolation=cv2.INTER_NEAREST)

        p_img = p_img.transpose(2, 0, 1)

        extra_dict = None

        return p_img, p_gt, extra_dict


def get_train_loader(engine, dataset):
    data_setting = {'img_root': config.img_root_folder,
                    'gt_root': config.gt_root_folder,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source}
    train_preprocess = TrainPre(config.image_mean, config.image_std)

    train_dataset = dataset(data_setting, "train", train_preprocess,
                            config.batch_size * config.niters_per_epoch)

    train_sampler = None
    is_shuffle = True
    batch_size = config.batch_size


    if engine.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
        batch_size = config.batch_size // engine.world_size
        is_shuffle = False

        def _worker_init_fn(id):
            np.random.seed(seed=config.seed + engine.local_rank + id)
            random.seed(config.seed + engine.local_rank + id)
    else:
        _worker_init_fn = None

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   num_workers=config.num_workers,
                                   drop_last=True,
                                   shuffle=is_shuffle,
                                   pin_memory=True,
                                   sampler=train_sampler,
                                   worker_init_fn=_worker_init_fn)

    return train_loader, train_sampler
