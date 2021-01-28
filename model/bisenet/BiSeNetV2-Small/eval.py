#!/usr/bin/env python3
# encoding: utf-8
import os
import cv2
import argparse
import numpy as np

import torch
import torch.multiprocessing as mp

from config import config
from furnace.utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from furnace.utils.visualize import print_iou, show_img, show_prediction
from furnace.engine.evaluator import Evaluator
from furnace.engine.logger import get_logger
from furnace.seg_opr.metric import hist_info, compute_score
from furnace.utils.img_utils import pad_image_to_shape
from furnace.tools.benchmark import compute_speed, stat
from furnace.datasets.cityscapes import Cityscapes
from network import BiSeNetV2

logger = get_logger()


class SegEvaluator(Evaluator):
    def process_image(self, img, crop_size=None):
        p_img = img

        if img.shape[2] < 3:
            im_b = p_img
            im_g = p_img
            im_r = p_img
            p_img = np.concatenate((im_b, im_g, im_r), axis=2)

        if crop_size is not None:
            p_img, margin = pad_image_to_shape(p_img, crop_size,
                                               cv2.BORDER_CONSTANT, value=0)
            p_img = p_img.transpose(2, 0, 1)

            return p_img, margin

        p_img = p_img.transpose(2, 0, 1)

        return p_img

    def func_per_iteration(self, data, device):
        img = data['data']
        label = data['label']
        name = data['fn']

        # img = cv2.resize(img, (config.eval_width, config.eval_height),
        #                  interpolation=cv2.INTER_LINEAR)
        # label = cv2.resize(label, (config.eval_width, config.eval_height),
        #                    interpolation=cv2.INTER_NEAREST)

        pred = self.whole_eval(img,
                               (config.eval_height, config.eval_width),
                               device=device)
        # pred = cv2.resize(pred, dsize=(config.image_width, config.image_height),
        #                   interpolation=cv2.INTER_NEAREST)

        pred = cv2.resize(pred, dsize=(label.shape[1], label.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
        hist_tmp, labeled_tmp, correct_tmp = hist_info(config.num_classes,
                                                       pred,
                                                       label)
        results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp,
                        'correct': correct_tmp}

        if self.save_path is not None:
            fn = name + '.png'
            pred = show_prediction(Cityscapes.get_class_colors(), -1, img,
                                   label, label, show255=True)
            cv2.imwrite(os.path.join(self.save_path, fn), pred)
            logger.info('Save the image ' + fn)

        if self.show_image:
            colors = self.dataset.get_class_colors()
            image = img
            clean = np.zeros(label.shape)
            comp_img = show_img(colors, config.background, image, clean,
                                label,
                                pred)
            cv2.imshow('comp_image', comp_img)
            cv2.waitKey(0)

        return results_dict

    def compute_metric(self, results):
        hist = np.zeros((config.num_classes, config.num_classes))
        correct = 0
        labeled = 0
        count = 0
        for d in results:
            hist += d['hist']
            correct += d['correct']
            labeled += d['labeled']
            count += 1

        iu, mean_IU, _, mean_pixel_acc = compute_score(hist, correct,
                                                       labeled)
        result_line = print_iou(iu, mean_pixel_acc,
                                dataset.get_class_names(), True)
        return result_line


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='last', type=str)
    parser.add_argument('-d', '--devices', default='1', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--show_image', '-s', default=False,
                        action='store_true')
    parser.add_argument('--save_path', '-p', default=None)
    parser.add_argument('--input_size', type=str, default='1x3x512x1024',
                        help='Input size. '
                             'channels x height x width (default: 1x3x224x224)')
    parser.add_argument('-speed', '--speed_test', action='store_true')
    parser.add_argument('--iteration', type=int, default=5000)
    parser.add_argument('-summary', '--summary', action='store_true')

    args = parser.parse_args()
    all_dev = parse_devices(args.devices)

    network = BiSeNetV2(config.num_classes, is_training=False, criterion=None)
    data_setting = {'img_root': config.img_root_folder,
                    'gt_root': config.gt_root_folder,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source}
    dataset = Cityscapes(data_setting, 'val', None)

    if args.speed_test:
        device = all_dev[0]
        logger.info("=========DEVICE:%s SIZE:%s=========" % (
            torch.cuda.get_device_name(device), args.input_size))
        input_size = tuple(int(x) for x in args.input_size.split('x'))
        compute_speed(network, input_size, device, args.iteration)
    elif args.summary:
        input_size = tuple(int(x) for x in args.input_size.split('x'))
        stat(network, input_size)
    else:
        with torch.no_grad():
            segmentor = SegEvaluator(dataset, config.num_classes,
                                     config.image_mean,
                                     config.image_std, network,
                                     config.eval_scale_array, config.eval_flip,
                                     all_dev, args.verbose, args.save_path,
                                     args.show_image)
            segmentor.run(config.snapshot_dir, args.epochs, config.val_log_file,
                          config.link_val_log_file)
