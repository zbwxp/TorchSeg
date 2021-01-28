from __future__ import division
import os.path as osp
import sys
import argparse
from tqdm import tqdm
import numpy as np
import random

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
# from torch.nn.parallel import DistributedDataParallel

from config import config
from dataloader import get_train_loader
from network import BiSeNetV2
from furnace.datasets import Cityscapes
from furnace.utils.init_func import init_weight, group_weight
from furnace.engine.lr_policy import PolyLR  #WarmupPolyLR
from furnace.engine.engine import Engine
from furnace.seg_opr.loss_opr import ProbOhemCrossEntropy2d

try:
    from apex.parallel import DistributedDataParallel, SyncBatchNorm
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex .")

parser = argparse.ArgumentParser()

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not engine.distributed:
        engine.local_rank = 0

    torch.manual_seed(config.seed + engine.local_rank)
    torch.cuda.manual_seed(config.seed + engine.local_rank)
    torch.cuda.manual_seed_all(config.seed + engine.local_rank)
    np.random.seed(seed=config.seed + engine.local_rank)
    random.seed(config.seed + engine.local_rank)

    # data loader
    train_loader, train_sampler = get_train_loader(engine, Cityscapes)

    min_kept = 100000
    criterion = ProbOhemCrossEntropy2d(ignore_label=255, thresh=0.7,
                                       min_kept=min_kept,
                                       use_weight=False)

    if engine.distributed:
        BatchNorm2d = SyncBatchNorm
    else:
        BatchNorm2d = nn.BatchNorm2d
    model = BiSeNetV2(config.num_classes, is_training=True, criterion=criterion,
                      norm_layer=BatchNorm2d)

    # group weight and config optimizer
    base_lr = config.lr

    params_list = []
    params_list = group_weight(params_list, model.semantic_branch,
                               BatchNorm2d, base_lr)
    params_list = group_weight(params_list, model.detail_branch,
                               BatchNorm2d, base_lr)
    params_list = group_weight(params_list, model.ffm, BatchNorm2d,
                               base_lr)
    params_list = group_weight(params_list, model.heads, BatchNorm2d,
                               base_lr)

    optimizer = torch.optim.SGD(params_list,
                                lr=base_lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)

    # config lr policy
    warm_iteration = config.warm_epochs * config.niters_per_epoch
    total_iteration = config.nepochs * config.niters_per_epoch + warm_iteration
    # lr_policy = WarmupPolyLR(config.warm_lr, warm_iteration, base_lr,
    #                          config.lr_power, total_iteration)
    lr_policy = PolyLR(base_lr, config.lr_power, total_iteration)


    if engine.distributed:
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(model)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model = DataParallelModel(model, device_ids=engine.devices)
        model.to(device)

    engine.register_state(dataloader=train_loader, model=model,
                          optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    model.train()

    total_epochs = config.nepochs + config.warm_epochs
    for epoch in range(engine.state.epoch, total_epochs):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout,
                    bar_format=bar_format)
        dataloader = iter(train_loader)
        for idx in pbar:
            optimizer.zero_grad()
            engine.update_iteration(epoch, idx)

            minibatch = dataloader.next()
            imgs = minibatch['data']
            gts = minibatch['label']

            imgs = imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)

            loss = model(imgs, gts)

            # reduce the whole loss over multi-gpu
            if engine.distributed:
                reduced_loss = loss.clone()
                dist.all_reduce(reduced_loss, dist.ReduceOp.SUM)
                reduced_loss.div_(engine.world_size)
            else:
                reduced_loss = loss.clone()

            current_idx = epoch * config.niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)

            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr
            # for i in range(6, len(optimizer.param_groups)):
            #     optimizer.param_groups[i]['lr'] = lr

            loss.backward()
            optimizer.step()
            print_str = 'Epoch{}/{}'.format(epoch, total_epochs) \
                        + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.2e' % lr \
                        + ' loss=%.2f' % reduced_loss.item()

            pbar.set_description(print_str, refresh=False)

        if (epoch > total_epochs - 20) or (epoch % config.snapshot_iter == 0):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
            elif not engine.distributed:
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
