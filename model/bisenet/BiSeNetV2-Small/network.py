# encoding: utf-8
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models import resnet50, resnet101, resnet152

from config import config
from _utils import SemanticBranch, DetailBranch, FeatureFusion, SegmentationHead


class BiSeNetV2(nn.Module):
    def __init__(self, out_planes, is_training, criterion,
                 norm_layer=nn.BatchNorm2d):
        super(BiSeNetV2, self).__init__()
        self.is_training = is_training

        self.semantic_branch = SemanticBranch([2, 2, 4], [32, 64, 128],
                                              expansion=6,
                                              norm_layer=norm_layer)
        self.detail_branch = DetailBranch(3, 128, norm_layer)

        conv_channel = 128

        self.ffm = FeatureFusion(conv_channel, conv_channel, norm_layer)

        if is_training:
            heads = [SegmentationHead(16, out_planes, 4, True, norm_layer),
                     SegmentationHead(32, out_planes, 8, True, norm_layer),
                     SegmentationHead(64, out_planes, 16, True, norm_layer),
                     SegmentationHead(conv_channel, out_planes, 32, True,
                                      norm_layer),
                     SegmentationHead(conv_channel, out_planes, 8, False,
                                      norm_layer)]
        else:
            heads = [None, None, None, None,
                     SegmentationHead(conv_channel, out_planes, 8, False,
                                      norm_layer)]
        self.heads = nn.ModuleList(heads)

        self.business_layer = []
        self.business_layer.append(self.heads)
        self.business_layer.append(self.ffm)

        self._initialize_weights()

        if is_training:
            self.criterion = criterion

    def forward(self, data, label=None):
        data = F.interpolate(data, scale_factor=0.5, mode='bilinear',
                             align_corners=True)
        detail_blocks = self.detail_branch(data)
        semantic_blocks = self.semantic_branch(data)

        fianl_fm = self.ffm(detail_blocks, semantic_blocks[-1])

        if self.is_training:
            aux_loss0 = self.criterion(self.heads[0](semantic_blocks[-5]),
                                       label)
            aux_loss1 = self.criterion(self.heads[1](semantic_blocks[-4]),
                                       label)
            aux_loss2 = self.criterion(self.heads[2](semantic_blocks[-3]),
                                       label)
            aux_loss3 = self.criterion(self.heads[3](semantic_blocks[-2]),
                                       label)
            # aux_loss4 = self.criterion(self.heads[4](context_blocks[-1]),
            #                            label)

            main_loss = self.criterion(self.heads[-1](fianl_fm), label)

            loss = main_loss + 0.4 * aux_loss0 + 0.4 * aux_loss1 + \
                   0.4 * aux_loss2 + 0.4 * aux_loss3

            return loss


        # return F.log_softmax(self.heads[3](semantic_blocks[-2]), dim=1)
        return F.log_softmax(self.heads[-1](fianl_fm), dim=1)

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'stem' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
