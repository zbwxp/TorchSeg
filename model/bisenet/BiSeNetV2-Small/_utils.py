from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F

from furnace.seg_opr.seg_oprs import ConvBnRelu, SELayer


class SeparableConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=1, stride=1, padding=0, dilation=1,
                 has_relu=True, norm_layer=nn.BatchNorm2d):
        super(SeparableConvBnRelu, self).__init__()

        self.stride = stride
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                               padding, dilation, groups=in_channels,
                               bias=False)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, 1, 1, 1,
                               groups=in_channels,
                               bias=False)
        self.bn = norm_layer(in_channels)
        self.point_wise_cbr = ConvBnRelu(in_channels, out_channels, 1, 1, 0,
                                         has_bn=True, norm_layer=norm_layer,
                                         has_relu=has_relu, has_bias=False)

    def forward(self, x):
        x = self.conv1(x)
        if self.stride == 2:
            x = self.conv2(x)
        x = self.bn(x)
        x = self.point_wise_cbr(x)
        return x


class ContextEmbedding(nn.Module):
    def __init__(self, in_channels, norm_layer=nn.BatchNorm2d):
        super(ContextEmbedding, self).__init__()

        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            norm_layer(in_channels),
            ConvBnRelu(in_channels, in_channels, 1, 1, 0,
                       has_bn=True, norm_layer=norm_layer,
                       has_relu=False, has_bias=False)
        )

        self.point_wise_cbr = ConvBnRelu(in_channels, in_channels, 3, 1, 1,
                                         has_bn=True, norm_layer=norm_layer,
                                         has_relu=True, has_bias=False)

    def forward(self, x):
        global_x = self.global_branch(x)
        x = self.point_wise_cbr(
            x + F.interpolate(global_x, size=x.size()[2:],
                              mode='bilinear', align_corners=True))

        return x


class GatherExciteLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, expansion,
                 stride, norm_layer=nn.BatchNorm2d):
        super(GatherExciteLayer, self).__init__()
        self.in_channels = in_channels
        self.stride = stride
        mid_channels = round(in_channels * expansion)

        if self.stride == 1:
            self.out_channels = in_channels
            self.down_sample = nn.Identity()
        elif self.stride == 2:
            self.out_channels = out_channels
            self.down_sample = SeparableConvBnRelu(self.in_channels,
                                                   self.out_channels, 3,
                                                   stride, 1, 1,
                                                   has_relu=False,
                                                   norm_layer=norm_layer)
        self.residual_branch = nn.Sequential(
            ConvBnRelu(self.in_channels, mid_channels, 3, 1, 1,
                       has_relu=True, norm_layer=norm_layer),
            SeparableConvBnRelu(mid_channels, self.out_channels,
                                kernel_size, stride, kernel_size // 2, 1,
                                has_relu=False, norm_layer=norm_layer))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = self.down_sample(x)
        residual = self.residual_branch(x)
        return self.relu(shortcut + residual)


class StemBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, stride,
                 norm_layer=nn.BatchNorm2d):
        super(StemBlock, self).__init__()
        self.in_channels = in_channels

        self.steam_conv1 = ConvBnRelu(3, self.in_channels, 3, 2, 1, has_bn=True,
                                      norm_layer=norm_layer, has_relu=True,
                                      has_bias=False)
        self.steam_branch_a = nn.Sequential(
            ConvBnRelu(self.in_channels, self.in_channels // 2, 1, 1, 0,
                       has_bn=True, norm_layer=norm_layer,
                       has_relu=True, has_bias=False),
            ConvBnRelu(self.in_channels // 2, self.in_channels, kernel_size,
                       stride, kernel_size // 2,
                       has_bn=True, norm_layer=norm_layer,
                       has_relu=True, has_bias=False)
        )
        self.steam_branch_b = nn.MaxPool2d(kernel_size=3, stride=stride,
                                           padding=1)
        self.steam_conv3 = ConvBnRelu(self.in_channels * 2, self.in_channels,
                                      1, 1, 0,
                                      has_bn=True, norm_layer=norm_layer,
                                      has_relu=True, has_bias=False)

    def forward(self, x):
        x = self.steam_conv1(x)
        branch_a = self.steam_branch_a(x)
        branch_b = self.steam_branch_b(x)
        x = torch.cat([branch_a, branch_b], dim=1)
        x = self.steam_conv3(x)

        return x


class SemanticBranch(nn.Module):
    def __init__(self, layers, channels, expansion, norm_layer=nn.BatchNorm2d):
        super(SemanticBranch, self).__init__()
        self.in_channels = 16

        self.stem_block = StemBlock(self.in_channels, 3, 2, norm_layer)

        self.layer1 = self._make_layer(layers[0], channels[0], expansion,
                                       norm_layer, stride=2)
        self.layer2 = self._make_layer(layers[1], channels[1], expansion,
                                       norm_layer, stride=2)
        self.layer3 = self._make_layer(layers[2], channels[2], expansion,
                                       norm_layer, stride=2)
        self.context_embedding = ContextEmbedding(channels[2], norm_layer)

    def _make_layer(self, blocks, out_channels, expansion, norm_layer,
                    stride=1):
        layers = []
        layers.append(
            GatherExciteLayer(self.in_channels, out_channels, 3, expansion,
                              stride, norm_layer=norm_layer))
        for i in range(1, blocks):
            layers.append(
                GatherExciteLayer(out_channels, out_channels, 3, expansion,
                                  1, norm_layer=norm_layer))
        self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        blocks = []

        x = self.stem_block(x)
        blocks.append(x)

        x = self.layer1(x);
        blocks.append(x)
        x = self.layer2(x);
        blocks.append(x)
        x = self.layer3(x);
        blocks.append(x)
        x = self.context_embedding(x);
        blocks.append(x)

        return blocks


class DetailBranch(nn.Module):
    def __init__(self, in_planes, out_channels, norm_layer=nn.BatchNorm2d):
        super(DetailBranch, self).__init__()
        inner_channel = 64
        self.conv1_1 = ConvBnRelu(in_planes, inner_channel, 3, 2, 1,
                                  norm_layer=norm_layer,
                                  has_bn=True, has_relu=True, has_bias=False)
        self.conv1_2 = ConvBnRelu(inner_channel, inner_channel, 3, 1, 1,
                                  norm_layer=norm_layer,
                                  has_bn=True, has_relu=True, has_bias=False)

        self.conv2_1 = ConvBnRelu(inner_channel, inner_channel, 3, 2, 1,
                                  norm_layer=norm_layer,
                                  has_bn=True, has_relu=True, has_bias=False)
        self.conv2_2 = ConvBnRelu(inner_channel, inner_channel, 3, 1, 1,
                                  norm_layer=norm_layer,
                                  has_bn=True, has_relu=True, has_bias=False)
        self.conv2_3 = ConvBnRelu(inner_channel, inner_channel, 3, 1, 1,
                                  norm_layer=norm_layer,
                                  has_bn=True, has_relu=True, has_bias=False)

        self.conv3_1 = ConvBnRelu(inner_channel, out_channels, 3, 2, 1,
                                  norm_layer=norm_layer,
                                  has_bn=True, has_relu=True, has_bias=False)
        self.conv3_2 = ConvBnRelu(out_channels, out_channels, 3, 1, 1,
                                  norm_layer=norm_layer,
                                  has_bn=True, has_relu=True, has_bias=False)
        self.conv3_3 = ConvBnRelu(out_channels, out_channels, 3, 1, 1,
                                  norm_layer=norm_layer,
                                  has_bn=True, has_relu=True, has_bias=False)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)

        return x


class BilinearAggregation(nn.Module):
    def __init__(self, in_planes, out_planes, base_scale, kernel_size, stride=1,
                 norm_layer=nn.BatchNorm2d):
        super(BilinearAggregation, self).__init__()
        self.out_planes = out_planes
        self.in_planes = in_planes
        self.base_scale = base_scale
        self.kernel_size = kernel_size
        self.stride = stride

        self.conv_d2d = SeparableConvBnRelu(in_planes, out_planes,
                                            kernel_size=3, stride=1, padding=1,
                                            dilation=1,
                                            has_relu=False,
                                            norm_layer=norm_layer)
        self.conv_d2c = nn.Sequential(
            ConvBnRelu(in_planes, out_planes, kernel_size, 2,
                       kernel_size // 2,
                       has_bn=True,
                       norm_layer=norm_layer,
                       has_relu=False, has_bias=False),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1,
                         ceil_mode=False,
                         count_include_pad=False)
        )
        self.conv_c2c = SeparableConvBnRelu(in_planes, out_planes,
                                            kernel_size=3, stride=1, padding=1,
                                            dilation=1,
                                            has_relu=False,
                                            norm_layer=norm_layer)
        self.conv_c2d = ConvBnRelu(in_planes, out_planes, kernel_size, 1,
                                   kernel_size // 2,
                                   has_bn=True,
                                   norm_layer=norm_layer,
                                   has_relu=False, has_bias=False)
        self.refine_conv = ConvBnRelu(out_planes, out_planes, 3, 1, 1,
                                      has_bn=True, norm_layer=norm_layer,
                                      has_relu=False, has_bias=False)

    def forward(self, x_detail, x_context):
        x_d2d = self.conv_d2d(x_detail)
        x_d2c = self.conv_d2c(x_detail)

        x_c2c = self.conv_c2c(x_context)
        x_c2d = F.interpolate(self.conv_c2d(x_context), size=x_d2d.size()[2:],
                              mode='bilinear', align_corners=True)

        x_d = torch.sigmoid(x_c2d) * x_d2d
        x_c = x_d2c * torch.sigmoid(x_c2c)

        x = self.refine_conv(
            x_d + F.interpolate(x_c, size=x_d.size()[2:], mode='bilinear',
                                align_corners=True))

        return x


class FeatureFusion(nn.Module):
    def __init__(self, in_planes, out_planes, norm_layer=nn.BatchNorm2d):
        super(FeatureFusion, self).__init__()
        self.out_planes = out_planes

        self.bilinear_aggregation = BilinearAggregation(in_planes, out_planes,
                                                        base_scale=4,
                                                        kernel_size=3, stride=1,
                                                        norm_layer=norm_layer)

        self.block1 = GatherExciteLayer(out_planes, out_planes, 3, 1, 1,
                                        norm_layer)
        self.block2 = GatherExciteLayer(out_planes, out_planes, 3, 1, 1,
                                        norm_layer)

    def forward(self, x_detail, x_context):
        x_ensemble = self.bilinear_aggregation(x_detail, x_context)
        x = self.block1(x_ensemble)
        x = self.block2(x)

        return x


class SegmentationHead(nn.Module):
    def __init__(self, in_planes, out_planes, scale, is_aux=False,
                 norm_layer=nn.BatchNorm2d):
        super(SegmentationHead, self).__init__()
        if is_aux:
            self.conv_3x3 = ConvBnRelu(in_planes, 256, 3, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False)
        else:
            self.conv_3x3 = ConvBnRelu(in_planes, 64, 3, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False)
        if is_aux:
            self.conv_1x1 = nn.Conv2d(256, out_planes, kernel_size=1,
                                      stride=1, padding=0)
        else:
            self.conv_1x1 = nn.Conv2d(64, out_planes, kernel_size=1,
                                      stride=1, padding=0)
        self.scale = scale

    def forward(self, x):
        fm = self.conv_3x3(x)
        output = self.conv_1x1(fm)
        if self.scale > 1:
            output = F.interpolate(output, scale_factor=self.scale,
                                   mode='bilinear',
                                   align_corners=True)

        return output
