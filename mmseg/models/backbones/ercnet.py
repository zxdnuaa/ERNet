# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmcv.ops import DeformConv2dPack as DCN
from mmseg.ops import resize
from ..builder import BACKBONES, build_backbone
from .bisenetv1 import AttentionRefinementModule
from ..utils import InvertedResidualV3
import cv2 as cv
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair


class DWConv(nn.Module):
    def __init__(self, in_chans, out_chans, k=3, s=1, p=1):
        super(DWConv, self).__init__()
        self.depth_conv = nn.Sequential(
            nn.Conv2d(in_chans, in_chans, kernel_size=k, stride=s,
                      padding=p, groups=in_chans, bias=False),
            nn.BatchNorm2d(in_chans),
            nn.ReLU(inplace=True)
        )
        self.point_conv = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial1 = nn.Conv2d(
            dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv_spatial2 = nn.Conv2d(
            dim, dim, 3, stride=1, padding=3, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn1 = self.conv_spatial1(attn)
        attn2 = self.conv_spatial2(attn)
        attn = attn1 + attn2
        attn = self.conv1(attn)
        return u * attn


class SpatialAttention(nn.Module):
    def __init__(self, d_model, out_channels):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, out_channels, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        return x


class SobelBlur(nn.Module):
    def __init__(self):
        super(SobelBlur, self).__init__()
        kernel = [[-1, -1, -1],
                  [-1, 8, -1],
                  [-1, -1, -1]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x1 = F.conv2d(x1.unsqueeze(1), self.weight, padding=1)
        x2 = F.conv2d(x2.unsqueeze(1), self.weight, padding=1)
        x3 = F.conv2d(x3.unsqueeze(1), self.weight, padding=1)
        x = torch.cat([x1, x2, x3], dim=1)
        return x


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor=4,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, in_channels // 32)
        self.DW = DWConv(out_channels, out_channels)
        self.DWc = DWConv(out_channels, out_channels)
        self.DWs = DWConv(in_channels - 128, out_channels)
        self.conv0s = nn.Conv2d(out_channels, 1, kernel_size=1, stride=1, padding=0)
        self.conv0c = nn.Conv2d(out_channels, 1, kernel_size=1, stride=1, padding=0)

        self.conv0 = nn.Conv2d(in_channels - 128, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(in_channels - 128, mip, kernel_size=1, stride=1, padding=0)
        self.conv1c = nn.Conv2d(out_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()

    def forward(self, space, arms_out, boundary_targets):
        cx = torch.cat([arms_out, boundary_targets], dim=1)
        sx = torch.cat([space, boundary_targets], dim=1)
        cx0 = self.DWc(cx)
        cx0 = self.conv0c(cx0)
        sx0 = self.DWs(sx)
        sx0 = self.conv0s(sx0)

        n, c, h, w = sx.size()
        x_h = self.pool_h(cx)
        x_w = self.pool_w(cx).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        n1, c1, h1, w1 = cx.size()
        x_h1 = self.pool_h(cx)
        x_w1 = self.pool_w(cx).permute(0, 1, 3, 2)
        y1 = torch.cat([x_h1, x_w1], dim=2)
        y1 = self.conv1c(y1)
        y1 = self.bn1(y1)
        y1 = self.relu(y1)
        x_h1, x_w1 = torch.split(y1, [h1, w1], dim=2)
        x_w1 = x_w1.permute(0, 1, 3, 2)

        x_h1 = self.conv2(x_h1).sigmoid()
        x_w1 = self.conv3(x_w1).sigmoid()
        x_h1 = x_h1.expand(-1, -1, h1, w1)
        x_w1 = x_w1.expand(-1, -1, h1, w1)
        out = self.DW(cx0 * x_h * x_w + sx0 * x_h1 * x_w1)
        return out


class STDCModule(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 norm_cfg=None,
                 act_cfg=None,
                 num_convs=4,
                 fusion_type='add',
                 init_cfg=None):
        super(STDCModule, self).__init__(init_cfg=init_cfg)
        assert num_convs > 1
        assert fusion_type in ['add', 'cat']
        self.stride = stride
        self.with_downsample = True if self.stride == 2 else False
        self.fusion_type = fusion_type

        self.layers = ModuleList()
        conv_0 = ConvModule(
            in_channels, out_channels // 2, kernel_size=1, norm_cfg=norm_cfg)

        if self.with_downsample:
            self.downsample = ConvModule(
                out_channels // 2,
                out_channels // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=out_channels // 2,
                norm_cfg=norm_cfg,
                act_cfg=None)

            if self.fusion_type == 'add':
                self.layers.append(nn.Sequential(conv_0, self.downsample))
                self.skip = Sequential(
                    ConvModule(
                        in_channels,
                        in_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        groups=in_channels,
                        norm_cfg=norm_cfg,
                        act_cfg=None),
                    ConvModule(
                        in_channels,
                        out_channels,
                        1,
                        norm_cfg=norm_cfg,
                        act_cfg=None))
            else:
                self.layers.append(conv_0)
                self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.layers.append(conv_0)

        for i in range(1, num_convs):
            out_factor = 2 ** (i + 1) if i != num_convs - 1 else 2 ** i
            self.layers.append(
                ConvModule(
                    out_channels // 2 ** i,
                    out_channels // out_factor,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

    def forward(self, inputs):
        if self.fusion_type == 'add':
            out = self.forward_add(inputs)
        else:
            out = self.forward_cat(inputs)
        return out

    def forward_add(self, inputs):
        layer_outputs = []
        x = inputs.clone()
        for layer in self.layers:
            x = layer(x)
            layer_outputs.append(x)
        if self.with_downsample:
            inputs = self.skip(inputs)

        return torch.cat(layer_outputs, dim=1) + inputs

    def forward_cat(self, inputs):
        x0 = self.layers[0](inputs)
        layer_outputs = [x0]
        for i, layer in enumerate(self.layers[1:]):
            if i == 0:
                if self.with_downsample:
                    x = layer(self.downsample(x0))
                else:
                    x = layer(x0)
            else:
                x = layer(x)
            layer_outputs.append(x)
        if self.with_downsample:
            layer_outputs[0] = self.skip(x0)
        return torch.cat(layer_outputs, dim=1)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class GatedSpatialConv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(GatedSpatialConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, 'zeros')

        self._gate_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels + 1),
            nn.Conv2d(in_channels + 1, in_channels + 1, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels + 1, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, input_features, gating_features):
        alphas = self._gate_conv(torch.cat([input_features, gating_features], dim=1))

        input_features = (input_features * (alphas + 1))
        return F.conv2d(input_features, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class BasicBlock1(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock1, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        residual = x

        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)

        out1 = self.conv2(out1)
        out1 = self.bn2(out1)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = residual + out1
        out1 = self.relu(out1)
        out = self.relu(out)

        return out, out1


class ERM(nn.Module):
    def __init__(self, inch, outch):
        super(ERM, self).__init__()
        self.res1 = BasicBlock1(128, outch, stride=1, downsample=None)
        self.res2 = BasicBlock1(128, outch, stride=1, downsample=None)
        self.conv1 = nn.Conv2d(inch, 1, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(64, outch, kernel_size=1, stride=1)
        self.gate = GatedSpatialConv2d(outch, outch)
        self.bdam = SpatialAttention(outch, outch)

    def forward(self, x, f_x):
        u_0 = x
        f_x = self.conv1(f_x)
        u_1, delta_u_0 = self.res1(u_0)
        _, u_2 = self.res2(u_1)
        u_3_pre = self.gate(u_2, f_x)
        u_3 = 3 * delta_u_0 + u_2 + u_3_pre
        u_3 = self.bdam(u_3)
        return u_3


@BACKBONES.register_module()
class ERCNet(BaseModule):
    arch_settings = {
        'STDCNet1': [(2, 1), (2, 1), (2, 1)],
        'STDCNet2': [(2, 1, 1, 1), (2, 1, 1, 1, 1), (2, 1, 1)]
    }

    def __init__(self,
                 stdc_type,
                 in_channels,
                 channels,
                 bottleneck_type,
                 norm_cfg,
                 act_cfg,
                 num_convs=4,
                 with_final_conv=False,
                 pretrained=None,
                 init_cfg=None):
        super(ERCNet, self).__init__(init_cfg=init_cfg)
        assert stdc_type in self.arch_settings, \
            f'invalid structure {stdc_type} for STDCNet.'
        assert bottleneck_type in ['add', 'cat'], \
            f'bottleneck_type must be `add` or `cat`, got {bottleneck_type}'

        assert len(channels) == 5, \
            f'invalid channels length {len(channels)} for STDCNet.'

        self.in_channels = in_channels
        self.channels = channels
        self.stage_strides = self.arch_settings[stdc_type]
        self.prtrained = pretrained
        self.num_convs = num_convs
        self.with_final_conv = with_final_conv

        self.stages = ModuleList([
            ConvModule(
                self.in_channels,
                self.channels[0],
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                self.channels[0],
                self.channels[1],
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        ])
        self.num_shallow_features = len(self.stages)

        for strides in self.stage_strides:
            idx = len(self.stages) - 1
            self.stages.append(
                self._make_stage(self.channels[idx], self.channels[idx + 1],
                                 strides, norm_cfg, act_cfg, bottleneck_type))
        if self.with_final_conv:
            self.final_conv = ConvModule(
                self.channels[-1],
                max(1024, self.channels[-1]),
                1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)

    def _make_stage(self, in_channels, out_channels, strides, norm_cfg,
                    act_cfg, bottleneck_type):
        layers = []
        for i, stride in enumerate(strides):
            layers.append(
                STDCModule(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    stride,
                    norm_cfg,
                    act_cfg,
                    num_convs=self.num_convs,
                    fusion_type=bottleneck_type))
        if strides != (2, 1, 1):
            layers.append(AttentionRefinementModule(out_channels, out_channels))
        else:
            layers.append(SpatialAttention(out_channels, out_channels))
        return Sequential(*layers)

    def forward(self, x):

        outs = []
        for stage in self.stages:
            x = stage(x)
            outs.append(x)
        if self.with_final_conv:
            outs[-1] = self.final_conv(outs[-1])
        outs = outs[self.num_shallow_features:]
        return tuple(outs)


@BACKBONES.register_module()
class ERCNetPathNet(BaseModule):
    def __init__(self,
                 backbone_cfg,
                 last_in_channels=(1024, 512),
                 out_channels=128,
                 ffm_cfg=dict(in_channels=512, out_channels=256, scale_factor=4),
                 upsample_mode='nearest',
                 align_corners=None,
                 norm_cfg=dict(type='BN'),
                 init_cfg=None):
        super(ERCNetPathNet, self).__init__(init_cfg=init_cfg)
        self.backbone = build_backbone(backbone_cfg)
        self.arms = ModuleList()
        self.convs = ModuleList()
        for channels in (512, 1024):
            self.arms.append(ERM(channels, out_channels))
            self.convs.append(
                ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    norm_cfg=norm_cfg))

        self.conv_avg_stage5 = ConvModule(
            last_in_channels[0], out_channels, 1, norm_cfg=norm_cfg)
        self.conv_avg_stage4 = ConvModule(
            last_in_channels[1], out_channels, 1, norm_cfg=norm_cfg)
        self.upsample_mode = upsample_mode
        self.align_corners = align_corners
        self.sigmoid = nn.Sigmoid()
        self.SobelBlur = SobelBlur()
        self.stages = ModuleList([
            ConvModule(
                3,
                32,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU'), ),
            ConvModule(
                32,
                128,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU'), )
        ])
        self.conv_fuse = ConvModule(
            last_in_channels[0], out_channels, 1, norm_cfg=norm_cfg)
        self.cff = CoordAtt(**ffm_cfg)

    def forward(self, x):
        boundary_edge = F.interpolate(x, scale_factor=1 / 8, mode='bilinear', align_corners=True)
        boundary_edge = self.SobelBlur(boundary_edge)
        for stage in self.stages:
            boundary_edge = stage(boundary_edge)
        outs = list(self.backbone(x))
        boundary_edge = F.interpolate(boundary_edge, size=(outs[0].shape[2], outs[0].shape[3]), mode='bilinear')
        avg_feat = self.conv_avg_stage5(outs[-1])
        avg_feat2 = self.conv_avg_stage4(outs[1])
        feature_up = resize(
            avg_feat,
            size=outs[1].shape[2:],
            mode=self.upsample_mode,
            align_corners=self.align_corners)
        arms_out = []
        for i in range(len(self.arms)):
            out_up = F.interpolate(outs[i + 1], size=(boundary_edge.shape[2], boundary_edge.shape[3]), mode='bilinear')
            boundary_edge = self.arms[i](boundary_edge, out_up)
            arms_out.append(feature_up)
            feature_up = resize(
                avg_feat2,
                size=outs[0].shape[2:],
                mode=self.upsample_mode,
                align_corners=self.align_corners)
        arms_out[0], arms_out[1] = arms_out[1], arms_out[0]
        feat_fuse = self.cff(outs[0], arms_out[0], boundary_edge)
        outputs = [outs[0]] + list(arms_out) + [feat_fuse]
        return tuple(outputs)
