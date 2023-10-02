    # Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F

from ..builder import HEADS
from .fcn_head import FCNHead
import cv2

@HEADS.register_module()
class STDCHead(FCNHead):
    """This head is the implementation of `Rethinking BiSeNet For Real-time
    Semantic Segmentation <https://arxiv.org/abs/2104.13188>`_.

    Args:
        boundary_threshold (float): The threshold of calculating boundary.
            Default: 0.1.
    """

    def __init__(self, boundary_threshold=0.1, **kwargs):
        super(STDCHead, self).__init__(**kwargs)
        self.boundary_threshold = boundary_threshold
        # Using register buffer to make laplacian kernel on the same
        # device of `seg_label`.
        self.register_buffer(
            'laplacian_kernel',
            torch.tensor([-1, -1, -1, -1, 8, -1, -1, -1, -1],
                         dtype=torch.float32,
                         requires_grad=False).reshape((1, 1, 3, 3)))
        self.fusion_kernel = torch.nn.Parameter(
            torch.tensor([[6. / 10], [3. / 10], [1. / 10]],
                         dtype=torch.float32).reshape(1, 3, 1, 1),
            requires_grad=False)

    def losses(self, seg_logit, seg_label):
        """Compute Detail Aggregation Loss."""
        # Note: The paper claims `fusion_kernel` is a trainable 1x1 conv
        # parameters. However, it is a constant in original repo and other
        # codebase because it would not be added into computation graph
        # after threshold operation.
        # print('seg_logit=',seg_logit.shape,'seg_label=',seg_label.shape)
        seg_label = seg_label.to(self.laplacian_kernel)
        boundary_targets = F.conv2d(
            seg_label, self.laplacian_kernel, padding=1)  # 做拉普拉斯卷积，同时进行stride=1的卷积
        boundary_targets = boundary_targets.clamp(min=0)  # clamp(min=0),就相当于relu函数的等效。boundary_targets=0时，clamp选择导数为1
        boundary_targets[boundary_targets > self.boundary_threshold] = 1
        boundary_targets[boundary_targets <= self.boundary_threshold] = 0
        # seg_label2 = cv2.medianBlur(seg_label,3)
        boundary_targets_x2 = F.conv2d(
            seg_label, self.laplacian_kernel, stride=2, padding=1)  # 做拉普拉斯卷积，同时进行stride=2的卷积
        boundary_targets_x2 = boundary_targets_x2.clamp(min=0)

        boundary_targets_x4 = F.conv2d(
            seg_label, self.laplacian_kernel, stride=4, padding=1)  # 做拉普拉斯卷积，同时进行stride=4的卷积
        boundary_targets_x4 = boundary_targets_x4.clamp(min=0)

        boundary_targets_x4_up = F.interpolate(
            boundary_targets_x4, boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x2_up = F.interpolate(
            boundary_targets_x2, boundary_targets.shape[2:], mode='nearest')

        boundary_targets_x2_up[
            boundary_targets_x2_up > self.boundary_threshold] = 1
        boundary_targets_x2_up[
            boundary_targets_x2_up <= self.boundary_threshold] = 0

        boundary_targets_x4_up[
            boundary_targets_x4_up > self.boundary_threshold] = 1
        boundary_targets_x4_up[
            boundary_targets_x4_up <= self.boundary_threshold] = 0

        boudary_targets_pyramids = torch.stack(
            (boundary_targets, boundary_targets_x2_up, boundary_targets_x4_up),
            dim=1)  # 拼接

        boudary_targets_pyramids = boudary_targets_pyramids.squeeze(2)
        boudary_targets_pyramid = F.conv2d(boudary_targets_pyramids,
                                           self.fusion_kernel)

        boudary_targets_pyramid[
            boudary_targets_pyramid > self.boundary_threshold] = 1
        boudary_targets_pyramid[
            boudary_targets_pyramid <= self.boundary_threshold] = 0

        target = torch.squeeze(boudary_targets_pyramid, 1)
        print("seg_logit=", seg_logit.size(), "target=", target.size())
        loss = super(STDCHead, self).losses(seg_logit,
                                            boudary_targets_pyramid.long())
        return loss
