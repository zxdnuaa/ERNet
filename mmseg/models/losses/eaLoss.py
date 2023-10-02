import torch
import torch.nn as nn

from ..builder import LOSSES
from .utils import weighted_loss
import torch.nn.functional as F



@weighted_loss
def myloss(logits, label, n_classes, **kwargs):
    radius = 1
    alpha = 0.01
    smooth = 1
    p = 2
    prediction = F.softmax(logits, dim=1)
    label = label.unsqueeze(1)
    ks = 2 * radius + 1
    filt1 = torch.ones(1, 1, ks, ks)
    filt1[:, :, radius:2 * radius, radius:2 * radius] = -8
    filt1.requires_grad = False
    filt1 = filt1.cuda()
    lbedge = F.conv2d(label.float(), filt1, bias=None, stride=1, padding=radius)
    lbedge = 1 - torch.eq(lbedge, 0).float()
    filt2 = torch.ones(n_classes, 1, ks, ks)
    filt2[:, :, radius:2 * radius, radius:2 * radius] = -8
    filt2.requires_grad = False
    filt2 = filt2.cuda()
    prededge = F.conv2d(prediction.float(), filt2, bias=None,
                        stride=1, padding=radius, groups=n_classes)
    norm = torch.sum(torch.pow(prededge, 2), 1).unsqueeze(1)
    prededge = norm / (norm + alpha)
    predict, target = prededge.float(), lbedge.float()
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)

    num = 2 * torch.sum(torch.mul(predict, target), dim=1) + smooth
    den = torch.sum(predict.pow(p) + target.pow(p), dim=1) + smooth

    loss = 1 - num / den
    return loss.sum()


@LOSSES.register_module
class eaLoss(nn.Module):

    def __init__(self, use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.5,
                 reduction='mean', loss_weight=1.0, class_weight=None, loss_name='loss_focal'):
        super(eaLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self._loss_name = loss_name

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255,
                **kwargs
                ):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        target = torch.squeeze(target, 1)
        num_classes = pred.shape[1]
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * myloss(
            pred, target, n_classes=num_classes, reduction=reduction, avg_factor=avg_factor)
        return loss

    @property
    def loss_name(self):
        return self._loss_name
