import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from lovasz_losses import lovasz_hinge

ALPHA = 0.5  # < 0.5 penalises FP more, > 0.5 penalises FN more
BETA = 0.5
GAMMA = 1
CE_RATIO = 0.5  # weighted contribution of modified CE loss compared to Dice loss


class DiceLoss(_Loss):
    def __init__(self, weight=None, size_avg=True):
        super(DiceLoss, self).__init__()

    @staticmethod
    def forward(logits, targets, smooth=1):
        logits = torch.sigmoid(logits)

        intersect = (logits * targets).sum()
        dice = (2. * intersect + smooth) / (logits.sum() + targets.sum() + smooth)

        return 1 - dice


class DiceBCELoss(_Loss):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    @staticmethod
    def forward(logits, targets, smooth=1):
        logits = torch.sigmoid(logits)

        intersection = (logits * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (logits.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(logits, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


class IoULoss(_Loss):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    @staticmethod
    def forward(logits, targets, smooth=1):
        logits = torch.sigmoid(logits)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (logits * targets).sum()
        total = (logits + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU


class FocalLoss(_Loss):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    @staticmethod
    def forward(logits, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        logits = torch.sigmoid(logits)

        BCE = F.binary_cross_entropy(logits, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE

        return focal_loss


class TverskyLoss(_Loss):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    @staticmethod
    def forward(logits, targets, smooth=1, alpha=ALPHA, beta=BETA):
        logits = torch.sigmoid(logits)

        # True Positives, False Positives & False Negatives
        TP = (logits * targets).sum()
        FP = ((1 - targets) * logits).sum()
        FN = (targets * (1 - logits)).sum()

        Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

        return 1 - Tversky


class FocalTverskyLoss(_Loss):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    @staticmethod
    def forward(logits, targets, smooth=1, alpha=ALPHA, beta=BETA, gamma=GAMMA):
        logits = torch.sigmoid(logits)

        # True Positives, False Positives & False Negatives
        TP = (logits * targets).sum()
        FP = ((1 - targets) * logits).sum()
        FN = (targets * (1 - logits)).sum()

        Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
        FocalTversky = (1 - Tversky) ** gamma

        return FocalTversky


class LovaszHingeLoss(_Loss):
    def __init__(self, weight=None, size_average=True):
        super(LovaszHingeLoss, self).__init__()

    @staticmethod
    def forward(logits, targets):
        logits = torch.sigmoid(logits)
        Lovasz = lovasz_hinge(logits, targets, per_image=False)
        return Lovasz


class ComboLoss(_Loss):
    def __init__(self, weight=None, size_average=True):
        super(ComboLoss, self).__init__()

    @staticmethod
    def forward(logits, targets, smooth=1, alpha=ALPHA, beta=BETA, eps=1e-9):
        logits = torch.sigmoid(logits)

        # True Positives, False Positives & False Negatives
        intersection = (logits * targets).sum()
        dice = (2. * intersection + smooth) / (logits.sum() + targets.sum() + smooth)

        logits = torch.clamp(logits, eps, 1.0 - eps)
        out = - (ALPHA * ((targets * torch.log(logits)) + ((1 - ALPHA) * (1.0 - targets) * torch.log(1.0 - logits))))
        weighted_ce = out.mean(-1)
        combo = (CE_RATIO * weighted_ce) - ((1 - CE_RATIO) * dice)

        return combo
