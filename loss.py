import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        smooth = 1.0
        iflat = input.flatten()
        tflat = target.flatten()
        intersection = torch.sum(iflat * tflat)
        return 1 - ((2. * intersection + smooth) / (torch.sum(iflat) + torch.sum(tflat) + smooth))


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return F.cross_entropy(input, target)