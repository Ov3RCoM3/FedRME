import torch
import torch.nn as nn
import torch.nn.functional as F

def to_one_hot(tensor, n_classes):
    n, h, w = tensor.size()
    one_hot = torch.zeros(n, n_classes, h, w).to(tensor.device).scatter_(1, tensor.view(n, 1, h, w), 1)
    return one_hot

def calc_iou(input, target):
    # logit => N x Classes x H x W
    # target => N x H x W

    N = len(input)
    n_classes = input.shape[1]

    pred = F.softmax(input, dim=1)
    target_onehot = to_one_hot(target, n_classes)

    # Numerator Product
    inter = pred * target_onehot
    # Sum over all pixels N x C x H x W => N x C
    inter = inter.view(N, n_classes, -1).sum(2)

    # Denominator
    union = pred + target_onehot - (pred * target_onehot)
    # Sum over all pixels N x C x H x W => N x C
    union = union.view(N, n_classes, -1).sum(2)

    loss = inter / (union + 1e-16)

    return loss.mean().item()