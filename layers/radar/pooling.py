# Code imported from https://github.com/filipradenovic/cnnimageretrieval-pytorch/blob/master/cirtorch/layers/pooling.py

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class MAC(nn.Module):
    def __init__(self):
        super(MAC, self).__init__()

    def forward(self, x):
        return mac(x)


class SPoC(nn.Module):
    def __init__(self):
        super(SPoC, self).__init__()

    def forward(self, x):
        return spoc(x)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)


def mac(x):
    return F.max_pool2d(x, (x.size(-2), x.size(-1)))
    # return F.adaptive_max_pool2d(x, (1,1)) # alternative


def spoc(x):
    return F.avg_pool2d(x, (x.size(-2), x.size(-1)))
    # return F.adaptive_avg_pool2d(x, (1,1)) # alternative


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)
    # return F.lp_pool2d(F.threshold(x, eps, eps), p, (x.size(-2), x.size(-1))) # alternative


