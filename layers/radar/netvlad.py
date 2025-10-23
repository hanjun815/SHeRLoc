"""
PointNet code taken from PointNetVLAD Pytorch implementation.
"""

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import math

"""
NOTE: The toolbox can only pool lists of features of the same length. It was specifically optimized to efficiently
do so. One way to handle multiple lists of features of variable length is to create, via a data augmentation
technique, a tensor of shape: 'batch_size'x'max_samples'x'feature_size'. Where 'max_samples' would be the maximum
number of feature per list. Then for each list, you would fill the tensor with 0 values.
"""

class NetVLADLoupe(nn.Module):
    def __init__(self, feature_size, cluster_size, add_batch_norm=True):
        super().__init__()
        self.feature_size = feature_size
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size
        self.softmax = nn.Softmax(dim=-1)
        self.cluster_weights = nn.Parameter(torch.randn(feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        self.cluster_weights2 = nn.Parameter(torch.randn(1, feature_size, cluster_size) * 1 / math.sqrt(feature_size))

        if add_batch_norm:
            self.cluster_biases = None
            self.bn1 = nn.BatchNorm1d(cluster_size)
        else:
            self.cluster_biases = nn.Parameter(torch.randn(cluster_size) * 1 / math.sqrt(feature_size))
            self.bn1 = None



    def forward(self, x):
        # Expects (batch_size, num_points, channels) tensor
        assert x.dim() == 3
        num_points = x.shape[1]
        activation = torch.matmul(x, self.cluster_weights)
        if self.add_batch_norm:
            # activation = activation.transpose(1,2).contiguous()
            activation = activation.view(-1, self.cluster_size)
            activation = self.bn1(activation)
            activation = activation.view(-1, num_points, self.cluster_size)
            # activation = activation.transpose(1,2).contiguous()
        else:
            activation = activation + self.cluster_biases
        activation = self.softmax(activation)
        activation = activation.view((-1, num_points, self.cluster_size))

        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weights2

        activation = torch.transpose(activation, 2, 1)
        x = x.view((-1, num_points, self.feature_size))
        vlad = torch.matmul(activation, x)
        vlad = torch.transpose(vlad, 2, 1)
        vlad = vlad - a

        vlad = F.normalize(vlad, dim=1, p=2)
        vlad = vlad.reshape((-1, self.cluster_size * self.feature_size))
        vlad = F.normalize(vlad, dim=1, p=2)


        return vlad
