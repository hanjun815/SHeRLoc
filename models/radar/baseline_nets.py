import torch
import torch.nn as nn
from models.radar.my_resnet import ResNetFeatureExtractor
from layers.radar.netvlad import NetVLADLoupe
from models.HOLMES import HOLMES, HOLMES_S
from layers.radar.pooling import MAC, SPoC, GeM

class SHeRLoc(nn.Module):
    def __init__(self):
        super().__init__()
        self.fe = ResNetFeatureExtractor(resnet_type="resnet18", in_channels=1)
        # self.fe = ResNetFeatureExtractor(resnet_type="resnet34", in_channels=3)
        # self.fe = ResNetFeatureExtractor(resnet_type="resnet50", in_channels=3)

        self.pool = HOLMES(num_channels=512, num_clusters=64, cluster_dim=256, token_dim=256, dropout=0)
        # self.pool = NetVLADLoupe(feature_size=512, cluster_size=64, add_batch_norm=True)
        # self.pool = GeM()
        # self.pool = MAC()
        # self.pool = SPoC()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fe(x)
        x = self.pool(x)
        return x
    
class SHeRLoc_S(nn.Module):
    def __init__(self):
        super().__init__()
        self.fe = ResNetFeatureExtractor(resnet_type="resnet18", in_channels=1)
        self.pool = HOLMES_S(num_channels=512, num_clusters=64, cluster_dim=256, token_dim=256, dropout=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fe(x)
        x = self.pool(x)
        return x