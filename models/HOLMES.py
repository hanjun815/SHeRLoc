import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

def log_otp_solver(log_a, log_b, M, num_iters: int = 20, reg: float = 1.0) -> torch.Tensor:
    batch_size = M.size(0)
    M = M / reg.view(batch_size, 1, 1)
    u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)

    for _ in range(num_iters):
        u = log_a - torch.logsumexp(M + v.unsqueeze(1), dim=2).squeeze()
        v = log_b - torch.logsumexp(M + u.unsqueeze(2), dim=1).squeeze()

    return M + u.unsqueeze(2) + v.unsqueeze(1)

def get_matching_probs(S, ghostbin_score =1.0, dustbin_score = 1.0, num_iters=3, reg=1.0):
    batch_size, m, n = S.size()

    S_aug = torch.empty(batch_size, m + 2, n, dtype=S.dtype, device=S.device)
    S_aug[:, :m, :n] = S
    S_aug[:, m, :] = ghostbin_score
    S_aug[:, m+1, :] = dustbin_score

    norm = -torch.tensor(math.log(n + m + 1), device=S.device)
    log_a, log_b = norm.expand(m + 2).contiguous(), norm.expand(n).contiguous()
    log_a[-1] = log_a[-1] + math.log(n - m)
    log_a[-2] = log_a[-2] + math.log(n - m)
    log_a, log_b = log_a.expand(batch_size, -1), log_b.expand(batch_size, -1)

    log_P = log_otp_solver(
        log_a,
        log_b,
        S_aug,
        num_iters=num_iters,
        reg=reg
    )
    return log_P - norm

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(
        x.clamp(min=eps).pow(p),
        (x.size(-2), x.size(-1))
    ).pow(1. / p)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class HOLMES(nn.Module):
    def __init__(self,
                 num_channels=1536,
                 num_clusters=64,
                 cluster_dim=128,
                 token_dim=256,
                 dropout=0.3,
                 p=3):
        super().__init__()

        self.num_channels = num_channels
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim
        self.gem = GeM(p=p)

        if dropout > 0:
            dropout_layer = nn.Dropout(dropout)
        else:
            dropout_layer = nn.Identity()

        self.conv_layer_1 = BasicBlock(num_channels, 256, stride=2)
        self.conv_layer_2 = BasicBlock(256, 128, stride=2)

        self.cluster_features = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout_layer,
            nn.ReLU(),
            nn.Conv2d(512, self.cluster_dim, 1)
        )
        self.cluster_features2 = nn.Sequential(
            nn.Conv2d(256, 512, 1),
            dropout_layer,
            nn.ReLU(),
            nn.Conv2d(512, 64, 1)
        )
        self.cluster_features3 = nn.Sequential(
            nn.Conv2d(128, 512, 1),
            dropout_layer,
            nn.ReLU(),
            nn.Conv2d(512, 64, 1)
        )

        self.score = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout_layer,
            nn.ReLU(),
            nn.Conv2d(512, self.num_clusters, 1),
        )

        self.score2 = nn.Sequential(
            nn.Conv2d(256, 512, 1),
            dropout_layer,
            nn.ReLU(),
            nn.Conv2d(512, 16, 1), 
        )

        self.ghost_bin = nn.Parameter(torch.tensor(1.0))
        self.ghost_bin2 = nn.Parameter(torch.tensor(1.0))

        self.dust_bin = nn.Parameter(torch.tensor(1.0))
        self.dust_bin2 = nn.Parameter(torch.tensor(1.0))


        self.token_features = nn.Sequential(
            nn.Linear(self.num_channels, 512),
            nn.ReLU(),
            nn.Linear(512, self.token_dim)
        )

        self.token_features2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 64)
        )



        self.hidden1_weights = nn.Parameter(
            torch.randn(64 * 256 + 256, 256) * (1.0 / math.sqrt(256))
        )
        self.hidden2_weights = nn.Parameter(
            torch.randn(16 * 64 + 64, 64) * (1.0 / math.sqrt(64))
        )

        self.bn = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(64)


    def dynamic_regularization(self, x):
        var_per_batch = x.var(dim=(1, 2, 3))  # [B]
        mean_per_batch = x.abs().mean(dim=(1, 2, 3))  # [B]
        ratio = var_per_batch / (mean_per_batch + 1e-6)  # [B]
        reg = 1.0 + 2.0 * torch.tanh(ratio / 2.0)  # [B]
        return reg
    
    def forward(self, x):
        global_feature = self.gem(x).squeeze(-1).squeeze(-1)  # [B, C]
        t1 = self.token_features(global_feature)               # [B, token_dim]
        f = self.cluster_features(x).flatten(2)  # [B, 256, 12*6]
        S = self.score(x).flatten(2)             # [B, 64, 12*6]
        reg = self.dynamic_regularization(x)
        log_p = get_matching_probs(S, self.ghost_bin, self.dust_bin, num_iters=3, reg=reg)
        p = torch.exp(log_p)[:, :-2, :]         # [B, 64, 12*6]
        p = p.unsqueeze(1).expand(-1, self.cluster_dim, -1, -1)  # [B, 256, 64, 12*6]
        f = f.unsqueeze(2).expand(-1, -1, 64, -1)           # [B, 256, 64, 12*6]
        f = torch.cat([
            nn.functional.normalize(t1, p=2, dim=-1),
            nn.functional.normalize((f * p).sum(dim=-1), p=2, dim=1).flatten(1)
        ], dim=-1)
        f = torch.matmul(f, self.hidden1_weights)
        f1 = self.bn(f)
        D1 = nn.functional.normalize(f1, p=2, dim=-1)


        x2 = self.conv_layer_1(x)
        global_feature2 = self.gem(x2).squeeze(-1).squeeze(-1)  # [B, C]
        t2 = self.token_features2(global_feature2)               # [B, token_dim]
        f2 = self.cluster_features2(x2).flatten(2)  # [B, 64, 6*3]
        S2 = self.score2(x2).flatten(2)            # [B, 16, 6*3]
        reg2 = self.dynamic_regularization(x2)
        log_p2 = get_matching_probs(S2, self.ghost_bin2, self.dust_bin2, num_iters=3, reg=reg2)
        p2 = torch.exp(log_p2)[:, :-2, :]          # [B, 16, 6*3]
        p2 = p2.unsqueeze(1).expand(-1, 64, -1, -1)  # [B, 64, 16, 6*3]
        f2 = f2.unsqueeze(2).expand(-1, -1, 16, -1)           # [B, 64, 16, 6*3]
        f2 = torch.cat([
            nn.functional.normalize(t2, p=2, dim=-1),
            nn.functional.normalize((f2 * p2).sum(dim=-1), p=2, dim=1).flatten(1)
        ], dim=-1)
        f2 = torch.matmul(f2, self.hidden2_weights)  # [B, 64]
        f2 = self.bn2(f2)                                # [B, 64]
        D2 = nn.functional.normalize(f2, p=2, dim=-1)

        f_combined = torch.cat([D1, D2], dim=-1)  # [B, 256]


        return nn.functional.normalize(f_combined, p=2, dim=-1)   


class HOLMES_S(nn.Module):
    def __init__(self,
                 num_channels=1536,
                 num_clusters=64,
                 cluster_dim=128,
                 token_dim=256,
                 dropout=0.3,
                 p=3):
        super().__init__()

        self.num_channels = num_channels
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim
        self.gem = GeM(p=p)

        if dropout > 0:
            dropout_layer = nn.Dropout(dropout)
        else:
            dropout_layer = nn.Identity()

        self.conv_layer_1 = BasicBlock(num_channels, 256, stride=2)
        self.conv_layer_2 = BasicBlock(256, 128, stride=2)

        self.cluster_features = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout_layer,
            nn.ReLU(),
            nn.Conv2d(512, self.cluster_dim, 1)
        )
        self.cluster_features2 = nn.Sequential(
            nn.Conv2d(256, 512, 1),
            dropout_layer,
            nn.ReLU(),
            nn.Conv2d(512, 64, 1)
        )
        self.cluster_features3 = nn.Sequential(
            nn.Conv2d(128, 512, 1),
            dropout_layer,
            nn.ReLU(),
            nn.Conv2d(512, 64, 1)
        )

        self.score = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout_layer,
            nn.ReLU(),
            nn.Conv2d(512, self.num_clusters, 1),
        )

        self.score2 = nn.Sequential(
            nn.Conv2d(256, 512, 1),
            dropout_layer,
            nn.ReLU(),
            nn.Conv2d(512, 16, 1), 
        )

        self.ghost_bin = nn.Parameter(torch.tensor(1.0))
        self.ghost_bin2 = nn.Parameter(torch.tensor(1.0))

        self.dust_bin = nn.Parameter(torch.tensor(1.0))
        self.dust_bin2 = nn.Parameter(torch.tensor(1.0))


        self.token_features = nn.Sequential(
            nn.Linear(self.num_channels, 512),
            nn.ReLU(),
            nn.Linear(512, self.token_dim)
        )

        self.token_features2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 64)
        )

        self.hidden1_weights = nn.Parameter(
            torch.randn(64 * 256 + 256, 256) * (1.0 / math.sqrt(256))
        )
        self.hidden2_weights = nn.Parameter(
            torch.randn(16 * 64 + 64, 64) * (1.0 / math.sqrt(64))
        )

        self.bn = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(64)

    def dynamic_regularization(self, x):
        var_per_batch = x.var(dim=(1, 2, 3))  # [B]
        mean_per_batch = x.abs().mean(dim=(1, 2, 3))  # [B]
        ratio = var_per_batch / (mean_per_batch + 1e-6)  # [B]
        reg = 1.0 + 2.0 * torch.tanh(ratio / 2.0)  # [B]
        return reg
    
    def forward(self, x):
        global_feature = self.gem(x).squeeze(-1).squeeze(-1)  # [B, C]
        t1 = self.token_features(global_feature)               # [B, token_dim]
        f = self.cluster_features(x).flatten(2)  # [B, 256, 12*6]
        S = self.score(x).flatten(2)             # [B, 64, 12*6]
        reg = self.dynamic_regularization(x)
        log_p = get_matching_probs(S, self.ghost_bin, self.dust_bin, num_iters=3, reg=reg)
        p = torch.exp(log_p)[:, :-2, :]         # [B, 64, 12*6]
        p = p.unsqueeze(1).expand(-1, self.cluster_dim, -1, -1)  # [B, 256, 64, 12*6]
        f = f.unsqueeze(2).expand(-1, -1, 64, -1)           # [B, 256, 64, 12*6]
        f = torch.cat([
            nn.functional.normalize(t1, p=2, dim=-1),
            nn.functional.normalize((f * p).sum(dim=-1), p=2, dim=1).flatten(1)
        ], dim=-1)
        f = torch.matmul(f, self.hidden1_weights)
        f1 = self.bn(f)
        D1 = nn.functional.normalize(f1, p=2, dim=-1)

        return D1