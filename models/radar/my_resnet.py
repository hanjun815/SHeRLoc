import torch
import torch.nn as nn
from typing import Union, List, Dict, Tuple

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes: int, out_planes: int, stride: int = 1, downsample: nn.Module = None,
                 padding_mode: str = 'constant'):
        super().__init__()
        self.conv1 = self._make_conv(in_planes, out_planes, kernel_size=3, stride=stride,
                                     padding=1, padding_mode=padding_mode)
        self.bn1 = nn.BatchNorm2d(out_planes)

        self.conv2 = self._make_conv(out_planes, out_planes, kernel_size=3, stride=1,
                                     padding=1, padding_mode=padding_mode)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def _make_conv(self, in_ch, out_ch, kernel_size, stride, padding, padding_mode):
        conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride,
                         padding=padding, bias=False)
        return conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes: int, out_planes: int, stride: int = 1, downsample: nn.Module = None,
                 padding_mode: str = 'constant'):
        super().__init__()
        mid_planes = out_planes
        # 1x1
        self.conv1 = self._make_conv(in_planes, mid_planes, kernel_size=1, stride=1,
                                     padding=0, padding_mode=padding_mode)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        # 3x3
        self.conv2 = self._make_conv(mid_planes, mid_planes, kernel_size=3, stride=stride,
                                     padding=1, padding_mode=padding_mode)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        # 1x1
        self.conv3 = self._make_conv(mid_planes, out_planes * self.expansion, kernel_size=1, stride=1,
                                     padding=0, padding_mode=padding_mode)
        self.bn3 = nn.BatchNorm2d(out_planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def _make_conv(self, in_ch, out_ch, kernel_size, stride, padding, padding_mode):
        conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride,
                         padding=padding, bias=False)
        return conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


resnet_cfgs: Dict[str, Tuple[Union[BasicBlock, Bottleneck], List[int]]] = {
    "resnet18":  (BasicBlock,  [2, 2, 2, 2]),
    "resnet34":  (BasicBlock,  [3, 4, 6, 3]),
    "resnet50":  (Bottleneck, [3, 4, 6, 3]),
}


def make_resnet_layers(
    resnet_type: str = "resnet18",
    in_channels: int = 1,
    padding_mode: str = "constant"
) -> nn.Sequential:
    block_class, layer_cfg = resnet_cfgs[resnet_type]


    layers: List[nn.Module] = []

    in_planes = 64
    conv1 = nn.Conv2d(in_channels, in_planes, kernel_size=7, stride=2, padding=3, bias=False)
    bn1 = nn.BatchNorm2d(in_planes)
    relu = nn.ReLU(inplace=True)
    maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    layers += [conv1, bn1, relu, maxpool]

    out_planes_arr = [64, 128, 256, 512]
    
    def _make_layer(
        block: nn.Module,
        in_ch: int,
        out_ch: int,
        num_blocks: int,
        stride: int = 1,
        padding_mode: str = "constant"
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or in_ch != out_ch * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch * block.expansion),
            )

        blocks = []
        blocks.append(block(in_ch, out_ch, stride=stride, downsample=downsample, padding_mode=padding_mode))
        in_ch = out_ch * block.expansion
        for _ in range(1, num_blocks):
            blocks.append(block(in_ch, out_ch, stride=1, downsample=None, padding_mode=padding_mode))

        return nn.Sequential(*blocks)

    current_in_ch = in_planes

    strides = [1, 2, 2, 2]  
    for layer_idx, n_blocks in enumerate(layer_cfg):
        out_planes = out_planes_arr[layer_idx]
        layer = _make_layer(block_class, current_in_ch, out_planes, n_blocks, stride=strides[layer_idx], padding_mode=padding_mode)
        layers.append(layer)
        current_in_ch = out_planes * block_class.expansion

    return nn.Sequential(*layers)


class ResNetFeatureExtractor(nn.Module):
    def __init__(
        self,
        resnet_type: str = "resnet18",
        in_channels: int = 1,
        padding_mode: str = 'constant'
    ):
        super().__init__()
        assert resnet_type in resnet_cfgs, f"Not supported ResNet type: {resnet_type}"
        
        self.resnet_type = resnet_type
        self.in_channels = in_channels
        self.padding_mode = padding_mode

        self.feature_extractor = make_resnet_layers(resnet_type, in_channels, padding_mode)
        
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)