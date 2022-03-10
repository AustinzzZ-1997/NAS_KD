# """resnet in pytorch



# [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

#     Deep Residual Learning for Image Recognition
#     https://arxiv.org/abs/1512.03385v1
# """

# import torch
# import torch.nn as nn

# class BasicBlock(nn.Module):
#     """Basic Block for resnet 18 and resnet 34

#     """

#     #BasicBlock and BottleNeck block
#     #have different output size
#     #we use class attribute expansion
#     #to distinct
#     expansion = 1

#     def __init__(self, in_channels, out_channels, stride=1):
#         super().__init__()

#         #residual function
#         self.residual_function = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels * BasicBlock.expansion)
#         )

#         #shortcut
#         self.shortcut = nn.Sequential()

#         #the shortcut output dimension is not the same with residual function
#         #use 1*1 convolution to match the dimension
#         if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels * BasicBlock.expansion)
#             )

#     def forward(self, x):
#         return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

# class BottleNeck(nn.Module):
#     """Residual block for resnet over 50 layers

#     """
#     expansion = 4
#     def __init__(self, in_channels, out_channels, stride=1):
#         super().__init__()
#         self.residual_function = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
#             nn.BatchNorm2d(out_channels * BottleNeck.expansion),
#         )

#         self.shortcut = nn.Sequential()

#         if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
#                 nn.BatchNorm2d(out_channels * BottleNeck.expansion)
#             )

#     def forward(self, x):
#         return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

# class ResNet(nn.Module):

#     def __init__(self, block, num_block, num_classes=100):
#         super().__init__()

#         self.in_channels = 64

#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True))
#         #we use a different inputsize than the original paper
#         #so conv2_x's stride is 1
#         self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
#         self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
#         self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
#         self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
#         self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512 * block.expansion, num_classes)

#     def _make_layer(self, block, out_channels, num_blocks, stride):
#         """make resnet layers(by layer i didnt mean this 'layer' was the
#         same as a neuron netowork layer, ex. conv layer), one layer may
#         contain more than one residual block

#         Args:
#             block: block type, basic block or bottle neck block
#             out_channels: output depth channel number of this layer
#             num_blocks: how many blocks per layer
#             stride: the stride of the first block of this layer

#         Return:
#             return a resnet layer
#         """

#         # we have num_block blocks per layer, the first block
#         # could be 1 or 2, other blocks would always be 1
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_channels, out_channels, stride))
#             self.in_channels = out_channels * block.expansion

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         output = self.conv1(x)
#         output = self.conv2_x(output)
#         output = self.conv3_x(output)
#         output = self.conv4_x(output)
#         output = self.conv5_x(output)
#         # print(output.shape)
#         output = self.avg_pool(output)
#         # print(output.shape)
#         output = output.view(output.size(0), -1)
#         # print(output.shape)
#         output = self.fc(output)

#         return output

# def resnet18():
#     """ return a ResNet 18 object
#     """
#     return ResNet(BasicBlock, [2, 2, 2, 2])

# def resnet34():
#     """ return a ResNet 34 object
#     """
#     return ResNet(BasicBlock, [3, 4, 6, 3])

# def resnet50(num_classes):
#     """ return a ResNet 50 object
#     """
#     return ResNet(BottleNeck, [3, 4, 6, 3],num_classes=num_classes)

# def resnet101():
#     """ return a ResNet 101 object
#     """
#     return ResNet(BottleNeck, [3, 4, 23, 3])

# def resnet152(num_classes):
#     """ return a ResNet 152 object
#     """
#     return ResNet(BottleNeck, [3, 8, 36, 3],num_classes=num_classes)

import torch.nn as nn
import torch


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)

def resnet152(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, include_top=include_top)

def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


