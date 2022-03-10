# import re
# from typing import Any, List, Tuple
# from collections import OrderedDict

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.utils.checkpoint as cp
# from torch import Tensor

# # 蔬菜
# class _DenseLayer(nn.Module):
#     def __init__(self,
#                  input_c: int,
#                  growth_rate: int,
#                  bn_size: int,
#                  drop_rate: float,
#                  memory_efficient: bool = False):
#         super(_DenseLayer, self).__init__()

#         self.add_module("norm1", nn.BatchNorm2d(input_c))
#         self.add_module("relu1", nn.ReLU(inplace=True))
#         self.add_module("conv1", nn.Conv2d(in_channels=input_c,
#                                            out_channels=bn_size * growth_rate,
#                                            kernel_size=1,
#                                            stride=1,
#                                            bias=False))
#         self.add_module("norm2", nn.BatchNorm2d(bn_size * growth_rate))
#         self.add_module("relu2", nn.ReLU(inplace=True))
#         self.add_module("conv2", nn.Conv2d(bn_size * growth_rate,
#                                            growth_rate,
#                                            kernel_size=3,
#                                            stride=1,
#                                            padding=1,
#                                            bias=False))
#         self.drop_rate = drop_rate
#         self.memory_efficient = memory_efficient

#     def bn_function(self, inputs: List[Tensor]) -> Tensor:
#         concat_features = torch.cat(inputs, 1)
#         bottleneck_output = self.conv1(self.relu1(self.norm1(concat_features)))
#         return bottleneck_output

#     @staticmethod
#     def any_requires_grad(inputs: List[Tensor]) -> bool:
#         for tensor in inputs:
#             if tensor.requires_grad:
#                 return True

#         return False

#     @torch.jit.unused
#     def call_checkpoint_bottleneck(self, inputs: List[Tensor]) -> Tensor:
#         def closure(*inp):
#             return self.bn_function(inp)

#         return cp.checkpoint(closure, *inputs)

#     def forward(self, inputs: Tensor) -> Tensor:
#         if isinstance(inputs, Tensor):
#             prev_features = [inputs]
#         else:
#             prev_features = inputs

#         if self.memory_efficient and self.any_requires_grad(prev_features):
#             if torch.jit.is_scripting():
#                 raise Exception("memory efficient not supported in JIT")

#             bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
#         else:
#             bottleneck_output = self.bn_function(prev_features)

#         new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
#         if self.drop_rate > 0:
#             new_features = F.dropout(new_features,
#                                      p=self.drop_rate,
#                                      training=self.training)

#         return new_features


# class _DenseBlock(nn.ModuleDict):
#     _version = 2

#     def __init__(self,
#                  num_layers: int,
#                  input_c: int,
#                  bn_size: int,
#                  growth_rate: int,
#                  drop_rate: float,
#                  memory_efficient: bool = False):
#         super(_DenseBlock, self).__init__()
#         for i in range(num_layers):
#             layer = _DenseLayer(input_c + i * growth_rate,
#                                 growth_rate=growth_rate,
#                                 bn_size=bn_size,
#                                 drop_rate=drop_rate,
#                                 memory_efficient=memory_efficient)
#             self.add_module("denselayer%d" % (i + 1), layer)

#     def forward(self, init_features: Tensor) -> Tensor:
#         features = [init_features]
#         for name, layer in self.items():
#             new_features = layer(features)
#             features.append(new_features)
#         return torch.cat(features, 1)


# class _Transition(nn.Sequential):
#     def __init__(self,
#                  input_c: int,
#                  output_c: int):
#         super(_Transition, self).__init__()
#         self.add_module("norm", nn.BatchNorm2d(input_c))
#         self.add_module("relu", nn.ReLU(inplace=True))
#         self.add_module("conv", nn.Conv2d(input_c,
#                                           output_c,
#                                           kernel_size=1,
#                                           stride=1,
#                                           bias=False))
#         self.add_module("pool", nn.AvgPool2d(kernel_size=2, stride=2))


# class DenseNet(nn.Module):
#     """
#     Densenet-BC model class for imagenet

#     Args:
#         growth_rate (int) - how many filters to add each layer (`k` in paper)
#         block_config (list of 4 ints) - how many layers in each pooling block
#         num_init_features (int) - the number of filters to learn in the first convolution layer
#         bn_size (int) - multiplicative factor for number of bottle neck layers
#           (i.e. bn_size * k features in the bottleneck layer)
#         drop_rate (float) - dropout rate after each dense layer
#         num_classes (int) - number of classification classes
#         memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient
#     """

#     def __init__(self,
#                  growth_rate: int = 32,
#                  block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
#                  num_init_features: int = 64,
#                  bn_size: int = 4,
#                  drop_rate: float = 0,
#                  num_classes: int = 1000,
#                  memory_efficient: bool = False):
#         super(DenseNet, self).__init__()

#         # first conv+bn+relu+pool
#         self.features = nn.Sequential(OrderedDict([
#             ("conv0", nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
#             ("norm0", nn.BatchNorm2d(num_init_features)),
#             ("relu0", nn.ReLU(inplace=True)),
#             ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
#         ]))

#         # each dense block
#         num_features = num_init_features
#         for i, num_layers in enumerate(block_config):
#             block = _DenseBlock(num_layers=num_layers,
#                                 input_c=num_features,
#                                 bn_size=bn_size,
#                                 growth_rate=growth_rate,
#                                 drop_rate=drop_rate,
#                                 memory_efficient=memory_efficient)
#             self.features.add_module("denseblock%d" % (i + 1), block)
#             num_features = num_features + num_layers * growth_rate

#             if i != len(block_config) - 1:
#                 trans = _Transition(input_c=num_features,
#                                     output_c=num_features // 2)
#                 self.features.add_module("transition%d" % (i + 1), trans)
#                 num_features = num_features // 2

#         # finnal batch norm
#         self.features.add_module("norm5", nn.BatchNorm2d(num_features))

#         # fc layer
#         self.classifier = nn.Linear(num_features, num_classes)

#         # init weights
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x: Tensor) -> Tensor:
#         features = self.features(x)
#         out = F.relu(features, inplace=True)
#         out = F.adaptive_avg_pool2d(out, (1, 1))
#         out = torch.flatten(out, 1)
#         out = self.classifier(out)
#         return out


# def densenet121(**kwargs: Any) -> DenseNet:
#     # Top-1 error: 25.35%
#     # 'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth'
#     return DenseNet(growth_rate=32,
#                     block_config=(6, 12, 24, 16),
#                     num_init_features=64,
#                     **kwargs)


# def densenet169(**kwargs: Any) -> DenseNet:
#     # Top-1 error: 24.00%
#     # 'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth'
#     return DenseNet(growth_rate=32,
#                     block_config=(6, 12, 32, 32),
#                     num_init_features=64,
#                     **kwargs)


# def densenet201(**kwargs: Any) -> DenseNet:
#     # Top-1 error: 22.80%
#     # 'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth'
#     return DenseNet(growth_rate=32,
#                     block_config=(6, 12, 48, 32),
#                     num_init_features=64,
#                     **kwargs)


# def densenet161(**kwargs: Any) -> DenseNet:
#     # Top-1 error: 22.35%
#     # 'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth'
#     return DenseNet(growth_rate=48,
#                     block_config=(6, 12, 36, 24),
#                     num_init_features=96,
#                     **kwargs)







"""dense net in pytorch

    cifar100

[1] Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger.

    Densely Connected Convolutional Networks
    https://arxiv.org/abs/1608.06993v5
"""

import torch
import torch.nn as nn



#"""Bottleneck layers. Although each layer only produces k
#output feature-maps, it typically has many more inputs. It
#has been noted in [37, 11] that a 1×1 convolution can be in-
#troduced as bottleneck layer before each 3×3 convolution
#to reduce the number of input feature-maps, and thus to
#improve computational efficiency."""
class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        #"""In  our experiments, we let each 1×1 convolution
        #produce 4k feature-maps."""
        inner_channel = 4 * growth_rate

        #"""We find this design especially effective for DenseNet and
        #we refer to our network with such a bottleneck layer, i.e.,
        #to the BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3) version of H ` ,
        #as DenseNet-B."""
        self.bottle_neck = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inner_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channel, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return torch.cat([x, self.bottle_neck(x)], 1)

#"""We refer to layers between blocks as transition
#layers, which do convolution and pooling."""
class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #"""The transition layers used in our experiments
        #consist of a batch normalization layer and an 1×1
        #convolutional layer followed by a 2×2 average pooling
        #layer""".
        self.down_sample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.AvgPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.down_sample(x)

#DesneNet-BC
#B stands for bottleneck layer(BN-RELU-CONV(1x1)-BN-RELU-CONV(3x3))
#C stands for compression factor(0<=theta<=1)
class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_class=100):
        super().__init__()
        self.growth_rate = growth_rate

        #"""Before entering the first dense block, a convolution
        #with 16 (or twice the growth rate for DenseNet-BC)
        #output channels is performed on the input images."""
        inner_channels = 2 * growth_rate

        #For convolutional layers with kernel size 3×3, each
        #side of the inputs is zero-padded by one pixel to keep
        #the feature-map size fixed.
        self.conv1 = nn.Conv2d(3, inner_channels, kernel_size=3, padding=1, bias=False)

        self.features = nn.Sequential()

        for index in range(len(nblocks) - 1):
            self.features.add_module("dense_block_layer_{}".format(index), self._make_dense_layers(block, inner_channels, nblocks[index]))
            inner_channels += growth_rate * nblocks[index]

            #"""If a dense block contains m feature-maps, we let the
            #following transition layer generate θm output feature-
            #maps, where 0 < θ ≤ 1 is referred to as the compression
            #fac-tor.
            out_channels = int(reduction * inner_channels) # int() will automatic floor the value
            self.features.add_module("transition_layer_{}".format(index), Transition(inner_channels, out_channels))
            inner_channels = out_channels

        self.features.add_module("dense_block{}".format(len(nblocks) - 1), self._make_dense_layers(block, inner_channels, nblocks[len(nblocks)-1]))
        inner_channels += growth_rate * nblocks[len(nblocks) - 1]
        self.features.add_module('bn', nn.BatchNorm2d(inner_channels))
        self.features.add_module('relu', nn.ReLU(inplace=True))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(inner_channels, num_class)

    def forward(self, x):
        output = self.conv1(x)
        output = self.features(output)
        output = self.avgpool(output)
        output = output.view(output.size()[0], -1)
        output = self.linear(output)
        return output

    def _make_dense_layers(self, block, in_channels, nblocks):
        dense_block = nn.Sequential()
        for index in range(nblocks):
            dense_block.add_module('bottle_neck_layer_{}'.format(index), block(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return dense_block

def densenet121(num_class):
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32,num_class=num_class)

def densenet169():
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32)

def densenet201(num_class):
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32,num_class=num_class)

def densenet161():
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48)


