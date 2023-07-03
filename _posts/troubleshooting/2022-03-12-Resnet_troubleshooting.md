---
layout : single
title:  "[Troubleshooting] resnet34 model error handle"
excerpt: "resnet34 구현 시 발생한 에러 처리"

categories:
  - Troubleshooting
tags:
  - resnet34

toc: true
toc_sticky: true

author_profile: true
sidebar_main: true
 
date: 2022-03-12
last_modified_at: 2022-03-12
---

Troubleshooting 
=================


```python
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import itertools
import matplotlib.pyplot as plt
import os
import numpy as np
import torch.nn as nn
import pandas as pd
import random
```

    C:\Users\ggara\miniconda3\lib\site-packages\torchvision\io\image.py:11: UserWarning: Failed to load image Python extension: Could not find module 'C:\Users\ggara\miniconda3\Lib\site-packages\torchvision\image.pyd' (or one of its dependencies). Try using the full path with constructor syntax.
      warn(f"Failed to load image Python extension: {e}")
    


```python
seed = 111

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)
```


```python
class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, activation: bool = True):
        super().__init__()

        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        layers.append(nn.BatchNorm2d(out_channels))
        if activation:
            layers.append(nn.ReLU(inplace=True))

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
```


```python
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        layers = []

        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3,stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3,stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.relu = nn.ReLU()
        
        self.conv1 = nn.Sequential(*layers)
        self.resblk = nn.Identity()

    def forward(self, x):
        y = self.resblk(x)
        x = self.conv1(x)
        return x + y
```


```python
class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, nblk=[3,4,6,3]):
        super(ResNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.enc = ConvBlock(in_channels, nker, kernel_size=7, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(ResBlock,64,nblk[0],stride = 1)
        self.layer2 = self.make_layer(ResBlock,128,nblk[1],stride = 2)
        self.layer3 = self.make_layer(ResBlock,256,nblk[2],stride = 2)
        self.layer4 = self.make_layer(ResBlock,512,nblk[3],stride = 2)

        self.avg = nn.AdaptiveAvgPool2d((1,1))


        self.fc = nn.Linear(nker*2*2*2, 10)

    def make_layer(self,block,out_plane,num_block,stride):
        if out_plane == 64:
            layers = [block(64,out_plane,stride = 1)]
            self.in_channels = out_plane * 1
            for i in range(num_block -1):
                layers.append(block(self.in_channels,out_plane))
            return nn.Sequential(*layers)
        else:
            layers = []
            if stride != 1: 
                layers.append(nn.Sequential(
                    #nn.Conv2d(int(out_plane / 2), out_plane, kernel_size=1, stride=stride, bias=False),
                    nn.Conv2d(out_plane, out_plane, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_plane)
                ))
            self.in_channels = out_plane * 1
            for i in range(num_block):
                layers.append(block(self.in_channels,out_plane))
            return nn.Sequential(*layers)

        

    def forward(self, x):
        x = self.enc(x)
        x = self.max_pool(x)
        
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg(x)

        x = x.view(-1,512)

        out = self.fc(x)

        return out
```


```python
# Network
model_test = ResNet(3, 10)

# Random input
x = torch.randn((4, 3, 227, 227))

# Forward
out = model_test(x)

# Check the output shape
print("Output tensor shape is :", out.shape)
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    ~\AppData\Local\Temp/ipykernel_11728/2293773626.py in <module>
          6 
          7 # Forward
    ----> 8 out = model_test(x)
          9 
         10 # Check the output shape
    

    ~\miniconda3\lib\site-packages\torch\nn\modules\module.py in _call_impl(self, *input, **kwargs)
       1100         if not (self._backward_hooks or self._forward_hooks or self._forward_pre_hooks or _global_backward_hooks
       1101                 or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1102             return forward_call(*input, **kwargs)
       1103         # Do not call functions when jit is used
       1104         full_backward_hooks, non_full_backward_hooks = [], []
    

    ~\AppData\Local\Temp/ipykernel_11728/953636019.py in forward(self, x)
         46 
         47         x = self.layer1(x)
    ---> 48         x = self.layer2(x)
         49         x = self.layer3(x)
         50         x = self.layer4(x)
    

    ~\miniconda3\lib\site-packages\torch\nn\modules\module.py in _call_impl(self, *input, **kwargs)
       1100         if not (self._backward_hooks or self._forward_hooks or self._forward_pre_hooks or _global_backward_hooks
       1101                 or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1102             return forward_call(*input, **kwargs)
       1103         # Do not call functions when jit is used
       1104         full_backward_hooks, non_full_backward_hooks = [], []
    

    ~\miniconda3\lib\site-packages\torch\nn\modules\container.py in forward(self, input)
        139     def forward(self, input):
        140         for module in self:
    --> 141             input = module(input)
        142         return input
        143 
    

    ~\miniconda3\lib\site-packages\torch\nn\modules\module.py in _call_impl(self, *input, **kwargs)
       1100         if not (self._backward_hooks or self._forward_hooks or self._forward_pre_hooks or _global_backward_hooks
       1101                 or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1102             return forward_call(*input, **kwargs)
       1103         # Do not call functions when jit is used
       1104         full_backward_hooks, non_full_backward_hooks = [], []
    

    ~\miniconda3\lib\site-packages\torch\nn\modules\container.py in forward(self, input)
        139     def forward(self, input):
        140         for module in self:
    --> 141             input = module(input)
        142         return input
        143 
    

    ~\miniconda3\lib\site-packages\torch\nn\modules\module.py in _call_impl(self, *input, **kwargs)
       1100         if not (self._backward_hooks or self._forward_hooks or self._forward_pre_hooks or _global_backward_hooks
       1101                 or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1102             return forward_call(*input, **kwargs)
       1103         # Do not call functions when jit is used
       1104         full_backward_hooks, non_full_backward_hooks = [], []
    

    ~\miniconda3\lib\site-packages\torch\nn\modules\conv.py in forward(self, input)
        444 
        445     def forward(self, input: Tensor) -> Tensor:
    --> 446         return self._conv_forward(input, self.weight, self.bias)
        447 
        448 class Conv3d(_ConvNd):
    

    ~\miniconda3\lib\site-packages\torch\nn\modules\conv.py in _conv_forward(self, input, weight, bias)
        440                             weight, bias, self.stride,
        441                             _pair(0), self.dilation, self.groups)
    --> 442         return F.conv2d(input, weight, bias, self.stride,
        443                         self.padding, self.dilation, self.groups)
        444 
    

    RuntimeError: Given groups=1, weight of size [128, 128, 1, 1], expected input[4, 64, 56, 56] to have 128 channels, but got 64 channels instead


# Given groups=1, weight of size [128, 128, 1, 1], expected input[4, 64, 56, 56] to have 128 channels, but got 64 channels instead 

위처럼 다음 ResBlock 으로 넘어갈 때 channel 이 다르므로 (layer1 : 64 channel, layer2 : 128 channel) downsample을 해줘야 하는데, pretrained resnet model 을 출력해보면 아래와 같이 순서가 downsample 이 layer2의 가장 밑으로 가있다. <br>

이는 __init__ 단계에서 layer 을 설정해준 순서대로 model 출력시에 나오기 때문에, 이 순서와 헷갈려 model 구축 시에 이러한 에러를 보았고, downsample 순서를 바꾸어 해결하였다
 

```
ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  
    (0): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (1): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (2): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer2): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (2): BasicBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (3): BasicBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer3): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (2): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (3): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (4): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (5): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer4): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (2): BasicBlock(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Linear(in_features=512, out_features=3, bias=True)
)
```

