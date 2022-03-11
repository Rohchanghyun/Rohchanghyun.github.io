---
title:  "[Pytorch] resnet34 model"
excerpt: "pytorch 를 사용한 resnet 34 model 구현"

categories:
  - pytorch
tags:
  - resnet34
  - fine-tuning

toc: true
toc_sticky: true
 
date: 2022-03-11
last_modified_at: 2022-03-11
---

# Resnet34 model 구현


quickdraw dataset 에 대해 직접 구현한 Resnet34 model 을 사용하여 random weight 에서 시작해 train 

## Dataset

Quickdraw dataset [link](https://quickdraw.withgoogle.com/data)


```python
# install quickdraw python API
!pip3 install quickdraw
```

    Collecting quickdraw
      Downloading quickdraw-0.2.0-py3-none-any.whl (10 kB)
    Requirement already satisfied: requests in /opt/conda/lib/python3.8/site-packages (from quickdraw) (2.24.0)
    Requirement already satisfied: pillow in /opt/conda/lib/python3.8/site-packages (from quickdraw) (8.1.0)
    Requirement already satisfied: idna<3,>=2.5 in /opt/conda/lib/python3.8/site-packages (from requests->quickdraw) (2.10)
    Requirement already satisfied: chardet<4,>=3.0.2 in /opt/conda/lib/python3.8/site-packages (from requests->quickdraw) (3.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.8/site-packages (from requests->quickdraw) (2021.10.8)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/lib/python3.8/site-packages (from requests->quickdraw) (1.25.11)
    Installing collected packages: quickdraw
    Successfully installed quickdraw-0.2.0
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0m[33mWARNING: You are using pip version 22.0.3; however, version 22.0.4 is available.
    You should consider upgrading via the '/opt/conda/bin/python -m pip install --upgrade pip' command.[0m[33m
    [0m


```python
# import packages
from quickdraw import QuickDrawData, QuickDrawDataGroup
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
num_img_per_class = 5000
qd = QuickDrawData(max_drawings=num_img_per_class)
```

class mapping


```python
class_list = ['apple', 'wine bottle', 'spoon', 'rainbow', 'panda', 'hospital', 'scissors', 'toothpaste', 'baseball', 'hourglass']
class_dict = {'apple' : 0, 'wine bottle' : 1, 'spoon' : 2, 'rainbow' : 3, 'panda': 4, 'hospital' : 5, 'scissors' : 6, 'toothpaste' : 7, 'baseball' : 8, 'hourglass' : 9}
```


```python
qd.load_drawings(class_list)
```

    downloading apple from https://storage.googleapis.com/quickdraw_dataset/full/binary/apple.bin
    download complete
    loading apple drawings
    load complete
    downloading wine bottle from https://storage.googleapis.com/quickdraw_dataset/full/binary/wine bottle.bin
    download complete
    loading wine bottle drawings
    load complete
    downloading spoon from https://storage.googleapis.com/quickdraw_dataset/full/binary/spoon.bin
    download complete
    loading spoon drawings
    load complete
    downloading rainbow from https://storage.googleapis.com/quickdraw_dataset/full/binary/rainbow.bin
    download complete
    loading rainbow drawings
    load complete
    downloading panda from https://storage.googleapis.com/quickdraw_dataset/full/binary/panda.bin
    download complete
    loading panda drawings
    load complete
    downloading hospital from https://storage.googleapis.com/quickdraw_dataset/full/binary/hospital.bin
    download complete
    loading hospital drawings
    load complete
    downloading scissors from https://storage.googleapis.com/quickdraw_dataset/full/binary/scissors.bin
    download complete
    loading scissors drawings
    load complete
    downloading toothpaste from https://storage.googleapis.com/quickdraw_dataset/full/binary/toothpaste.bin
    download complete
    loading toothpaste drawings
    load complete
    downloading baseball from https://storage.googleapis.com/quickdraw_dataset/full/binary/baseball.bin
    download complete
    loading baseball drawings
    load complete
    downloading hourglass from https://storage.googleapis.com/quickdraw_dataset/full/binary/hourglass.bin
    download complete
    loading hourglass drawings
    load complete



```python
# get images, and append to train/validation data and label list
train_data = list()
val_data = list()
train_label = list()
val_label = list()
for class_name in class_list:
  qdgroup = QuickDrawDataGroup(class_name, max_drawings=num_img_per_class)
  for i, img in enumerate(qdgroup.drawings):
    if i < int(0.9 * num_img_per_class):
      train_data.append(np.asarray(img.get_image()))
      train_label.append(class_dict[class_name])
    else:
      val_data.append(np.asarray(img.get_image()))
      val_label.append(class_dict[class_name])
```

    loading apple drawings
    load complete
    loading wine bottle drawings
    load complete
    loading spoon drawings
    load complete
    loading rainbow drawings
    load complete
    loading panda drawings
    load complete
    loading hospital drawings
    load complete
    loading scissors drawings
    load complete
    loading toothpaste drawings
    load complete
    loading baseball drawings
    load complete
    loading hourglass drawings
    load complete



```python
# transformation, image to (227, 227) tensor

# NOTE : torchvision 0.8 이하에서 Tensor에 대해 transforms.Resize가 적용 불가능해 에러 발생
# torchvision update를 통해 문제를 해결하도록 안내!
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((227,227)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])
```

dataset 


```python
# custom dataset for Quickdraw
class QuickDrawDataset(Dataset):

    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        if self.transform:
          img = self.transform(img)
        return img, label
```

## model 구현

[paper](https://arxiv.org/pdf/1512.03385.pdf)

### Conv Block

ReLU 와 Batchnorm 의 순서는 상관 없다


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

### ResBlock

#### down sampling 의 이유 

ResBlock 의 skip connection 시에 paper 처럼 하려면 identity 를 더해주어야 한다. 이때 tensor 간의 합은 차원이 같아야 하는데, layer 2로 넘어갈 때 channel 수가 128로 바뀌는데, skip connection 을 통한 tensor는 channel 이 2배가 되지 않기 때문에 downsampling 을 해주어야 한다 (channel 2배 이외에도 image 크기가 1/2배 된다)


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

### ResNet

downsample 시에 maxpool 로 사이즈를 조정할 수 도 있지만 downsample stride를  2로 설정하면 maxpool 안해줘도 된다


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
                    nn.Conv2d(int(out_plane / 2), out_plane, kernel_size=1, stride=stride, bias=False),
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

model output size test


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

    Output tensor shape is : torch.Size([4, 10])


## Train


```python
# Build user-defined ResNet model
model_scratch = ResNet(3, 10).cuda()
model_scratch
```




    ResNet(
      (enc): ConvBlock(
        (layers): ModuleList(
          (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(1, 1))
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
        )
      )
      (max_pool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      (layer1): Sequential(
        (0): ResBlock(
          (relu): ReLU()
          (conv1): Sequential(
            (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (resblk): Identity()
        )
        (1): ResBlock(
          (relu): ReLU()
          (conv1): Sequential(
            (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (resblk): Identity()
        )
        (2): ResBlock(
          (relu): ReLU()
          (conv1): Sequential(
            (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (resblk): Identity()
        )
      )
      (layer2): Sequential(
        (0): Sequential(
          (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): ResBlock(
          (relu): ReLU()
          (conv1): Sequential(
            (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (resblk): Identity()
        )
        (2): ResBlock(
          (relu): ReLU()
          (conv1): Sequential(
            (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (resblk): Identity()
        )
        (3): ResBlock(
          (relu): ReLU()
          (conv1): Sequential(
            (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (resblk): Identity()
        )
        (4): ResBlock(
          (relu): ReLU()
          (conv1): Sequential(
            (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (resblk): Identity()
        )
      )
      (layer3): Sequential(
        (0): Sequential(
          (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): ResBlock(
          (relu): ReLU()
          (conv1): Sequential(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (resblk): Identity()
        )
        (2): ResBlock(
          (relu): ReLU()
          (conv1): Sequential(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (resblk): Identity()
        )
        (3): ResBlock(
          (relu): ReLU()
          (conv1): Sequential(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (resblk): Identity()
        )
        (4): ResBlock(
          (relu): ReLU()
          (conv1): Sequential(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (resblk): Identity()
        )
        (5): ResBlock(
          (relu): ReLU()
          (conv1): Sequential(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (resblk): Identity()
        )
        (6): ResBlock(
          (relu): ReLU()
          (conv1): Sequential(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (resblk): Identity()
        )
      )
      (layer4): Sequential(
        (0): Sequential(
          (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): ResBlock(
          (relu): ReLU()
          (conv1): Sequential(
            (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (resblk): Identity()
        )
        (2): ResBlock(
          (relu): ReLU()
          (conv1): Sequential(
            (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (resblk): Identity()
        )
        (3): ResBlock(
          (relu): ReLU()
          (conv1): Sequential(
            (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (resblk): Identity()
        )
      )
      (avg): AdaptiveAvgPool2d(output_size=(1, 1))
      (fc): Linear(in_features=512, out_features=10, bias=True)
    )




```python
# Loss function and Optimizer
from torch.optim import Adam

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model_scratch.parameters(), lr=1e-4)
```

graph 그리기 위한 log


```python
log_dir ='./log'
```


```python
# quickdraw train/validatoin dataset and dataloader
qd_train_dataset = QuickDrawDataset(train_data, train_label, transform)
qd_val_dataset = QuickDrawDataset(val_data, val_label, transform)

qd_train_dataloader = DataLoader(qd_train_dataset, batch_size=4, shuffle=True)
qd_val_dataloader = DataLoader(qd_val_dataset, batch_size=4, shuffle=True)
```


```python
# Misc
class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
      self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count
```


```python
# Main
os.makedirs(log_dir, exist_ok=True)

with open(os.path.join(log_dir, 'scratch_train_log.csv'), 'w') as log:
    # Training
    
    for iter, (img, label) in enumerate(qd_train_dataloader):
        img,label = img.float().cuda(),label.long().cuda()

        optimizer.zero_grad()
        pred = model_scratch(img)

        loss = criterion(pred,label)

        loss.backward()
        optimizer.step()

        pred_label = torch.argmax(pred, 1)
        acc = (pred_label == label).sum().item() / len(img)

        train_loss = loss.item()
        train_acc = acc

  
      # Validation 
        if (iter % 20 == 0) or (iter == len(qd_train_dataloader)-1):
            model_scratch.eval()
            valid_loss, valid_acc = AverageMeter(), AverageMeter()

            for img, label in qd_val_dataloader:
                img, label = img.float().cuda(), label.long().cuda()
                with torch.no_grad():
                    pred = model_scratch(img)
                loss = criterion(pred,label)
                pred_label = torch.argmax(pred,1)
                acc = (pred_label == label).sum().item() / len(img)

                valid_loss.update(loss.item(),len(img))
                valid_acc.update(acc,len(img))

            valid_loss = valid_loss.avg
            valid_acc = valid_acc.avg
          
            print("Iter [%3d/%3d] | Train Loss %.4f | Train Acc %.4f | Valid Loss %.4f | Valid Acc %.4f" % (iter, len(qd_train_dataloader), train_loss, train_acc, valid_loss, valid_acc))
          
            # Train Log Writing
            log.write('%d,%.4f,%.4f,%.4f,%.4f\n'%(iter, train_loss, train_acc, valid_loss, valid_acc))
        model_scratch.train()
```

    /opt/conda/lib/python3.8/site-packages/torchvision/transforms/functional.py:126: UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. This means you can write to the underlying (supposedly non-writeable) NumPy array using the tensor. You may want to copy the array to protect its data or make it writeable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at  ../torch/csrc/utils/tensor_numpy.cpp:189.)
      img = torch.from_numpy(pic.transpose((2, 0, 1))).contiguous()


    Iter [  0/11250] | Train Loss 2.4557 | Train Acc 0.0000 | Valid Loss 2.3090 | Valid Acc 0.0998
    Iter [ 20/11250] | Train Loss 1.4652 | Train Acc 0.7500 | Valid Loss 2.9662 | Valid Acc 0.1762
    Iter [ 40/11250] | Train Loss 1.6796 | Train Acc 0.2500 | Valid Loss 5.4018 | Valid Acc 0.2336
    Iter [ 60/11250] | Train Loss 1.8647 | Train Acc 0.2500 | Valid Loss 3.3267 | Valid Acc 0.3512
    Iter [ 80/11250] | Train Loss 2.4778 | Train Acc 0.2500 | Valid Loss 1.9507 | Valid Acc 0.4142
    Iter [100/11250] | Train Loss 1.0885 | Train Acc 0.5000 | Valid Loss 1.7938 | Valid Acc 0.4268
    Iter [120/11250] | Train Loss 0.9346 | Train Acc 0.7500 | Valid Loss 1.9187 | Valid Acc 0.4422
    Iter [140/11250] | Train Loss 1.5834 | Train Acc 0.5000 | Valid Loss 1.7617 | Valid Acc 0.4144
    Iter [160/11250] | Train Loss 1.3280 | Train Acc 0.5000 | Valid Loss 2.2629 | Valid Acc 0.4018
    Iter [180/11250] | Train Loss 0.3775 | Train Acc 1.0000 | Valid Loss 1.7191 | Valid Acc 0.5464
    Iter [200/11250] | Train Loss 1.6580 | Train Acc 0.2500 | Valid Loss 1.7635 | Valid Acc 0.5590
    Iter [220/11250] | Train Loss 1.0215 | Train Acc 0.7500 | Valid Loss 1.2892 | Valid Acc 0.5834
    Iter [240/11250] | Train Loss 0.5347 | Train Acc 0.7500 | Valid Loss 1.1908 | Valid Acc 0.6622
    Iter [260/11250] | Train Loss 1.0238 | Train Acc 0.7500 | Valid Loss 1.4810 | Valid Acc 0.5560
    Iter [280/11250] | Train Loss 0.5277 | Train Acc 0.7500 | Valid Loss 1.1641 | Valid Acc 0.6462
    Iter [300/11250] | Train Loss 1.1599 | Train Acc 0.7500 | Valid Loss 1.5112 | Valid Acc 0.6232
    Iter [320/11250] | Train Loss 0.7087 | Train Acc 0.5000 | Valid Loss 1.4871 | Valid Acc 0.5882
    Iter [340/11250] | Train Loss 0.3795 | Train Acc 1.0000 | Valid Loss 1.1931 | Valid Acc 0.6000
    Iter [360/11250] | Train Loss 0.5875 | Train Acc 0.7500 | Valid Loss 0.9855 | Valid Acc 0.7224
    Iter [380/11250] | Train Loss 2.0808 | Train Acc 0.5000 | Valid Loss 1.0004 | Valid Acc 0.6880
    Iter [400/11250] | Train Loss 0.9968 | Train Acc 0.5000 | Valid Loss 1.4406 | Valid Acc 0.6434
    Iter [420/11250] | Train Loss 0.6797 | Train Acc 0.7500 | Valid Loss 1.1868 | Valid Acc 0.5988
    Iter [440/11250] | Train Loss 0.1967 | Train Acc 1.0000 | Valid Loss 1.0924 | Valid Acc 0.6610
    Iter [460/11250] | Train Loss 2.5372 | Train Acc 0.2500 | Valid Loss 1.3341 | Valid Acc 0.6426
    Iter [480/11250] | Train Loss 0.8337 | Train Acc 0.5000 | Valid Loss 1.0865 | Valid Acc 0.6494
    Iter [500/11250] | Train Loss 1.1318 | Train Acc 0.7500 | Valid Loss 0.9894 | Valid Acc 0.7240
    Iter [520/11250] | Train Loss 0.6192 | Train Acc 0.7500 | Valid Loss 0.8890 | Valid Acc 0.7124
    Iter [540/11250] | Train Loss 0.7896 | Train Acc 0.7500 | Valid Loss 0.9677 | Valid Acc 0.7206
    Iter [560/11250] | Train Loss 0.6491 | Train Acc 0.7500 | Valid Loss 0.9721 | Valid Acc 0.7090
    Iter [580/11250] | Train Loss 0.6874 | Train Acc 0.7500 | Valid Loss 0.9922 | Valid Acc 0.7148
    Iter [600/11250] | Train Loss 1.2141 | Train Acc 0.5000 | Valid Loss 0.8797 | Valid Acc 0.7538
    Iter [620/11250] | Train Loss 0.8697 | Train Acc 0.7500 | Valid Loss 0.8036 | Valid Acc 0.7398
    Iter [640/11250] | Train Loss 0.8192 | Train Acc 0.7500 | Valid Loss 1.1653 | Valid Acc 0.6308
    Iter [660/11250] | Train Loss 1.1667 | Train Acc 0.5000 | Valid Loss 1.1679 | Valid Acc 0.6896
    Iter [680/11250] | Train Loss 0.4851 | Train Acc 1.0000 | Valid Loss 0.9343 | Valid Acc 0.7226
    Iter [700/11250] | Train Loss 0.6586 | Train Acc 0.7500 | Valid Loss 1.0328 | Valid Acc 0.6926
    Iter [720/11250] | Train Loss 1.9298 | Train Acc 0.5000 | Valid Loss 1.2105 | Valid Acc 0.7174
    Iter [740/11250] | Train Loss 0.3728 | Train Acc 1.0000 | Valid Loss 0.9047 | Valid Acc 0.7004
    Iter [760/11250] | Train Loss 1.1414 | Train Acc 0.7500 | Valid Loss 0.7371 | Valid Acc 0.7838
    Iter [780/11250] | Train Loss 1.0098 | Train Acc 0.5000 | Valid Loss 0.8479 | Valid Acc 0.7464
    Iter [800/11250] | Train Loss 0.9017 | Train Acc 0.7500 | Valid Loss 0.6548 | Valid Acc 0.8070
    Iter [820/11250] | Train Loss 1.5111 | Train Acc 0.5000 | Valid Loss 0.8737 | Valid Acc 0.7710
    Iter [840/11250] | Train Loss 3.5070 | Train Acc 0.2500 | Valid Loss 0.8665 | Valid Acc 0.7412
    Iter [860/11250] | Train Loss 2.5340 | Train Acc 0.2500 | Valid Loss 0.9015 | Valid Acc 0.7094
    Iter [880/11250] | Train Loss 0.7909 | Train Acc 0.7500 | Valid Loss 0.9195 | Valid Acc 0.6984
    Iter [900/11250] | Train Loss 2.2677 | Train Acc 0.5000 | Valid Loss 0.6912 | Valid Acc 0.8042
    Iter [920/11250] | Train Loss 0.8849 | Train Acc 0.7500 | Valid Loss 0.7821 | Valid Acc 0.7630
    Iter [940/11250] | Train Loss 0.5307 | Train Acc 1.0000 | Valid Loss 0.7600 | Valid Acc 0.7834
    Iter [960/11250] | Train Loss 0.9977 | Train Acc 0.5000 | Valid Loss 0.6937 | Valid Acc 0.8004
    Iter [980/11250] | Train Loss 0.7988 | Train Acc 0.7500 | Valid Loss 0.8060 | Valid Acc 0.7530
    Iter [1000/11250] | Train Loss 0.5606 | Train Acc 0.7500 | Valid Loss 0.8877 | Valid Acc 0.7088
    Iter [1020/11250] | Train Loss 0.3767 | Train Acc 1.0000 | Valid Loss 0.8540 | Valid Acc 0.7516
    Iter [1040/11250] | Train Loss 1.2609 | Train Acc 0.7500 | Valid Loss 0.8118 | Valid Acc 0.7508
    Iter [1060/11250] | Train Loss 0.1652 | Train Acc 1.0000 | Valid Loss 0.6801 | Valid Acc 0.7980
    Iter [1080/11250] | Train Loss 0.4207 | Train Acc 0.7500 | Valid Loss 0.7097 | Valid Acc 0.7798
    Iter [1100/11250] | Train Loss 0.6302 | Train Acc 0.7500 | Valid Loss 0.7176 | Valid Acc 0.7870
    Iter [1120/11250] | Train Loss 1.9659 | Train Acc 0.2500 | Valid Loss 0.6619 | Valid Acc 0.7884
    Iter [1140/11250] | Train Loss 1.5162 | Train Acc 0.7500 | Valid Loss 0.7231 | Valid Acc 0.7732
    Iter [1160/11250] | Train Loss 0.9247 | Train Acc 0.5000 | Valid Loss 0.7377 | Valid Acc 0.7658
    Iter [1180/11250] | Train Loss 0.3754 | Train Acc 1.0000 | Valid Loss 0.7123 | Valid Acc 0.7866
    Iter [1200/11250] | Train Loss 0.1180 | Train Acc 1.0000 | Valid Loss 0.6919 | Valid Acc 0.7902
    Iter [1220/11250] | Train Loss 0.7150 | Train Acc 0.7500 | Valid Loss 0.8221 | Valid Acc 0.7612
    Iter [1240/11250] | Train Loss 0.9434 | Train Acc 0.7500 | Valid Loss 0.6983 | Valid Acc 0.8106
    Iter [1260/11250] | Train Loss 0.1441 | Train Acc 1.0000 | Valid Loss 0.7179 | Valid Acc 0.7956
    Iter [1280/11250] | Train Loss 0.3171 | Train Acc 1.0000 | Valid Loss 0.5881 | Valid Acc 0.8246
    Iter [1300/11250] | Train Loss 0.3965 | Train Acc 1.0000 | Valid Loss 0.5743 | Valid Acc 0.8304
    Iter [1320/11250] | Train Loss 0.7544 | Train Acc 0.7500 | Valid Loss 0.6332 | Valid Acc 0.8088
    Iter [1340/11250] | Train Loss 0.8797 | Train Acc 0.5000 | Valid Loss 0.7051 | Valid Acc 0.7752
    Iter [1360/11250] | Train Loss 1.4560 | Train Acc 0.7500 | Valid Loss 0.7110 | Valid Acc 0.7658
    Iter [1380/11250] | Train Loss 1.3868 | Train Acc 0.5000 | Valid Loss 0.5635 | Valid Acc 0.8372
    Iter [1400/11250] | Train Loss 0.7472 | Train Acc 0.5000 | Valid Loss 0.6596 | Valid Acc 0.8032
    Iter [1420/11250] | Train Loss 0.6691 | Train Acc 0.7500 | Valid Loss 0.6054 | Valid Acc 0.8220
    Iter [1440/11250] | Train Loss 1.4114 | Train Acc 0.7500 | Valid Loss 0.5863 | Valid Acc 0.8212
    Iter [1460/11250] | Train Loss 0.2170 | Train Acc 1.0000 | Valid Loss 0.5752 | Valid Acc 0.8198
    Iter [1480/11250] | Train Loss 0.2714 | Train Acc 0.7500 | Valid Loss 0.5908 | Valid Acc 0.8142
    Iter [1500/11250] | Train Loss 1.9398 | Train Acc 0.5000 | Valid Loss 0.6643 | Valid Acc 0.8060
    Iter [1520/11250] | Train Loss 2.7779 | Train Acc 0.5000 | Valid Loss 0.6855 | Valid Acc 0.8096
    Iter [1540/11250] | Train Loss 0.3892 | Train Acc 0.7500 | Valid Loss 0.8190 | Valid Acc 0.7716
    Iter [1560/11250] | Train Loss 0.5040 | Train Acc 1.0000 | Valid Loss 0.7583 | Valid Acc 0.7572
    Iter [1580/11250] | Train Loss 0.5516 | Train Acc 0.7500 | Valid Loss 0.7050 | Valid Acc 0.7900
    Iter [1600/11250] | Train Loss 0.3752 | Train Acc 1.0000 | Valid Loss 0.6422 | Valid Acc 0.8090
    Iter [1620/11250] | Train Loss 1.0462 | Train Acc 0.7500 | Valid Loss 0.7782 | Valid Acc 0.7638
    Iter [1640/11250] | Train Loss 0.8950 | Train Acc 0.7500 | Valid Loss 0.6275 | Valid Acc 0.8030
    Iter [1660/11250] | Train Loss 0.2900 | Train Acc 1.0000 | Valid Loss 0.6840 | Valid Acc 0.7836
    Iter [1680/11250] | Train Loss 0.4997 | Train Acc 0.7500 | Valid Loss 0.6268 | Valid Acc 0.8168
    Iter [1700/11250] | Train Loss 0.4030 | Train Acc 0.7500 | Valid Loss 0.6245 | Valid Acc 0.8218
    Iter [1720/11250] | Train Loss 0.7512 | Train Acc 0.7500 | Valid Loss 0.5936 | Valid Acc 0.8088
    Iter [1740/11250] | Train Loss 0.6196 | Train Acc 0.7500 | Valid Loss 0.5674 | Valid Acc 0.8216
    Iter [1760/11250] | Train Loss 0.4511 | Train Acc 1.0000 | Valid Loss 0.5872 | Valid Acc 0.8236
    Iter [1780/11250] | Train Loss 0.3221 | Train Acc 1.0000 | Valid Loss 0.5729 | Valid Acc 0.8278
    Iter [1800/11250] | Train Loss 1.8114 | Train Acc 0.5000 | Valid Loss 0.5771 | Valid Acc 0.8268
    Iter [1820/11250] | Train Loss 0.1120 | Train Acc 1.0000 | Valid Loss 0.5772 | Valid Acc 0.8280
    Iter [1840/11250] | Train Loss 0.1424 | Train Acc 1.0000 | Valid Loss 0.5441 | Valid Acc 0.8380
    Iter [1860/11250] | Train Loss 0.7168 | Train Acc 0.7500 | Valid Loss 0.5806 | Valid Acc 0.8264
    Iter [1880/11250] | Train Loss 0.9154 | Train Acc 0.2500 | Valid Loss 0.6255 | Valid Acc 0.8258
    Iter [1900/11250] | Train Loss 1.0422 | Train Acc 0.7500 | Valid Loss 0.6168 | Valid Acc 0.8318
    Iter [1920/11250] | Train Loss 2.9096 | Train Acc 0.2500 | Valid Loss 0.5467 | Valid Acc 0.8350
    Iter [1940/11250] | Train Loss 1.0498 | Train Acc 0.5000 | Valid Loss 0.5826 | Valid Acc 0.8258
    Iter [1960/11250] | Train Loss 0.1919 | Train Acc 1.0000 | Valid Loss 0.5126 | Valid Acc 0.8398
    Iter [1980/11250] | Train Loss 0.7384 | Train Acc 0.7500 | Valid Loss 0.5540 | Valid Acc 0.8276
    Iter [2000/11250] | Train Loss 1.4997 | Train Acc 0.5000 | Valid Loss 0.6175 | Valid Acc 0.8230
    Iter [2020/11250] | Train Loss 0.3818 | Train Acc 1.0000 | Valid Loss 0.5217 | Valid Acc 0.8436
    Iter [2040/11250] | Train Loss 1.1251 | Train Acc 0.5000 | Valid Loss 0.6257 | Valid Acc 0.7960
    Iter [2060/11250] | Train Loss 1.1264 | Train Acc 0.5000 | Valid Loss 0.6758 | Valid Acc 0.7916
    Iter [2080/11250] | Train Loss 0.8104 | Train Acc 0.7500 | Valid Loss 0.6319 | Valid Acc 0.8014
    Iter [2100/11250] | Train Loss 1.0864 | Train Acc 0.5000 | Valid Loss 0.4901 | Valid Acc 0.8536
    Iter [2120/11250] | Train Loss 0.3828 | Train Acc 0.7500 | Valid Loss 0.4937 | Valid Acc 0.8476
    Iter [2140/11250] | Train Loss 2.1726 | Train Acc 0.7500 | Valid Loss 0.5439 | Valid Acc 0.8460
    Iter [2160/11250] | Train Loss 0.0468 | Train Acc 1.0000 | Valid Loss 0.5574 | Valid Acc 0.8422
    Iter [2180/11250] | Train Loss 0.4327 | Train Acc 0.7500 | Valid Loss 0.4630 | Valid Acc 0.8582
    Iter [2200/11250] | Train Loss 0.1405 | Train Acc 1.0000 | Valid Loss 0.5971 | Valid Acc 0.8168
    Iter [2220/11250] | Train Loss 0.5999 | Train Acc 0.7500 | Valid Loss 0.5815 | Valid Acc 0.8372
    Iter [2240/11250] | Train Loss 1.0745 | Train Acc 0.5000 | Valid Loss 0.5269 | Valid Acc 0.8466
    Iter [2260/11250] | Train Loss 0.1790 | Train Acc 1.0000 | Valid Loss 0.4670 | Valid Acc 0.8574
    Iter [2280/11250] | Train Loss 0.5883 | Train Acc 0.7500 | Valid Loss 0.4819 | Valid Acc 0.8532
    Iter [2300/11250] | Train Loss 0.1703 | Train Acc 1.0000 | Valid Loss 0.5096 | Valid Acc 0.8518
    Iter [2320/11250] | Train Loss 1.0593 | Train Acc 0.5000 | Valid Loss 0.7090 | Valid Acc 0.7788
    Iter [2340/11250] | Train Loss 0.2454 | Train Acc 1.0000 | Valid Loss 0.5138 | Valid Acc 0.8430
    Iter [2360/11250] | Train Loss 0.1961 | Train Acc 1.0000 | Valid Loss 0.5224 | Valid Acc 0.8486
    Iter [2380/11250] | Train Loss 0.8169 | Train Acc 0.7500 | Valid Loss 0.4655 | Valid Acc 0.8648
    Iter [2400/11250] | Train Loss 0.0316 | Train Acc 1.0000 | Valid Loss 0.4592 | Valid Acc 0.8720
    Iter [2420/11250] | Train Loss 0.1973 | Train Acc 1.0000 | Valid Loss 0.5456 | Valid Acc 0.8374
    Iter [2440/11250] | Train Loss 0.3090 | Train Acc 1.0000 | Valid Loss 0.4789 | Valid Acc 0.8642
    Iter [2460/11250] | Train Loss 0.5379 | Train Acc 0.7500 | Valid Loss 0.4975 | Valid Acc 0.8492
    Iter [2480/11250] | Train Loss 0.5649 | Train Acc 0.7500 | Valid Loss 0.5121 | Valid Acc 0.8504
    Iter [2500/11250] | Train Loss 0.7563 | Train Acc 0.7500 | Valid Loss 0.5624 | Valid Acc 0.8466
    Iter [2520/11250] | Train Loss 0.4819 | Train Acc 0.7500 | Valid Loss 0.6474 | Valid Acc 0.7996
    Iter [2540/11250] | Train Loss 0.0409 | Train Acc 1.0000 | Valid Loss 0.5792 | Valid Acc 0.8222
    Iter [2560/11250] | Train Loss 1.2121 | Train Acc 0.5000 | Valid Loss 0.5705 | Valid Acc 0.8370
    Iter [2580/11250] | Train Loss 0.5781 | Train Acc 1.0000 | Valid Loss 0.5631 | Valid Acc 0.8288
    Iter [2600/11250] | Train Loss 0.2937 | Train Acc 0.7500 | Valid Loss 0.6668 | Valid Acc 0.7900
    Iter [2620/11250] | Train Loss 0.4685 | Train Acc 0.7500 | Valid Loss 0.5294 | Valid Acc 0.8492
    Iter [2640/11250] | Train Loss 2.5506 | Train Acc 0.5000 | Valid Loss 0.5398 | Valid Acc 0.8270
    Iter [2660/11250] | Train Loss 0.5554 | Train Acc 0.7500 | Valid Loss 0.5211 | Valid Acc 0.8430
    Iter [2680/11250] | Train Loss 0.5783 | Train Acc 0.7500 | Valid Loss 0.6126 | Valid Acc 0.7964
    Iter [2700/11250] | Train Loss 1.1740 | Train Acc 0.5000 | Valid Loss 0.5015 | Valid Acc 0.8472
    Iter [2720/11250] | Train Loss 0.2219 | Train Acc 1.0000 | Valid Loss 0.4386 | Valid Acc 0.8654
    Iter [2740/11250] | Train Loss 0.8015 | Train Acc 0.7500 | Valid Loss 0.5391 | Valid Acc 0.8272
    Iter [2760/11250] | Train Loss 0.1356 | Train Acc 1.0000 | Valid Loss 0.4391 | Valid Acc 0.8672
    Iter [2780/11250] | Train Loss 1.0748 | Train Acc 0.5000 | Valid Loss 0.4264 | Valid Acc 0.8730
    Iter [2800/11250] | Train Loss 0.1819 | Train Acc 1.0000 | Valid Loss 0.4859 | Valid Acc 0.8616
    Iter [2820/11250] | Train Loss 0.6254 | Train Acc 0.7500 | Valid Loss 0.4738 | Valid Acc 0.8606
    Iter [2840/11250] | Train Loss 0.5717 | Train Acc 0.7500 | Valid Loss 0.4627 | Valid Acc 0.8580
    Iter [2860/11250] | Train Loss 0.4442 | Train Acc 0.7500 | Valid Loss 0.5071 | Valid Acc 0.8426
    Iter [2880/11250] | Train Loss 1.2458 | Train Acc 0.7500 | Valid Loss 0.7731 | Valid Acc 0.7606
    Iter [2900/11250] | Train Loss 0.7081 | Train Acc 0.7500 | Valid Loss 0.5452 | Valid Acc 0.8354
    Iter [2920/11250] | Train Loss 0.3968 | Train Acc 1.0000 | Valid Loss 0.4932 | Valid Acc 0.8506
    Iter [2940/11250] | Train Loss 0.2015 | Train Acc 1.0000 | Valid Loss 0.5001 | Valid Acc 0.8458
    Iter [2960/11250] | Train Loss 1.9253 | Train Acc 0.2500 | Valid Loss 0.4924 | Valid Acc 0.8598
    Iter [2980/11250] | Train Loss 0.7034 | Train Acc 0.5000 | Valid Loss 0.5002 | Valid Acc 0.8532
    Iter [3000/11250] | Train Loss 0.3115 | Train Acc 1.0000 | Valid Loss 0.4988 | Valid Acc 0.8542
    Iter [3020/11250] | Train Loss 0.3667 | Train Acc 0.7500 | Valid Loss 0.5178 | Valid Acc 0.8382
    Iter [3040/11250] | Train Loss 0.7830 | Train Acc 0.7500 | Valid Loss 0.5119 | Valid Acc 0.8460
    Iter [3060/11250] | Train Loss 0.1514 | Train Acc 1.0000 | Valid Loss 0.4948 | Valid Acc 0.8516
    Iter [3080/11250] | Train Loss 0.3779 | Train Acc 1.0000 | Valid Loss 0.4523 | Valid Acc 0.8662
    Iter [3100/11250] | Train Loss 0.1219 | Train Acc 1.0000 | Valid Loss 0.5237 | Valid Acc 0.8462
    Iter [3120/11250] | Train Loss 0.1195 | Train Acc 1.0000 | Valid Loss 0.5243 | Valid Acc 0.8440
    Iter [3140/11250] | Train Loss 2.1868 | Train Acc 0.5000 | Valid Loss 0.4556 | Valid Acc 0.8618
    Iter [3160/11250] | Train Loss 1.7417 | Train Acc 0.5000 | Valid Loss 0.5623 | Valid Acc 0.8326
    Iter [3180/11250] | Train Loss 0.3791 | Train Acc 0.7500 | Valid Loss 0.4763 | Valid Acc 0.8560
    Iter [3200/11250] | Train Loss 0.4706 | Train Acc 0.7500 | Valid Loss 0.4338 | Valid Acc 0.8682
    Iter [3220/11250] | Train Loss 0.2095 | Train Acc 1.0000 | Valid Loss 0.4293 | Valid Acc 0.8736
    Iter [3240/11250] | Train Loss 0.7879 | Train Acc 0.5000 | Valid Loss 0.5032 | Valid Acc 0.8380
    Iter [3260/11250] | Train Loss 0.6186 | Train Acc 0.7500 | Valid Loss 0.4709 | Valid Acc 0.8542
    Iter [3280/11250] | Train Loss 0.6284 | Train Acc 0.7500 | Valid Loss 0.5033 | Valid Acc 0.8524
    Iter [3300/11250] | Train Loss 0.0424 | Train Acc 1.0000 | Valid Loss 0.4320 | Valid Acc 0.8752
    Iter [3320/11250] | Train Loss 0.2906 | Train Acc 1.0000 | Valid Loss 0.4756 | Valid Acc 0.8550
    Iter [3340/11250] | Train Loss 0.2088 | Train Acc 1.0000 | Valid Loss 0.4564 | Valid Acc 0.8636
    Iter [3360/11250] | Train Loss 0.0469 | Train Acc 1.0000 | Valid Loss 0.7697 | Valid Acc 0.7470
    Iter [3380/11250] | Train Loss 1.4440 | Train Acc 0.7500 | Valid Loss 0.5219 | Valid Acc 0.8476
    Iter [3400/11250] | Train Loss 0.2370 | Train Acc 1.0000 | Valid Loss 0.4736 | Valid Acc 0.8560
    Iter [3420/11250] | Train Loss 0.1939 | Train Acc 1.0000 | Valid Loss 0.4645 | Valid Acc 0.8600
    Iter [3440/11250] | Train Loss 1.2617 | Train Acc 0.5000 | Valid Loss 0.5485 | Valid Acc 0.8232
    Iter [3460/11250] | Train Loss 0.1001 | Train Acc 1.0000 | Valid Loss 0.4637 | Valid Acc 0.8632
    Iter [3480/11250] | Train Loss 0.2470 | Train Acc 1.0000 | Valid Loss 0.4976 | Valid Acc 0.8682
    Iter [3500/11250] | Train Loss 0.9187 | Train Acc 0.7500 | Valid Loss 0.4423 | Valid Acc 0.8724
    Iter [3520/11250] | Train Loss 0.6177 | Train Acc 0.7500 | Valid Loss 0.4553 | Valid Acc 0.8646
    Iter [3540/11250] | Train Loss 0.5249 | Train Acc 0.7500 | Valid Loss 0.4696 | Valid Acc 0.8698
    Iter [3560/11250] | Train Loss 1.8092 | Train Acc 0.7500 | Valid Loss 0.4570 | Valid Acc 0.8684
    Iter [3580/11250] | Train Loss 0.1708 | Train Acc 1.0000 | Valid Loss 0.5690 | Valid Acc 0.8312
    Iter [3600/11250] | Train Loss 0.3388 | Train Acc 1.0000 | Valid Loss 0.5740 | Valid Acc 0.8292
    Iter [3620/11250] | Train Loss 1.2969 | Train Acc 0.2500 | Valid Loss 0.6154 | Valid Acc 0.8108
    Iter [3640/11250] | Train Loss 0.2501 | Train Acc 1.0000 | Valid Loss 0.4615 | Valid Acc 0.8652
    Iter [3660/11250] | Train Loss 2.2921 | Train Acc 0.7500 | Valid Loss 0.4898 | Valid Acc 0.8494
    Iter [3680/11250] | Train Loss 0.0257 | Train Acc 1.0000 | Valid Loss 0.4762 | Valid Acc 0.8494
    Iter [3700/11250] | Train Loss 0.2551 | Train Acc 0.7500 | Valid Loss 0.4663 | Valid Acc 0.8588
    Iter [3720/11250] | Train Loss 0.1797 | Train Acc 1.0000 | Valid Loss 0.4824 | Valid Acc 0.8560
    Iter [3740/11250] | Train Loss 0.3403 | Train Acc 1.0000 | Valid Loss 0.4837 | Valid Acc 0.8484
    Iter [3760/11250] | Train Loss 0.3730 | Train Acc 0.7500 | Valid Loss 0.4286 | Valid Acc 0.8696
    Iter [3780/11250] | Train Loss 0.3719 | Train Acc 0.7500 | Valid Loss 0.4416 | Valid Acc 0.8656
    Iter [3800/11250] | Train Loss 1.1846 | Train Acc 0.7500 | Valid Loss 0.4868 | Valid Acc 0.8520
    Iter [3820/11250] | Train Loss 0.8126 | Train Acc 0.5000 | Valid Loss 0.4808 | Valid Acc 0.8554
    Iter [3840/11250] | Train Loss 0.1759 | Train Acc 1.0000 | Valid Loss 0.4977 | Valid Acc 0.8464
    Iter [3860/11250] | Train Loss 0.0964 | Train Acc 1.0000 | Valid Loss 0.4382 | Valid Acc 0.8740
    Iter [3880/11250] | Train Loss 0.2028 | Train Acc 1.0000 | Valid Loss 0.4298 | Valid Acc 0.8702
    Iter [3900/11250] | Train Loss 0.5254 | Train Acc 1.0000 | Valid Loss 0.3979 | Valid Acc 0.8844
    Iter [3920/11250] | Train Loss 1.4985 | Train Acc 0.5000 | Valid Loss 0.4022 | Valid Acc 0.8812
    Iter [3940/11250] | Train Loss 0.0731 | Train Acc 1.0000 | Valid Loss 0.3785 | Valid Acc 0.8934
    Iter [3960/11250] | Train Loss 0.5252 | Train Acc 0.7500 | Valid Loss 0.5220 | Valid Acc 0.8278
    Iter [3980/11250] | Train Loss 0.1456 | Train Acc 1.0000 | Valid Loss 0.4009 | Valid Acc 0.8830
    Iter [4000/11250] | Train Loss 2.1613 | Train Acc 0.5000 | Valid Loss 0.3973 | Valid Acc 0.8808
    Iter [4020/11250] | Train Loss 0.3568 | Train Acc 1.0000 | Valid Loss 0.4444 | Valid Acc 0.8708
    Iter [4040/11250] | Train Loss 0.2101 | Train Acc 1.0000 | Valid Loss 0.4399 | Valid Acc 0.8646
    Iter [4060/11250] | Train Loss 0.7458 | Train Acc 0.7500 | Valid Loss 0.3952 | Valid Acc 0.8776
    Iter [4080/11250] | Train Loss 0.4133 | Train Acc 0.7500 | Valid Loss 0.4050 | Valid Acc 0.8820
    Iter [4100/11250] | Train Loss 1.5893 | Train Acc 0.7500 | Valid Loss 0.4269 | Valid Acc 0.8752
    Iter [4120/11250] | Train Loss 1.2005 | Train Acc 0.2500 | Valid Loss 0.4690 | Valid Acc 0.8546
    Iter [4140/11250] | Train Loss 1.4089 | Train Acc 0.5000 | Valid Loss 0.4662 | Valid Acc 0.8472
    Iter [4160/11250] | Train Loss 0.1833 | Train Acc 1.0000 | Valid Loss 0.5171 | Valid Acc 0.8320
    Iter [4180/11250] | Train Loss 0.3948 | Train Acc 0.7500 | Valid Loss 0.4422 | Valid Acc 0.8656
    Iter [4200/11250] | Train Loss 0.2135 | Train Acc 1.0000 | Valid Loss 0.4487 | Valid Acc 0.8686
    Iter [4220/11250] | Train Loss 0.7798 | Train Acc 0.7500 | Valid Loss 0.4722 | Valid Acc 0.8588
    Iter [4240/11250] | Train Loss 0.5656 | Train Acc 0.7500 | Valid Loss 0.4840 | Valid Acc 0.8472
    Iter [4260/11250] | Train Loss 0.9016 | Train Acc 0.7500 | Valid Loss 0.4491 | Valid Acc 0.8650
    Iter [4280/11250] | Train Loss 0.2483 | Train Acc 1.0000 | Valid Loss 0.3968 | Valid Acc 0.8776
    Iter [4300/11250] | Train Loss 0.9671 | Train Acc 0.7500 | Valid Loss 0.3797 | Valid Acc 0.8888
    Iter [4320/11250] | Train Loss 0.4375 | Train Acc 0.7500 | Valid Loss 0.4390 | Valid Acc 0.8798
    Iter [4340/11250] | Train Loss 0.1219 | Train Acc 1.0000 | Valid Loss 0.4580 | Valid Acc 0.8842
    Iter [4360/11250] | Train Loss 0.2995 | Train Acc 1.0000 | Valid Loss 0.4634 | Valid Acc 0.8670
    Iter [4380/11250] | Train Loss 1.7404 | Train Acc 0.7500 | Valid Loss 0.4506 | Valid Acc 0.8668
    Iter [4400/11250] | Train Loss 0.6310 | Train Acc 0.7500 | Valid Loss 0.5063 | Valid Acc 0.8454
    Iter [4420/11250] | Train Loss 0.5861 | Train Acc 0.7500 | Valid Loss 0.3850 | Valid Acc 0.8832
    Iter [4440/11250] | Train Loss 1.3695 | Train Acc 0.7500 | Valid Loss 0.4052 | Valid Acc 0.8796
    Iter [4460/11250] | Train Loss 0.1158 | Train Acc 1.0000 | Valid Loss 0.4588 | Valid Acc 0.8704
    Iter [4480/11250] | Train Loss 0.4250 | Train Acc 1.0000 | Valid Loss 0.3988 | Valid Acc 0.8822
    Iter [4500/11250] | Train Loss 0.1252 | Train Acc 1.0000 | Valid Loss 0.3870 | Valid Acc 0.8792
    Iter [4520/11250] | Train Loss 0.9274 | Train Acc 0.7500 | Valid Loss 0.4237 | Valid Acc 0.8726
    Iter [4540/11250] | Train Loss 0.0964 | Train Acc 1.0000 | Valid Loss 0.4076 | Valid Acc 0.8830
    Iter [4560/11250] | Train Loss 0.7920 | Train Acc 0.7500 | Valid Loss 0.4134 | Valid Acc 0.8892
    Iter [4580/11250] | Train Loss 0.6513 | Train Acc 0.7500 | Valid Loss 0.3885 | Valid Acc 0.8918
    Iter [4600/11250] | Train Loss 2.4282 | Train Acc 0.5000 | Valid Loss 0.3900 | Valid Acc 0.8838
    Iter [4620/11250] | Train Loss 0.4245 | Train Acc 1.0000 | Valid Loss 0.4696 | Valid Acc 0.8686
    Iter [4640/11250] | Train Loss 0.5740 | Train Acc 0.7500 | Valid Loss 0.4427 | Valid Acc 0.8674
    Iter [4660/11250] | Train Loss 0.3007 | Train Acc 0.7500 | Valid Loss 0.5012 | Valid Acc 0.8500
    Iter [4680/11250] | Train Loss 0.2198 | Train Acc 1.0000 | Valid Loss 0.3846 | Valid Acc 0.8940
    Iter [4700/11250] | Train Loss 0.3977 | Train Acc 0.7500 | Valid Loss 0.4271 | Valid Acc 0.8746
    Iter [4720/11250] | Train Loss 0.9243 | Train Acc 0.5000 | Valid Loss 0.4388 | Valid Acc 0.8746
    Iter [4740/11250] | Train Loss 0.6110 | Train Acc 0.7500 | Valid Loss 0.4808 | Valid Acc 0.8484
    Iter [4760/11250] | Train Loss 0.4931 | Train Acc 0.7500 | Valid Loss 0.3703 | Valid Acc 0.8924
    Iter [4780/11250] | Train Loss 0.5572 | Train Acc 0.7500 | Valid Loss 0.3843 | Valid Acc 0.8824
    Iter [4800/11250] | Train Loss 0.1967 | Train Acc 1.0000 | Valid Loss 0.3713 | Valid Acc 0.8978
    Iter [4820/11250] | Train Loss 0.6250 | Train Acc 0.7500 | Valid Loss 0.4565 | Valid Acc 0.8710
    Iter [4840/11250] | Train Loss 0.2410 | Train Acc 1.0000 | Valid Loss 0.5231 | Valid Acc 0.8282
    Iter [4860/11250] | Train Loss 0.0341 | Train Acc 1.0000 | Valid Loss 0.4379 | Valid Acc 0.8656
    Iter [4880/11250] | Train Loss 0.0341 | Train Acc 1.0000 | Valid Loss 0.4244 | Valid Acc 0.8736
    Iter [4900/11250] | Train Loss 0.5768 | Train Acc 0.7500 | Valid Loss 0.4735 | Valid Acc 0.8660
    Iter [4920/11250] | Train Loss 0.0444 | Train Acc 1.0000 | Valid Loss 0.4677 | Valid Acc 0.8626
    Iter [4940/11250] | Train Loss 0.6566 | Train Acc 0.7500 | Valid Loss 0.3707 | Valid Acc 0.8960
    Iter [4960/11250] | Train Loss 1.0096 | Train Acc 0.7500 | Valid Loss 0.4575 | Valid Acc 0.8732
    Iter [4980/11250] | Train Loss 1.1794 | Train Acc 0.5000 | Valid Loss 0.3642 | Valid Acc 0.8938
    Iter [5000/11250] | Train Loss 0.0137 | Train Acc 1.0000 | Valid Loss 0.3620 | Valid Acc 0.8972
    Iter [5020/11250] | Train Loss 0.6178 | Train Acc 1.0000 | Valid Loss 0.4447 | Valid Acc 0.8632
    Iter [5040/11250] | Train Loss 0.2807 | Train Acc 1.0000 | Valid Loss 0.4792 | Valid Acc 0.8510
    Iter [5060/11250] | Train Loss 0.3914 | Train Acc 0.7500 | Valid Loss 0.4210 | Valid Acc 0.8780
    Iter [5080/11250] | Train Loss 0.1697 | Train Acc 1.0000 | Valid Loss 0.4583 | Valid Acc 0.8574
    Iter [5100/11250] | Train Loss 2.5601 | Train Acc 0.5000 | Valid Loss 0.4194 | Valid Acc 0.8770
    Iter [5120/11250] | Train Loss 0.7158 | Train Acc 0.7500 | Valid Loss 0.4325 | Valid Acc 0.8720
    Iter [5140/11250] | Train Loss 0.1930 | Train Acc 1.0000 | Valid Loss 0.4213 | Valid Acc 0.8796
    Iter [5160/11250] | Train Loss 0.5577 | Train Acc 0.7500 | Valid Loss 0.3947 | Valid Acc 0.8804
    Iter [5180/11250] | Train Loss 0.0590 | Train Acc 1.0000 | Valid Loss 0.3707 | Valid Acc 0.8940
    Iter [5200/11250] | Train Loss 0.0681 | Train Acc 1.0000 | Valid Loss 0.3751 | Valid Acc 0.8908
    Iter [5220/11250] | Train Loss 0.2943 | Train Acc 1.0000 | Valid Loss 0.4014 | Valid Acc 0.8820
    Iter [5240/11250] | Train Loss 1.2697 | Train Acc 0.5000 | Valid Loss 0.4041 | Valid Acc 0.8802
    Iter [5260/11250] | Train Loss 2.0349 | Train Acc 0.5000 | Valid Loss 0.3807 | Valid Acc 0.8884
    Iter [5280/11250] | Train Loss 0.8448 | Train Acc 0.5000 | Valid Loss 0.3882 | Valid Acc 0.8876
    Iter [5300/11250] | Train Loss 1.3951 | Train Acc 0.7500 | Valid Loss 0.4112 | Valid Acc 0.8774
    Iter [5320/11250] | Train Loss 0.4613 | Train Acc 1.0000 | Valid Loss 0.4041 | Valid Acc 0.8804
    Iter [5340/11250] | Train Loss 1.1047 | Train Acc 0.7500 | Valid Loss 0.3465 | Valid Acc 0.8998
    Iter [5360/11250] | Train Loss 0.0301 | Train Acc 1.0000 | Valid Loss 0.3592 | Valid Acc 0.9026
    Iter [5380/11250] | Train Loss 1.3919 | Train Acc 0.7500 | Valid Loss 0.4497 | Valid Acc 0.8604
    Iter [5400/11250] | Train Loss 1.7171 | Train Acc 0.5000 | Valid Loss 0.4506 | Valid Acc 0.8670
    Iter [5420/11250] | Train Loss 0.4531 | Train Acc 1.0000 | Valid Loss 0.3815 | Valid Acc 0.8906
    Iter [5440/11250] | Train Loss 1.0091 | Train Acc 0.7500 | Valid Loss 0.4466 | Valid Acc 0.8738
    Iter [5460/11250] | Train Loss 0.4256 | Train Acc 0.7500 | Valid Loss 0.4965 | Valid Acc 0.8648
    Iter [5480/11250] | Train Loss 1.4662 | Train Acc 0.7500 | Valid Loss 0.3799 | Valid Acc 0.8840
    Iter [5500/11250] | Train Loss 0.3435 | Train Acc 1.0000 | Valid Loss 0.4402 | Valid Acc 0.8776
    Iter [5520/11250] | Train Loss 0.1688 | Train Acc 1.0000 | Valid Loss 0.4410 | Valid Acc 0.8780
    Iter [5540/11250] | Train Loss 0.0576 | Train Acc 1.0000 | Valid Loss 0.4375 | Valid Acc 0.8730
    Iter [5560/11250] | Train Loss 0.4574 | Train Acc 0.7500 | Valid Loss 0.4612 | Valid Acc 0.8618
    Iter [5580/11250] | Train Loss 0.0280 | Train Acc 1.0000 | Valid Loss 0.4286 | Valid Acc 0.8736
    Iter [5600/11250] | Train Loss 0.4399 | Train Acc 0.7500 | Valid Loss 0.3742 | Valid Acc 0.8896
    Iter [5620/11250] | Train Loss 0.5018 | Train Acc 0.7500 | Valid Loss 0.4392 | Valid Acc 0.8698
    Iter [5640/11250] | Train Loss 0.1862 | Train Acc 1.0000 | Valid Loss 0.4103 | Valid Acc 0.8756
    Iter [5660/11250] | Train Loss 0.3607 | Train Acc 0.7500 | Valid Loss 0.6797 | Valid Acc 0.7966
    Iter [5680/11250] | Train Loss 0.5704 | Train Acc 0.7500 | Valid Loss 0.3851 | Valid Acc 0.8906
    Iter [5700/11250] | Train Loss 0.2596 | Train Acc 1.0000 | Valid Loss 0.3793 | Valid Acc 0.8856
    Iter [5720/11250] | Train Loss 1.7461 | Train Acc 0.7500 | Valid Loss 0.3775 | Valid Acc 0.8936
    Iter [5740/11250] | Train Loss 0.1056 | Train Acc 1.0000 | Valid Loss 0.3687 | Valid Acc 0.8926
    Iter [5760/11250] | Train Loss 0.0808 | Train Acc 1.0000 | Valid Loss 0.3715 | Valid Acc 0.8908
    Iter [5780/11250] | Train Loss 1.3309 | Train Acc 0.7500 | Valid Loss 0.4372 | Valid Acc 0.8668
    Iter [5800/11250] | Train Loss 1.4323 | Train Acc 0.5000 | Valid Loss 0.4122 | Valid Acc 0.8734
    Iter [5820/11250] | Train Loss 0.0756 | Train Acc 1.0000 | Valid Loss 0.3789 | Valid Acc 0.8910
    Iter [5840/11250] | Train Loss 0.5131 | Train Acc 0.7500 | Valid Loss 0.4403 | Valid Acc 0.8708
    Iter [5860/11250] | Train Loss 0.1635 | Train Acc 1.0000 | Valid Loss 0.5340 | Valid Acc 0.8488
    Iter [5880/11250] | Train Loss 0.1643 | Train Acc 1.0000 | Valid Loss 0.4429 | Valid Acc 0.8768
    Iter [5900/11250] | Train Loss 0.0236 | Train Acc 1.0000 | Valid Loss 0.4052 | Valid Acc 0.8844
    Iter [5920/11250] | Train Loss 0.2318 | Train Acc 1.0000 | Valid Loss 0.3815 | Valid Acc 0.8890
    Iter [5940/11250] | Train Loss 0.1624 | Train Acc 1.0000 | Valid Loss 0.3930 | Valid Acc 0.8858
    Iter [5960/11250] | Train Loss 0.5241 | Train Acc 0.7500 | Valid Loss 0.3695 | Valid Acc 0.8906
    Iter [5980/11250] | Train Loss 0.5376 | Train Acc 0.7500 | Valid Loss 0.4152 | Valid Acc 0.8856
    Iter [6000/11250] | Train Loss 0.1292 | Train Acc 1.0000 | Valid Loss 0.4025 | Valid Acc 0.8902
    Iter [6020/11250] | Train Loss 0.4556 | Train Acc 1.0000 | Valid Loss 0.4188 | Valid Acc 0.8878
    Iter [6040/11250] | Train Loss 0.4931 | Train Acc 0.7500 | Valid Loss 0.4163 | Valid Acc 0.8840
    Iter [6060/11250] | Train Loss 1.2608 | Train Acc 0.7500 | Valid Loss 0.3927 | Valid Acc 0.8958
    Iter [6080/11250] | Train Loss 0.0393 | Train Acc 1.0000 | Valid Loss 0.3946 | Valid Acc 0.8806
    Iter [6100/11250] | Train Loss 0.1487 | Train Acc 1.0000 | Valid Loss 0.4329 | Valid Acc 0.8664
    Iter [6120/11250] | Train Loss 0.1731 | Train Acc 1.0000 | Valid Loss 0.4150 | Valid Acc 0.8766
    Iter [6140/11250] | Train Loss 0.3241 | Train Acc 1.0000 | Valid Loss 0.3780 | Valid Acc 0.8882
    Iter [6160/11250] | Train Loss 0.0245 | Train Acc 1.0000 | Valid Loss 0.3893 | Valid Acc 0.8828
    Iter [6180/11250] | Train Loss 0.2174 | Train Acc 1.0000 | Valid Loss 0.3991 | Valid Acc 0.8816
    Iter [6200/11250] | Train Loss 0.0437 | Train Acc 1.0000 | Valid Loss 0.4560 | Valid Acc 0.8712
    Iter [6220/11250] | Train Loss 0.3267 | Train Acc 1.0000 | Valid Loss 0.4485 | Valid Acc 0.8760
    Iter [6240/11250] | Train Loss 1.0553 | Train Acc 0.5000 | Valid Loss 0.3802 | Valid Acc 0.8886
    Iter [6260/11250] | Train Loss 0.1863 | Train Acc 1.0000 | Valid Loss 0.3797 | Valid Acc 0.8830
    Iter [6280/11250] | Train Loss 1.0208 | Train Acc 0.5000 | Valid Loss 0.3625 | Valid Acc 0.8948
    Iter [6300/11250] | Train Loss 0.0803 | Train Acc 1.0000 | Valid Loss 0.3501 | Valid Acc 0.9014
    Iter [6320/11250] | Train Loss 0.2288 | Train Acc 1.0000 | Valid Loss 0.3580 | Valid Acc 0.8968
    Iter [6340/11250] | Train Loss 0.9446 | Train Acc 0.7500 | Valid Loss 0.3603 | Valid Acc 0.8932
    Iter [6360/11250] | Train Loss 0.0067 | Train Acc 1.0000 | Valid Loss 0.3569 | Valid Acc 0.8982
    Iter [6380/11250] | Train Loss 0.7395 | Train Acc 0.5000 | Valid Loss 0.3407 | Valid Acc 0.9038
    Iter [6400/11250] | Train Loss 0.1684 | Train Acc 1.0000 | Valid Loss 0.3744 | Valid Acc 0.8896
    Iter [6420/11250] | Train Loss 0.4081 | Train Acc 0.7500 | Valid Loss 0.3825 | Valid Acc 0.8876
    Iter [6440/11250] | Train Loss 1.1351 | Train Acc 0.5000 | Valid Loss 0.3587 | Valid Acc 0.8930
    Iter [6460/11250] | Train Loss 1.4614 | Train Acc 0.5000 | Valid Loss 0.4046 | Valid Acc 0.8782
    Iter [6480/11250] | Train Loss 0.9320 | Train Acc 0.5000 | Valid Loss 0.4571 | Valid Acc 0.8564
    Iter [6500/11250] | Train Loss 0.0209 | Train Acc 1.0000 | Valid Loss 0.3647 | Valid Acc 0.8854
    Iter [6520/11250] | Train Loss 0.6617 | Train Acc 0.7500 | Valid Loss 0.3844 | Valid Acc 0.8838
    Iter [6540/11250] | Train Loss 0.6827 | Train Acc 0.7500 | Valid Loss 0.4073 | Valid Acc 0.8708
    Iter [6560/11250] | Train Loss 1.8165 | Train Acc 0.5000 | Valid Loss 0.3551 | Valid Acc 0.8948
    Iter [6580/11250] | Train Loss 0.4812 | Train Acc 0.7500 | Valid Loss 0.3649 | Valid Acc 0.8992
    Iter [6600/11250] | Train Loss 0.0947 | Train Acc 1.0000 | Valid Loss 0.3716 | Valid Acc 0.8892
    Iter [6620/11250] | Train Loss 1.6789 | Train Acc 0.7500 | Valid Loss 0.3961 | Valid Acc 0.8806
    Iter [6640/11250] | Train Loss 0.7920 | Train Acc 0.7500 | Valid Loss 0.4235 | Valid Acc 0.8770
    Iter [6660/11250] | Train Loss 0.5466 | Train Acc 0.7500 | Valid Loss 0.3803 | Valid Acc 0.8900
    Iter [6680/11250] | Train Loss 0.2214 | Train Acc 1.0000 | Valid Loss 0.3973 | Valid Acc 0.8744
    Iter [6700/11250] | Train Loss 1.0194 | Train Acc 0.7500 | Valid Loss 0.3748 | Valid Acc 0.8864
    Iter [6720/11250] | Train Loss 0.3261 | Train Acc 1.0000 | Valid Loss 0.3624 | Valid Acc 0.8954
    Iter [6740/11250] | Train Loss 0.6195 | Train Acc 0.7500 | Valid Loss 0.3994 | Valid Acc 0.8780
    Iter [6760/11250] | Train Loss 0.6718 | Train Acc 0.7500 | Valid Loss 0.4137 | Valid Acc 0.8778
    Iter [6780/11250] | Train Loss 0.5907 | Train Acc 0.7500 | Valid Loss 0.4041 | Valid Acc 0.8856
    Iter [6800/11250] | Train Loss 2.5247 | Train Acc 0.5000 | Valid Loss 0.3563 | Valid Acc 0.9002
    Iter [6820/11250] | Train Loss 0.3599 | Train Acc 0.7500 | Valid Loss 0.3768 | Valid Acc 0.8914
    Iter [6840/11250] | Train Loss 0.0550 | Train Acc 1.0000 | Valid Loss 0.4176 | Valid Acc 0.8738
    Iter [6860/11250] | Train Loss 1.6979 | Train Acc 0.7500 | Valid Loss 0.3838 | Valid Acc 0.8948
    Iter [6880/11250] | Train Loss 1.6230 | Train Acc 0.5000 | Valid Loss 0.4558 | Valid Acc 0.8782
    Iter [6900/11250] | Train Loss 0.1355 | Train Acc 1.0000 | Valid Loss 0.4047 | Valid Acc 0.8820
    Iter [6920/11250] | Train Loss 0.4936 | Train Acc 0.7500 | Valid Loss 0.3500 | Valid Acc 0.9038
    Iter [6940/11250] | Train Loss 0.9563 | Train Acc 0.5000 | Valid Loss 0.3664 | Valid Acc 0.8938
    Iter [6960/11250] | Train Loss 0.1461 | Train Acc 1.0000 | Valid Loss 0.3733 | Valid Acc 0.8950
    Iter [6980/11250] | Train Loss 0.0588 | Train Acc 1.0000 | Valid Loss 0.4106 | Valid Acc 0.8812
    Iter [7000/11250] | Train Loss 2.1927 | Train Acc 0.5000 | Valid Loss 0.3946 | Valid Acc 0.8878
    Iter [7020/11250] | Train Loss 0.1690 | Train Acc 1.0000 | Valid Loss 0.3824 | Valid Acc 0.8884
    Iter [7040/11250] | Train Loss 0.1344 | Train Acc 1.0000 | Valid Loss 0.4217 | Valid Acc 0.8800
    Iter [7060/11250] | Train Loss 0.0775 | Train Acc 1.0000 | Valid Loss 0.4319 | Valid Acc 0.8730
    Iter [7080/11250] | Train Loss 0.1307 | Train Acc 1.0000 | Valid Loss 0.5039 | Valid Acc 0.8570
    Iter [7100/11250] | Train Loss 1.2761 | Train Acc 0.5000 | Valid Loss 0.3998 | Valid Acc 0.8832
    Iter [7120/11250] | Train Loss 1.1164 | Train Acc 0.7500 | Valid Loss 0.3645 | Valid Acc 0.8892
    Iter [7140/11250] | Train Loss 1.9755 | Train Acc 0.5000 | Valid Loss 0.4339 | Valid Acc 0.8630
    Iter [7160/11250] | Train Loss 0.0487 | Train Acc 1.0000 | Valid Loss 0.3937 | Valid Acc 0.8804
    Iter [7180/11250] | Train Loss 1.0371 | Train Acc 0.7500 | Valid Loss 0.3807 | Valid Acc 0.8904
    Iter [7200/11250] | Train Loss 0.0329 | Train Acc 1.0000 | Valid Loss 0.3561 | Valid Acc 0.8932
    Iter [7220/11250] | Train Loss 1.0926 | Train Acc 0.7500 | Valid Loss 0.3643 | Valid Acc 0.8874
    Iter [7240/11250] | Train Loss 1.2978 | Train Acc 0.2500 | Valid Loss 0.4047 | Valid Acc 0.8778
    Iter [7260/11250] | Train Loss 0.0587 | Train Acc 1.0000 | Valid Loss 0.3915 | Valid Acc 0.8874
    Iter [7280/11250] | Train Loss 0.3301 | Train Acc 1.0000 | Valid Loss 0.4804 | Valid Acc 0.8542
    Iter [7300/11250] | Train Loss 0.7470 | Train Acc 0.7500 | Valid Loss 0.4255 | Valid Acc 0.8740
    Iter [7320/11250] | Train Loss 0.1180 | Train Acc 1.0000 | Valid Loss 0.3725 | Valid Acc 0.8952
    Iter [7340/11250] | Train Loss 0.1063 | Train Acc 1.0000 | Valid Loss 0.3618 | Valid Acc 0.9010
    Iter [7360/11250] | Train Loss 0.3569 | Train Acc 0.7500 | Valid Loss 0.3801 | Valid Acc 0.8932
    Iter [7380/11250] | Train Loss 0.0772 | Train Acc 1.0000 | Valid Loss 0.3577 | Valid Acc 0.9004
    Iter [7400/11250] | Train Loss 0.2651 | Train Acc 1.0000 | Valid Loss 0.3530 | Valid Acc 0.8974
    Iter [7420/11250] | Train Loss 0.4468 | Train Acc 0.7500 | Valid Loss 0.3583 | Valid Acc 0.8962
    Iter [7440/11250] | Train Loss 0.0483 | Train Acc 1.0000 | Valid Loss 0.3441 | Valid Acc 0.9008
    Iter [7460/11250] | Train Loss 0.2179 | Train Acc 1.0000 | Valid Loss 0.3310 | Valid Acc 0.8974
    Iter [7480/11250] | Train Loss 1.1019 | Train Acc 0.2500 | Valid Loss 0.3231 | Valid Acc 0.9026
    Iter [7500/11250] | Train Loss 0.3180 | Train Acc 0.7500 | Valid Loss 0.3727 | Valid Acc 0.8946
    Iter [7520/11250] | Train Loss 1.0243 | Train Acc 0.7500 | Valid Loss 0.3575 | Valid Acc 0.8994
    Iter [7540/11250] | Train Loss 0.0494 | Train Acc 1.0000 | Valid Loss 0.4100 | Valid Acc 0.8782
    Iter [7560/11250] | Train Loss 0.0478 | Train Acc 1.0000 | Valid Loss 0.3909 | Valid Acc 0.8810
    Iter [7580/11250] | Train Loss 0.2214 | Train Acc 1.0000 | Valid Loss 0.3494 | Valid Acc 0.8970
    Iter [7600/11250] | Train Loss 0.3541 | Train Acc 1.0000 | Valid Loss 0.3701 | Valid Acc 0.8948
    Iter [7620/11250] | Train Loss 0.0188 | Train Acc 1.0000 | Valid Loss 0.3338 | Valid Acc 0.9072
    Iter [7640/11250] | Train Loss 0.0929 | Train Acc 1.0000 | Valid Loss 0.3543 | Valid Acc 0.8976
    Iter [7660/11250] | Train Loss 0.0366 | Train Acc 1.0000 | Valid Loss 0.4203 | Valid Acc 0.8808
    Iter [7680/11250] | Train Loss 0.3280 | Train Acc 1.0000 | Valid Loss 0.3913 | Valid Acc 0.8924
    Iter [7700/11250] | Train Loss 0.2387 | Train Acc 1.0000 | Valid Loss 0.3514 | Valid Acc 0.9008
    Iter [7720/11250] | Train Loss 0.6643 | Train Acc 1.0000 | Valid Loss 0.3957 | Valid Acc 0.8866
    Iter [7740/11250] | Train Loss 0.4254 | Train Acc 0.7500 | Valid Loss 0.3734 | Valid Acc 0.8930
    Iter [7760/11250] | Train Loss 0.0358 | Train Acc 1.0000 | Valid Loss 0.3642 | Valid Acc 0.8978
    Iter [7780/11250] | Train Loss 0.3546 | Train Acc 0.7500 | Valid Loss 0.3530 | Valid Acc 0.8980
    Iter [7800/11250] | Train Loss 0.4106 | Train Acc 0.7500 | Valid Loss 0.3690 | Valid Acc 0.8898
    Iter [7820/11250] | Train Loss 0.0803 | Train Acc 1.0000 | Valid Loss 0.3538 | Valid Acc 0.9000
    Iter [7840/11250] | Train Loss 1.1157 | Train Acc 0.7500 | Valid Loss 0.3241 | Valid Acc 0.9046
    Iter [7860/11250] | Train Loss 1.4757 | Train Acc 0.7500 | Valid Loss 0.3450 | Valid Acc 0.9014
    Iter [7880/11250] | Train Loss 0.3723 | Train Acc 0.7500 | Valid Loss 0.3768 | Valid Acc 0.8878
    Iter [7900/11250] | Train Loss 0.2481 | Train Acc 1.0000 | Valid Loss 0.3878 | Valid Acc 0.8902
    Iter [7920/11250] | Train Loss 0.0979 | Train Acc 1.0000 | Valid Loss 0.3675 | Valid Acc 0.8918
    Iter [7940/11250] | Train Loss 0.0839 | Train Acc 1.0000 | Valid Loss 0.3276 | Valid Acc 0.9082
    Iter [7960/11250] | Train Loss 1.4070 | Train Acc 0.7500 | Valid Loss 0.3707 | Valid Acc 0.8926
    Iter [7980/11250] | Train Loss 0.4205 | Train Acc 0.7500 | Valid Loss 0.3375 | Valid Acc 0.9060
    Iter [8000/11250] | Train Loss 0.2830 | Train Acc 1.0000 | Valid Loss 0.3494 | Valid Acc 0.8960
    Iter [8020/11250] | Train Loss 0.2498 | Train Acc 1.0000 | Valid Loss 0.3207 | Valid Acc 0.9060
    Iter [8040/11250] | Train Loss 0.5232 | Train Acc 0.7500 | Valid Loss 0.3526 | Valid Acc 0.8944
    Iter [8060/11250] | Train Loss 0.3204 | Train Acc 1.0000 | Valid Loss 0.3431 | Valid Acc 0.9026
    Iter [8080/11250] | Train Loss 0.1649 | Train Acc 1.0000 | Valid Loss 0.3859 | Valid Acc 0.8846
    Iter [8100/11250] | Train Loss 0.0400 | Train Acc 1.0000 | Valid Loss 0.3831 | Valid Acc 0.8862
    Iter [8120/11250] | Train Loss 0.1384 | Train Acc 1.0000 | Valid Loss 0.3403 | Valid Acc 0.9032
    Iter [8140/11250] | Train Loss 0.0527 | Train Acc 1.0000 | Valid Loss 0.3591 | Valid Acc 0.8946
    Iter [8160/11250] | Train Loss 0.6732 | Train Acc 0.7500 | Valid Loss 0.3918 | Valid Acc 0.8838
    Iter [8180/11250] | Train Loss 0.0274 | Train Acc 1.0000 | Valid Loss 0.3537 | Valid Acc 0.9016
    Iter [8200/11250] | Train Loss 0.3239 | Train Acc 1.0000 | Valid Loss 0.3844 | Valid Acc 0.8914
    Iter [8220/11250] | Train Loss 0.2710 | Train Acc 1.0000 | Valid Loss 0.3676 | Valid Acc 0.8942
    Iter [8240/11250] | Train Loss 0.4410 | Train Acc 0.7500 | Valid Loss 0.3428 | Valid Acc 0.9016
    Iter [8260/11250] | Train Loss 1.0113 | Train Acc 0.5000 | Valid Loss 0.3409 | Valid Acc 0.8998
    Iter [8280/11250] | Train Loss 0.0243 | Train Acc 1.0000 | Valid Loss 0.3565 | Valid Acc 0.9020
    Iter [8300/11250] | Train Loss 0.5302 | Train Acc 0.7500 | Valid Loss 0.3704 | Valid Acc 0.8942
    Iter [8320/11250] | Train Loss 0.1967 | Train Acc 1.0000 | Valid Loss 0.3352 | Valid Acc 0.9028
    Iter [8340/11250] | Train Loss 0.4561 | Train Acc 0.7500 | Valid Loss 0.3313 | Valid Acc 0.9020
    Iter [8360/11250] | Train Loss 1.1716 | Train Acc 0.7500 | Valid Loss 0.3389 | Valid Acc 0.9034
    Iter [8380/11250] | Train Loss 0.3326 | Train Acc 0.7500 | Valid Loss 0.3506 | Valid Acc 0.8968
    Iter [8400/11250] | Train Loss 0.4574 | Train Acc 0.7500 | Valid Loss 0.3632 | Valid Acc 0.8882
    Iter [8420/11250] | Train Loss 0.1623 | Train Acc 1.0000 | Valid Loss 0.3556 | Valid Acc 0.8958
    Iter [8440/11250] | Train Loss 1.4745 | Train Acc 0.7500 | Valid Loss 0.3319 | Valid Acc 0.9020
    Iter [8460/11250] | Train Loss 0.4454 | Train Acc 0.7500 | Valid Loss 0.3236 | Valid Acc 0.9056
    Iter [8480/11250] | Train Loss 0.4189 | Train Acc 0.7500 | Valid Loss 0.3034 | Valid Acc 0.9138
    Iter [8500/11250] | Train Loss 0.1194 | Train Acc 1.0000 | Valid Loss 0.3024 | Valid Acc 0.9134
    Iter [8520/11250] | Train Loss 0.2984 | Train Acc 0.7500 | Valid Loss 0.3434 | Valid Acc 0.9024
    Iter [8540/11250] | Train Loss 0.3714 | Train Acc 0.7500 | Valid Loss 0.3028 | Valid Acc 0.9132
    Iter [8560/11250] | Train Loss 0.1524 | Train Acc 1.0000 | Valid Loss 0.3070 | Valid Acc 0.9118
    Iter [8580/11250] | Train Loss 0.9981 | Train Acc 0.5000 | Valid Loss 0.3040 | Valid Acc 0.9172
    Iter [8600/11250] | Train Loss 0.3272 | Train Acc 0.7500 | Valid Loss 0.3325 | Valid Acc 0.9044
    Iter [8620/11250] | Train Loss 0.0489 | Train Acc 1.0000 | Valid Loss 0.3151 | Valid Acc 0.9092
    Iter [8640/11250] | Train Loss 0.2726 | Train Acc 1.0000 | Valid Loss 0.3225 | Valid Acc 0.9104
    Iter [8660/11250] | Train Loss 0.3023 | Train Acc 1.0000 | Valid Loss 0.3492 | Valid Acc 0.9028
    Iter [8680/11250] | Train Loss 0.0054 | Train Acc 1.0000 | Valid Loss 0.3200 | Valid Acc 0.9058
    Iter [8700/11250] | Train Loss 0.0314 | Train Acc 1.0000 | Valid Loss 0.3238 | Valid Acc 0.9090
    Iter [8720/11250] | Train Loss 0.0137 | Train Acc 1.0000 | Valid Loss 0.3586 | Valid Acc 0.8954
    Iter [8740/11250] | Train Loss 0.8341 | Train Acc 0.7500 | Valid Loss 0.3320 | Valid Acc 0.9074
    Iter [8760/11250] | Train Loss 1.3986 | Train Acc 0.7500 | Valid Loss 0.3806 | Valid Acc 0.8904
    Iter [8780/11250] | Train Loss 0.0176 | Train Acc 1.0000 | Valid Loss 0.3780 | Valid Acc 0.8926
    Iter [8800/11250] | Train Loss 0.4678 | Train Acc 0.7500 | Valid Loss 0.3585 | Valid Acc 0.8926
    Iter [8820/11250] | Train Loss 1.5112 | Train Acc 0.7500 | Valid Loss 0.3340 | Valid Acc 0.9022
    Iter [8840/11250] | Train Loss 0.1262 | Train Acc 1.0000 | Valid Loss 0.3391 | Valid Acc 0.8976
    Iter [8860/11250] | Train Loss 0.8398 | Train Acc 0.5000 | Valid Loss 0.3342 | Valid Acc 0.8994
    Iter [8880/11250] | Train Loss 0.1165 | Train Acc 1.0000 | Valid Loss 0.3941 | Valid Acc 0.8792
    Iter [8900/11250] | Train Loss 1.1242 | Train Acc 0.5000 | Valid Loss 0.3337 | Valid Acc 0.9050
    Iter [8920/11250] | Train Loss 0.3960 | Train Acc 1.0000 | Valid Loss 0.3286 | Valid Acc 0.9058
    Iter [8940/11250] | Train Loss 1.4097 | Train Acc 0.7500 | Valid Loss 0.3250 | Valid Acc 0.9092
    Iter [8960/11250] | Train Loss 0.0113 | Train Acc 1.0000 | Valid Loss 0.3422 | Valid Acc 0.9024
    Iter [8980/11250] | Train Loss 0.0305 | Train Acc 1.0000 | Valid Loss 0.3389 | Valid Acc 0.9020
    Iter [9000/11250] | Train Loss 0.0089 | Train Acc 1.0000 | Valid Loss 0.3453 | Valid Acc 0.8980
    Iter [9020/11250] | Train Loss 0.0351 | Train Acc 1.0000 | Valid Loss 0.3736 | Valid Acc 0.8922
    Iter [9040/11250] | Train Loss 0.9622 | Train Acc 0.7500 | Valid Loss 0.3495 | Valid Acc 0.8996
    Iter [9060/11250] | Train Loss 0.1556 | Train Acc 1.0000 | Valid Loss 0.3284 | Valid Acc 0.9006
    Iter [9080/11250] | Train Loss 0.6400 | Train Acc 0.7500 | Valid Loss 0.3078 | Valid Acc 0.9116
    Iter [9100/11250] | Train Loss 0.0945 | Train Acc 1.0000 | Valid Loss 0.3436 | Valid Acc 0.9010
    Iter [9120/11250] | Train Loss 0.9943 | Train Acc 0.5000 | Valid Loss 0.3506 | Valid Acc 0.9000
    Iter [9140/11250] | Train Loss 0.0094 | Train Acc 1.0000 | Valid Loss 0.3149 | Valid Acc 0.9098
    Iter [9160/11250] | Train Loss 0.5925 | Train Acc 0.7500 | Valid Loss 0.3215 | Valid Acc 0.9084
    Iter [9180/11250] | Train Loss 0.2413 | Train Acc 1.0000 | Valid Loss 0.3343 | Valid Acc 0.9014
    Iter [9200/11250] | Train Loss 0.1377 | Train Acc 1.0000 | Valid Loss 0.3281 | Valid Acc 0.9068
    Iter [9220/11250] | Train Loss 0.0175 | Train Acc 1.0000 | Valid Loss 0.3524 | Valid Acc 0.9054
    Iter [9240/11250] | Train Loss 0.1307 | Train Acc 1.0000 | Valid Loss 0.3524 | Valid Acc 0.8982
    Iter [9260/11250] | Train Loss 0.8910 | Train Acc 0.7500 | Valid Loss 0.3795 | Valid Acc 0.8778
    Iter [9280/11250] | Train Loss 0.4891 | Train Acc 0.7500 | Valid Loss 0.3202 | Valid Acc 0.9064
    Iter [9300/11250] | Train Loss 0.4120 | Train Acc 0.7500 | Valid Loss 0.3321 | Valid Acc 0.9052
    Iter [9320/11250] | Train Loss 0.1855 | Train Acc 1.0000 | Valid Loss 0.3484 | Valid Acc 0.8990
    Iter [9340/11250] | Train Loss 1.4398 | Train Acc 0.5000 | Valid Loss 0.3608 | Valid Acc 0.8982
    Iter [9360/11250] | Train Loss 0.3903 | Train Acc 0.7500 | Valid Loss 0.3454 | Valid Acc 0.8960
    Iter [9380/11250] | Train Loss 0.8742 | Train Acc 0.7500 | Valid Loss 0.3217 | Valid Acc 0.9056
    Iter [9400/11250] | Train Loss 0.0121 | Train Acc 1.0000 | Valid Loss 0.3080 | Valid Acc 0.9146
    Iter [9420/11250] | Train Loss 0.0266 | Train Acc 1.0000 | Valid Loss 0.3235 | Valid Acc 0.9082
    Iter [9440/11250] | Train Loss 1.1230 | Train Acc 0.7500 | Valid Loss 0.3307 | Valid Acc 0.9048
    Iter [9460/11250] | Train Loss 0.1644 | Train Acc 1.0000 | Valid Loss 0.3958 | Valid Acc 0.8794
    Iter [9480/11250] | Train Loss 0.5149 | Train Acc 0.7500 | Valid Loss 0.3287 | Valid Acc 0.9042
    Iter [9500/11250] | Train Loss 0.1123 | Train Acc 1.0000 | Valid Loss 0.3590 | Valid Acc 0.8966
    Iter [9520/11250] | Train Loss 0.0687 | Train Acc 1.0000 | Valid Loss 0.3315 | Valid Acc 0.9052
    Iter [9540/11250] | Train Loss 0.2629 | Train Acc 1.0000 | Valid Loss 0.3232 | Valid Acc 0.9066
    Iter [9560/11250] | Train Loss 0.1146 | Train Acc 1.0000 | Valid Loss 0.3248 | Valid Acc 0.9024
    Iter [9580/11250] | Train Loss 0.1818 | Train Acc 1.0000 | Valid Loss 0.3352 | Valid Acc 0.9012
    Iter [9600/11250] | Train Loss 0.0497 | Train Acc 1.0000 | Valid Loss 0.3305 | Valid Acc 0.9082
    Iter [9620/11250] | Train Loss 1.5105 | Train Acc 0.5000 | Valid Loss 0.3502 | Valid Acc 0.9052
    Iter [9640/11250] | Train Loss 0.9168 | Train Acc 0.7500 | Valid Loss 0.3338 | Valid Acc 0.9106
    Iter [9660/11250] | Train Loss 0.4375 | Train Acc 0.7500 | Valid Loss 0.3283 | Valid Acc 0.9052
    Iter [9680/11250] | Train Loss 0.0565 | Train Acc 1.0000 | Valid Loss 0.3360 | Valid Acc 0.9054
    Iter [9700/11250] | Train Loss 0.8458 | Train Acc 0.7500 | Valid Loss 0.3211 | Valid Acc 0.9112
    Iter [9720/11250] | Train Loss 0.5441 | Train Acc 0.7500 | Valid Loss 0.3262 | Valid Acc 0.9024
    Iter [9740/11250] | Train Loss 0.9463 | Train Acc 0.5000 | Valid Loss 0.3397 | Valid Acc 0.9002
    Iter [9760/11250] | Train Loss 0.2703 | Train Acc 0.7500 | Valid Loss 0.3183 | Valid Acc 0.9094
    Iter [9780/11250] | Train Loss 0.5664 | Train Acc 0.7500 | Valid Loss 0.3194 | Valid Acc 0.9086
    Iter [9800/11250] | Train Loss 0.0222 | Train Acc 1.0000 | Valid Loss 0.3210 | Valid Acc 0.9112
    Iter [9820/11250] | Train Loss 0.0303 | Train Acc 1.0000 | Valid Loss 0.3042 | Valid Acc 0.9174
    Iter [9840/11250] | Train Loss 0.0531 | Train Acc 1.0000 | Valid Loss 0.3219 | Valid Acc 0.9098
    Iter [9860/11250] | Train Loss 0.8629 | Train Acc 0.7500 | Valid Loss 0.3269 | Valid Acc 0.9014
    Iter [9880/11250] | Train Loss 1.9320 | Train Acc 0.5000 | Valid Loss 0.3184 | Valid Acc 0.9144
    Iter [9900/11250] | Train Loss 0.5911 | Train Acc 0.7500 | Valid Loss 0.2852 | Valid Acc 0.9214
    Iter [9920/11250] | Train Loss 1.6397 | Train Acc 0.7500 | Valid Loss 0.3004 | Valid Acc 0.9204
    Iter [9940/11250] | Train Loss 0.7595 | Train Acc 0.7500 | Valid Loss 0.3252 | Valid Acc 0.9114
    Iter [9960/11250] | Train Loss 1.2463 | Train Acc 0.7500 | Valid Loss 0.3235 | Valid Acc 0.9054
    Iter [9980/11250] | Train Loss 0.9159 | Train Acc 0.7500 | Valid Loss 0.3362 | Valid Acc 0.9116
    Iter [10000/11250] | Train Loss 0.1102 | Train Acc 1.0000 | Valid Loss 0.3437 | Valid Acc 0.9084
    Iter [10020/11250] | Train Loss 0.3090 | Train Acc 1.0000 | Valid Loss 0.3270 | Valid Acc 0.9080
    Iter [10040/11250] | Train Loss 0.3017 | Train Acc 0.7500 | Valid Loss 0.3625 | Valid Acc 0.8914
    Iter [10060/11250] | Train Loss 0.0335 | Train Acc 1.0000 | Valid Loss 0.3278 | Valid Acc 0.9024
    Iter [10080/11250] | Train Loss 0.1881 | Train Acc 1.0000 | Valid Loss 0.3234 | Valid Acc 0.9040
    Iter [10100/11250] | Train Loss 0.1951 | Train Acc 1.0000 | Valid Loss 0.3080 | Valid Acc 0.9080
    Iter [10120/11250] | Train Loss 0.0501 | Train Acc 1.0000 | Valid Loss 0.3209 | Valid Acc 0.9054
    Iter [10140/11250] | Train Loss 0.4436 | Train Acc 0.7500 | Valid Loss 0.3374 | Valid Acc 0.9020
    Iter [10160/11250] | Train Loss 0.2201 | Train Acc 1.0000 | Valid Loss 0.3540 | Valid Acc 0.9008
    Iter [10180/11250] | Train Loss 0.2043 | Train Acc 1.0000 | Valid Loss 0.3126 | Valid Acc 0.9100
    Iter [10200/11250] | Train Loss 0.3313 | Train Acc 0.7500 | Valid Loss 0.3256 | Valid Acc 0.9042
    Iter [10220/11250] | Train Loss 0.1235 | Train Acc 1.0000 | Valid Loss 0.3420 | Valid Acc 0.9010
    Iter [10240/11250] | Train Loss 0.1405 | Train Acc 1.0000 | Valid Loss 0.3407 | Valid Acc 0.9114
    Iter [10260/11250] | Train Loss 0.1501 | Train Acc 1.0000 | Valid Loss 0.3254 | Valid Acc 0.9168
    Iter [10280/11250] | Train Loss 0.3248 | Train Acc 0.7500 | Valid Loss 0.3383 | Valid Acc 0.9106
    Iter [10300/11250] | Train Loss 1.4255 | Train Acc 0.7500 | Valid Loss 0.3165 | Valid Acc 0.9174
    Iter [10320/11250] | Train Loss 1.5911 | Train Acc 0.2500 | Valid Loss 0.3259 | Valid Acc 0.9074
    Iter [10340/11250] | Train Loss 1.0925 | Train Acc 0.7500 | Valid Loss 0.3570 | Valid Acc 0.9006
    Iter [10360/11250] | Train Loss 0.7225 | Train Acc 0.5000 | Valid Loss 0.3187 | Valid Acc 0.9104
    Iter [10380/11250] | Train Loss 1.1041 | Train Acc 0.5000 | Valid Loss 0.3754 | Valid Acc 0.8872
    Iter [10400/11250] | Train Loss 0.6192 | Train Acc 0.7500 | Valid Loss 0.3631 | Valid Acc 0.8858
    Iter [10420/11250] | Train Loss 2.0572 | Train Acc 0.5000 | Valid Loss 0.3252 | Valid Acc 0.9042
    Iter [10440/11250] | Train Loss 0.0142 | Train Acc 1.0000 | Valid Loss 0.3497 | Valid Acc 0.9030
    Iter [10460/11250] | Train Loss 0.1779 | Train Acc 1.0000 | Valid Loss 0.3132 | Valid Acc 0.9104
    Iter [10480/11250] | Train Loss 0.0285 | Train Acc 1.0000 | Valid Loss 0.2833 | Valid Acc 0.9186
    Iter [10500/11250] | Train Loss 0.2248 | Train Acc 1.0000 | Valid Loss 0.2849 | Valid Acc 0.9186
    Iter [10520/11250] | Train Loss 0.0555 | Train Acc 1.0000 | Valid Loss 0.3298 | Valid Acc 0.9008
    Iter [10540/11250] | Train Loss 0.8606 | Train Acc 0.7500 | Valid Loss 0.3018 | Valid Acc 0.9146
    Iter [10560/11250] | Train Loss 0.0395 | Train Acc 1.0000 | Valid Loss 0.2974 | Valid Acc 0.9174
    Iter [10580/11250] | Train Loss 0.1022 | Train Acc 1.0000 | Valid Loss 0.3381 | Valid Acc 0.9010
    Iter [10600/11250] | Train Loss 0.7638 | Train Acc 0.7500 | Valid Loss 0.3135 | Valid Acc 0.9092
    Iter [10620/11250] | Train Loss 0.0542 | Train Acc 1.0000 | Valid Loss 0.2871 | Valid Acc 0.9186
    Iter [10640/11250] | Train Loss 0.0257 | Train Acc 1.0000 | Valid Loss 0.2959 | Valid Acc 0.9166
    Iter [10660/11250] | Train Loss 0.2383 | Train Acc 1.0000 | Valid Loss 0.3078 | Valid Acc 0.9150
    Iter [10680/11250] | Train Loss 0.1976 | Train Acc 1.0000 | Valid Loss 0.3072 | Valid Acc 0.9122
    Iter [10700/11250] | Train Loss 0.3851 | Train Acc 1.0000 | Valid Loss 0.3184 | Valid Acc 0.9104
    Iter [10720/11250] | Train Loss 2.0561 | Train Acc 0.7500 | Valid Loss 0.3361 | Valid Acc 0.8986
    Iter [10740/11250] | Train Loss 0.5665 | Train Acc 0.7500 | Valid Loss 0.3235 | Valid Acc 0.9050
    Iter [10760/11250] | Train Loss 0.3703 | Train Acc 0.7500 | Valid Loss 0.2983 | Valid Acc 0.9196
    Iter [10780/11250] | Train Loss 0.4198 | Train Acc 0.7500 | Valid Loss 0.3027 | Valid Acc 0.9170
    Iter [10800/11250] | Train Loss 0.0552 | Train Acc 1.0000 | Valid Loss 0.3300 | Valid Acc 0.9068
    Iter [10820/11250] | Train Loss 0.0749 | Train Acc 1.0000 | Valid Loss 0.3444 | Valid Acc 0.9024
    Iter [10840/11250] | Train Loss 0.0741 | Train Acc 1.0000 | Valid Loss 0.3238 | Valid Acc 0.9096
    Iter [10860/11250] | Train Loss 0.2522 | Train Acc 0.7500 | Valid Loss 0.3795 | Valid Acc 0.8978
    Iter [10880/11250] | Train Loss 0.7439 | Train Acc 0.7500 | Valid Loss 0.3494 | Valid Acc 0.8990
    Iter [10900/11250] | Train Loss 0.5068 | Train Acc 0.7500 | Valid Loss 0.3079 | Valid Acc 0.9150
    Iter [10920/11250] | Train Loss 0.4184 | Train Acc 1.0000 | Valid Loss 0.3172 | Valid Acc 0.9106
    Iter [10940/11250] | Train Loss 0.4333 | Train Acc 0.7500 | Valid Loss 0.3129 | Valid Acc 0.9176
    Iter [10960/11250] | Train Loss 0.2175 | Train Acc 1.0000 | Valid Loss 0.3285 | Valid Acc 0.9062
    Iter [10980/11250] | Train Loss 0.0888 | Train Acc 1.0000 | Valid Loss 0.3728 | Valid Acc 0.8998
    Iter [11000/11250] | Train Loss 0.9651 | Train Acc 0.7500 | Valid Loss 0.3368 | Valid Acc 0.9094
    Iter [11020/11250] | Train Loss 0.7993 | Train Acc 0.7500 | Valid Loss 0.3115 | Valid Acc 0.9078
    Iter [11040/11250] | Train Loss 0.0156 | Train Acc 1.0000 | Valid Loss 0.3138 | Valid Acc 0.9092
    Iter [11060/11250] | Train Loss 0.0100 | Train Acc 1.0000 | Valid Loss 0.3110 | Valid Acc 0.9090
    Iter [11080/11250] | Train Loss 0.9557 | Train Acc 0.7500 | Valid Loss 0.3096 | Valid Acc 0.9112
    Iter [11100/11250] | Train Loss 0.0857 | Train Acc 1.0000 | Valid Loss 0.3317 | Valid Acc 0.9100
    Iter [11120/11250] | Train Loss 0.8044 | Train Acc 0.7500 | Valid Loss 0.2834 | Valid Acc 0.9260
    Iter [11140/11250] | Train Loss 0.2208 | Train Acc 1.0000 | Valid Loss 0.3012 | Valid Acc 0.9146
    Iter [11160/11250] | Train Loss 1.5477 | Train Acc 0.7500 | Valid Loss 0.3119 | Valid Acc 0.9104
    Iter [11180/11250] | Train Loss 0.8139 | Train Acc 0.7500 | Valid Loss 0.3191 | Valid Acc 0.9096
    Iter [11200/11250] | Train Loss 0.1602 | Train Acc 1.0000 | Valid Loss 0.3478 | Valid Acc 0.8958
    Iter [11220/11250] | Train Loss 0.0802 | Train Acc 1.0000 | Valid Loss 0.3382 | Valid Acc 0.9036
    Iter [11240/11250] | Train Loss 0.0258 | Train Acc 1.0000 | Valid Loss 0.3252 | Valid Acc 0.9094
    Iter [11249/11250] | Train Loss 0.1314 | Train Acc 1.0000 | Valid Loss 0.3246 | Valid Acc 0.9040


## Fine-tuning resnet model

resnet34 pre-trained model 가져와 마지막 layer 제외하고 parameter freeze

### model 선언


```python
from torchvision.models import resnet34

model_finetune = resnet34(pretrained=True)
num_classes = 10
num_ftrs = model_finetune.fc.in_features
model_finetune.fc = torch.nn.Linear(num_ftrs, num_classes)


for param in model_finetune.parameters():
    param.requires_grad = False

model_finetune.fc.weight.requires_grad = True

model_finetune.cuda()
```

    Downloading: "https://download.pytorch.org/models/resnet34-b627a593.pth" to /opt/ml/.cache/torch/hub/checkpoints/resnet34-b627a593.pth



    HBox(children=(HTML(value=''), FloatProgress(value=0.0, max=87319819.0), HTML(value='')))


    





    ResNet(
      (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      (layer1): Sequential(
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
      (fc): Linear(in_features=512, out_features=10, bias=True)
    )



### loss,criterion


```python
# Loss function and Optimizer
from torch.optim import Adam

criterion = nn.CrossEntropyLoss()
optimizer_ft = Adam(model_finetune.parameters(), lr=1e-4)
```

### Train


```python
# Main
os.makedirs(log_dir, exist_ok=True)

with open(os.path.join(log_dir, 'fine_tuned_train_log.csv'), 'w') as log:
  # Training
  model_finetune.train()
  for iter, (img, label) in enumerate(qd_train_dataloader):

    # 학습에 사용하기 위한 image, label 처리 (필요한 경우, data type도 변경해주세요)
    img, label = img.float().cuda(), label.long().cuda()

    # implementing zero_grad ~ step
    optimizer_ft.zero_grad()

    # 모델에 이미지 forward
    pred_logit = model_finetune(img)

    # loss 값 계산
    loss = criterion(pred_logit, label)

    # Backpropagation
    loss.backward()
    optimizer_ft.step()

    # Accuracy 계산
    pred_label = torch.argmax(pred_logit, 1)
    acc = (pred_label == label).sum().item() / len(img)

    train_loss = loss.item()
    train_acc = acc

    # Validation
    if (iter % 20 == 0) or (iter == len(qd_train_dataloader)-1):
      model_finetune.eval()
      valid_loss, valid_acc = AverageMeter(), AverageMeter()

      for img, label in qd_val_dataloader:
        # Validation에 사용하기 위한 image, label 처리 (필요한 경우, data type도 변경해주세요)
        img, label = img.float().cuda(), label.long().cuda()

        # 모델에 이미지 forward (gradient 계산 X)
        with torch.no_grad():
          pred_logit = model_finetune(img)

        # loss 값 계산
        loss = criterion(pred_logit, label)

        # Accuracy 계산
        pred_label = torch.argmax(pred_logit, 1)
        acc = (pred_label == label).sum().item() / len(img)

        valid_loss.update(loss.item(), len(img))
        valid_acc.update(acc, len(img))

      valid_loss = valid_loss.avg
      valid_acc = valid_acc.avg

      print("Iter [%3d/%3d] | Train Loss %.4f | Train Acc %.4f | Valid Loss %.4f | Valid Acc %.4f" %
            (iter, len(qd_train_dataloader), train_loss, train_acc, valid_loss, valid_acc))
      
      # Train Log Writing
      log.write('%d,%.4f,%.4f,%.4f,%.4f\n'%(iter, train_loss, train_acc, valid_loss, valid_acc))
```

    Iter [  0/11250] | Train Loss 2.0591 | Train Acc 0.0000 | Valid Loss 2.4477 | Valid Acc 0.0988
    Iter [ 20/11250] | Train Loss 2.2279 | Train Acc 0.0000 | Valid Loss 2.3648 | Valid Acc 0.0888
    Iter [ 40/11250] | Train Loss 2.5071 | Train Acc 0.0000 | Valid Loss 2.3175 | Valid Acc 0.1024
    Iter [ 60/11250] | Train Loss 2.4160 | Train Acc 0.0000 | Valid Loss 2.2830 | Valid Acc 0.1206
    Iter [ 80/11250] | Train Loss 2.2929 | Train Acc 0.0000 | Valid Loss 2.2525 | Valid Acc 0.1510
    Iter [100/11250] | Train Loss 2.2165 | Train Acc 0.0000 | Valid Loss 2.2219 | Valid Acc 0.1960
    Iter [120/11250] | Train Loss 2.3550 | Train Acc 0.2500 | Valid Loss 2.1967 | Valid Acc 0.2208
    Iter [140/11250] | Train Loss 2.3591 | Train Acc 0.0000 | Valid Loss 2.1699 | Valid Acc 0.2268
    Iter [160/11250] | Train Loss 1.9484 | Train Acc 0.5000 | Valid Loss 2.1374 | Valid Acc 0.2466
    Iter [180/11250] | Train Loss 1.9446 | Train Acc 0.5000 | Valid Loss 2.1069 | Valid Acc 0.2780
    Iter [200/11250] | Train Loss 1.9569 | Train Acc 0.5000 | Valid Loss 2.0752 | Valid Acc 0.2994
    Iter [220/11250] | Train Loss 1.6027 | Train Acc 0.7500 | Valid Loss 2.0503 | Valid Acc 0.3094
    Iter [240/11250] | Train Loss 1.8786 | Train Acc 0.5000 | Valid Loss 2.0309 | Valid Acc 0.3134
    Iter [260/11250] | Train Loss 2.2872 | Train Acc 0.2500 | Valid Loss 2.0072 | Valid Acc 0.3490
    Iter [280/11250] | Train Loss 1.9658 | Train Acc 0.5000 | Valid Loss 1.9751 | Valid Acc 0.3602
    Iter [300/11250] | Train Loss 1.7093 | Train Acc 1.0000 | Valid Loss 1.9522 | Valid Acc 0.3636
    Iter [320/11250] | Train Loss 1.7833 | Train Acc 0.5000 | Valid Loss 1.9238 | Valid Acc 0.3984
    Iter [340/11250] | Train Loss 1.9999 | Train Acc 0.2500 | Valid Loss 1.8990 | Valid Acc 0.4488
    Iter [360/11250] | Train Loss 1.8246 | Train Acc 0.5000 | Valid Loss 1.8774 | Valid Acc 0.4720
    Iter [380/11250] | Train Loss 1.9225 | Train Acc 0.2500 | Valid Loss 1.8562 | Valid Acc 0.4938
    Iter [400/11250] | Train Loss 1.9509 | Train Acc 0.2500 | Valid Loss 1.8333 | Valid Acc 0.5176
    Iter [420/11250] | Train Loss 1.7058 | Train Acc 0.7500 | Valid Loss 1.8118 | Valid Acc 0.4748
    Iter [440/11250] | Train Loss 1.8134 | Train Acc 0.5000 | Valid Loss 1.7865 | Valid Acc 0.5046
    Iter [460/11250] | Train Loss 1.9296 | Train Acc 0.2500 | Valid Loss 1.7714 | Valid Acc 0.4918
    Iter [480/11250] | Train Loss 1.6216 | Train Acc 0.5000 | Valid Loss 1.7469 | Valid Acc 0.4944
    Iter [500/11250] | Train Loss 1.7342 | Train Acc 0.5000 | Valid Loss 1.7200 | Valid Acc 0.5626
    Iter [520/11250] | Train Loss 1.7983 | Train Acc 0.5000 | Valid Loss 1.7056 | Valid Acc 0.5654
    Iter [540/11250] | Train Loss 1.4740 | Train Acc 0.5000 | Valid Loss 1.6913 | Valid Acc 0.5544
    Iter [560/11250] | Train Loss 1.5064 | Train Acc 0.5000 | Valid Loss 1.6674 | Valid Acc 0.5850
    Iter [580/11250] | Train Loss 1.9752 | Train Acc 0.2500 | Valid Loss 1.6500 | Valid Acc 0.6098
    Iter [600/11250] | Train Loss 1.6236 | Train Acc 1.0000 | Valid Loss 1.6301 | Valid Acc 0.6414
    Iter [620/11250] | Train Loss 1.4108 | Train Acc 0.7500 | Valid Loss 1.6143 | Valid Acc 0.6758
    Iter [640/11250] | Train Loss 1.6802 | Train Acc 0.5000 | Valid Loss 1.5968 | Valid Acc 0.6722
    Iter [660/11250] | Train Loss 1.2353 | Train Acc 0.7500 | Valid Loss 1.5781 | Valid Acc 0.6370
    Iter [680/11250] | Train Loss 1.3141 | Train Acc 0.7500 | Valid Loss 1.5736 | Valid Acc 0.5790
    Iter [700/11250] | Train Loss 1.4469 | Train Acc 0.7500 | Valid Loss 1.5568 | Valid Acc 0.5936
    Iter [720/11250] | Train Loss 1.5748 | Train Acc 1.0000 | Valid Loss 1.5343 | Valid Acc 0.6442
    Iter [740/11250] | Train Loss 1.4584 | Train Acc 1.0000 | Valid Loss 1.5177 | Valid Acc 0.6996
    Iter [760/11250] | Train Loss 1.4361 | Train Acc 0.7500 | Valid Loss 1.5033 | Valid Acc 0.7252
    Iter [780/11250] | Train Loss 1.3676 | Train Acc 1.0000 | Valid Loss 1.4847 | Valid Acc 0.7240
    Iter [800/11250] | Train Loss 1.4051 | Train Acc 0.7500 | Valid Loss 1.4701 | Valid Acc 0.7272
    Iter [820/11250] | Train Loss 1.4336 | Train Acc 0.7500 | Valid Loss 1.4549 | Valid Acc 0.7364
    Iter [840/11250] | Train Loss 1.1964 | Train Acc 1.0000 | Valid Loss 1.4410 | Valid Acc 0.7186
    Iter [860/11250] | Train Loss 1.2944 | Train Acc 0.5000 | Valid Loss 1.4304 | Valid Acc 0.6860
    Iter [880/11250] | Train Loss 1.5738 | Train Acc 0.5000 | Valid Loss 1.4208 | Valid Acc 0.6540
    Iter [900/11250] | Train Loss 1.3473 | Train Acc 0.7500 | Valid Loss 1.4043 | Valid Acc 0.7132
    Iter [920/11250] | Train Loss 1.2739 | Train Acc 1.0000 | Valid Loss 1.3921 | Valid Acc 0.7288
    Iter [940/11250] | Train Loss 1.7524 | Train Acc 0.5000 | Valid Loss 1.3861 | Valid Acc 0.7362
    Iter [960/11250] | Train Loss 1.3646 | Train Acc 0.7500 | Valid Loss 1.3695 | Valid Acc 0.7618
    Iter [980/11250] | Train Loss 1.3402 | Train Acc 0.7500 | Valid Loss 1.3618 | Valid Acc 0.7480
    Iter [1000/11250] | Train Loss 1.3554 | Train Acc 1.0000 | Valid Loss 1.3401 | Valid Acc 0.7676
    Iter [1020/11250] | Train Loss 1.2602 | Train Acc 0.7500 | Valid Loss 1.3317 | Valid Acc 0.7740
    Iter [1040/11250] | Train Loss 2.0795 | Train Acc 0.2500 | Valid Loss 1.3225 | Valid Acc 0.7694
    Iter [1060/11250] | Train Loss 1.7672 | Train Acc 0.7500 | Valid Loss 1.3091 | Valid Acc 0.7794
    Iter [1080/11250] | Train Loss 1.3725 | Train Acc 0.7500 | Valid Loss 1.3067 | Valid Acc 0.7618
    Iter [1100/11250] | Train Loss 1.0507 | Train Acc 0.7500 | Valid Loss 1.2988 | Valid Acc 0.7522
    Iter [1120/11250] | Train Loss 1.1073 | Train Acc 0.7500 | Valid Loss 1.2829 | Valid Acc 0.7700
    Iter [1140/11250] | Train Loss 1.3328 | Train Acc 0.7500 | Valid Loss 1.2684 | Valid Acc 0.7692
    Iter [1160/11250] | Train Loss 1.2895 | Train Acc 0.7500 | Valid Loss 1.2605 | Valid Acc 0.7778
    Iter [1180/11250] | Train Loss 0.9139 | Train Acc 1.0000 | Valid Loss 1.2527 | Valid Acc 0.7790
    Iter [1200/11250] | Train Loss 1.1328 | Train Acc 0.7500 | Valid Loss 1.2417 | Valid Acc 0.7788
    Iter [1220/11250] | Train Loss 1.4650 | Train Acc 0.5000 | Valid Loss 1.2299 | Valid Acc 0.7768
    Iter [1240/11250] | Train Loss 1.5060 | Train Acc 0.5000 | Valid Loss 1.2239 | Valid Acc 0.7676
    Iter [1260/11250] | Train Loss 1.5119 | Train Acc 0.5000 | Valid Loss 1.2140 | Valid Acc 0.7772
    Iter [1280/11250] | Train Loss 1.4875 | Train Acc 0.5000 | Valid Loss 1.1991 | Valid Acc 0.7846
    Iter [1300/11250] | Train Loss 1.4897 | Train Acc 0.5000 | Valid Loss 1.1915 | Valid Acc 0.7904
    Iter [1320/11250] | Train Loss 1.1193 | Train Acc 0.7500 | Valid Loss 1.1914 | Valid Acc 0.7766
    Iter [1340/11250] | Train Loss 1.0093 | Train Acc 1.0000 | Valid Loss 1.1780 | Valid Acc 0.7712
    Iter [1360/11250] | Train Loss 1.2071 | Train Acc 0.5000 | Valid Loss 1.1722 | Valid Acc 0.7450
    Iter [1380/11250] | Train Loss 1.2593 | Train Acc 1.0000 | Valid Loss 1.1680 | Valid Acc 0.7350
    Iter [1400/11250] | Train Loss 1.2438 | Train Acc 0.7500 | Valid Loss 1.1516 | Valid Acc 0.7770
    Iter [1420/11250] | Train Loss 1.0130 | Train Acc 0.7500 | Valid Loss 1.1430 | Valid Acc 0.7932
    Iter [1440/11250] | Train Loss 1.4470 | Train Acc 0.7500 | Valid Loss 1.1345 | Valid Acc 0.7860
    Iter [1460/11250] | Train Loss 1.8587 | Train Acc 0.7500 | Valid Loss 1.1300 | Valid Acc 0.7782
    Iter [1480/11250] | Train Loss 1.1333 | Train Acc 0.7500 | Valid Loss 1.1205 | Valid Acc 0.7834
    Iter [1500/11250] | Train Loss 1.0270 | Train Acc 0.7500 | Valid Loss 1.1107 | Valid Acc 0.7832
    Iter [1520/11250] | Train Loss 0.6471 | Train Acc 1.0000 | Valid Loss 1.1023 | Valid Acc 0.7866
    Iter [1540/11250] | Train Loss 0.9966 | Train Acc 0.7500 | Valid Loss 1.0980 | Valid Acc 0.7780
    Iter [1560/11250] | Train Loss 0.7269 | Train Acc 1.0000 | Valid Loss 1.0930 | Valid Acc 0.7704
    Iter [1580/11250] | Train Loss 1.3074 | Train Acc 0.7500 | Valid Loss 1.0831 | Valid Acc 0.7820
    Iter [1600/11250] | Train Loss 1.5609 | Train Acc 0.7500 | Valid Loss 1.0735 | Valid Acc 0.8018
    Iter [1620/11250] | Train Loss 0.9092 | Train Acc 1.0000 | Valid Loss 1.0711 | Valid Acc 0.7996
    Iter [1640/11250] | Train Loss 1.0940 | Train Acc 0.7500 | Valid Loss 1.0696 | Valid Acc 0.7882
    Iter [1660/11250] | Train Loss 1.3981 | Train Acc 0.5000 | Valid Loss 1.0677 | Valid Acc 0.7774
    Iter [1680/11250] | Train Loss 1.5554 | Train Acc 0.5000 | Valid Loss 1.0552 | Valid Acc 0.8028
    Iter [1700/11250] | Train Loss 0.6198 | Train Acc 1.0000 | Valid Loss 1.0435 | Valid Acc 0.8172
    Iter [1720/11250] | Train Loss 0.8106 | Train Acc 1.0000 | Valid Loss 1.0368 | Valid Acc 0.8070
    Iter [1740/11250] | Train Loss 0.9881 | Train Acc 0.7500 | Valid Loss 1.0336 | Valid Acc 0.8122
    Iter [1760/11250] | Train Loss 1.1102 | Train Acc 1.0000 | Valid Loss 1.0271 | Valid Acc 0.8216
    Iter [1780/11250] | Train Loss 0.8113 | Train Acc 1.0000 | Valid Loss 1.0274 | Valid Acc 0.8160
    Iter [1800/11250] | Train Loss 0.5481 | Train Acc 1.0000 | Valid Loss 1.0181 | Valid Acc 0.8178
    Iter [1820/11250] | Train Loss 1.3690 | Train Acc 0.5000 | Valid Loss 1.0160 | Valid Acc 0.8060
    Iter [1840/11250] | Train Loss 0.9353 | Train Acc 1.0000 | Valid Loss 1.0117 | Valid Acc 0.8118
    Iter [1860/11250] | Train Loss 1.3549 | Train Acc 0.7500 | Valid Loss 1.0041 | Valid Acc 0.8098
    Iter [1880/11250] | Train Loss 0.7556 | Train Acc 1.0000 | Valid Loss 0.9940 | Valid Acc 0.8142
    Iter [1900/11250] | Train Loss 0.7258 | Train Acc 0.7500 | Valid Loss 0.9865 | Valid Acc 0.8186
    Iter [1920/11250] | Train Loss 1.6029 | Train Acc 0.5000 | Valid Loss 0.9811 | Valid Acc 0.8200
    Iter [1940/11250] | Train Loss 0.5613 | Train Acc 1.0000 | Valid Loss 0.9742 | Valid Acc 0.8176
    Iter [1960/11250] | Train Loss 1.4883 | Train Acc 0.5000 | Valid Loss 0.9739 | Valid Acc 0.8070
    Iter [1980/11250] | Train Loss 1.1443 | Train Acc 0.7500 | Valid Loss 0.9687 | Valid Acc 0.8154
    Iter [2000/11250] | Train Loss 1.3162 | Train Acc 0.7500 | Valid Loss 0.9687 | Valid Acc 0.8074
    Iter [2020/11250] | Train Loss 1.3257 | Train Acc 0.5000 | Valid Loss 0.9604 | Valid Acc 0.8156
    Iter [2040/11250] | Train Loss 1.1973 | Train Acc 0.2500 | Valid Loss 0.9591 | Valid Acc 0.8184
    Iter [2060/11250] | Train Loss 1.0312 | Train Acc 0.7500 | Valid Loss 0.9514 | Valid Acc 0.8198
    Iter [2080/11250] | Train Loss 1.1499 | Train Acc 0.7500 | Valid Loss 0.9472 | Valid Acc 0.8144
    Iter [2100/11250] | Train Loss 0.7136 | Train Acc 0.7500 | Valid Loss 0.9389 | Valid Acc 0.8192
    Iter [2120/11250] | Train Loss 0.7135 | Train Acc 1.0000 | Valid Loss 0.9387 | Valid Acc 0.8140
    Iter [2140/11250] | Train Loss 0.8459 | Train Acc 0.7500 | Valid Loss 0.9318 | Valid Acc 0.8218
    Iter [2160/11250] | Train Loss 0.8350 | Train Acc 0.7500 | Valid Loss 0.9271 | Valid Acc 0.8272
    Iter [2180/11250] | Train Loss 0.9735 | Train Acc 0.7500 | Valid Loss 0.9276 | Valid Acc 0.8252
    Iter [2200/11250] | Train Loss 0.5276 | Train Acc 1.0000 | Valid Loss 0.9243 | Valid Acc 0.8194
    Iter [2220/11250] | Train Loss 0.8451 | Train Acc 1.0000 | Valid Loss 0.9171 | Valid Acc 0.8222
    Iter [2240/11250] | Train Loss 0.6817 | Train Acc 0.7500 | Valid Loss 0.9103 | Valid Acc 0.8260
    Iter [2260/11250] | Train Loss 0.8070 | Train Acc 1.0000 | Valid Loss 0.9083 | Valid Acc 0.8230
    Iter [2280/11250] | Train Loss 1.0338 | Train Acc 0.7500 | Valid Loss 0.9021 | Valid Acc 0.8262
    Iter [2300/11250] | Train Loss 0.7003 | Train Acc 0.7500 | Valid Loss 0.8961 | Valid Acc 0.8322
    Iter [2320/11250] | Train Loss 0.5455 | Train Acc 1.0000 | Valid Loss 0.8914 | Valid Acc 0.8306
    Iter [2340/11250] | Train Loss 0.9798 | Train Acc 0.7500 | Valid Loss 0.8889 | Valid Acc 0.8214
    Iter [2360/11250] | Train Loss 1.0309 | Train Acc 0.7500 | Valid Loss 0.8914 | Valid Acc 0.8106
    Iter [2380/11250] | Train Loss 0.5524 | Train Acc 1.0000 | Valid Loss 0.8888 | Valid Acc 0.8102
    Iter [2400/11250] | Train Loss 0.8944 | Train Acc 1.0000 | Valid Loss 0.8787 | Valid Acc 0.8250
    Iter [2420/11250] | Train Loss 1.0116 | Train Acc 0.7500 | Valid Loss 0.8724 | Valid Acc 0.8294
    Iter [2440/11250] | Train Loss 1.6238 | Train Acc 0.5000 | Valid Loss 0.8719 | Valid Acc 0.8210
    Iter [2460/11250] | Train Loss 0.5179 | Train Acc 1.0000 | Valid Loss 0.8677 | Valid Acc 0.8296
    Iter [2480/11250] | Train Loss 0.8827 | Train Acc 0.7500 | Valid Loss 0.8654 | Valid Acc 0.8288
    Iter [2500/11250] | Train Loss 0.7003 | Train Acc 0.7500 | Valid Loss 0.8634 | Valid Acc 0.8284
    Iter [2520/11250] | Train Loss 1.4029 | Train Acc 0.5000 | Valid Loss 0.8580 | Valid Acc 0.8350
    Iter [2540/11250] | Train Loss 0.9038 | Train Acc 0.7500 | Valid Loss 0.8548 | Valid Acc 0.8342
    Iter [2560/11250] | Train Loss 0.7290 | Train Acc 0.7500 | Valid Loss 0.8635 | Valid Acc 0.8168
    Iter [2580/11250] | Train Loss 0.6540 | Train Acc 1.0000 | Valid Loss 0.8603 | Valid Acc 0.8160
    Iter [2600/11250] | Train Loss 0.5731 | Train Acc 1.0000 | Valid Loss 0.8544 | Valid Acc 0.8196
    Iter [2620/11250] | Train Loss 0.5269 | Train Acc 1.0000 | Valid Loss 0.8441 | Valid Acc 0.8272
    Iter [2640/11250] | Train Loss 0.9680 | Train Acc 1.0000 | Valid Loss 0.8416 | Valid Acc 0.8280
    Iter [2660/11250] | Train Loss 0.5847 | Train Acc 1.0000 | Valid Loss 0.8379 | Valid Acc 0.8302
    Iter [2680/11250] | Train Loss 0.8487 | Train Acc 1.0000 | Valid Loss 0.8343 | Valid Acc 0.8364
    Iter [2700/11250] | Train Loss 0.4818 | Train Acc 1.0000 | Valid Loss 0.8312 | Valid Acc 0.8374
    Iter [2720/11250] | Train Loss 0.4330 | Train Acc 1.0000 | Valid Loss 0.8313 | Valid Acc 0.8320
    Iter [2740/11250] | Train Loss 0.5375 | Train Acc 1.0000 | Valid Loss 0.8270 | Valid Acc 0.8314
    Iter [2760/11250] | Train Loss 0.8155 | Train Acc 0.7500 | Valid Loss 0.8255 | Valid Acc 0.8358
    Iter [2780/11250] | Train Loss 0.8069 | Train Acc 0.7500 | Valid Loss 0.8169 | Valid Acc 0.8362
    Iter [2800/11250] | Train Loss 0.9438 | Train Acc 0.7500 | Valid Loss 0.8139 | Valid Acc 0.8346
    Iter [2820/11250] | Train Loss 1.4869 | Train Acc 0.7500 | Valid Loss 0.8120 | Valid Acc 0.8330
    Iter [2840/11250] | Train Loss 0.8090 | Train Acc 0.7500 | Valid Loss 0.8108 | Valid Acc 0.8280
    Iter [2860/11250] | Train Loss 0.7519 | Train Acc 0.7500 | Valid Loss 0.8050 | Valid Acc 0.8304
    Iter [2880/11250] | Train Loss 0.8143 | Train Acc 0.7500 | Valid Loss 0.8078 | Valid Acc 0.8322
    Iter [2900/11250] | Train Loss 0.5562 | Train Acc 1.0000 | Valid Loss 0.8037 | Valid Acc 0.8326
    Iter [2920/11250] | Train Loss 1.5683 | Train Acc 0.5000 | Valid Loss 0.7992 | Valid Acc 0.8346
    Iter [2940/11250] | Train Loss 0.3723 | Train Acc 1.0000 | Valid Loss 0.7970 | Valid Acc 0.8348
    Iter [2960/11250] | Train Loss 0.9393 | Train Acc 1.0000 | Valid Loss 0.7904 | Valid Acc 0.8404
    Iter [2980/11250] | Train Loss 0.8193 | Train Acc 0.7500 | Valid Loss 0.7860 | Valid Acc 0.8414
    Iter [3000/11250] | Train Loss 0.9265 | Train Acc 0.5000 | Valid Loss 0.7806 | Valid Acc 0.8418
    Iter [3020/11250] | Train Loss 1.0808 | Train Acc 0.5000 | Valid Loss 0.7802 | Valid Acc 0.8390
    Iter [3040/11250] | Train Loss 2.0606 | Train Acc 0.2500 | Valid Loss 0.7784 | Valid Acc 0.8380
    Iter [3060/11250] | Train Loss 0.7943 | Train Acc 0.7500 | Valid Loss 0.7750 | Valid Acc 0.8402
    Iter [3080/11250] | Train Loss 0.5602 | Train Acc 1.0000 | Valid Loss 0.7703 | Valid Acc 0.8416
    Iter [3100/11250] | Train Loss 0.4642 | Train Acc 1.0000 | Valid Loss 0.7704 | Valid Acc 0.8408
    Iter [3120/11250] | Train Loss 0.5712 | Train Acc 0.7500 | Valid Loss 0.7710 | Valid Acc 0.8416
    Iter [3140/11250] | Train Loss 0.7664 | Train Acc 0.7500 | Valid Loss 0.7664 | Valid Acc 0.8404
    Iter [3160/11250] | Train Loss 0.7019 | Train Acc 1.0000 | Valid Loss 0.7641 | Valid Acc 0.8384
    Iter [3180/11250] | Train Loss 0.4551 | Train Acc 1.0000 | Valid Loss 0.7654 | Valid Acc 0.8332
    Iter [3200/11250] | Train Loss 0.4359 | Train Acc 1.0000 | Valid Loss 0.7683 | Valid Acc 0.8270
    Iter [3220/11250] | Train Loss 1.3124 | Train Acc 0.7500 | Valid Loss 0.7617 | Valid Acc 0.8336
    Iter [3240/11250] | Train Loss 0.8802 | Train Acc 1.0000 | Valid Loss 0.7589 | Valid Acc 0.8388
    Iter [3260/11250] | Train Loss 0.9781 | Train Acc 0.5000 | Valid Loss 0.7573 | Valid Acc 0.8376
    Iter [3280/11250] | Train Loss 1.4524 | Train Acc 0.7500 | Valid Loss 0.7576 | Valid Acc 0.8410
    Iter [3300/11250] | Train Loss 0.6160 | Train Acc 1.0000 | Valid Loss 0.7587 | Valid Acc 0.8424
    Iter [3320/11250] | Train Loss 0.8157 | Train Acc 1.0000 | Valid Loss 0.7456 | Valid Acc 0.8494
    Iter [3340/11250] | Train Loss 0.3284 | Train Acc 1.0000 | Valid Loss 0.7432 | Valid Acc 0.8464
    Iter [3360/11250] | Train Loss 1.3157 | Train Acc 0.7500 | Valid Loss 0.7387 | Valid Acc 0.8470
    Iter [3380/11250] | Train Loss 1.6155 | Train Acc 0.7500 | Valid Loss 0.7362 | Valid Acc 0.8464
    Iter [3400/11250] | Train Loss 0.5488 | Train Acc 1.0000 | Valid Loss 0.7355 | Valid Acc 0.8440
    Iter [3420/11250] | Train Loss 1.9107 | Train Acc 0.5000 | Valid Loss 0.7352 | Valid Acc 0.8406
    Iter [3440/11250] | Train Loss 0.2982 | Train Acc 1.0000 | Valid Loss 0.7341 | Valid Acc 0.8392
    Iter [3460/11250] | Train Loss 1.2070 | Train Acc 0.5000 | Valid Loss 0.7299 | Valid Acc 0.8446
    Iter [3480/11250] | Train Loss 0.7420 | Train Acc 0.7500 | Valid Loss 0.7311 | Valid Acc 0.8402
    Iter [3500/11250] | Train Loss 1.0170 | Train Acc 0.7500 | Valid Loss 0.7247 | Valid Acc 0.8484
    Iter [3520/11250] | Train Loss 0.8868 | Train Acc 0.7500 | Valid Loss 0.7241 | Valid Acc 0.8446
    Iter [3540/11250] | Train Loss 0.6123 | Train Acc 1.0000 | Valid Loss 0.7225 | Valid Acc 0.8456
    Iter [3560/11250] | Train Loss 0.8683 | Train Acc 1.0000 | Valid Loss 0.7186 | Valid Acc 0.8494
    Iter [3580/11250] | Train Loss 0.8948 | Train Acc 0.7500 | Valid Loss 0.7204 | Valid Acc 0.8494
    Iter [3600/11250] | Train Loss 0.8089 | Train Acc 0.7500 | Valid Loss 0.7207 | Valid Acc 0.8404
    Iter [3620/11250] | Train Loss 0.3082 | Train Acc 1.0000 | Valid Loss 0.7190 | Valid Acc 0.8384
    Iter [3640/11250] | Train Loss 0.7032 | Train Acc 0.7500 | Valid Loss 0.7148 | Valid Acc 0.8432
    Iter [3660/11250] | Train Loss 0.6600 | Train Acc 0.7500 | Valid Loss 0.7146 | Valid Acc 0.8436
    Iter [3680/11250] | Train Loss 0.5219 | Train Acc 1.0000 | Valid Loss 0.7175 | Valid Acc 0.8422
    Iter [3700/11250] | Train Loss 1.3487 | Train Acc 0.7500 | Valid Loss 0.7172 | Valid Acc 0.8414
    Iter [3720/11250] | Train Loss 0.2985 | Train Acc 1.0000 | Valid Loss 0.7131 | Valid Acc 0.8418
    Iter [3740/11250] | Train Loss 0.5318 | Train Acc 1.0000 | Valid Loss 0.7082 | Valid Acc 0.8460
    Iter [3760/11250] | Train Loss 0.5357 | Train Acc 1.0000 | Valid Loss 0.7033 | Valid Acc 0.8500
    Iter [3780/11250] | Train Loss 0.1903 | Train Acc 1.0000 | Valid Loss 0.7091 | Valid Acc 0.8414
    Iter [3800/11250] | Train Loss 0.6249 | Train Acc 0.7500 | Valid Loss 0.6998 | Valid Acc 0.8480
    Iter [3820/11250] | Train Loss 0.8509 | Train Acc 0.7500 | Valid Loss 0.6973 | Valid Acc 0.8498
    Iter [3840/11250] | Train Loss 1.1319 | Train Acc 0.7500 | Valid Loss 0.6945 | Valid Acc 0.8494
    Iter [3860/11250] | Train Loss 0.3840 | Train Acc 1.0000 | Valid Loss 0.6932 | Valid Acc 0.8484
    Iter [3880/11250] | Train Loss 0.4881 | Train Acc 0.7500 | Valid Loss 0.6963 | Valid Acc 0.8432
    Iter [3900/11250] | Train Loss 0.3611 | Train Acc 1.0000 | Valid Loss 0.6878 | Valid Acc 0.8522
    Iter [3920/11250] | Train Loss 1.0599 | Train Acc 0.5000 | Valid Loss 0.6848 | Valid Acc 0.8520
    Iter [3940/11250] | Train Loss 1.0419 | Train Acc 0.7500 | Valid Loss 0.6855 | Valid Acc 0.8502
    Iter [3960/11250] | Train Loss 0.5321 | Train Acc 1.0000 | Valid Loss 0.6830 | Valid Acc 0.8518
    Iter [3980/11250] | Train Loss 0.7839 | Train Acc 0.7500 | Valid Loss 0.6813 | Valid Acc 0.8496
    Iter [4000/11250] | Train Loss 0.4693 | Train Acc 1.0000 | Valid Loss 0.6805 | Valid Acc 0.8506
    Iter [4020/11250] | Train Loss 1.4966 | Train Acc 0.5000 | Valid Loss 0.6806 | Valid Acc 0.8508
    Iter [4040/11250] | Train Loss 0.2599 | Train Acc 1.0000 | Valid Loss 0.6778 | Valid Acc 0.8526
    Iter [4060/11250] | Train Loss 0.7919 | Train Acc 1.0000 | Valid Loss 0.6776 | Valid Acc 0.8476
    Iter [4080/11250] | Train Loss 0.4891 | Train Acc 0.7500 | Valid Loss 0.6797 | Valid Acc 0.8456
    Iter [4100/11250] | Train Loss 0.9422 | Train Acc 0.7500 | Valid Loss 0.6835 | Valid Acc 0.8404
    Iter [4120/11250] | Train Loss 0.4479 | Train Acc 0.7500 | Valid Loss 0.6798 | Valid Acc 0.8434
    Iter [4140/11250] | Train Loss 0.4616 | Train Acc 1.0000 | Valid Loss 0.6743 | Valid Acc 0.8492
    Iter [4160/11250] | Train Loss 0.3022 | Train Acc 1.0000 | Valid Loss 0.6717 | Valid Acc 0.8464
    Iter [4180/11250] | Train Loss 1.0461 | Train Acc 0.7500 | Valid Loss 0.6689 | Valid Acc 0.8476
    Iter [4200/11250] | Train Loss 0.5655 | Train Acc 1.0000 | Valid Loss 0.6669 | Valid Acc 0.8516
    Iter [4220/11250] | Train Loss 0.2467 | Train Acc 1.0000 | Valid Loss 0.6679 | Valid Acc 0.8444
    Iter [4240/11250] | Train Loss 0.8756 | Train Acc 0.7500 | Valid Loss 0.6603 | Valid Acc 0.8560
    Iter [4260/11250] | Train Loss 0.4909 | Train Acc 1.0000 | Valid Loss 0.6575 | Valid Acc 0.8556
    Iter [4280/11250] | Train Loss 1.1380 | Train Acc 0.7500 | Valid Loss 0.6587 | Valid Acc 0.8518
    Iter [4300/11250] | Train Loss 0.1792 | Train Acc 1.0000 | Valid Loss 0.6585 | Valid Acc 0.8536
    Iter [4320/11250] | Train Loss 0.1811 | Train Acc 1.0000 | Valid Loss 0.6579 | Valid Acc 0.8536
    Iter [4340/11250] | Train Loss 0.4022 | Train Acc 1.0000 | Valid Loss 0.6567 | Valid Acc 0.8560
    Iter [4360/11250] | Train Loss 0.7166 | Train Acc 0.7500 | Valid Loss 0.6532 | Valid Acc 0.8562
    Iter [4380/11250] | Train Loss 0.6016 | Train Acc 1.0000 | Valid Loss 0.6519 | Valid Acc 0.8570
    Iter [4400/11250] | Train Loss 0.6420 | Train Acc 0.7500 | Valid Loss 0.6511 | Valid Acc 0.8590
    Iter [4420/11250] | Train Loss 0.2980 | Train Acc 1.0000 | Valid Loss 0.6530 | Valid Acc 0.8530
    Iter [4440/11250] | Train Loss 0.5821 | Train Acc 0.7500 | Valid Loss 0.6484 | Valid Acc 0.8594
    Iter [4460/11250] | Train Loss 0.9802 | Train Acc 0.5000 | Valid Loss 0.6508 | Valid Acc 0.8548
    Iter [4480/11250] | Train Loss 0.4092 | Train Acc 1.0000 | Valid Loss 0.6506 | Valid Acc 0.8552
    Iter [4500/11250] | Train Loss 0.4769 | Train Acc 1.0000 | Valid Loss 0.6468 | Valid Acc 0.8566
    Iter [4520/11250] | Train Loss 0.2956 | Train Acc 1.0000 | Valid Loss 0.6421 | Valid Acc 0.8604
    Iter [4540/11250] | Train Loss 1.0443 | Train Acc 0.5000 | Valid Loss 0.6394 | Valid Acc 0.8612
    Iter [4560/11250] | Train Loss 1.1611 | Train Acc 0.7500 | Valid Loss 0.6373 | Valid Acc 0.8612
    Iter [4580/11250] | Train Loss 0.9851 | Train Acc 0.5000 | Valid Loss 0.6375 | Valid Acc 0.8604
    Iter [4600/11250] | Train Loss 0.3880 | Train Acc 1.0000 | Valid Loss 0.6474 | Valid Acc 0.8522
    Iter [4620/11250] | Train Loss 0.4707 | Train Acc 1.0000 | Valid Loss 0.6464 | Valid Acc 0.8510
    Iter [4640/11250] | Train Loss 0.8099 | Train Acc 0.5000 | Valid Loss 0.6380 | Valid Acc 0.8550
    Iter [4660/11250] | Train Loss 1.1040 | Train Acc 0.7500 | Valid Loss 0.6355 | Valid Acc 0.8548
    Iter [4680/11250] | Train Loss 0.8862 | Train Acc 0.7500 | Valid Loss 0.6341 | Valid Acc 0.8542
    Iter [4700/11250] | Train Loss 0.1677 | Train Acc 1.0000 | Valid Loss 0.6301 | Valid Acc 0.8558
    Iter [4720/11250] | Train Loss 0.6289 | Train Acc 1.0000 | Valid Loss 0.6292 | Valid Acc 0.8568
    Iter [4740/11250] | Train Loss 0.1860 | Train Acc 1.0000 | Valid Loss 0.6265 | Valid Acc 0.8582
    Iter [4760/11250] | Train Loss 0.3321 | Train Acc 1.0000 | Valid Loss 0.6254 | Valid Acc 0.8598
    Iter [4780/11250] | Train Loss 0.6741 | Train Acc 0.7500 | Valid Loss 0.6305 | Valid Acc 0.8542
    Iter [4800/11250] | Train Loss 0.2838 | Train Acc 1.0000 | Valid Loss 0.6278 | Valid Acc 0.8512
    Iter [4820/11250] | Train Loss 0.0817 | Train Acc 1.0000 | Valid Loss 0.6218 | Valid Acc 0.8594
    Iter [4840/11250] | Train Loss 0.9373 | Train Acc 0.7500 | Valid Loss 0.6224 | Valid Acc 0.8604
    Iter [4860/11250] | Train Loss 0.5262 | Train Acc 0.7500 | Valid Loss 0.6207 | Valid Acc 0.8598
    Iter [4880/11250] | Train Loss 0.7314 | Train Acc 0.7500 | Valid Loss 0.6205 | Valid Acc 0.8590
    Iter [4900/11250] | Train Loss 0.4941 | Train Acc 0.7500 | Valid Loss 0.6195 | Valid Acc 0.8584
    Iter [4920/11250] | Train Loss 1.0372 | Train Acc 0.5000 | Valid Loss 0.6210 | Valid Acc 0.8596
    Iter [4940/11250] | Train Loss 0.4193 | Train Acc 1.0000 | Valid Loss 0.6173 | Valid Acc 0.8608
    Iter [4960/11250] | Train Loss 0.5757 | Train Acc 0.7500 | Valid Loss 0.6151 | Valid Acc 0.8602
    Iter [4980/11250] | Train Loss 0.9820 | Train Acc 0.7500 | Valid Loss 0.6179 | Valid Acc 0.8582
    Iter [5000/11250] | Train Loss 0.5282 | Train Acc 1.0000 | Valid Loss 0.6155 | Valid Acc 0.8600
    Iter [5020/11250] | Train Loss 0.4743 | Train Acc 0.7500 | Valid Loss 0.6126 | Valid Acc 0.8618
    Iter [5040/11250] | Train Loss 0.3448 | Train Acc 1.0000 | Valid Loss 0.6097 | Valid Acc 0.8626
    Iter [5060/11250] | Train Loss 0.1669 | Train Acc 1.0000 | Valid Loss 0.6102 | Valid Acc 0.8628
    Iter [5080/11250] | Train Loss 0.3563 | Train Acc 1.0000 | Valid Loss 0.6075 | Valid Acc 0.8628
    Iter [5100/11250] | Train Loss 0.9881 | Train Acc 0.7500 | Valid Loss 0.6071 | Valid Acc 0.8622
    Iter [5120/11250] | Train Loss 0.0731 | Train Acc 1.0000 | Valid Loss 0.6057 | Valid Acc 0.8586
    Iter [5140/11250] | Train Loss 1.1569 | Train Acc 0.5000 | Valid Loss 0.6051 | Valid Acc 0.8578
    Iter [5160/11250] | Train Loss 0.1412 | Train Acc 1.0000 | Valid Loss 0.6086 | Valid Acc 0.8550
    Iter [5180/11250] | Train Loss 0.5774 | Train Acc 1.0000 | Valid Loss 0.6061 | Valid Acc 0.8546
    Iter [5200/11250] | Train Loss 0.7955 | Train Acc 0.5000 | Valid Loss 0.6060 | Valid Acc 0.8542
    Iter [5220/11250] | Train Loss 0.5601 | Train Acc 1.0000 | Valid Loss 0.6055 | Valid Acc 0.8564
    Iter [5240/11250] | Train Loss 0.6676 | Train Acc 0.7500 | Valid Loss 0.6042 | Valid Acc 0.8564
    Iter [5260/11250] | Train Loss 1.0312 | Train Acc 0.7500 | Valid Loss 0.5999 | Valid Acc 0.8588
    Iter [5280/11250] | Train Loss 0.3916 | Train Acc 1.0000 | Valid Loss 0.5981 | Valid Acc 0.8588
    Iter [5300/11250] | Train Loss 0.9138 | Train Acc 0.5000 | Valid Loss 0.5966 | Valid Acc 0.8652
    Iter [5320/11250] | Train Loss 0.6269 | Train Acc 0.7500 | Valid Loss 0.5951 | Valid Acc 0.8642
    Iter [5340/11250] | Train Loss 1.2493 | Train Acc 0.7500 | Valid Loss 0.5964 | Valid Acc 0.8646
    Iter [5360/11250] | Train Loss 0.5892 | Train Acc 1.0000 | Valid Loss 0.5941 | Valid Acc 0.8658
    Iter [5380/11250] | Train Loss 0.8208 | Train Acc 0.7500 | Valid Loss 0.5927 | Valid Acc 0.8660
    Iter [5400/11250] | Train Loss 1.4732 | Train Acc 0.7500 | Valid Loss 0.5900 | Valid Acc 0.8620
    Iter [5420/11250] | Train Loss 0.2724 | Train Acc 1.0000 | Valid Loss 0.5886 | Valid Acc 0.8614
    Iter [5440/11250] | Train Loss 0.4813 | Train Acc 1.0000 | Valid Loss 0.5874 | Valid Acc 0.8646
    Iter [5460/11250] | Train Loss 0.1990 | Train Acc 1.0000 | Valid Loss 0.5915 | Valid Acc 0.8632
    Iter [5480/11250] | Train Loss 0.8545 | Train Acc 0.7500 | Valid Loss 0.5845 | Valid Acc 0.8650
    Iter [5500/11250] | Train Loss 0.7901 | Train Acc 0.5000 | Valid Loss 0.5828 | Valid Acc 0.8694
    Iter [5520/11250] | Train Loss 1.3672 | Train Acc 0.5000 | Valid Loss 0.5826 | Valid Acc 0.8662
    Iter [5540/11250] | Train Loss 0.2384 | Train Acc 1.0000 | Valid Loss 0.5885 | Valid Acc 0.8606
    Iter [5560/11250] | Train Loss 0.6421 | Train Acc 1.0000 | Valid Loss 0.5908 | Valid Acc 0.8580
    Iter [5580/11250] | Train Loss 0.5777 | Train Acc 0.7500 | Valid Loss 0.5887 | Valid Acc 0.8618
    Iter [5600/11250] | Train Loss 0.6581 | Train Acc 0.7500 | Valid Loss 0.5808 | Valid Acc 0.8654
    Iter [5620/11250] | Train Loss 0.5289 | Train Acc 1.0000 | Valid Loss 0.5810 | Valid Acc 0.8694
    Iter [5640/11250] | Train Loss 0.6136 | Train Acc 0.7500 | Valid Loss 0.5843 | Valid Acc 0.8648
    Iter [5660/11250] | Train Loss 0.5413 | Train Acc 1.0000 | Valid Loss 0.5828 | Valid Acc 0.8620
    Iter [5680/11250] | Train Loss 0.2718 | Train Acc 1.0000 | Valid Loss 0.5827 | Valid Acc 0.8630
    Iter [5700/11250] | Train Loss 0.8607 | Train Acc 0.7500 | Valid Loss 0.5819 | Valid Acc 0.8612
    Iter [5720/11250] | Train Loss 0.3592 | Train Acc 1.0000 | Valid Loss 0.5809 | Valid Acc 0.8606
    Iter [5740/11250] | Train Loss 0.5108 | Train Acc 1.0000 | Valid Loss 0.5841 | Valid Acc 0.8612
    Iter [5760/11250] | Train Loss 0.1935 | Train Acc 1.0000 | Valid Loss 0.5770 | Valid Acc 0.8658
    Iter [5780/11250] | Train Loss 0.7232 | Train Acc 0.7500 | Valid Loss 0.5752 | Valid Acc 0.8688
    Iter [5800/11250] | Train Loss 0.4238 | Train Acc 0.7500 | Valid Loss 0.5724 | Valid Acc 0.8668
    Iter [5820/11250] | Train Loss 0.6761 | Train Acc 0.7500 | Valid Loss 0.5736 | Valid Acc 0.8660
    Iter [5840/11250] | Train Loss 0.4787 | Train Acc 1.0000 | Valid Loss 0.5743 | Valid Acc 0.8672
    Iter [5860/11250] | Train Loss 0.1408 | Train Acc 1.0000 | Valid Loss 0.5740 | Valid Acc 0.8636
    Iter [5880/11250] | Train Loss 0.4750 | Train Acc 1.0000 | Valid Loss 0.5744 | Valid Acc 0.8608
    Iter [5900/11250] | Train Loss 1.9203 | Train Acc 0.2500 | Valid Loss 0.5697 | Valid Acc 0.8672
    Iter [5920/11250] | Train Loss 0.3556 | Train Acc 1.0000 | Valid Loss 0.5654 | Valid Acc 0.8692
    Iter [5940/11250] | Train Loss 0.4757 | Train Acc 0.7500 | Valid Loss 0.5636 | Valid Acc 0.8688
    Iter [5960/11250] | Train Loss 1.9849 | Train Acc 0.5000 | Valid Loss 0.5638 | Valid Acc 0.8706
    Iter [5980/11250] | Train Loss 0.6652 | Train Acc 0.7500 | Valid Loss 0.5614 | Valid Acc 0.8724
    Iter [6000/11250] | Train Loss 0.5291 | Train Acc 1.0000 | Valid Loss 0.5624 | Valid Acc 0.8704
    Iter [6020/11250] | Train Loss 0.1524 | Train Acc 1.0000 | Valid Loss 0.5664 | Valid Acc 0.8666
    Iter [6040/11250] | Train Loss 0.3985 | Train Acc 1.0000 | Valid Loss 0.5652 | Valid Acc 0.8696
    Iter [6060/11250] | Train Loss 0.7752 | Train Acc 0.7500 | Valid Loss 0.5604 | Valid Acc 0.8696
    Iter [6080/11250] | Train Loss 0.6901 | Train Acc 0.7500 | Valid Loss 0.5589 | Valid Acc 0.8690
    Iter [6100/11250] | Train Loss 1.1087 | Train Acc 0.5000 | Valid Loss 0.5592 | Valid Acc 0.8686
    Iter [6120/11250] | Train Loss 0.5596 | Train Acc 0.7500 | Valid Loss 0.5595 | Valid Acc 0.8678
    Iter [6140/11250] | Train Loss 0.3637 | Train Acc 1.0000 | Valid Loss 0.5582 | Valid Acc 0.8700
    Iter [6160/11250] | Train Loss 0.8779 | Train Acc 0.5000 | Valid Loss 0.5572 | Valid Acc 0.8726
    Iter [6180/11250] | Train Loss 0.5419 | Train Acc 0.7500 | Valid Loss 0.5562 | Valid Acc 0.8704
    Iter [6200/11250] | Train Loss 0.3005 | Train Acc 1.0000 | Valid Loss 0.5561 | Valid Acc 0.8716
    Iter [6220/11250] | Train Loss 1.4431 | Train Acc 0.7500 | Valid Loss 0.5626 | Valid Acc 0.8628
    Iter [6240/11250] | Train Loss 0.9316 | Train Acc 0.7500 | Valid Loss 0.5623 | Valid Acc 0.8622
    Iter [6260/11250] | Train Loss 0.6379 | Train Acc 0.7500 | Valid Loss 0.5602 | Valid Acc 0.8610
    Iter [6280/11250] | Train Loss 0.4583 | Train Acc 1.0000 | Valid Loss 0.5589 | Valid Acc 0.8634
    Iter [6300/11250] | Train Loss 0.5505 | Train Acc 0.7500 | Valid Loss 0.5515 | Valid Acc 0.8700
    Iter [6320/11250] | Train Loss 0.6738 | Train Acc 0.7500 | Valid Loss 0.5512 | Valid Acc 0.8706
    Iter [6340/11250] | Train Loss 0.2728 | Train Acc 1.0000 | Valid Loss 0.5502 | Valid Acc 0.8724
    Iter [6360/11250] | Train Loss 0.2099 | Train Acc 1.0000 | Valid Loss 0.5524 | Valid Acc 0.8696
    Iter [6380/11250] | Train Loss 0.6392 | Train Acc 0.7500 | Valid Loss 0.5532 | Valid Acc 0.8668
    Iter [6400/11250] | Train Loss 1.4369 | Train Acc 0.7500 | Valid Loss 0.5556 | Valid Acc 0.8656
    Iter [6420/11250] | Train Loss 0.4551 | Train Acc 1.0000 | Valid Loss 0.5543 | Valid Acc 0.8688
    Iter [6440/11250] | Train Loss 0.6996 | Train Acc 1.0000 | Valid Loss 0.5520 | Valid Acc 0.8694
    Iter [6460/11250] | Train Loss 0.1484 | Train Acc 1.0000 | Valid Loss 0.5550 | Valid Acc 0.8678
    Iter [6480/11250] | Train Loss 0.7382 | Train Acc 0.7500 | Valid Loss 0.5525 | Valid Acc 0.8688
    Iter [6500/11250] | Train Loss 0.5258 | Train Acc 0.7500 | Valid Loss 0.5497 | Valid Acc 0.8700
    Iter [6520/11250] | Train Loss 1.5711 | Train Acc 0.5000 | Valid Loss 0.5475 | Valid Acc 0.8708
    Iter [6540/11250] | Train Loss 0.8705 | Train Acc 0.7500 | Valid Loss 0.5448 | Valid Acc 0.8698
    Iter [6560/11250] | Train Loss 0.3441 | Train Acc 0.7500 | Valid Loss 0.5432 | Valid Acc 0.8698
    Iter [6580/11250] | Train Loss 0.3866 | Train Acc 1.0000 | Valid Loss 0.5428 | Valid Acc 0.8696
    Iter [6600/11250] | Train Loss 0.3392 | Train Acc 1.0000 | Valid Loss 0.5417 | Valid Acc 0.8716
    Iter [6620/11250] | Train Loss 0.9118 | Train Acc 0.7500 | Valid Loss 0.5397 | Valid Acc 0.8734
    Iter [6640/11250] | Train Loss 0.6457 | Train Acc 0.7500 | Valid Loss 0.5402 | Valid Acc 0.8726
    Iter [6660/11250] | Train Loss 0.7987 | Train Acc 0.7500 | Valid Loss 0.5387 | Valid Acc 0.8714
    Iter [6680/11250] | Train Loss 0.7687 | Train Acc 0.5000 | Valid Loss 0.5374 | Valid Acc 0.8726
    Iter [6700/11250] | Train Loss 0.2348 | Train Acc 1.0000 | Valid Loss 0.5371 | Valid Acc 0.8720
    Iter [6720/11250] | Train Loss 0.1519 | Train Acc 1.0000 | Valid Loss 0.5375 | Valid Acc 0.8724
    Iter [6740/11250] | Train Loss 0.1123 | Train Acc 1.0000 | Valid Loss 0.5352 | Valid Acc 0.8746
    Iter [6760/11250] | Train Loss 0.7623 | Train Acc 0.7500 | Valid Loss 0.5343 | Valid Acc 0.8734
    Iter [6780/11250] | Train Loss 0.4555 | Train Acc 0.7500 | Valid Loss 0.5347 | Valid Acc 0.8730
    Iter [6800/11250] | Train Loss 0.0386 | Train Acc 1.0000 | Valid Loss 0.5365 | Valid Acc 0.8730
    Iter [6820/11250] | Train Loss 1.0858 | Train Acc 0.5000 | Valid Loss 0.5352 | Valid Acc 0.8732
    Iter [6840/11250] | Train Loss 0.7082 | Train Acc 0.7500 | Valid Loss 0.5358 | Valid Acc 0.8730
    Iter [6860/11250] | Train Loss 0.3500 | Train Acc 1.0000 | Valid Loss 0.5335 | Valid Acc 0.8724
    Iter [6880/11250] | Train Loss 0.8300 | Train Acc 0.7500 | Valid Loss 0.5352 | Valid Acc 0.8694
    Iter [6900/11250] | Train Loss 0.2557 | Train Acc 1.0000 | Valid Loss 0.5299 | Valid Acc 0.8748
    Iter [6920/11250] | Train Loss 0.2431 | Train Acc 1.0000 | Valid Loss 0.5279 | Valid Acc 0.8760
    Iter [6940/11250] | Train Loss 0.8081 | Train Acc 0.7500 | Valid Loss 0.5285 | Valid Acc 0.8740
    Iter [6960/11250] | Train Loss 0.3366 | Train Acc 1.0000 | Valid Loss 0.5270 | Valid Acc 0.8766
    Iter [6980/11250] | Train Loss 0.7598 | Train Acc 1.0000 | Valid Loss 0.5285 | Valid Acc 0.8750
    Iter [7000/11250] | Train Loss 0.5124 | Train Acc 0.7500 | Valid Loss 0.5274 | Valid Acc 0.8758
    Iter [7020/11250] | Train Loss 1.5315 | Train Acc 0.7500 | Valid Loss 0.5258 | Valid Acc 0.8750
    Iter [7040/11250] | Train Loss 0.1804 | Train Acc 1.0000 | Valid Loss 0.5241 | Valid Acc 0.8740
    Iter [7060/11250] | Train Loss 0.9937 | Train Acc 0.7500 | Valid Loss 0.5248 | Valid Acc 0.8724
    Iter [7080/11250] | Train Loss 0.3348 | Train Acc 0.7500 | Valid Loss 0.5287 | Valid Acc 0.8698
    Iter [7100/11250] | Train Loss 0.3869 | Train Acc 1.0000 | Valid Loss 0.5292 | Valid Acc 0.8658
    Iter [7120/11250] | Train Loss 0.8571 | Train Acc 0.7500 | Valid Loss 0.5296 | Valid Acc 0.8664
    Iter [7140/11250] | Train Loss 0.4620 | Train Acc 0.7500 | Valid Loss 0.5263 | Valid Acc 0.8710
    Iter [7160/11250] | Train Loss 1.1210 | Train Acc 0.7500 | Valid Loss 0.5248 | Valid Acc 0.8702
    Iter [7180/11250] | Train Loss 0.6333 | Train Acc 0.7500 | Valid Loss 0.5240 | Valid Acc 0.8708
    Iter [7200/11250] | Train Loss 0.4408 | Train Acc 0.7500 | Valid Loss 0.5249 | Valid Acc 0.8702
    Iter [7220/11250] | Train Loss 0.0693 | Train Acc 1.0000 | Valid Loss 0.5227 | Valid Acc 0.8734
    Iter [7240/11250] | Train Loss 0.1793 | Train Acc 1.0000 | Valid Loss 0.5224 | Valid Acc 0.8746
    Iter [7260/11250] | Train Loss 1.1042 | Train Acc 0.7500 | Valid Loss 0.5199 | Valid Acc 0.8750
    Iter [7280/11250] | Train Loss 0.1915 | Train Acc 1.0000 | Valid Loss 0.5200 | Valid Acc 0.8754
    Iter [7300/11250] | Train Loss 0.4814 | Train Acc 0.7500 | Valid Loss 0.5197 | Valid Acc 0.8734
    Iter [7320/11250] | Train Loss 0.1381 | Train Acc 1.0000 | Valid Loss 0.5222 | Valid Acc 0.8694
    Iter [7340/11250] | Train Loss 1.2595 | Train Acc 0.2500 | Valid Loss 0.5198 | Valid Acc 0.8722
    Iter [7360/11250] | Train Loss 0.2784 | Train Acc 1.0000 | Valid Loss 0.5176 | Valid Acc 0.8730
    Iter [7380/11250] | Train Loss 0.1546 | Train Acc 1.0000 | Valid Loss 0.5195 | Valid Acc 0.8692
    Iter [7400/11250] | Train Loss 0.2899 | Train Acc 1.0000 | Valid Loss 0.5171 | Valid Acc 0.8712
    Iter [7420/11250] | Train Loss 0.9742 | Train Acc 0.7500 | Valid Loss 0.5199 | Valid Acc 0.8736
    Iter [7440/11250] | Train Loss 0.2649 | Train Acc 1.0000 | Valid Loss 0.5168 | Valid Acc 0.8736
    Iter [7460/11250] | Train Loss 0.6721 | Train Acc 1.0000 | Valid Loss 0.5185 | Valid Acc 0.8726
    Iter [7480/11250] | Train Loss 0.4387 | Train Acc 1.0000 | Valid Loss 0.5200 | Valid Acc 0.8718
    Iter [7500/11250] | Train Loss 0.4559 | Train Acc 1.0000 | Valid Loss 0.5131 | Valid Acc 0.8774
    Iter [7520/11250] | Train Loss 0.7792 | Train Acc 0.5000 | Valid Loss 0.5151 | Valid Acc 0.8778
    Iter [7540/11250] | Train Loss 0.5608 | Train Acc 0.7500 | Valid Loss 0.5151 | Valid Acc 0.8758
    Iter [7560/11250] | Train Loss 0.5609 | Train Acc 0.7500 | Valid Loss 0.5162 | Valid Acc 0.8790
    Iter [7580/11250] | Train Loss 0.4176 | Train Acc 1.0000 | Valid Loss 0.5168 | Valid Acc 0.8786
    Iter [7600/11250] | Train Loss 0.6485 | Train Acc 0.7500 | Valid Loss 0.5109 | Valid Acc 0.8780
    Iter [7620/11250] | Train Loss 0.3069 | Train Acc 0.7500 | Valid Loss 0.5085 | Valid Acc 0.8788
    Iter [7640/11250] | Train Loss 0.4015 | Train Acc 0.7500 | Valid Loss 0.5104 | Valid Acc 0.8752
    Iter [7660/11250] | Train Loss 0.3686 | Train Acc 1.0000 | Valid Loss 0.5128 | Valid Acc 0.8722
    Iter [7680/11250] | Train Loss 1.0373 | Train Acc 0.5000 | Valid Loss 0.5126 | Valid Acc 0.8728
    Iter [7700/11250] | Train Loss 1.5212 | Train Acc 0.7500 | Valid Loss 0.5111 | Valid Acc 0.8728
    Iter [7720/11250] | Train Loss 0.5522 | Train Acc 0.7500 | Valid Loss 0.5093 | Valid Acc 0.8740
    Iter [7740/11250] | Train Loss 0.2444 | Train Acc 1.0000 | Valid Loss 0.5117 | Valid Acc 0.8732
    Iter [7760/11250] | Train Loss 0.5071 | Train Acc 0.7500 | Valid Loss 0.5145 | Valid Acc 0.8698
    Iter [7780/11250] | Train Loss 1.0094 | Train Acc 0.5000 | Valid Loss 0.5078 | Valid Acc 0.8736
    Iter [7800/11250] | Train Loss 0.2609 | Train Acc 1.0000 | Valid Loss 0.5071 | Valid Acc 0.8744
    Iter [7820/11250] | Train Loss 1.0301 | Train Acc 0.7500 | Valid Loss 0.5074 | Valid Acc 0.8746
    Iter [7840/11250] | Train Loss 0.2248 | Train Acc 1.0000 | Valid Loss 0.5060 | Valid Acc 0.8756
    Iter [7860/11250] | Train Loss 0.3840 | Train Acc 1.0000 | Valid Loss 0.5059 | Valid Acc 0.8780
    Iter [7880/11250] | Train Loss 0.2360 | Train Acc 1.0000 | Valid Loss 0.5046 | Valid Acc 0.8770
    Iter [7900/11250] | Train Loss 1.4719 | Train Acc 0.5000 | Valid Loss 0.5108 | Valid Acc 0.8752
    Iter [7920/11250] | Train Loss 0.7102 | Train Acc 0.7500 | Valid Loss 0.5108 | Valid Acc 0.8768
    Iter [7940/11250] | Train Loss 0.4564 | Train Acc 1.0000 | Valid Loss 0.5107 | Valid Acc 0.8734
    Iter [7960/11250] | Train Loss 0.1122 | Train Acc 1.0000 | Valid Loss 0.5025 | Valid Acc 0.8760
    Iter [7980/11250] | Train Loss 0.7875 | Train Acc 0.5000 | Valid Loss 0.5012 | Valid Acc 0.8782
    Iter [8000/11250] | Train Loss 0.4444 | Train Acc 0.7500 | Valid Loss 0.5011 | Valid Acc 0.8808
    Iter [8020/11250] | Train Loss 0.1777 | Train Acc 1.0000 | Valid Loss 0.4990 | Valid Acc 0.8814
    Iter [8040/11250] | Train Loss 0.3841 | Train Acc 0.7500 | Valid Loss 0.4977 | Valid Acc 0.8788
    Iter [8060/11250] | Train Loss 0.1554 | Train Acc 1.0000 | Valid Loss 0.5003 | Valid Acc 0.8770
    Iter [8080/11250] | Train Loss 1.2645 | Train Acc 0.5000 | Valid Loss 0.5018 | Valid Acc 0.8756
    Iter [8100/11250] | Train Loss 0.9963 | Train Acc 0.7500 | Valid Loss 0.5012 | Valid Acc 0.8746
    Iter [8120/11250] | Train Loss 1.1099 | Train Acc 0.2500 | Valid Loss 0.5018 | Valid Acc 0.8740
    Iter [8140/11250] | Train Loss 0.4203 | Train Acc 1.0000 | Valid Loss 0.5024 | Valid Acc 0.8758
    Iter [8160/11250] | Train Loss 0.6682 | Train Acc 0.7500 | Valid Loss 0.5023 | Valid Acc 0.8748
    Iter [8180/11250] | Train Loss 0.2706 | Train Acc 1.0000 | Valid Loss 0.4999 | Valid Acc 0.8756
    Iter [8200/11250] | Train Loss 1.2080 | Train Acc 0.7500 | Valid Loss 0.5031 | Valid Acc 0.8752
    Iter [8220/11250] | Train Loss 0.3969 | Train Acc 0.7500 | Valid Loss 0.4959 | Valid Acc 0.8780
    Iter [8240/11250] | Train Loss 0.1909 | Train Acc 1.0000 | Valid Loss 0.4954 | Valid Acc 0.8786
    Iter [8260/11250] | Train Loss 0.4548 | Train Acc 0.7500 | Valid Loss 0.4957 | Valid Acc 0.8764
    Iter [8280/11250] | Train Loss 0.5971 | Train Acc 0.7500 | Valid Loss 0.4977 | Valid Acc 0.8742
    Iter [8300/11250] | Train Loss 0.4023 | Train Acc 0.7500 | Valid Loss 0.4937 | Valid Acc 0.8784
    Iter [8320/11250] | Train Loss 0.4359 | Train Acc 1.0000 | Valid Loss 0.4958 | Valid Acc 0.8804
    Iter [8340/11250] | Train Loss 0.5144 | Train Acc 1.0000 | Valid Loss 0.4962 | Valid Acc 0.8794
    Iter [8360/11250] | Train Loss 0.1375 | Train Acc 1.0000 | Valid Loss 0.4973 | Valid Acc 0.8822
    Iter [8380/11250] | Train Loss 0.5495 | Train Acc 0.7500 | Valid Loss 0.4943 | Valid Acc 0.8812
    Iter [8400/11250] | Train Loss 0.1810 | Train Acc 1.0000 | Valid Loss 0.4924 | Valid Acc 0.8784
    Iter [8420/11250] | Train Loss 0.3161 | Train Acc 1.0000 | Valid Loss 0.4905 | Valid Acc 0.8810
    Iter [8440/11250] | Train Loss 0.3239 | Train Acc 1.0000 | Valid Loss 0.4915 | Valid Acc 0.8776
    Iter [8460/11250] | Train Loss 1.1806 | Train Acc 0.7500 | Valid Loss 0.4905 | Valid Acc 0.8770
    Iter [8480/11250] | Train Loss 0.4304 | Train Acc 0.7500 | Valid Loss 0.4882 | Valid Acc 0.8784
    Iter [8500/11250] | Train Loss 0.0641 | Train Acc 1.0000 | Valid Loss 0.4895 | Valid Acc 0.8774
    Iter [8520/11250] | Train Loss 0.6559 | Train Acc 0.7500 | Valid Loss 0.4877 | Valid Acc 0.8824
    Iter [8540/11250] | Train Loss 0.2650 | Train Acc 1.0000 | Valid Loss 0.4898 | Valid Acc 0.8804
    Iter [8560/11250] | Train Loss 0.1504 | Train Acc 1.0000 | Valid Loss 0.4873 | Valid Acc 0.8818
    Iter [8580/11250] | Train Loss 0.1757 | Train Acc 1.0000 | Valid Loss 0.4867 | Valid Acc 0.8804
    Iter [8600/11250] | Train Loss 0.2100 | Train Acc 1.0000 | Valid Loss 0.4878 | Valid Acc 0.8808
    Iter [8620/11250] | Train Loss 0.3779 | Train Acc 1.0000 | Valid Loss 0.4875 | Valid Acc 0.8804
    Iter [8640/11250] | Train Loss 0.1644 | Train Acc 1.0000 | Valid Loss 0.4904 | Valid Acc 0.8780
    Iter [8660/11250] | Train Loss 0.8213 | Train Acc 0.7500 | Valid Loss 0.4897 | Valid Acc 0.8786
    Iter [8680/11250] | Train Loss 1.2152 | Train Acc 0.2500 | Valid Loss 0.4860 | Valid Acc 0.8790
    Iter [8700/11250] | Train Loss 0.3043 | Train Acc 1.0000 | Valid Loss 0.4831 | Valid Acc 0.8800
    Iter [8720/11250] | Train Loss 0.6636 | Train Acc 0.7500 | Valid Loss 0.4837 | Valid Acc 0.8800
    Iter [8740/11250] | Train Loss 0.8064 | Train Acc 0.7500 | Valid Loss 0.4880 | Valid Acc 0.8784
    Iter [8760/11250] | Train Loss 0.0979 | Train Acc 1.0000 | Valid Loss 0.4884 | Valid Acc 0.8782
    Iter [8780/11250] | Train Loss 0.6773 | Train Acc 0.5000 | Valid Loss 0.4849 | Valid Acc 0.8780
    Iter [8800/11250] | Train Loss 0.3243 | Train Acc 1.0000 | Valid Loss 0.4847 | Valid Acc 0.8796
    Iter [8820/11250] | Train Loss 0.3418 | Train Acc 0.7500 | Valid Loss 0.4831 | Valid Acc 0.8796
    Iter [8840/11250] | Train Loss 0.0848 | Train Acc 1.0000 | Valid Loss 0.4822 | Valid Acc 0.8794
    Iter [8860/11250] | Train Loss 0.1669 | Train Acc 1.0000 | Valid Loss 0.4835 | Valid Acc 0.8786
    Iter [8880/11250] | Train Loss 0.4963 | Train Acc 0.7500 | Valid Loss 0.4815 | Valid Acc 0.8786
    Iter [8900/11250] | Train Loss 0.8907 | Train Acc 0.7500 | Valid Loss 0.4821 | Valid Acc 0.8808
    Iter [8920/11250] | Train Loss 0.1541 | Train Acc 1.0000 | Valid Loss 0.4820 | Valid Acc 0.8816
    Iter [8940/11250] | Train Loss 0.1417 | Train Acc 1.0000 | Valid Loss 0.4845 | Valid Acc 0.8822
    Iter [8960/11250] | Train Loss 0.3712 | Train Acc 0.7500 | Valid Loss 0.4805 | Valid Acc 0.8830
    Iter [8980/11250] | Train Loss 0.4459 | Train Acc 0.7500 | Valid Loss 0.4791 | Valid Acc 0.8804
    Iter [9000/11250] | Train Loss 0.2666 | Train Acc 0.7500 | Valid Loss 0.4784 | Valid Acc 0.8824
    Iter [9020/11250] | Train Loss 0.2853 | Train Acc 0.7500 | Valid Loss 0.4774 | Valid Acc 0.8828
    Iter [9040/11250] | Train Loss 0.3380 | Train Acc 1.0000 | Valid Loss 0.4777 | Valid Acc 0.8832
    Iter [9060/11250] | Train Loss 0.2218 | Train Acc 1.0000 | Valid Loss 0.4788 | Valid Acc 0.8814
    Iter [9080/11250] | Train Loss 0.1145 | Train Acc 1.0000 | Valid Loss 0.4779 | Valid Acc 0.8840
    Iter [9100/11250] | Train Loss 0.3495 | Train Acc 1.0000 | Valid Loss 0.4762 | Valid Acc 0.8832
    Iter [9120/11250] | Train Loss 0.4468 | Train Acc 1.0000 | Valid Loss 0.4776 | Valid Acc 0.8836
    Iter [9140/11250] | Train Loss 0.7755 | Train Acc 0.7500 | Valid Loss 0.4799 | Valid Acc 0.8824
    Iter [9160/11250] | Train Loss 0.6954 | Train Acc 0.5000 | Valid Loss 0.4777 | Valid Acc 0.8826
    Iter [9180/11250] | Train Loss 0.3417 | Train Acc 1.0000 | Valid Loss 0.4747 | Valid Acc 0.8830
    Iter [9200/11250] | Train Loss 0.5563 | Train Acc 1.0000 | Valid Loss 0.4736 | Valid Acc 0.8838
    Iter [9220/11250] | Train Loss 0.5085 | Train Acc 0.7500 | Valid Loss 0.4730 | Valid Acc 0.8842
    Iter [9240/11250] | Train Loss 0.6276 | Train Acc 1.0000 | Valid Loss 0.4714 | Valid Acc 0.8832
    Iter [9260/11250] | Train Loss 0.1193 | Train Acc 1.0000 | Valid Loss 0.4722 | Valid Acc 0.8832
    Iter [9280/11250] | Train Loss 1.1675 | Train Acc 0.5000 | Valid Loss 0.4724 | Valid Acc 0.8830
    Iter [9300/11250] | Train Loss 0.3446 | Train Acc 0.7500 | Valid Loss 0.4712 | Valid Acc 0.8830
    Iter [9320/11250] | Train Loss 1.8077 | Train Acc 0.7500 | Valid Loss 0.4703 | Valid Acc 0.8840
    Iter [9340/11250] | Train Loss 0.3868 | Train Acc 1.0000 | Valid Loss 0.4720 | Valid Acc 0.8816
    Iter [9360/11250] | Train Loss 1.0356 | Train Acc 0.7500 | Valid Loss 0.4748 | Valid Acc 0.8808
    Iter [9380/11250] | Train Loss 0.2114 | Train Acc 1.0000 | Valid Loss 0.4778 | Valid Acc 0.8824
    Iter [9400/11250] | Train Loss 0.1541 | Train Acc 1.0000 | Valid Loss 0.4736 | Valid Acc 0.8842
    Iter [9420/11250] | Train Loss 0.2746 | Train Acc 1.0000 | Valid Loss 0.4719 | Valid Acc 0.8832
    Iter [9440/11250] | Train Loss 0.1338 | Train Acc 1.0000 | Valid Loss 0.4733 | Valid Acc 0.8794
    Iter [9460/11250] | Train Loss 1.1830 | Train Acc 0.5000 | Valid Loss 0.4703 | Valid Acc 0.8806
    Iter [9480/11250] | Train Loss 0.5878 | Train Acc 0.7500 | Valid Loss 0.4717 | Valid Acc 0.8794
    Iter [9500/11250] | Train Loss 0.2411 | Train Acc 1.0000 | Valid Loss 0.4710 | Valid Acc 0.8794
    Iter [9520/11250] | Train Loss 0.9206 | Train Acc 0.7500 | Valid Loss 0.4721 | Valid Acc 0.8812
    Iter [9540/11250] | Train Loss 0.2916 | Train Acc 1.0000 | Valid Loss 0.4720 | Valid Acc 0.8820
    Iter [9560/11250] | Train Loss 0.0672 | Train Acc 1.0000 | Valid Loss 0.4698 | Valid Acc 0.8848
    Iter [9580/11250] | Train Loss 0.4815 | Train Acc 0.7500 | Valid Loss 0.4706 | Valid Acc 0.8860
    Iter [9600/11250] | Train Loss 0.0508 | Train Acc 1.0000 | Valid Loss 0.4668 | Valid Acc 0.8854
    Iter [9620/11250] | Train Loss 0.7052 | Train Acc 0.7500 | Valid Loss 0.4677 | Valid Acc 0.8846
    Iter [9640/11250] | Train Loss 0.3764 | Train Acc 1.0000 | Valid Loss 0.4687 | Valid Acc 0.8864
    Iter [9660/11250] | Train Loss 0.2744 | Train Acc 1.0000 | Valid Loss 0.4675 | Valid Acc 0.8848
    Iter [9680/11250] | Train Loss 0.1250 | Train Acc 1.0000 | Valid Loss 0.4716 | Valid Acc 0.8824
    Iter [9700/11250] | Train Loss 0.6679 | Train Acc 0.7500 | Valid Loss 0.4673 | Valid Acc 0.8844
    Iter [9720/11250] | Train Loss 0.2216 | Train Acc 1.0000 | Valid Loss 0.4651 | Valid Acc 0.8828
    Iter [9740/11250] | Train Loss 0.8262 | Train Acc 0.5000 | Valid Loss 0.4662 | Valid Acc 0.8820
    Iter [9760/11250] | Train Loss 1.1213 | Train Acc 0.5000 | Valid Loss 0.4658 | Valid Acc 0.8810
    Iter [9780/11250] | Train Loss 0.7553 | Train Acc 0.7500 | Valid Loss 0.4655 | Valid Acc 0.8814
    Iter [9800/11250] | Train Loss 1.3534 | Train Acc 0.5000 | Valid Loss 0.4623 | Valid Acc 0.8836
    Iter [9820/11250] | Train Loss 0.1841 | Train Acc 1.0000 | Valid Loss 0.4651 | Valid Acc 0.8812
    Iter [9840/11250] | Train Loss 0.0960 | Train Acc 1.0000 | Valid Loss 0.4643 | Valid Acc 0.8808
    Iter [9860/11250] | Train Loss 1.6293 | Train Acc 0.5000 | Valid Loss 0.4633 | Valid Acc 0.8842
    Iter [9880/11250] | Train Loss 0.5477 | Train Acc 0.7500 | Valid Loss 0.4639 | Valid Acc 0.8862
    Iter [9900/11250] | Train Loss 0.2993 | Train Acc 1.0000 | Valid Loss 0.4649 | Valid Acc 0.8828
    Iter [9920/11250] | Train Loss 1.8394 | Train Acc 0.7500 | Valid Loss 0.4648 | Valid Acc 0.8848
    Iter [9940/11250] | Train Loss 0.4902 | Train Acc 0.7500 | Valid Loss 0.4650 | Valid Acc 0.8848
    Iter [9960/11250] | Train Loss 0.4606 | Train Acc 1.0000 | Valid Loss 0.4629 | Valid Acc 0.8850
    Iter [9980/11250] | Train Loss 0.6730 | Train Acc 0.7500 | Valid Loss 0.4620 | Valid Acc 0.8860
    Iter [10000/11250] | Train Loss 0.1729 | Train Acc 1.0000 | Valid Loss 0.4625 | Valid Acc 0.8852
    Iter [10020/11250] | Train Loss 1.3086 | Train Acc 0.5000 | Valid Loss 0.4662 | Valid Acc 0.8820
    Iter [10040/11250] | Train Loss 0.4644 | Train Acc 1.0000 | Valid Loss 0.4677 | Valid Acc 0.8822
    Iter [10060/11250] | Train Loss 0.5722 | Train Acc 0.7500 | Valid Loss 0.4673 | Valid Acc 0.8814
    Iter [10080/11250] | Train Loss 0.2200 | Train Acc 1.0000 | Valid Loss 0.4591 | Valid Acc 0.8854
    Iter [10100/11250] | Train Loss 0.5675 | Train Acc 0.7500 | Valid Loss 0.4578 | Valid Acc 0.8870
    Iter [10120/11250] | Train Loss 0.3030 | Train Acc 1.0000 | Valid Loss 0.4576 | Valid Acc 0.8860
    Iter [10140/11250] | Train Loss 0.8005 | Train Acc 0.7500 | Valid Loss 0.4605 | Valid Acc 0.8844
    Iter [10160/11250] | Train Loss 0.7034 | Train Acc 0.7500 | Valid Loss 0.4626 | Valid Acc 0.8828
    Iter [10180/11250] | Train Loss 0.0746 | Train Acc 1.0000 | Valid Loss 0.4636 | Valid Acc 0.8822
    Iter [10200/11250] | Train Loss 0.3830 | Train Acc 1.0000 | Valid Loss 0.4621 | Valid Acc 0.8820
    Iter [10220/11250] | Train Loss 1.0225 | Train Acc 0.7500 | Valid Loss 0.4573 | Valid Acc 0.8858
    Iter [10240/11250] | Train Loss 0.3780 | Train Acc 1.0000 | Valid Loss 0.4553 | Valid Acc 0.8874
    Iter [10260/11250] | Train Loss 0.7159 | Train Acc 0.7500 | Valid Loss 0.4556 | Valid Acc 0.8862
    Iter [10280/11250] | Train Loss 0.4555 | Train Acc 1.0000 | Valid Loss 0.4566 | Valid Acc 0.8866
    Iter [10300/11250] | Train Loss 0.7207 | Train Acc 0.7500 | Valid Loss 0.4560 | Valid Acc 0.8870
    Iter [10320/11250] | Train Loss 0.1342 | Train Acc 1.0000 | Valid Loss 0.4560 | Valid Acc 0.8886
    Iter [10340/11250] | Train Loss 0.4264 | Train Acc 0.7500 | Valid Loss 0.4544 | Valid Acc 0.8894
    Iter [10360/11250] | Train Loss 1.0603 | Train Acc 0.5000 | Valid Loss 0.4551 | Valid Acc 0.8880
    Iter [10380/11250] | Train Loss 0.2422 | Train Acc 1.0000 | Valid Loss 0.4552 | Valid Acc 0.8864
    Iter [10400/11250] | Train Loss 0.3837 | Train Acc 0.7500 | Valid Loss 0.4570 | Valid Acc 0.8864
    Iter [10420/11250] | Train Loss 0.3002 | Train Acc 1.0000 | Valid Loss 0.4568 | Valid Acc 0.8868
    Iter [10440/11250] | Train Loss 2.4549 | Train Acc 0.5000 | Valid Loss 0.4557 | Valid Acc 0.8878
    Iter [10460/11250] | Train Loss 0.3255 | Train Acc 1.0000 | Valid Loss 0.4588 | Valid Acc 0.8854
    Iter [10480/11250] | Train Loss 0.9022 | Train Acc 0.7500 | Valid Loss 0.4568 | Valid Acc 0.8842
    Iter [10500/11250] | Train Loss 0.4520 | Train Acc 0.7500 | Valid Loss 0.4546 | Valid Acc 0.8844
    Iter [10520/11250] | Train Loss 0.6378 | Train Acc 0.7500 | Valid Loss 0.4546 | Valid Acc 0.8848
    Iter [10540/11250] | Train Loss 0.2737 | Train Acc 1.0000 | Valid Loss 0.4576 | Valid Acc 0.8822
    Iter [10560/11250] | Train Loss 0.1441 | Train Acc 1.0000 | Valid Loss 0.4578 | Valid Acc 0.8832
    Iter [10580/11250] | Train Loss 0.5796 | Train Acc 0.7500 | Valid Loss 0.4560 | Valid Acc 0.8836
    Iter [10600/11250] | Train Loss 0.9689 | Train Acc 0.7500 | Valid Loss 0.4523 | Valid Acc 0.8852
    Iter [10620/11250] | Train Loss 0.2970 | Train Acc 1.0000 | Valid Loss 0.4536 | Valid Acc 0.8864
    Iter [10640/11250] | Train Loss 0.1739 | Train Acc 1.0000 | Valid Loss 0.4549 | Valid Acc 0.8854
    Iter [10660/11250] | Train Loss 0.3390 | Train Acc 1.0000 | Valid Loss 0.4546 | Valid Acc 0.8856
    Iter [10680/11250] | Train Loss 0.4824 | Train Acc 1.0000 | Valid Loss 0.4607 | Valid Acc 0.8834
    Iter [10700/11250] | Train Loss 0.5850 | Train Acc 0.7500 | Valid Loss 0.4608 | Valid Acc 0.8834
    Iter [10720/11250] | Train Loss 0.4415 | Train Acc 0.7500 | Valid Loss 0.4635 | Valid Acc 0.8796
    Iter [10740/11250] | Train Loss 0.3986 | Train Acc 1.0000 | Valid Loss 0.4584 | Valid Acc 0.8802
    Iter [10760/11250] | Train Loss 0.2657 | Train Acc 1.0000 | Valid Loss 0.4570 | Valid Acc 0.8822
    Iter [10780/11250] | Train Loss 0.3923 | Train Acc 1.0000 | Valid Loss 0.4495 | Valid Acc 0.8888
    Iter [10800/11250] | Train Loss 0.4651 | Train Acc 0.7500 | Valid Loss 0.4478 | Valid Acc 0.8906
    Iter [10820/11250] | Train Loss 0.1926 | Train Acc 1.0000 | Valid Loss 0.4472 | Valid Acc 0.8874
    Iter [10840/11250] | Train Loss 0.7728 | Train Acc 0.5000 | Valid Loss 0.4494 | Valid Acc 0.8862
    Iter [10860/11250] | Train Loss 0.0341 | Train Acc 1.0000 | Valid Loss 0.4469 | Valid Acc 0.8858
    Iter [10880/11250] | Train Loss 0.1808 | Train Acc 1.0000 | Valid Loss 0.4457 | Valid Acc 0.8876
    Iter [10900/11250] | Train Loss 0.4231 | Train Acc 0.7500 | Valid Loss 0.4458 | Valid Acc 0.8868
    Iter [10920/11250] | Train Loss 0.6330 | Train Acc 0.7500 | Valid Loss 0.4461 | Valid Acc 0.8858
    Iter [10940/11250] | Train Loss 0.4058 | Train Acc 0.7500 | Valid Loss 0.4483 | Valid Acc 0.8854
    Iter [10960/11250] | Train Loss 0.0860 | Train Acc 1.0000 | Valid Loss 0.4467 | Valid Acc 0.8860
    Iter [10980/11250] | Train Loss 0.3309 | Train Acc 1.0000 | Valid Loss 0.4462 | Valid Acc 0.8862
    Iter [11000/11250] | Train Loss 0.1287 | Train Acc 1.0000 | Valid Loss 0.4489 | Valid Acc 0.8846
    Iter [11020/11250] | Train Loss 0.1114 | Train Acc 1.0000 | Valid Loss 0.4504 | Valid Acc 0.8866
    Iter [11040/11250] | Train Loss 1.0553 | Train Acc 0.5000 | Valid Loss 0.4486 | Valid Acc 0.8848
    Iter [11060/11250] | Train Loss 0.6951 | Train Acc 0.5000 | Valid Loss 0.4529 | Valid Acc 0.8854
    Iter [11080/11250] | Train Loss 0.2845 | Train Acc 1.0000 | Valid Loss 0.4468 | Valid Acc 0.8850
    Iter [11100/11250] | Train Loss 0.3955 | Train Acc 1.0000 | Valid Loss 0.4444 | Valid Acc 0.8876
    Iter [11120/11250] | Train Loss 0.6192 | Train Acc 1.0000 | Valid Loss 0.4416 | Valid Acc 0.8896
    Iter [11140/11250] | Train Loss 0.0741 | Train Acc 1.0000 | Valid Loss 0.4411 | Valid Acc 0.8892
    Iter [11160/11250] | Train Loss 1.4547 | Train Acc 0.7500 | Valid Loss 0.4431 | Valid Acc 0.8900
    Iter [11180/11250] | Train Loss 0.1664 | Train Acc 1.0000 | Valid Loss 0.4445 | Valid Acc 0.8896
    Iter [11200/11250] | Train Loss 0.4328 | Train Acc 1.0000 | Valid Loss 0.4464 | Valid Acc 0.8856
    Iter [11220/11250] | Train Loss 0.3245 | Train Acc 1.0000 | Valid Loss 0.4442 | Valid Acc 0.8860
    Iter [11240/11250] | Train Loss 0.6281 | Train Acc 0.7500 | Valid Loss 0.4432 | Valid Acc 0.8868
    Iter [11249/11250] | Train Loss 0.2686 | Train Acc 1.0000 | Valid Loss 0.4435 | Valid Acc 0.8846


## Visualization


```python
# Load log file
scratch_train_log = pd.read_csv(os.path.join(log_dir, 'scratch_train_log.csv'), index_col=0, header=None)
fine_tuned_train_log = pd.read_csv(os.path.join(log_dir, 'fine_tuned_train_log.csv'), index_col=0, header=None)
```


```python
# Visualize training log
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,8))

ax1.plot(scratch_train_log.iloc[:,0], label='Scratch Training')
ax1.plot(fine_tuned_train_log.iloc[:,0], label='Fine Tuning')
ax1.set_title('Training Loss Graph', fontsize=15)
ax1.set_xlabel('Iteration', fontsize=15)
ax1.set_ylabel('Loss', fontsize=15)

fig.legend(fontsize=15)

ax2.plot(scratch_train_log.iloc[:,1], label='Scratch Training')
ax2.plot(fine_tuned_train_log.iloc[:,1], label='Fine Tuning')
ax2.set_title('Training Accuracy Graph', fontsize=15)
ax2.set_xlabel('Iteration', fontsize=15)
ax2.set_ylabel('Accuracy', fontsize=15)

plt.show()
```


    
![png](/assets/images/resnet34/train_graph.png)
    



```python
# Visualize validation log
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,8))

ax1.plot(scratch_train_log.iloc[:,2], label='Scratch Training')
ax1.plot(fine_tuned_train_log.iloc[:,2], label='Fine Tuning')
ax1.set_title('Validation Loss Graph', fontsize=15)
ax1.set_xlabel('Iteration', fontsize=15)
ax1.set_ylabel('Loss', fontsize=15)

fig.legend(fontsize=15)

ax2.plot(scratch_train_log.iloc[:,3], label='Scratch Training')
ax2.plot(fine_tuned_train_log.iloc[:,3], label='Fine Tuning')
ax2.set_title('Validation Accuracy Graph', fontsize=15)
ax2.set_xlabel('Iteration', fontsize=15)
ax2.set_ylabel('Accuracy', fontsize=15)

plt.show()
```


    
![png](/assets/images/resnet34/val_graph.png)
    



```python

```
