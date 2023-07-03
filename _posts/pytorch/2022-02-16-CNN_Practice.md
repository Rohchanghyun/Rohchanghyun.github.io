---
layout : single
title:  "[Pytorch] CNN Practice"
excerpt: "pytorch 를 통한 CNN model 구현"

categories:
  - pytorch
tags:
  - CNN

toc: true
toc_sticky: true

author_profile: true
sidebar_main: true
 
date: 2022-02-16
last_modified_at: 2022-02-16
---
# My CNN Model


```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
```


```python
from torchvision import datasets,transforms
mnist_train = datasets.MNIST(root='./data/',train=True,transform=transforms.ToTensor(),download=True)
mnist_test = datasets.MNIST(root='./data/',train=False,transform=transforms.ToTensor(),download=True)
print ("mnist_train:\n",mnist_train,"\n")
print ("mnist_test:\n",mnist_test,"\n")
print ("Done.")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

    mnist_train:
     Dataset MNIST
        Number of datapoints: 60000
        Root location: ./data/
        Split: Train
        StandardTransform
    Transform: ToTensor() 
    
    mnist_test:
     Dataset MNIST
        Number of datapoints: 10000
        Root location: ./data/
        Split: Test
        StandardTransform
    Transform: ToTensor() 
    
    Done.
    


```python
learning_rate = 0.001
training_epochs = 10
batch_size = 600
```


```python
dataloader =  torch.utils.data.DataLoader(dataset = mnist_train,batch_size = batch_size,shuffle = True)
```


```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = []

        #layer1
        self.layers.append(torch.nn.Conv2d(in_channels = 1,out_channels = 32,kernel_size = 3,padding = 1))
        self.layers.append(torch.nn.ReLU())
        self.layers.append(torch.nn.MaxPool2d(2))

        #layer2
        self.layers.append(torch.nn.Conv2d(in_channels = 32,out_channels = 64,kernel_size = 3,padding = 1))
        self.layers.append(torch.nn.ReLU())
        self.layers.append(torch.nn.MaxPool2d(2))

        #layer3
        self.fc = torch.nn.Linear(7 * 7 * 64, 10, bias=True)
        self.net = torch.nn.Sequential()

        #layer 1,2 sequential 에 넣음
        for l_idx,layer in enumerate(self.layers):
            layer_name = "%s_%02d"%(type(layer).__name__.lower(),l_idx)
            self.net.add_module(layer_name,layer)

        self.init_param() # initialize parameters

    def init_param(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d): # init conv
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m,nn.Linear): # lnit dense
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

        #layer 3 view 뒤에 linear 실행
    def forward(self,x):
        out =  self.net(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
```


```python
model = CNN().to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
```


```python
total_batch = len(dataloader)
print('총 배치의 수 : {}'.format(total_batch))
```

    총 배치의 수 : 100
    


```python
for epoch in range(training_epochs):
    avg_cost = 0
    for x,y in dataloader:
        x_train = x
        y_train = y

        x_train = x_train.cuda()
        y_train = y_train.cuda()

        prediction = model(x_train)

        loss = criterion(prediction,y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_cost += loss / total_batch

    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))


```

    [Epoch:    1] cost = 0.385076791
    [Epoch:    2] cost = 0.095616281
    [Epoch:    3] cost = 0.065636225
    [Epoch:    4] cost = 0.0524492934
    [Epoch:    5] cost = 0.0437104665
    [Epoch:    6] cost = 0.0382238925
    [Epoch:    7] cost = 0.0326808654
    [Epoch:    8] cost = 0.0290127192
    [Epoch:    9] cost = 0.0277520046
    [Epoch:   10] cost = 0.0250096172
    [Epoch:   11] cost = 0.0217923447
    [Epoch:   12] cost = 0.0199851617
    [Epoch:   13] cost = 0.0186407752
    [Epoch:   14] cost = 0.0158275999
    [Epoch:   15] cost = 0.0141217988
    

# Wikidocs CNN Model


```python
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
```


```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 랜덤 시드 고정
torch.manual_seed(777)

# GPU 사용 가능일 경우 랜덤 시드 고정
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
```


```python
learning_rate = 0.001
training_epochs = 15
batch_size = 100
```


```python
mnist_train = dsets.MNIST(root='MNIST_data/', # 다운로드 경로 지정
                          train=True, # True를 지정하면 훈련 데이터로 다운로드
                          transform=transforms.ToTensor(), # 텐서로 변환
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/', # 다운로드 경로 지정
                         train=False, # False를 지정하면 테스트 데이터로 다운로드
                         transform=transforms.ToTensor(), # 텐서로 변환
                         download=True)
```

    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to MNIST_data/MNIST/raw/train-images-idx3-ubyte.gz
    


      0%|          | 0/9912422 [00:00<?, ?it/s]


    Extracting MNIST_data/MNIST/raw/train-images-idx3-ubyte.gz to MNIST_data/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to MNIST_data/MNIST/raw/train-labels-idx1-ubyte.gz
    


      0%|          | 0/28881 [00:00<?, ?it/s]


    Extracting MNIST_data/MNIST/raw/train-labels-idx1-ubyte.gz to MNIST_data/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to MNIST_data/MNIST/raw/t10k-images-idx3-ubyte.gz
    


      0%|          | 0/1648877 [00:00<?, ?it/s]


    Extracting MNIST_data/MNIST/raw/t10k-images-idx3-ubyte.gz to MNIST_data/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to MNIST_data/MNIST/raw/t10k-labels-idx1-ubyte.gz
    


      0%|          | 0/4542 [00:00<?, ?it/s]


    Extracting MNIST_data/MNIST/raw/t10k-labels-idx1-ubyte.gz to MNIST_data/MNIST/raw
    
    


```python
data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)
```


```python
class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        # 첫번째층
        # ImgIn shape=(?, 28, 28, 1)
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 두번째층
        # ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 전결합층 7x7x64 inputs -> 10 outputs
        self.fc = torch.nn.Linear(7 * 7 * 64, 10, bias=True)

        # 전결합층 한정으로 가중치 초기화
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)   # 전결합층을 위해서 Flatten
        out = self.fc(out)
        return out
```


```python
# CNN 모델 정의
model = CNN().to(device)
```


```python
criterion = torch.nn.CrossEntropyLoss().to(device)    # 비용 함수에 소프트맥스 함수 포함되어져 있음.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
```


```python
total_batch = len(data_loader)
print('총 배치의 수 : {}'.format(total_batch))
```

    총 배치의 수 : 600
    


```python
for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader: # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y느 ㄴ레이블.
        # image is already size of (28x28), no reshape
        # label is not one-hot encoded
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))
```

    [Epoch:    1] cost = 0.225602433
    [Epoch:    2] cost = 0.0630259961
    [Epoch:    3] cost = 0.0462435372
    [Epoch:    4] cost = 0.0374970697
    [Epoch:    5] cost = 0.0314780995
    [Epoch:    6] cost = 0.0262787808
    [Epoch:    7] cost = 0.0219403766
    [Epoch:    8] cost = 0.0184826776
    [Epoch:    9] cost = 0.0160270929
    [Epoch:   10] cost = 0.0135485623
    [Epoch:   11] cost = 0.010235521
    [Epoch:   12] cost = 0.0099008102
    [Epoch:   13] cost = 0.00884327386
    [Epoch:   14] cost = 0.00605778443
    [Epoch:   15] cost = 0.00655947533
    


```python

```
