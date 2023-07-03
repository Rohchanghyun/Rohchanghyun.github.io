---
layout : single
title:  "[Pytorch] Convolutional Neural Network"
excerpt: "CNN 구현"

categories:
  - pytorch
tags:
  - CNN

toc: true
toc_sticky: true

author_profile: true
sidebar_main: true

date: 2022-06-28
last_modified_at: 2022-06-28
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


# Convolutional Neural Network(CNN)



```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
%matplotlib inline
%config InlineBackend.figure_format='retina'
print ("PyTorch version:[%s]."%(torch.__version__))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print ("device:[%s]."%(device))
```

<pre>
PyTorch version:[1.13.1+cu116].
device:[cuda:0].
</pre>
# Dataset



```python
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
```


```python
train_data = datasets.CIFAR10(
    root = './data',
    train = True,
    download = True,
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor() 
    ])
)

test_data = datasets.CIFAR10(
    root = './data',
    train = False,
    download = True,
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor() 
    ])
)
print ("CIFAR10_train:\n",train_data,"\n")
print ("CIFAR10_test:\n",test_data,"\n")
print ("Done.")
```

<pre>
Files already downloaded and verified
Files already downloaded and verified
CIFAR10_train:
 Dataset CIFAR10
    Number of datapoints: 50000
    Root location: ./data
    Split: Train
    StandardTransform
Transform: Compose(
               Grayscale(num_output_channels=1)
               ToTensor()
           ) 

CIFAR10_test:
 Dataset CIFAR10
    Number of datapoints: 10000
    Root location: ./data
    Split: Test
    StandardTransform
Transform: Compose(
               Grayscale(num_output_channels=1)
               ToTensor()
           ) 

Done.
</pre>

```python
BATCH_SIZE = 16
```


```python
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE,shuffle=True,num_workers=1)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE,shuffle=True,num_workers=1)
print ("Done.")
list(train_loader)[0][0].shape
```

<pre>
Done.
</pre>
<pre>
torch.Size([16, 1, 32, 32])
</pre>
# Model



```python
class custom_model_cnn(nn.Module):
    def __init__(self,xdim = [1,32,32],cdim = [32,64],hdim = [1028,512],ydim = 10,ksize = 3):
        super(custom_model_cnn,self).__init__()
        self.xdim = xdim
        self.cdim = cdim
        self.hdim = hdim
        self.ydim = ydim
        self.ksize = ksize

        self.layers = []
        prev_cdim = self.xdim[0]
        for c in self.cdim:
            self.layers.append(nn.Conv2d(prev_cdim,c,kernel_size=self.ksize,stride = (1,1),padding = self.ksize//2))
            self.layers.append(nn.BatchNorm2d(c))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.MaxPool2d(kernel_size=(2,2),stride = (2,2)))
            self.layers.append(nn.Dropout())
            prev_cdim = c

        self.layers.append(nn.Flatten())
        prev_hdim = prev_cdim * (self.xdim[1]//2**len(self.cdim)) * (self.xdim[2]//2**len(self.cdim))
        
        for h in hdim:
            self.layers.append(nn.Linear(prev_hdim,h))
            self.layers.append(nn.ReLU())
            prev_hdim = h
        
        self.layers.append(nn.Linear(prev_hdim,self.ydim))

        self.net = nn.Sequential()
        for m_idx,layer in enumerate(self.layers):
            layer_name = "%s_%02d"%(type(layer).__name__.lower(),m_idx)
            self.net.add_module(layer_name,layer)


        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d): # init conv
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m,nn.BatchNorm2d): # init BN
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear): # lnit dense
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
        
    def forward(self,x):
        return self.net(x)
    
```


```python
custom_model = custom_model_cnn()
custom_model.to(device)
criterion = nn.CrossEntropyLoss()
optm = optim.Adam(custom_model.parameters(),lr=1e-3)
EPOCHS = 20
```


```python
np.set_printoptions(precision=3)
n_param = 0
for p_idx,(param_name,param) in enumerate(custom_model.named_parameters()):
    if param.requires_grad:
        param_numpy = param.detach().cpu().numpy() # to numpy array 
        n_param += len(param_numpy.reshape(-1))
        print ("[%d] name:[%s] shape:[%s]."%(p_idx,param_name,param_numpy.shape))
        print ("    val:%s"%(param_numpy.reshape(-1)[:5]))
print ("Total number of parameters:[%s]."%(format(n_param,',d')))
```

<pre>
[0] name:[net.conv2d_00.weight] shape:[(32, 1, 3, 3)].
    val:[-0.209  0.025 -0.421 -0.301 -0.601]
[1] name:[net.conv2d_00.bias] shape:[(32,)].
    val:[0. 0. 0. 0. 0.]
[2] name:[net.batchnorm2d_01.weight] shape:[(32,)].
    val:[1. 1. 1. 1. 1.]
[3] name:[net.batchnorm2d_01.bias] shape:[(32,)].
    val:[0. 0. 0. 0. 0.]
[4] name:[net.conv2d_05.weight] shape:[(64, 32, 3, 3)].
    val:[-0.072  0.139 -0.051 -0.044  0.106]
[5] name:[net.conv2d_05.bias] shape:[(64,)].
    val:[0. 0. 0. 0. 0.]
[6] name:[net.batchnorm2d_06.weight] shape:[(64,)].
    val:[1. 1. 1. 1. 1.]
[7] name:[net.batchnorm2d_06.bias] shape:[(64,)].
    val:[0. 0. 0. 0. 0.]
[8] name:[net.linear_11.weight] shape:[(1028, 4096)].
    val:[-0.023  0.018  0.014 -0.028  0.015]
[9] name:[net.linear_11.bias] shape:[(1028,)].
    val:[0. 0. 0. 0. 0.]
[10] name:[net.linear_13.weight] shape:[(512, 1028)].
    val:[ 0.066 -0.014 -0.001  0.048 -0.082]
[11] name:[net.linear_13.bias] shape:[(512,)].
    val:[0. 0. 0. 0. 0.]
[12] name:[net.linear_15.weight] shape:[(10, 512)].
    val:[ 0.044 -0.051  0.003 -0.064  0.078]
[13] name:[net.linear_15.bias] shape:[(10,)].
    val:[0. 0. 0. 0. 0.]
Total number of parameters:[4,762,702].
</pre>

```python
custom_model
```

<pre>
custom_model_cnn(
  (net): Sequential(
    (conv2d_00): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (batchnorm2d_01): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu_02): ReLU()
    (maxpool2d_03): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    (dropout_04): Dropout(p=0.5, inplace=False)
    (conv2d_05): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (batchnorm2d_06): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu_07): ReLU()
    (maxpool2d_08): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    (dropout_09): Dropout(p=0.5, inplace=False)
    (flatten_10): Flatten(start_dim=1, end_dim=-1)
    (linear_11): Linear(in_features=4096, out_features=1028, bias=True)
    (relu_12): ReLU()
    (linear_13): Linear(in_features=1028, out_features=512, bias=True)
    (relu_14): ReLU()
    (linear_15): Linear(in_features=512, out_features=10, bias=True)
  )
)
</pre>
# Evaluation function



```python
def func_eval(model,data_iter,device):
    with torch.no_grad():
        n_total,n_correct = 0,0
        model.eval() # evaluate (affects DropOut and BN)
        for batch_in,batch_out in data_iter:
            y_trgt = batch_out.to(device)
            model_pred = model(batch_in.view(-1,1,32,32).to(device))
            _,y_pred = torch.max(model_pred.data,1)
            n_correct += (y_pred==y_trgt).sum().item()
            n_total += batch_in.size(0)
        val_accr = (n_correct/n_total)
        model.train() # back to train mode 
    return val_accr
print ("Done")
```

<pre>
Done
</pre>
# Train



```python
custom_model.init_params()
custom_model.train()
for epoch in range(EPOCHS):
    loss_val_sum = 0
    for x,y in train_loader:
        x = x.to(device)
        y = y.to(device)

        pred = custom_model.forward(x.view(-1,1,32,32))
        loss = criterion(pred,y)

        optm.zero_grad()
        loss.backward()
        optm.step()

        loss_val_sum += loss

    loss_val_avg = loss_val_sum / len(train_loader)


    train_accr = func_eval(custom_model,train_loader,device)
    test_accr = func_eval(custom_model,test_loader,device)
    print ("epoch:[%d] loss:[%.3f] train_accr:[%.3f] test_accr:[%.3f]."%
            (epoch,loss_val_avg,train_accr,test_accr))
```

<pre>
epoch:[0] loss:[1.706] train_accr:[0.546] test_accr:[0.534].
epoch:[1] loss:[1.347] train_accr:[0.614] test_accr:[0.597].
epoch:[2] loss:[1.230] train_accr:[0.645] test_accr:[0.615].
epoch:[3] loss:[1.144] train_accr:[0.689] test_accr:[0.656].
epoch:[4] loss:[1.083] train_accr:[0.704] test_accr:[0.664].
epoch:[5] loss:[1.039] train_accr:[0.725] test_accr:[0.671].
epoch:[6] loss:[1.006] train_accr:[0.735] test_accr:[0.681].
epoch:[7] loss:[0.969] train_accr:[0.742] test_accr:[0.685].
epoch:[8] loss:[0.945] train_accr:[0.746] test_accr:[0.686].
epoch:[9] loss:[0.916] train_accr:[0.766] test_accr:[0.692].
epoch:[10] loss:[0.895] train_accr:[0.774] test_accr:[0.697].
epoch:[11] loss:[0.871] train_accr:[0.784] test_accr:[0.695].
epoch:[12] loss:[0.853] train_accr:[0.780] test_accr:[0.701].
epoch:[13] loss:[0.834] train_accr:[0.798] test_accr:[0.706].
epoch:[14] loss:[0.815] train_accr:[0.807] test_accr:[0.705].
epoch:[15] loss:[0.801] train_accr:[0.827] test_accr:[0.720].
epoch:[16] loss:[0.775] train_accr:[0.828] test_accr:[0.717].
epoch:[17] loss:[0.756] train_accr:[0.829] test_accr:[0.713].
epoch:[18] loss:[0.744] train_accr:[0.840] test_accr:[0.714].
epoch:[19] loss:[0.729] train_accr:[0.856] test_accr:[0.724].
</pre>

```python
import random

sample_indices = 25
sample_num = random.sample(range(1,10000),sample_indices)

fig = plt.figure(figsize=(10,10))

for idx,f_idx in enumerate(sample_num):
    test_x,test_y = test_data.__getitem__(f_idx)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor() 
    ])
    
    with torch.no_grad():
        custom_model.eval() # to evaluation mode 
        y_pred = custom_model.forward(test_x.view(-1,1,32,32).to(device))
    y_pred = y_pred.argmax()

    temp = fig.add_subplot(5, 5, 1 + idx)
    temp.imshow(test_x.squeeze(),cmap = 'gray')
    temp.axis('off')
    temp.set_title(f"Pred:{y_pred}, Label:{test_y}")

plt.show()    
print ("Done")


```

<p align="center"><img src="/assets/images/Pytorch/cnn/figure_1.png"></p>

<pre>
<Figure size 720x720 with 25 Axes>
</pre>
<pre>
Done
</pre>

```python
```


```python
```
