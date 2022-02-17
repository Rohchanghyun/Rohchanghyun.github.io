---
title:  "[Pytorch] Imagenet Practice"
excerpt: "pre-trained ResNet34 를 통한 Imagenet의 데이터 훈련 "

categories:
  - pytorch
tags:
  - ResNet
  - drive_mount
  - kaggle_dataset

toc: true
toc_sticky: true
 
date: 2022-02-17
last_modified_at: 2022-02-17
---



```python
!pip install --upgrade --force-reinstall --no-deps kaggle
```

    Collecting kaggle
      Downloading kaggle-1.5.12.tar.gz (58 kB)
    [K     |████████████████████████████████| 58 kB 1.8 MB/s 
    [?25hBuilding wheels for collected packages: kaggle
      Building wheel for kaggle (setup.py) ... [?25l[?25hdone
      Created wheel for kaggle: filename=kaggle-1.5.12-py3-none-any.whl size=73051 sha256=17d6294af68d8ebe30614dc0b532c54f553b59c13d570451b8c5cfe2c0584bbb
      Stored in directory: /root/.cache/pip/wheels/62/d6/58/5853130f941e75b2177d281eb7e44b4a98ed46dd155f556dc5
    Successfully built kaggle
    Installing collected packages: kaggle
      Attempting uninstall: kaggle
        Found existing installation: kaggle 1.5.12
        Uninstalling kaggle-1.5.12:
          Successfully uninstalled kaggle-1.5.12
    Successfully installed kaggle-1.5.12
    


```python
!pip install kaggle
from google.colab import files
files.upload()
```

    Requirement already satisfied: kaggle in /usr/local/lib/python3.7/dist-packages (1.5.12)
    Requirement already satisfied: six>=1.10 in /usr/local/lib/python3.7/dist-packages (from kaggle) (1.15.0)
    Requirement already satisfied: certifi in /usr/local/lib/python3.7/dist-packages (from kaggle) (2021.10.8)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.7/dist-packages (from kaggle) (4.62.3)
    Requirement already satisfied: urllib3 in /usr/local/lib/python3.7/dist-packages (from kaggle) (1.24.3)
    Requirement already satisfied: python-slugify in /usr/local/lib/python3.7/dist-packages (from kaggle) (5.0.2)
    Requirement already satisfied: python-dateutil in /usr/local/lib/python3.7/dist-packages (from kaggle) (2.8.2)
    Requirement already satisfied: requests in /usr/local/lib/python3.7/dist-packages (from kaggle) (2.23.0)
    Requirement already satisfied: text-unidecode>=1.3 in /usr/local/lib/python3.7/dist-packages (from python-slugify->kaggle) (1.3)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests->kaggle) (2.10)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests->kaggle) (3.0.4)
    



<input type="file" id="files-9777bff1-c81b-408d-889c-3c25307ad697" name="files[]" multiple disabled
   style="border:none" />
<output id="result-9777bff1-c81b-408d-889c-3c25307ad697">
 Upload widget is only available when the cell has been executed in the
 current browser session. Please rerun this cell to enable.
 </output>
 <script src="/nbextensions/google.colab/files.js"></script> 



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-6-8090cb07c344> in <module>()
          1 get_ipython().system('pip install kaggle')
          2 from google.colab import files
    ----> 3 files.upload()
    

    /usr/local/lib/python3.7/dist-packages/google/colab/files.py in upload()
         61   result = _output.eval_js(
         62       'google.colab._files._uploadFiles("{input_id}", "{output_id}")'.format(
    ---> 63           input_id=input_id, output_id=output_id))
         64   files = _collections.defaultdict(_six.binary_type)
         65   # Mapping from original filename to filename as saved locally.
    

    /usr/local/lib/python3.7/dist-packages/google/colab/output/_js.py in eval_js(script, ignore_result, timeout_sec)
         38   if ignore_result:
         39     return
    ---> 40   return _message.read_reply_from_input(request_id, timeout_sec)
         41 
         42 
    

    /usr/local/lib/python3.7/dist-packages/google/colab/_message.py in read_reply_from_input(message_id, timeout_sec)
         99     reply = _read_next_input_message()
        100     if reply == _NOT_READY or not isinstance(reply, dict):
    --> 101       time.sleep(0.025)
        102       continue
        103     if (reply.get('type') == 'colab_reply' and
    

    KeyboardInterrupt: 



```python
ls -1ha kaggle.json
```

    kaggle.json
    

![image.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAALgAAAAoCAYAAABNVTCEAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAAG5SURBVHhe7dwNaoNAEIZh7alyn5hTGc9qGXDIMKw/EbdrP94HlmrGXSP7abaFtH88HnMHiPpZfgKSCDikEXBII+CQRsAhjYBDGgGHNAIOaQQc0gg4pP15wKdp6vq+X/basfdRU+3xcQxPcEgj4JU8n89lCy3dIuD549z2va2J9dJxXrflUKm+xfvGMb61dc48fsla7f1+7/bFR/OA2yTFp53veytNYjxmrz6O4/LqMfn8PsZVvzeUxs/XEI/JNTMMw2YdH00D7hMZ5f0s98nH57qF4e7i+7UndL4+Qnxes4DniYxsQr21EM9v7cq/+nhgraG+JgG3cK+FxibeQuCthXh+b/N83ReffEyCXl+TgPsa8szk5n55jFy3m+muYtBRR9M1+N7krtW8nzXbzmL99Xotr55X+sRZe2979vrZzR+Pse3SNeKYpgE3OeSl8OYJN1YrTXyuf7u0sD5+fm92k1y1RCmNn68jHpNr+M6//Fa9TXyUQ7BXr8nPTTDvgX8bAWnNlyhATQQc0gg4pBFwSCPgkEbAIY2AQxoBhzQCDmkEHNIIOKQRcEgj4JBGwCGNgEMaAYewrvsFBlcUV2GPRsUAAAAASUVORK5CYII=)


```python
!pip install -q kaggle
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/

# Permission Warning 방지
!chmod 600 ~/.kaggle/kaggle.json
```


```python
#!kaggle datasets download -d ambityga/imagenet100
```

    404 - Not Found
    


```python
!ls
```

     imagenet100.zip   imagenet_resnet.ipynb  'kaggle (1).json'   kaggle.json
    


```python
!unzip imagenet100.zip
```


```python
cd drive/My\Drive/imagenet

```


```python
mkdir drive/My\Drive/imagenet
```


```python
cd drive/My\Drive/imagenet
```


```python
from torchvision import models
import torch

resnet34_pretrained = models.resnet34(pretrained=True)

print(resnet34_pretrained)
```

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
      (fc): Linear(in_features=512, out_features=1000, bias=True)
    )
    


```python
num_classes = 25
num_ftrs = resnet34_pretrained.fc.in_features
resnet34_pretrained.fc = torch.nn.Linear(num_ftrs, num_classes)

device = torch.device('cuda:0')
resnet34_pretrained.to(device)
```

![image.png](data:image/png;base64,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)


```python
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split,DataLoader
import os
import os.path
```


```python
trans = transforms.Compose([transforms.ToTensor(),
                            transforms.Resize((224,224))])

dataset = ImageFolder(root = '/content/drive/MyDrive/imagenet/train.X1',transform = trans)
```


```python

val_num = int(len(dataset) * 0.1)
train_num = int(len(dataset)) - train_num

print(train_num,val_num)
```

    29250 3250
    


```python
train_data,val_data = random_split(dataset,[train_num,val_num])
```


```python
train_data.__getitem__(2)
```




    (tensor([[[0.3980, 0.4118, 0.4255,  ..., 0.0824, 0.1000, 0.1093],
              [0.4010, 0.4147, 0.4255,  ..., 0.0676, 0.0868, 0.1064],
              [0.4069, 0.4167, 0.4304,  ..., 0.0525, 0.0681, 0.0926],
              ...,
              [0.5730, 0.5775, 0.6206,  ..., 0.3049, 0.2971, 0.2946],
              [0.5569, 0.5642, 0.5956,  ..., 0.2995, 0.2936, 0.2892],
              [0.5451, 0.5627, 0.5676,  ..., 0.2966, 0.2907, 0.2863]],
     
             [[0.5275, 0.5412, 0.5549,  ..., 0.1059, 0.1235, 0.1328],
              [0.5304, 0.5441, 0.5549,  ..., 0.0912, 0.1103, 0.1299],
              [0.5363, 0.5461, 0.5598,  ..., 0.0760, 0.0917, 0.1162],
              ...,
              [0.4564, 0.4608, 0.5039,  ..., 0.3284, 0.3206, 0.3181],
              [0.4471, 0.4544, 0.4887,  ..., 0.3230, 0.3172, 0.3127],
              [0.4353, 0.4529, 0.4681,  ..., 0.3201, 0.3142, 0.3098]],
     
             [[0.5000, 0.5137, 0.5275,  ..., 0.0902, 0.1078, 0.1172],
              [0.5029, 0.5167, 0.5275,  ..., 0.0755, 0.0946, 0.1142],
              [0.5088, 0.5186, 0.5324,  ..., 0.0603, 0.0760, 0.1005],
              ...,
              [0.3005, 0.3049, 0.3480,  ..., 0.3284, 0.3206, 0.3181],
              [0.2912, 0.2985, 0.3333,  ..., 0.3230, 0.3172, 0.3127],
              [0.2824, 0.3000, 0.3157,  ..., 0.3201, 0.3142, 0.3098]]]), 4)




```python
trainloader = DataLoader(train_data,batch_size = 64,shuffle = True)
valloader = DataLoader(val_data,batch_size = 64,shuffle = True)
```


```python
dataiter = iter(trainloader)
images,labels = dataiter.next()
print(labels)
```

    tensor([12, 16, 22,  9, 13,  2,  4,  3, 23, 21,  2,  1,  8, 14, 11, 12,  4,  9,
            23, 21,  1, 14, 10, 15,  9, 18, 21, 15, 15, 22,  9,  2,  2, 11, 17, 11,
            14,  0, 18,  0, 14, 22,  3,  4, 12, 11,  5, 10, 18, 24,  4,  2, 13, 19,
            24, 24, 17, 19, 14, 18, 23, 19, 12, 16])
    


```python
learning_rate = 0.001
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adadelta(resnet34_pretrained.parameters(),lr = learning_rate)
```


```python
total_batch = len(trainloader)
print('총 배치의 수 : {}'.format(total_batch))
```

    총 배치의 수 : 458
    


```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
```


```python
nb_epochs = 10
for epoch in range(nb_epochs):
    avg_cost = 0
    for x,y in trainloader:
        x_train = x
        y_train = y

        x_train = x_train.cuda()
        y_train = y_train.cuda()

        prediction = resnet34_pretrained(x_train)

        loss = criterion(prediction,y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_cost += loss / total_batch

    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))
```

    [Epoch:    1] cost = 0.408272564
    [Epoch:    2] cost = 0.395183712
    [Epoch:    3] cost = 0.381901205
    [Epoch:    4] cost = 0.371450126
    [Epoch:    5] cost = 0.351915836
    [Epoch:    6] cost = 0.340945542
    


```python
correct = 0
total = len(valloader)
for x,y in valloader:
    x_val = x
    y_val = y

    x_val = x_val.cuda()
    y_val = y_val.cuda()

    prediction = resnet34_pretrained(x_val)
    if y_val == prediction:
        correct += 1

print("accuracy:",correct/total)



```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-2-d5e2a692d4d6> in <module>()
          1 correct = 0
    ----> 2 total = len(valloader)
          3 for x,y in valloader:
          4     x_val = x
          5     y_val = y
    

    NameError: name 'valloader' is not defined



```python

```
