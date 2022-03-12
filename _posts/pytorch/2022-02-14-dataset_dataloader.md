---
title:  "[Pytorch] batch_dataloader"
excerpt: "batch,epoch 의 개념과 dataloader 학습"

categories:
  - Theory
  - pytorch
tags:
  - pytorch
  - batch
  - dataloader
  - dataset

toc: true
toc_sticky: true
 
date: 2022-02-14
last_modified_at: 2022-02-14
---
# mini batch and dataload

## minibatch and batch size


```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
```


```python
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
```

**epoch** : 전체 훈련 데이터가 학습에 1번 사용된 횟수<br>
**batch_size** : minibatch 크기<br>
**iteration** : 1epoch 내에서 w,b update 횟수

>ex) 데이터가 2000, 배치 크기가 200이면 1배치당 경사하강법 실행 시 w,b update 되므로 1epoch 동안 10번 일어나게 된다

이 데이터의 sample 의 갯수는 5개다.<br>
전체 데이터를 하나의 행렬로  선언하여 전체 데이터에 대해 경사 하강법을 수행해야 학습할 수 있다<br>
하지만 이러한 방법은 데이터가 많을 때 일부에 대해 학습하려 해도 많은 계산량을 필요로 하기 때문에 **전체 데이터를 작은 단위로 나누어 해당 단위로 학습하는 개념이 Mini batch** 이다

미니 배치 학습을 하게 되면 미니 배치 만큼만 데이터를 가져가서 미니 배치에 대한 비용을 계산하고, 경사 하강법을 수행한다<br>
그리고 다음 미니 배치로 넘어가는 형식으로 전체 데이터에 대한 학습이 1회 끝나면 1 에포크가 끝나게 된다

미니 배치는 보통 2의 제곱수로 cpu 와 gpu 의 메모리가 2의 배수이므로 배치 크기가 2의 제곱수일 경우에 데이터 송수신 효율이 높다 

## iteration

iteration 은 한번의 에포크 내에서 이루어지는 매개변수인 가중치 w와 b 의 업데이트 횟수이다. <br>
전체 데이터가 2000일 때 배치 크기를 200으로 한다면 iter의 수는 총 10개이다 -> 한번의 에포크 당 매개변수 업데이트가 10번 이루어진다는것을 의미

## data load

dataset을 먼저 정의하고, dataloader 에 전달하는 방법


```python
from torch.utils.data import TensorDataset # 텐서데이터셋 -> tensor를 입력으로 받는다
from torch.utils.data import DataLoader # 데이터로더
```


```python
x_train  =  torch.FloatTensor([[73,  80,  75], 
                               [93,  88,  93], 
                               [89,  91,  90], 
                               [96,  98,  100],   
                               [73,  66,  70]])  
y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])
```

위 데이터를 텐서형식의 데이터셋으로 저장


```python
dataset = TensorDataset(x_train, y_train)
```

data loader는 기본적으로 2개의 인자를 입력받는데 하나는 dataset, 하나는 minibatch의 크기이다<br>
shuffle 인자는 모델이 데이터셋의 순서에 익숙해지는 것을 방지하여 학습하기 위함이다


```python
dataloader = DataLoader(dataset,batch_size = 2,shuffle = True)
```

optimizer 의 정확한 개념과 기능을 알고 싶었다 

optimizer 의 원리는 학습 데이터 셋을 이용하여 모델을 학습할 때 데이터의  실제 결과와 모델이 예측한 결과를 기반으로 잘 줄일 수 있게 만들어주는 역할을 한다

손실 함수의 결과값을 최소화 하는 모델의 **파라미터(가중치) 를 찾는 것**을 의미한다. 

가장 중요한 것은 optimizer 에 사용할 방법을 지정하여 지정한 방법으로 parameter()(requires_grad = True) 들을 미분하여 backward를 진행해 파라미터를 찾는 것이다


```python
model = nn.Linear(3,1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5) 
```

**Train**


```python
nb_epochs = 20
for epoch in range(nb_epochs + 1):
    
    for idx,samples in enumerate(dataloader):

        x_train, y_train = samples
        prediction = model(x_train)

        loss = F.mse_loss(prediction,y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, idx+1, len(dataloader),
            loss.item()
            ))
        
```

    Epoch    0/20 Batch 1/3 Cost: 34093.496094
    Epoch    0/20 Batch 2/3 Cost: 24011.878906
    Epoch    0/20 Batch 3/3 Cost: 6854.709961
    Epoch    1/20 Batch 1/3 Cost: 1009.365112
    Epoch    1/20 Batch 2/3 Cost: 352.783875
    Epoch    1/20 Batch 3/3 Cost: 114.595100
    Epoch    2/20 Batch 1/3 Cost: 26.848852
    Epoch    2/20 Batch 2/3 Cost: 16.310459
    Epoch    2/20 Batch 3/3 Cost: 5.742991
    Epoch    3/20 Batch 1/3 Cost: 3.812428
    Epoch    3/20 Batch 2/3 Cost: 2.044864
    Epoch    3/20 Batch 3/3 Cost: 0.021691
    Epoch    4/20 Batch 1/3 Cost: 1.782309
    Epoch    4/20 Batch 2/3 Cost: 0.565831
    Epoch    4/20 Batch 3/3 Cost: 4.151369
    Epoch    5/20 Batch 1/3 Cost: 0.350563
    Epoch    5/20 Batch 2/3 Cost: 1.204121
    Epoch    5/20 Batch 3/3 Cost: 7.463378
    Epoch    6/20 Batch 1/3 Cost: 0.852928
    Epoch    6/20 Batch 2/3 Cost: 3.523688
    Epoch    6/20 Batch 3/3 Cost: 1.035993
    Epoch    7/20 Batch 1/3 Cost: 0.129534
    Epoch    7/20 Batch 2/3 Cost: 3.720951
    Epoch    7/20 Batch 3/3 Cost: 0.677050
    Epoch    8/20 Batch 1/3 Cost: 0.255363
    Epoch    8/20 Batch 2/3 Cost: 2.435822
    Epoch    8/20 Batch 3/3 Cost: 4.399394
    Epoch    9/20 Batch 1/3 Cost: 0.271395
    Epoch    9/20 Batch 2/3 Cost: 3.936068
    Epoch    9/20 Batch 3/3 Cost: 0.523137
    Epoch   10/20 Batch 1/3 Cost: 0.124593
    Epoch   10/20 Batch 2/3 Cost: 3.192162
    Epoch   10/20 Batch 3/3 Cost: 4.627527
    Epoch   11/20 Batch 1/3 Cost: 2.953442
    Epoch   11/20 Batch 2/3 Cost: 2.436041
    Epoch   11/20 Batch 3/3 Cost: 0.373144
    Epoch   12/20 Batch 1/3 Cost: 1.864235
    Epoch   12/20 Batch 2/3 Cost: 3.247612
    Epoch   12/20 Batch 3/3 Cost: 0.009394
    Epoch   13/20 Batch 1/3 Cost: 0.402194
    Epoch   13/20 Batch 2/3 Cost: 1.891348
    Epoch   13/20 Batch 3/3 Cost: 5.968439
    Epoch   14/20 Batch 1/3 Cost: 0.713147
    Epoch   14/20 Batch 2/3 Cost: 4.469697
    Epoch   14/20 Batch 3/3 Cost: 0.694767
    Epoch   15/20 Batch 1/3 Cost: 1.457504
    Epoch   15/20 Batch 2/3 Cost: 3.414327
    Epoch   15/20 Batch 3/3 Cost: 0.040704
    Epoch   16/20 Batch 1/3 Cost: 0.191910
    Epoch   16/20 Batch 2/3 Cost: 3.622986
    Epoch   16/20 Batch 3/3 Cost: 0.775080
    Epoch   17/20 Batch 1/3 Cost: 3.791192
    Epoch   17/20 Batch 2/3 Cost: 0.156609
    Epoch   17/20 Batch 3/3 Cost: 0.506409
    Epoch   18/20 Batch 1/3 Cost: 2.644931
    Epoch   18/20 Batch 2/3 Cost: 0.448790
    Epoch   18/20 Batch 3/3 Cost: 3.629195
    Epoch   19/20 Batch 1/3 Cost: 0.428613
    Epoch   19/20 Batch 2/3 Cost: 1.295067
    Epoch   19/20 Batch 3/3 Cost: 7.746899
    Epoch   20/20 Batch 1/3 Cost: 0.993079
    Epoch   20/20 Batch 2/3 Cost: 1.592968
    Epoch   20/20 Batch 3/3 Cost: 5.216989
    


```python
# 임의의 입력 [73, 80, 75]를 선언
new_var =  torch.FloatTensor([[73, 80, 75]]) 
# 입력한 값 [73, 80, 75]에 대해서 예측값 y를 리턴받아서 pred_y에 저장
pred_y = model(new_var) 
print("훈련 후 입력이 73, 80, 75일 때의 예측값 :", pred_y) 
```

    훈련 후 입력이 73, 80, 75일 때의 예측값 : tensor([[152.2070]], grad_fn=<AddmmBackward0>)
    

# Custom dataset

torch.utils.data.Dataset 을 상속받아 커스텀 데이터셋을 만드는 경우도 있다 

custom dataset 을 만들 때 가장 기본적인 구조는 이렇다

<pre><code>
class CustomDataset(torch.utils.data.Dataset): 
  def __init__(self):

  def __len__(self):

  def __getitem__(self, idx): 
  </code>
</pre>

* __init__ : 데이터셋의 전처리를 해주는 부분
* __len__ : 데이터셋의 길이, 총 샘플의 수를 적는 부분
* __getitem__ : 데이터셋에서 특정 1개의 sample 을 가져오는 함수

## Custom dataset으로 선형회귀 구현하기 


```python
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
```

dataset 상속하여 custom dataset 만들기


```python
class CustomDataset(Dataset):
    def __init__(self):
        self.x_data = [[73, 80, 75],
                   [93, 88, 93],
                   [89, 91, 90],
                   [96, 98, 100],
                   [73, 66, 70]]
        self.y_data = [[152], [185], [180], [196], [142]]

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self,idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x,y
```

custom dataset 선언 및 dataloader 선언


```python
dataset = CustomDataset()
dataloader = DataLoader(dataset,batch_size = 2,shuffle = True)

model = torch.nn.Linear(3,1)
optimizer = torch.optim.SGD(model.parameters(),lr = 1e-5)
```

**Train**


```python
nb_epochs = 20
for epoch in range(nb_epochs + 1):
    for idx,sample in enumerate(dataloader):
        x_train,y_train = sample
        prediction = model(x_train)

        loss = F.mse_loss(prediction,y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
        epoch, nb_epochs, idx+1, len(dataloader),
        loss.item()
        ))
```

    Epoch    0/20 Batch 1/3 Cost: 1.420335
    Epoch    0/20 Batch 2/3 Cost: 3.741541
    Epoch    0/20 Batch 3/3 Cost: 0.029400
    Epoch    1/20 Batch 1/3 Cost: 3.538527
    Epoch    1/20 Batch 2/3 Cost: 0.261736
    Epoch    1/20 Batch 3/3 Cost: 0.636397
    Epoch    2/20 Batch 1/3 Cost: 2.610863
    Epoch    2/20 Batch 2/3 Cost: 2.310172
    Epoch    2/20 Batch 3/3 Cost: 0.325013
    Epoch    3/20 Batch 1/3 Cost: 1.695668
    Epoch    3/20 Batch 2/3 Cost: 0.118379
    Epoch    3/20 Batch 3/3 Cost: 5.945349
    Epoch    4/20 Batch 1/3 Cost: 1.400968
    Epoch    4/20 Batch 2/3 Cost: 1.132549
    Epoch    4/20 Batch 3/3 Cost: 4.582798
    Epoch    5/20 Batch 1/3 Cost: 1.380931
    Epoch    5/20 Batch 2/3 Cost: 3.296896
    Epoch    5/20 Batch 3/3 Cost: 0.279284
    Epoch    6/20 Batch 1/3 Cost: 0.198359
    Epoch    6/20 Batch 2/3 Cost: 4.002492
    Epoch    6/20 Batch 3/3 Cost: 0.084752
    Epoch    7/20 Batch 1/3 Cost: 2.036601
    Epoch    7/20 Batch 2/3 Cost: 0.404775
    Epoch    7/20 Batch 3/3 Cost: 3.892396
    Epoch    8/20 Batch 1/3 Cost: 4.321496
    Epoch    8/20 Batch 2/3 Cost: 0.263295
    Epoch    8/20 Batch 3/3 Cost: 0.072672
    Epoch    9/20 Batch 1/3 Cost: 0.269006
    Epoch    9/20 Batch 2/3 Cost: 1.350770
    Epoch    9/20 Batch 3/3 Cost: 6.942151
    Epoch   10/20 Batch 1/3 Cost: 1.580529
    Epoch   10/20 Batch 2/3 Cost: 0.918494
    Epoch   10/20 Batch 3/3 Cost: 4.460413
    Epoch   11/20 Batch 1/3 Cost: 2.986574
    Epoch   11/20 Batch 2/3 Cost: 1.995296
    Epoch   11/20 Batch 3/3 Cost: 0.354971
    Epoch   12/20 Batch 1/3 Cost: 4.045423
    Epoch   12/20 Batch 2/3 Cost: 0.309503
    Epoch   12/20 Batch 3/3 Cost: 0.300292
    Epoch   13/20 Batch 1/3 Cost: 3.630369
    Epoch   13/20 Batch 2/3 Cost: 0.557874
    Epoch   13/20 Batch 3/3 Cost: 0.180200
    Epoch   14/20 Batch 1/3 Cost: 0.314500
    Epoch   14/20 Batch 2/3 Cost: 2.136276
    Epoch   14/20 Batch 3/3 Cost: 4.539651
    Epoch   15/20 Batch 1/3 Cost: 1.208744
    Epoch   15/20 Batch 2/3 Cost: 4.019029
    Epoch   15/20 Batch 3/3 Cost: 0.043483
    Epoch   16/20 Batch 1/3 Cost: 1.794060
    Epoch   16/20 Batch 2/3 Cost: 2.456654
    Epoch   16/20 Batch 3/3 Cost: 0.547382
    Epoch   17/20 Batch 1/3 Cost: 0.127236
    Epoch   17/20 Batch 2/3 Cost: 3.182272
    Epoch   17/20 Batch 3/3 Cost: 4.532957
    Epoch   18/20 Batch 1/3 Cost: 1.366302
    Epoch   18/20 Batch 2/3 Cost: 3.252215
    Epoch   18/20 Batch 3/3 Cost: 0.753815
    Epoch   19/20 Batch 1/3 Cost: 2.464535
    Epoch   19/20 Batch 2/3 Cost: 1.097194
    Epoch   19/20 Batch 3/3 Cost: 3.425798
    Epoch   20/20 Batch 1/3 Cost: 0.445936
    Epoch   20/20 Batch 2/3 Cost: 1.066578
    Epoch   20/20 Batch 3/3 Cost: 7.676813
    


```python

```
