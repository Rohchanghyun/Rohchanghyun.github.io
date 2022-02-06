---
title:  "[Pytorch] Module"
excerpt: "nn.module class 를 통한 생성 공부"

categories:
  - pytorch
tags:
  - nn.Module
  - linearregression

toc: true
toc_sticky: true
 
date: 2022-02-06
last_modified_at: 2022-02-06
---
# Tensor 초기화


##텐서 생성

**데이터로부터 텐서 생성**


```python
import torch
import numpy as np
```


```python
data = [[1,2],[3,4]]
x_data = torch.tensor(data)
```


```python
x_data
```




    tensor([[1, 2],
            [3, 4]])



**Numpy 배열로부터 생성하기**


```python
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
```


```python
x_np
```




    tensor([[1, 2],
            [3, 4]])



**다른 텐서 모양 따오기**


```python
x_ones = torch.ones_like(x_data)
x_ones
```




    tensor([[1, 1],
            [1, 1]])




```python
x_rand = torch.rand_like(x_data,dtype = torch.float)
x_rand
```




    tensor([[0.2966, 0.8495],
            [0.4505, 0.5238]])




```python
x = torch.randn(3,4)
x
```




    tensor([[-1.1679, -0.0261, -0.9898,  0.2107],
            [ 1.6062,  1.5683, -0.2032,  2.2659],
            [ 3.0382,  0.5172,  0.0423, -1.8426]])



#텐서 연산


##텐서 사칙연산

**텐서 사칙연산**

- `+` : [torch.add](https://pytorch.org/docs/stable/generated/torch.add.html?highlight=add#torch.add)
- `-` : [torch.sub](https://pytorch.org/docs/stable/generated/torch.sub.html?highlight=torch%20sub#torch.sub)
- `*` : [torch.mul](https://pytorch.org/docs/stable/generated/torch.mul.html?highlight=torch%20mul#torch.mul)
- `/` : [torch.div](https://pytorch.org/docs/stable/search.html?q=torch.div&check_keywords=yes&area=default#)


```python
a = torch.tensor([5])
b = torch.tensor([7])
```


```python
torch.add(a,b)
```




    tensor([12])




```python
a + b
```




    tensor([12])



피 연산자중 하나라고 tensor 로 입력되면 결과는 tensor


```python
a + 7
```




    tensor([12])



##텐서 인덱싱

- [torch.index_select - PyTorch 공식 문서](https://pytorch.org/docs/stable/generated/torch.index_select.html?highlight=index#torch.index_select)

torch.index_select(input, dim, index, *, out=None) → Tensor

- tensor를 원하는 모양으로 바꾸고 싶을때 [torch.Tensor.view](https://pytorch.org/docs/stable/generated/torch.Tensor.view.html?highlight=view#torch.Tensor.view)라는 함수를 사용하자


```python
A = torch.Tensor([[1, 2],
                  [3, 4]])

# TODO : [1, 3]을 만드세요!

# torch.index_select 함수를 써서 해보세요!
output = torch.index_select(A.t(),0,torch.tensor([0]))
```


```python
output
```




    tensor([[1., 3.]])




```python
A = torch.Tensor([[1, 2],
                  [3, 4]])

# torch.gather 함수를 써서 해보세요!
output = torch.gather(A,0,torch.tensor([[0,1]]))
```

# torch.nn

##nn.Linear

nn.linear -> linear transform 구현

공식 문서에는 들어오는 float32 형 input 데이터에 대해 y = wx + b 형태의 선형 변환을 수행하는 메소드이다

**parameter** 

*   in-features : size of each input sample
*   out-features: size of each output sample
*   bias : if set to `False` the layer will not learn an additive bias default = `True`



**그냥 차원을 맞춰주는 함수가 아닌가?**





```python
from torch import nn

X = torch.Tensor([[1,2],[3,4]])
m = nn.Linear(2,5)
output = m(X)
print(output.size())
```

    torch.Size([2, 5])
    

#nn.Module

- `nn.Module`이라는 상자에 `기능`들을 가득 모아놓은 경우 `basic building block`
- `nn.Module`이라는 상자에 `basic building block`인 `nn.Module`들을 가득 모아놓은 경우 `딥러닝 모델`
- `nn.Module`이라는 상자에 `딥러닝 모델`인 `nn.Module`들을 가득 모아놓은 경우 `더욱 큰 딥러닝 모델`

pytorch 를 통한 신경망 모델 설계


1.   Design yout model using class with variables
2.   construct loss and optim
3.   Train cycle(forward,backward,update)



**torch.nn.Module 상속 해야한다**<br>
**__init__ forward() override**<br>



*   `__init__` : 모델에서 사용될 module(nn.Linear,nn.Conv2d), activation function 정의
*   `forward()`: 실행할 계산을 정의 `backward()`는 알아서 해준다고 함<br>
input 을 넣어 어떤 계산을 진행하여 output을 출력할지 정의한다고 생각





**nn.Module**<br>
pytorch 의 nn 라이브러리는 모든 신경망 모델의 baseclass<br>

*   __init__(self): 이 모델에 사용될 구성품들을 연결하는 메소드
*   forward(self,x): init에서 정의된 구성품들을 연결하는 메소드



##모델의 클래스를 통한 구현


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```


```python
#model = nn.Linear(1,1) 의 단순 선형 모델을 class 로 구현해보자
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,1)# 선형 회귀 함수의 정의

    def forward(self,x):
        return self.linear(x)#__init__에서 선언한 연산 실행 

model = LinearRegressionModel()
```


```python

```
