---
layout : single
title:  "[Theory]view(),reshape() 차이"
excerpt: "view(),reshape() 차이"

categories:
  - Theory
tags:
  - view
  - reshape

toc: true
toc_sticky: true

author_profile: true
sidebar_main: true

date: 2022-06-21
last_modified_at: 2022-06-21
---

# torch.view(), torch.reshape()

공식 문서: 

- torch.view() : [Link](https://pytorch.org/docs/stable/generated/torch.Tensor.view.html)
- torch.reshape() : [Link](https://pytorch.org/docs/stable/generated/torch.reshape.html)

#  사용 방법

- view 와 reshape 모두 tensor의 차원 변환에 사용
- 원하는 1개의 차원에 한해 -1로 설정하면 자동으로 변환 가능한 차원이 설정
  - -1값이 2개 혹은 변환 불가능한 차원일때는 RuntimeError 발생



```python
import torch

x = torch.randn(128,1,28,28)
print(x.size())
print(x.view(-1,28*28).size())
```



```python
import torch

x = torch.randn(128,1,28,28)
print(x.size())
print(x.reshape(-1,28*28).size())
```

결과

```python
torch.Size([128, 1, 28, 28])
torch.Size([128, 784])
```



## 두 함수의 차이

### contiguous

- view() -> contiguous 속성을 만족하지 않는 텐서에 적용 불가
- reshape() -> contiguous 속성을 만족하지 않는 텐서에 적용 가능

### contiguous 속성

- contiguous : 메모리 내에서 자료형 저장 상태

<p align="center"><img src="/assets/images/Theory/view/figure_1.png"></p>



