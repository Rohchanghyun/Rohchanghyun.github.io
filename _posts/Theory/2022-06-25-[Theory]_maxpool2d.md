---
layout : single
title:  "[Theory]Max pooling"
excerpt: "Maxpool 에 대한 설명"

categories:
  - Theory
tags:
  - Maxpool2d

toc: true
toc_sticky: true

author_profile: true
sidebar_main: true

date: 2022-06-25
last_modified_at: 2022-06-25
---

# Maxpool2d()

공식문서 : [Link](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html)

## Parameters

- kernel_size
- stride : default = 1
- padding : default = 0
- dilation : 커널 사이 간격 사이즈 조절
- return_indices : True 일 경우 최대 인덱스 반환
- ceil_mode



## pooling의 종류

- max pooling: 정해진 크기 안에서 가장 큰 값만 뽑아낸다
- average pooling : 정해진 크기 안의 값들의 평균을 뽑아낸다



<p align="center"><img src="/assets/images/Theory/maxpool/figure_1.png"></p>

## pooling의 목적

- input size 줄여준다(down sampling) -> 필요없는 parameter 줄여 overfitting 방지
- 일반화 성능 향상

