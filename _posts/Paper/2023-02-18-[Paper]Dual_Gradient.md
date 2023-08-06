---
layout : single
title:  "Dual Gradient Based Snow Attentive Desnowing"
excerpt: "Dual Gradient Based Snow Attentive Desnowing 논문 정리"

categories:
  - Paper
tags:
  - dehazing
  - dual gradient

toc: true
toc_sticky: true

author_profile: true
sidebar_main: true

date: 2023-02-18
last_modified_at: 2023-02-18
---

## Abstract

눈 입자를 특성화하여 장면에서 눈을 정확하게 제거할 수 있는 새로운 dual gradient-based desnowing 알고리즘을 제안.

  

- 이미지에서 눈의 위치를 파악하기 위해 눈 분류를 사용하여 추정할 수 있는 gradient 기반 눈 활성화 맵을 제시.
- 눈 입자의 모양과 궤적에서 다양한 패턴을 인식하기 위해 gradient 기반 snow edge map 도입

  → 이 두가지 gradient를 사용하여 desnowing에 사용되는 정확한 snow attention mask를 추정.

- 다양한 수준의 눈 투명도를 처리하여 desnowing중에 이미지 컨텍스트 정보가 손실되지 않도록 하는 반투명 인식 컨텍스트 복원 네트워크를 제안

  

## Introduction

deraining → 심층 신경망을 사용하여 강우량을 추정하는 것을 목표

dehazing → 아지랑이 바로 뒤에 있는 정보를 복원하는 것을 목표

  

각 작업은 이미지의 가시성을 개선하여 컴퓨터 비전 응용 프로그램의 성능을 향상시키는데 중요

desnowing의 문제점

- 눈 입자는 다양한 모양과 크기 패턴을 가지고 있기 떄문에 눈 입자를 추정하는것은 빗줄기보다 더 어렵다.
- 눈 입자는 이미지에서 불균일하게 분포
- 다양한 수준의 눈 투명도

  

본 논문에서는 기존 image degradation 모델을 사용하여 snow image를 꺠끗한 눈 마스크 레이어의 조합으로 만든다.

  

GSAM(gradient-based snow activation map)+ GSEM(gradient-based snow edge map)

- 눈 입자의 불규칙한 크기, 모양 및 투명성 처리

gsam을 추정하기 위해, gradient class activation map을 사용하여 대상 네트워크 계층에서 스노우 레이블에 대한 활성화된 gradient map을 추출한다.

gsem을 얻기 위해 이미지 수준의 gradient를 사용하여 예지 정보를 활용(hand-crafted convolution filters + 몇가지 layer 사용)

  

이 2개의 gradient 기반 맵은 마스크 추정 모듈에 공급되어 snow attention mask를 생성.

  

그러나 눈 투명도가 다양하기 때문에 이후 반투명 인식 컨텍스트 복원 네트워크를 사용

  

### key ideas

- GSAM → 눈 입자 국소화하고 GSEM이 모양과 크기를 결정하는 2중 gradient-based 기반 desnowing 방법 제안
- 2개의 gradient기반 맵을을 사용하여 SAM을 생성하는 새로운 마스크 추정 네트워크 도입
- 불투명도로 인해 손실된 정보를 복구하고 다양한 눈 투명도를 처리하기 위해 투명도 인식 컨텍스트 복원 네트워크 도입

  

### Proposed Method

### Snow Attention Mask Estimation

#### Gradient-Based Snow Activation Map

- WSOL 알고리즘을 기반으로 추정되는 제안된 GSAM을 사용하여 눈 위치를 정확하게 결정
- 훈련 데이터은 240,240,3과 해당 레이블인 0,1을 사용하여 눈을 분류하는 이진 분류 네트워크를 훈련
- 그 다음 대상 컨벌루션 레이어에 대한 레이블의 gradient를 계산
- 눈에 대한 이미지 정보를 포함하기 떄문에 conv 3x3 ReLU레이어를 대상 레이어로 사용한다.

  

#### Gradient-Based Snow Edge map

- 제안된 gsem을 사용하여 복잡한 모양과 다양한 크기의 눈 입자를 고려한다.
- Sovel filter를 사용
- 이 fixed kernel은 눈 입자의 고주파 정보를 쉽게 추출할 수 있다.
- 다른 물체의 edge를 억제하고 눈입자의 edgeㅡㄹ 증폭하기 위해 2개의 convolution layer와 3개의 residual block을 가진 residual group을 사용

  

#### Snow Attention Mask

- 정확한 sam을 추정하기 위해 2중 gradient-based method를 사용한다.
- 최종 눈 활성화 맵은 2중 gradient-based map과 눈 이미지를 사용하여 마스크 추정 모듈을 통해 생성된다.

  

### Translucency-Aware Context Restoration

- 실제 환경에서는 마스크 이미지가 다른 요소와 섞일 수 있음.
- inpainting 알고리즘을 사용하여 완화할 수 있지만 snow mask는 일반적으로 inpainting성능을 저하시키는 non-hole모양을 가지고 있다,
- 따라서 반투명 인식 컨텍스트 복원 알고리즈을 사용하여 advanced snow mask를 추론
-