---
layout : single
title:  "[Paper] Deep Dehazing Powered by Image Processing Network"
excerpt: "Deep Dehazing Powered by Image Processing Network 논문 정리"

categories:
  - Paper
tags:
  - dehazing

toc: true
toc_sticky: true

author_profile: true
sidebar_main: true

date: 2023-02-05
last_modified_at: 2023-02-05
---

## Abstract

최근의 dehazing 방법들은 새로운 DNN 구조 개발에만 초점을 맞추는 대신에 기존의 이미지 처리 기법을 사용한다. 하지만 이러한 추세와는 달리 이미지 처리 기술이 DNN과 통합되면 경쟁력이 있음을 보여준다.

본 논문에서는 

- 정확한 dehazing을 기존 이미지 처리 기술
- 안정적인 dehazing performance를 위한 direct learning

을 사용한다.

제안된 방법은 낮은 계산 비용과 학습이 쉬움을 보이고, 실험 결과에서 최근 알고리즘과 비교하여 정확한 dehazing 결과를 생성한다는 것을 보여준다.

  

## Introduction

Image Processing → 픽셀값을 직접 처리하는 대부분의 low-level vision방법에서 좋은 성능을 낼 수 있다. 하지만 딥러닝의 발달로 인해 이미지 처리 기술관련 연구는 점차 감소했다.

  

Dehazing : image Processing의 대표적인 작업이며 haze image에서 dehaze image를 얻는 task

training data가 부족하기 떄문에, 딥러닝 기반 dehazing방법은 일반적으로 haze model을 사용하여 haze region을 합성한다.

  

본 논문에서는 

- Image Processing이 DNN 전에 적용되어 dehazing 작업의 적절한 특성을 유지할 수 있다고 주장.
- Image Processing methods
    - curve adjustment
        - 밝기와 톤을 조절하여 다른 스타일의 이미지 생성
    - retinex decompositio
        - 반사율 및 조명 성분을 사용하여 이미지를 설명하기 위해 low light enhancement에 사용된다.
    - image fusion modules
        - 여러 영상을 서로 다른 속성으로 결합

  

### Pipeline

- curve adjustment, retinex decomposition 및 image fusion 기술과 같은 간단한 image processing 기술을 네트워크에 통합하면 최근의 복잡한 네트워크에 비해 경쟁령 있는 결과를 얻을 수 있음
- 이 기술들은 상호 보완적인 특성을 가지며 그 장점들은 이미지 융합 방법에 의해 결합되어 dehazing 정확도를 향상시킨다.
- 실험결과에서 논문의 방법이 haze dataset에서 최신 알고리즘을 능가함

  

## Proposed Methods

### Feature Extraction Module

- 흐릿한 이미지의 형상을 추출하는데 사용
- 정밀한 feature extraction을 위해 FFANEt 과 RCAN에서 높은 성능을 보여주는 핵심 구성요소 선택
- feature\_1 은 residual group이 5개의 feature attention bolck으로 구성되어있음을 보여준다.
- FFANet 과 RCAN에서와는 달리 논문의 FEM은 인코더-디코더 구조를 가지고 있다.
- FEM모듈을 통과한 feature map은 direct learning + 3가지의 Image processing의 입력으로 사용된다.

  

### Direct Learning

- 추출된 feature map을 간단한 컨볼루션 레이어를 통해 전달하고 global skip connection을 조정.
- curve adjustmnet와 retinex decomposition이 dehazing 결과를 크게변화시키는 것을 방지하여 안정적인 성능을 유도

  

### Curve Adjustment

- Zero-DCE는 곡선 맵을 사용하여 Image Adjustment의 중요한 요소를 보여준다.
- 모든 픽셀에 전체적으로 곡선을 적용하여 데이터 손실 방지
    - 하지만 특정 영역이 지나치게 어두워지거나 주변 영역의 밝기가 비슷해져 포화 문제가 발생
    - RGB중 하나의 채널에 대해서만 조정하면 색상 왜곡이 발생하고 픽셀 색상 비율이 축소될 수 있음
- 하지만 논문에서 제안된 곡선 맵 c는 각 채널과 각 픽셀에 대해 개별적으로 추정
- 곡선 맵은 복구된 이미지를 실측값과 비교하여 학습하기 때문에 포화와 왜곡이 억제되는 픽셀당 곡선 함수를 얻을 수 있다.

 → haze분포가 균일하지 않은 비균질 dehazing에 적합하다.

  

### Retienx Decomposition

- 일반적으로 꺠끗한 이미지를 조명 및 반사 성분으로 나누는 low-light enhancement에 사용되었다.
- 이와 같이 dehazing image를 입력 haze(illumination) image와 반사성분으로 나눌 수 있다고 가정.
- 조명이 이미ㅣㅈ의 밝기를 나타내므로 이미지에서 헤이즈 영역을 밝은 영역으로 간주할 수 있다.
- 반사율 구성 요소에는 조명조건에 관계 없이 이미지의 고유한 특성이 포함되어 있음.

  

### Multi-Dehazed Image Fusion

- direct learning, curve adjustmnet, retinex decomposition를 사용하여 세가지 중간 dehazing 결과 생성. 
- Image Fusion을 이용해 세 결과물을 결합
- 이를 위해 융합 맵을 생성한다.
- 2d softmax → 결합해야 할 영역을 쉽게 결정할 수 있다.