---
layout : single
title:  "FrePGAN: Robust Deepfake Detection Using Frequency-level Perturbations"
excerpt: "FrePGAN: Robust Deepfake Detection Using Frequency-level Perturbations 논문 정리"

categories:
  - Paper
tags:
  - GAN
  - FIngerprint

toc: true
toc_sticky: true

author_profile: true
sidebar_main: true

date: 2023-05-18
last_modified_at: 2023-05-18
---

  

## Abstract

다양한 딥페이크 탐지기가 제안되었지만 training 설정 이외 혹은 알려지지 않은 범주에서는 여전히 어려움이 있다.

본 논문의 목적은 known and unseen GAN 모두에 대해 딥페이크 탐지기를 일반화하는 프레임워크를 설계하는것.

생성된 이미지를 실제 이미지와 구별할 수 없도록 주파수 레벨의 perturbation map을 생성한다.

perturbation generator의 훈련과 함께 딥페이크 탐지기를 업데이트 함으로써, 모델은 초기 반복 시 주파수 레벨의 아티팩트를 감직하고 마지막 반복 시 이미지 수준 불규칙성을 고려하도록 훈련된다.

  

## Introduction

최근 GAN이 떠오름에 따라 딥페이크를 쉽고 광범위하게 생성할 수 있게 됨으로써, 악용의 가능성 떄문에 딥페이크 탐지의 중요성이 결정적이게 되었다.

이전 연구에서 확인한 바와 같이 CNN기반 생성 모델은 고주파 구성 요소를 재구성하는 데 한계가 있는 것으로 알려졌다.

그러나 특정 GAN모델에 대해 생성된 이미지를 감지하는 데에 주파수 레벨 아티팩트가 효과적이지만 CNN구조 및 훈련 범주에 따라 달라지는 주파수 레벨 아티팩트의 고유한 외관 떄문에 검출기가 훈련 설정에 overfitting되기 쉽다.

따라서 딥페이크 검출기의 일반화의 핵심은 훈련 중 주파수 레벨 아티팩트의 영향을 크게 줄이는 것이다.

  

이에 따라 본 논문에서는 주파수 perturbation GAN과(FrePGAN) 딥페이크 분류기 두 모듈로 구성된 새로운 프레임워크를 제안.

  

### key ideas

- 가짜 이미지의 도메인별 아티팩트를 무시하기 위해 주파수 레벨의 perturbation map을 생성하기 위해 FrePGAN을 개발
- FrePGAN에서 얻은 perturnbation map은 주어진 입력 이미지에 추가되어 도메인별 아티팩트의 영향을 줄이고 딥페이크 탐지기의 일반화 능력을 향상시킬 수 있다
- FrePGAN과 딥페이크 분류기는 주파수 레벨 아티팩트와 일반 기능을 모두 고려하도록 딥페이크 분류기를 훈련시키기 위해 교대로 업데이트 된다

  

## Method

주파수 레벨 아티팩트의 효과를 줄이기 위해 실제 이미지와 가짜 이미지 모두 FrePGAN의 생성된 perturbation map에 각각 추가된다.

<p align="center"><img src="/assets/images/Paper/FrepGAN/figure_1.png"></p>

  

### Training of Deepfake Detection Framework

perturbation map의 다양한 특성에 대해 포괄적인 훈련을 위해 두 네트워크(FrePGAN,딥페이크 분류기)를 하나의 반복에서 의도적으로 훈련시킨다.

또한 교대 업데이트를 통해 입력 데이터의 다양성을 확장하여 딥페이크 분류기의 일반화를 향상시킬 수 있다.

초기 업데이트에서는 FrePGAN이 주파수 레벨 아티팩트의 효과를 무시하기 위해 적절한 perturbation map을 생성하지 못하기 때문에 딥페이크 분류기는 탐지하기 쉬운 아티팩트를 사용하여 가짜 이미지를 실제 이미지와 구별하도록 훈련된다.

반대로, FrePGAN이 실제 이미지와 가짜 이미지를 혼동하는 perturbation map을 생성하도록 충분히 훈련되면 딥페이크 분류기는 다양한 유형의 GAN모델에서 일반적으로 작동하는 새로운 feature를 추출해야 한다.

  

### Frequency Perturbation GAN

FrePGAN을 훈련하기 위해 perturbation generation loss에 의해 훈련된 generator와 perturbation별 판별기의 두가지 주요 부분으로 구성된 새로운 구조 제안.

> 입력: w\*h\*c  
> y = 0 or 1(real or fake)  


#### Perturbation Map Generator

<p align="center"><img src="/assets/images/Paper/FrepGAN/figure_2.png"></p>

그림과 같이 주파수 영역으로 변환할 때 실제 이미지와 가짜 이미지를 쉽게 구분할 수 있다.

또한 주파수 레벨의 아티팩트가 주로 고주파 성분에 위치하는것을 확인 → 주파수 레벨의 perturbation을 추가하면 도메인별 아티팩트의 영향을 줄일 수 있다.

주파수 레벨의 아티팩트를 무시하려면 주파수 영역에서도 perturbation이 생성되어 있어야 한다. 따라서 원본 이미지에서 변환된 주파수 맵을 perturbation map 생성기의 입력으로 활용한다.

<span style="color: #88c8ff">perturbation generator</span> : 주파수 레벨 변환기 + 주파수 레벨 생성기 + 역 주파수 레벨 변환기

> 주파수 레벨 변환기

`xe = F(x)` 로 고속 푸리에 변환을 사용하여 입력 영상을 주파수 맵으로 변환

`xe (w*h*2c)` 는 x로부터 변환된 주파수 맵. 각 영상 채널이 주파수 맵의 실제 부분과 가상 부분에 대해 두개의 채널로 분리되기 때문에 xe의 채널 수는 두배가 된다.

> 주파수 레벨 생성기  

`xe` 를 받아 동일한 크기의 `xe` 를 갖는 주파수 맵을 생성.

생성기의 스킴은 인코더 및 디코더를 포함하는 image to image 변환 GAN의 스킴과 유사.

주파수 레벨 생성기를 `H` 라 하면,출력은 `ez = H(xe)` 

마지막으로 생성된 맵은 픽셀 수준의 perturbation map으로 변환된다.

  

`G(x) = F^-1 (H(F(x)))` 

  

### Perturbation Discriminator

생성된 perturbation map의 효과를 향상시키기 위해 perturbation discriminator를 추가하여 적대적으로 훈련시킨다.

전체 구조는 연속 컨볼루션 레이어에 의해 입력 특징을 하향 변환하고 마지막 컨볼루션 레이어에서 이진 분류를 수행하는 기전의 GAN 판별기를 따른다.

마지막으로 FC layer에 의해 perturbation 판별기는 perturbation map 생성기의 출력을 원래 이미지와 구별

estimation = `D(xr) = 0 or D(G(x)) = 1` 로 표현될 수 있는 확률

  

### Deepfake Classifier

입력 이미지가 생성된 가짜인지 아닌지를 구별하기 위한 네트워크.

전체 프레임워크는 딥페이크 탐지를 위한 이진 레이블을 예측하기 위해 ResNet-50을 사용하는 분류 네트워크

  

> 입력 : `AG(x) = x + G(x)`