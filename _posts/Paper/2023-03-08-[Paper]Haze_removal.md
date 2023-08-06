---
layout : single
title:  "Haze removal using deep convolutional neural network for Korea Multi-Purpose Satellite-3A (KOMPSAT-3A) multispectral remote sensing imagery"
excerpt: "Haze removal using deep convolutional neural network for Korea Multi-Purpose Satellite-3A (KOMPSAT-3A) multispectral remote sensing imagery 논문 정리"

categories:
  - Paper
tags:
  - dehazing
  - multispectral

toc: true
toc_sticky: true

author_profile: true
sidebar_main: true

date: 2023-03-08
last_modified_at: 2023-03-08
---

## Abstract

본 논문은 raw 파일 형식의 단일 다중 스펙트럼 원격 감지 이미지를 사용하여 헤이즈 분포를 자동으로 제거하기 위한 컨볼루션 신경망을 제시한다.

제안된 디헤이즈 네트워크를 훈련하기 위해 헤이즈 두꼐 맵과 헤이즈 분포의 파장 의존적 산란 특성을 나타내는 상대적 산란 모델을 사용하여 multispectral hazy image를 합성하였다.

raw multispectral image는 동적 범위가 낮기 떄문에 이들로부터 직접 헤이즈 분포를 정확하게 추정할 수 없다.

haze한 영역과 그렇지 않은 영역에 대해 차등적으로 attention을 부과하기 위해 imput으로 들어온 haze image의 대비를 높인 후 HTM을 사용하였다.

  

제안된 dehaze network는 4개의 하위 네트워크로 구성된다.

1. shallow feature extraction network(SFEN)
2. cascaded residual dense block network(CRDBN)
- 풍부한 로컬 특징 추출 가능
- 계단식 구조 → 로컬 정보와 gradient의 전파를 더욱 향상시킨다
4. multiscale feature extraction network(MFEN)
- haze 분포 및 haze가 없는 영역에 대한 계층 정보를 나타내는 다중 스케일 로컬 특징을 추출하는데 사용
6. refinement network(RN)

  

## Introduction

일반적인 RGB 컬러 영상과 달리 다중 스펙트럼 위성 영상의 dehaze는 확장된 범위의 파장을 활용할 수 있다.

기존 제안된 방법들

- DOS
    - 획득된 장면에서 어두운 영역이 존재하지 않을 수 있기 떄문에 오버헤이즈 및 스펙트럼 왜곡을 피할 수 없다
- kaufman 
    - 장면 내의 어두운 식생 픽셀의 일부 가정에 민감
- Makarau
    - 장면 내 헤이즈 입자의 양에 민감하기 때문에 획득한 원격 감지 이미지에 따라 추정되어야 한다.

  

본 논문에서는 raw file 형식의 다중 스펙트럼 원격 감지 이미지의 헤이즈 분포를 제거하기 위한 심층 컨볼루션 신경망을 제안

- 파장에 따라 다른 산란 특성을 부과하기 위해 실제 안개 영상에서 산란 계수와 추정 HTM을 사용하여 다중 스펙트럼 안개 영상을 합성하는 새로운 데이터 세트 합성 방법 제시
- 제안된 디헤이징 네트워크는 로컬 및 전역 정보를 활용하기 위해 조밀한 연결과 redisual 아키텍쳐를 사용한다.
- 네트워크는 raw file의 낮은 동적 범위에서 다른 attentnion을 부과하기 위해 대비가 향상된 버전에서 추정된 HTM을 입력으로 받는다.

  

## Multispectral hazy image synthesis

multispectral 영상을 합성하기 위한 팢아 의존적 헤이즈 시뮬레이션 방법 제시

  

### Estimation of real haze distribution

- 훈련 데이터의 크기는 over-fitting을 방지하기 위해 커야한다
- 다중 스펙트럼 haze 이미지는 실제 haze 패치로부터 파장 의존적인 헤이즈 분포를 사용하여 합성된다
- 각 스펙트럼 밴드의 추정된 헤이즈 분포는 헤이즈 없는 패치에 추가된다
- 제안된 방법은 각 스펙트럼 밴드에 대한 산란계수를 선형계수로 사용하여 추정HTM과 곱해 사용하고 있다.

  

## Proposed multispectral dehazing method

### Pre-processing

#### Contrast enhancement

- raw multispectral image는 동적 범위가 매우 낮기 때문에 젗너리 없이 흐릿하고 안개가 없는 영역을 정확하게 분리하기가 어렵다.
- 이 문제를 해결하기 위해 먼저 입력 multispectral haze 이미지의 대비 강화 버전에서 각 스펙트럼 대역의 파장 의존 HTM을 추정
- 추정된 HTM을 제안된 dehaze network의 입력으로 받아 안개가 없는 영역보다 안개 분포에 더 많은 주의를 기울인다
- 제안된 방법은 상한 및 하한 임계값을 사용하여 히스토그램을 선형적으로 스패닝하여 대비를 향상시키는 간단한 방법인 히스토그램을 늘림으로써 대비를 향상시킨다.

  

- 그러나 입력된 흐릿한 영상에 구름과 같은 밝은 물체가 존재하면 히스토그램이 밝은 영역으로 모여들기 떄문에, 구름 영역의 밝기 대비가 향상된다.
- 이를 해결하기 위해 밝은 물체를 배재하기 위한 이진 마스크를 통하여 히스토그램을 추정하며 이진 마스크는 평균된 스펙트럼 밴드의 평균을 이용하여 추정된다.

  

#### HTM estimation  

- 헤이즈 분포에 더 많은 attention을 주기 위해 입력 다중 스펙트럼 흐림 영상의 contrast 강화 ㅂ전으로부터 HTM을 추정

  

### Network Architecture

헤이즈 없는 이미지에 헤이즈 분포를 추가하여 관찰된 다중분광 haze image를 얻는다.

관찰된 다중분광 haze image에서 추정된 헤이즈 분포를 subtracting하여 잠재적 헤이즈 없는 이미지를 예측할 수 있다.

노이즈 제거 문제는 residual learning을 채택하여 디헤이즈된 이미지를 추정하는 대신 입력 다중 스펙트럼 haze image에서 헤이즈 분포를 추정