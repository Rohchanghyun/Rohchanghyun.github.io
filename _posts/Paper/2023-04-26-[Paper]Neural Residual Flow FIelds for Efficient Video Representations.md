---
layout : single
title:  "Neural Residual Flow FIelds for Efficient Video Representations"
excerpt: "Neural Residual Flow FIelds for Efficient Video Representations 논문 정리"

categories:
  - Paper
tags:
  - Neural fields
  - representations

toc: true
toc_sticky: true

author_profile: true
sidebar_main: true

date: 2023-04-26
last_modified_at: 2023-04-26
---

  

## Abstract

neural field는 비디오를 포함한 다양한 신호를 재현하기 위한 방법으로 많이 연구되지만 neural field의 매개 변수 효율성을 향상시키는 연구는 아직 초기 단계에 있다.

본 논문에서는 표준 비디오 압축 알고리즘에서 영감을 받아 비디오 프레임 전반에 걸친 모션 정보의 사용을 통해 의도적으로 데이터 중복을 제거하는 비디오를 다시 보내고 압축하기 위한 neural field 구조를 제안한다.

모션 정보를 유지하는 것은 훨씬 적은 수의 매개변수를 필요로 한다.

또한 모션 정보를 통해 색상 값을 재사용하는 것은 네트워크 매개 변수 효율성을 보여준다.

  

## Introduction

이산적으로 샘플링된 데이터를 저장하는 다른 표현 기술과 마찬가지로 neural field는 연속 좌표를 입력으로 사용하여 임의의 해상도 및 임의의 좌표에서 신호를 표현 할 수 있다.

이 표현 방법은 컴퓨터 그래픽, 물리적 시뮬레이션 등 많은 영역에서 상당한 가능성을 보여주었다.

하지만 이에 대한 매개변수 효율성은 자세히 연구되지 않았기 때문에, 본 논문에서는 새로운 표현 접근법을 사용하여 비디오를 효과적으로 표현하는 방법을 연구한다.

비디오 신호의 공간 및 시간적 중복을 활용하지 않으며, 목표는 중복을 명시적으로 제거하여 매개변수 효율성을 향상시키는것.

  

raw color 대신 optical flow와 residual을 사용하는 새로운 neural field체계인 NRFF를 제안한다.

모션 정보를 사용하여 프레임 전반에 걸쳐 제시된 신호를 중복 제거하는 표준 비디오 압축 알고리즘에서 영감을 받았다.

  

optical flow는 미세한 세부 정보를 보존하여 다른 참조 프레임의 색상 값을 재사용할 수 있도록 하고, 이는 섬세한 패턴이 있는 경우 네트워크 매개변수 효율성을 향상시킨다.

  

비디오 프렝미에서 원본 신호를 높은 정밀도로 복구하기 위해 residual을 사용한다

  

원시 색상 신호가 아닌 optical flow와 residual을 캡쳐하도록 네트워크를 훈련시켜 매개변수의 효율성을 크게 향상시킨다

  

네트워크를 2개의 하위 네트워크로 분할

1. optical flow용
2. residual

  

### Key ideas

- color 대신 optical flow와 residual를 출력으로 사용함으로써 비디오 품질을 크게 향상시킬 수 있음을 보여준다
- 프레임 재구성을 위해 여러 참조 프레임을 사용할 것을 제안, 이는 네트워크 크기를 늘리지 않고 비디오 품질을 향상
- 각 사진 그룸에 대해 공유 네트워크를 사용하는 것 외에 2개의 neural field를 사용함으로써 매개 변수 효율을 향상

  

## Method

주어진 공간 및 시간 좌표에 대한 optical flow를 생성

이 외에도 residual도 neural field에 의해 생성되고 이렇게 생성된 redisual은 reconstruction을 위해 warped frame에 추가된다

  

### Dense Oprical Flow Estimation

- 표준 비디오 압축방법 → 압축 효율성을 향상시키기 위해 블록별 움직임 정보를 사용 → 종종 블록 모양의 왜곡이나 아티팩트가 포함되므로 디블록킹 필터가 필요
- neural field를 통해 조밀한 픽셀 단위의 움직임 벡터를 사용
- 작은 크기의 신경망을 neural field로 사용하여 움직임 벡터를 조밀하게 샘플링

  

### Image warping and Completion

- 생성된 flow field를 사용하여 참조 프레임을 warp하여 대상 비디오를 예측
- 소스 비디오 프레임을 단순히 warp하여 비디오 프레임을 재구성 하는 것은 아티팩트를 포함할 가능성이 있다. 이는 부정확한 optical flow 추정을 야기할 수 있기 떄문에 이를 완화하기 위해 residual를 함께 사용

  

###