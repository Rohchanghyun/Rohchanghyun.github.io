---
layout : single
title:  "Stacked encoder-decoder transformer with boundary smoothing for action segmentation"
excerpt: "Stacked encoder-decoder transformer with boundary smoothing for action segmentation 논문 정리"

categories:
  - Paper
tags:
  - transformer
  - action segmentation

toc: true
toc_sticky: true

author_profile: true
sidebar_main: true

date: 2023-04-01
last_modified_at: 2023-04-01
---

## Abstract

본 논문에서는 작업 분할을 위해 새로운 스택형 인코더-디코더 변환기 모델 제안(SEDT) 

- 일련의 인코더 디코더 모듈로 구성
- 각 모듈은 self-attention계층이 있는 인코더와 cross-attention계층이 있는 디코더로 구성된다.
- 모든 디코더 앞에 self-attention이 있는 인코더를 추가하여 글로벌 정보와 함께 로컬 정보 보존
- 디코더를 통해 전파될 떄 발생하는 오류의 누적을 인코더-디코더 쌍이 방지.

  

## Introduction

시간적 행동 세분화 작업에 중점을 둠

액션 세분화 → 비디오 프레임에 대한 분류일 뿐만 아니라 비디오 프레임 전체의 상황 이해

시간 컨볼루션 네트워크는 행동 세분화를 위핸 지배적인 접근법 중 하나였음.

- 인코더 → 변압기 기반 프레임워크의 각 디코더 앞에 추가되어 인코더-디코더 패턴을 형성
- 추가 인코더는 디코더 출력 기능에서 새로운 초기 예측을 반환하여 연속적인 디코더에서 나타날 수 있는 오류의 누적을 방지
- 행동 클래스의 모호성을 줄이는 새로운 boundary smoothing strategy를 도입.

  

## Proposed Method

SEDT : 적층된 N개의 인코더-디코더 모듈로 구성되며, 각 모듈은 인코더와 디코더로 구성된다

인코더 블록이 적층된 인코더는 self-attention 메커니즘으로 초기 예측을 예측하고, 디코더 블록이 적층된 디코더는 인코더에서 인코딩된 기능의 예측을 세분화.

디코더는 새로운 입력으로 인코더에 의한 클래스 예측 확률을 얻고, 인코더-디코더 모듈은 정제된 기능을 다음 인코더-디코더 모듈에 전달.

  

### Encoder

feed forward network와 self-attention 레이어로 구성.

확장된 시간적 컨벌루션 레이어를 feed forward network로 사용하여 변압기에 원래 사용된 fully connected layer대신 데이터 부족을 극복.

  

### Decoder

하나의 하위 계층을 제외하고 인코더 블록과 동일하다.

attention 메커니즘이 encoder과의 차이점

cross attention 계층은 이전 feed forward 레이어에서 쿼리를 가져오고, 이전 인코더에서 키와 값을 가져온다.

  

### Boundary smoothing

입력 비디오를 프레임 시퀀스로 사용하고 각 프레임에 대한 액션 클래스를 반환

action boudary 근처에서 작업 전환이 발생하는 프레임에서 단일 작업에 레이블을 지정하는 것은 모호하다.

떄문에 프레임이 동작 경계에 접근함에 따라 원래 GT클래스 확률이 점차 감소하는 접근방식 제시