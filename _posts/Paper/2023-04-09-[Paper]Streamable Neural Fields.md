---
layout : single
title:  "[Paper] Streamable Neural Fields"
excerpt: "Streamable Neural Fields 논문 정리"

categories:
  - Paper
tags:
  - Neural fields

toc: true
toc_sticky: true

author_profile: true
sidebar_main: true

date: 2023-04-09
last_modified_at: 2023-04-09
---

  

## Abstract

neural field → 새로운 데이터 표현 패러다임, 다양한 신호 표현에서 성공적인 퍼포먼스를 보여줬다.

본 논문에서는 다양한 width의 실행 가능한 sub network로 구성된 streamable neural network를 제안한다.

  

단일 네트워크를 시간에 따라 스트리밍 가능하게 하고 다양한 품질과 부분을 재구성할 수 있다.

  

## Introduction

neural field: 입력을 공간 또는 시간 좌표로 사용하고 임의의 해상도로 신호 값을 생성하는 mlp를 사용

  

현재 neural field가 가지고 있는 과제: 

- 신호 전송은 전체 매개 변수를 보내고 받는 방식으로 이루어지기 떄문에 지연 시간과 처리량을 줄이고 처리하는데 최적의 모델 크기를 찾는 것이 중요하다
- raw 신호는 종종 서로 다른 해상도 또는 품질로 전송되어야 한다. 이때 다른 품질을 나타내는 네트워크 여러개를 저장하는 방법은 storage공간의 낭비이다. 

  

이를 극복하기 위해 논문에서 제안한 방법

- 훈련된 단일 네트워크를 다양한 폭의 실행 가능한 하위 네트워크로 분리할 수 있도록 하는 훈련 기술 및 구조 설계
- 각 하위 네트워크는 신호의 일부를 나타낸다

  

### key ideas

- <span style="color: #88c8ff"> 여러 시각적 품질과 공간적 범위를 표현하고 스트림 라인에서 신호를 디코딩할 수 있는 단일 신경망을 제안</span>

  

## Method

다양한 width의 실행 가능한 sub network로 구성된 스트리밍 가능한 neural field의 훈련 기술 및 구조에 대해 설명.

훈련이 완료되면 단일 네트워크는 retraining없이 다양한 품질의 신호를 제공할 수 있다.

좁은 sub network는 저주파 신호를 보존하고 더 넓은 sub network는 고주파 세부 정보를 포함한다.

  

### Network architecture and progressive training

- 처음은 작고 좁은 MLP로 훈련을 시작하여 결과가 수렴하면 임의의 크기만큼 폭이 증가
- 점진적 신경망 구조와 유사하게 새로 추가된 hidden unit에서 가중치를 제거하여 추가된 유닛이 소규모 유닛의 출력에 영향을 미치는 것을 방지
- 이러한 방법으로 대규모 네트워크가 소규모 네트워크에서 학습한 지식을 사용하고 소규모 네트워크가 캡쳐할 수 없는 잔여 신호만 학습하도록 장려한다
- 원하는 신호 품질 또는 공간/시간 크기가 충족될 때까지 이 프로세스를 반복한다

  

### Progressive training vs slimmable training

- 슬림화 가능한 네트워크에서의 훈련기술 → 이미지 및 비디오 피팅 작업의 목표도 달성할 수 있다.
- 제안된 점진적 훈련과 달리 미리 정의된 폭에 걸쳐 반복하고 해당 폭의 하위 네트워크를 취한 후 목표 신호 예측을 사용하여 손실을 계산.
- 하위 네트워크의 그레디언트는 모든 폭을 방문할 떄까지 누적되고 가중치가 한번에 업데이트 된다.