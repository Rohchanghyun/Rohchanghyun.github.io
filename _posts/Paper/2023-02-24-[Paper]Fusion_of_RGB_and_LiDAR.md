---
layout : single
title:  "Fusion of an RGB camera and LiDAR sensor through a Graph CNN for 3D ibject detection"
excerpt: "Fusion of an RGB camera and LiDAR sensor through a Graph CNN for 3D ibject detection 논문 정리"

categories:
  - Paper
tags:
  - LiDAR
  - graph CNN

toc: true
toc_sticky: true

author_profile: true
sidebar_main: true

date: 2023-02-24
last_modified_at: 2023-02-24
---

## Abstract

RGB 카메라와 LiDAR 센서의 구조적 데이터 특성이 다르기 떄문에 정보의 손실 없는 융합이 어렵다.

이를 해결하기위해 

- Graph CNN을 이용하여 두 센서를 융합
- 각 특징의 기하학적 정보를 보완하여 융합 특징을 생성

  

## Introduction

광학 카메라 or Lidar로 부터 획득한 포인트 클라우드를 독립적으로 처리할 때 → 인식 작업에서 문제 발생

> ex)복잡한 장면이나 안개나 폭우와 같은 극한의 날씨에 일부 물체가 가려질 경우 데이터를 잘 획득할 수 없어 인식에 한계가 존재 

이러한 문제를 보완하기 위해 센서 융합 기술 사용

  

**3D****카메라 획득 영상**

- 격자형 구조
- 데이터 형식이 규칙적이고 밀도가 높아 시멘틱 정보가 포인트 클라우드에 의해 풍부하다

  

**2D카메라 획득 영상**

- 투영 없이 3D 정보가 자연스럽게 포함된 포인트 클라우드에 비해 깊이 정보나 구조 정보가 필요한 분야에서 정확한 결과를 얻기에는 한계가 있다.

  

이러한 장단점을 활용하기 위해 데이터 융합에 의한 보완관계를 이용하려는 연구 있음

  

이 문제를 해결하기 위해 Graph CNN을 사용하여 이미지 특징과 포인트 클라우드 특징이 융합될 떄 발생하는 의미적이고 구조적인 정보의 손실을 최소화하는 GFF(graph feature fusion)모듈 제안

- 그래프 컨볼루션 신경망을 사용하여 이미지와 포인트 클라우드에서 추출한 로컬 특징을 융합한 다음 각 특징 사이의 관계 정보를 추가
- 스케일과 기하학에서 강력한 포인트-이미지 융합 특징을 생성
- 다운스트림으로 3D 객체 검출을 수행하여 기여도 평가

  

### Key ideas

- 카메라와 라이다를 통해 얻은 각 로컬 특징을 융합하고 각 특징 간의 관계 정보를 추가하며 스케일과 기하학에서 강력한 융합 특징을 생성하는 GFF모듈 제안
- 멀리 있는 물체에 대한 포인트 클라우드의 희소성을 보완하기 위해 카메라 영상은 의미적인 정보를 제공
- GFF 모듈에서는 포인트 기반 graph CNN을 사용하여 각 특징의 정보 손실 없이 포인트 클라우드 특징과 영상 특징 수렴

  

## Background

### Association of the image and point cloud

라이다 센서에 의해 획득된 포인트 클라우드와 카메라에 의해 획득된 영상은 구조적 데이터 특성이 다르므로 카메라와 라이다로부터 획득된 특징은 서로 다른 두 맥락에서 표현되며, 대응 관계를 획득하기 위해 카메라 영상에 포인트 클라우드를 투사하여 점 위치와 영상 화소의 관게를 설정

  

## Proposed method

<p align="center"><img src="/assets/images/Paper/fusion_lidar/figure_1.png"></p>

지각 작업을 정확사게 수행하기 위해서는 카메라와 라이다 센서의 정보를 융합하여 사용하는 것이 중요하며, 라이다 센서에 의해 획득되는 포인트 클라우드의 경우 시멘틱 정보가 충분하지 않아 GFF모듈을 이용하여 시멘틱 정보가 풍부한 영상 특징을 포인트 특징과 효과적으로 융합하는 Graph CNN기반으로 카메라와 라이다 데이터를 결합하여 3차원 객체 검출 방법을 제안

- GFF모듈은 EPNet의 2스트림 RPN에 적용되어 영상의 로컬 특징과 포인트 클라우드의 로컬 특징을 융합
- 기준선에서 특징 융합을 위해 FC계층을 사용하는 방법과 달리 본 모듈은 엣지 컨볼루션 계층을 사용하여 특징점 사이의 기하학적 관계 정보를 구성

  

### Point stream in RPN

- 각 포인트에 대한 특징을 학습하고 3D proposal을 생성
- 특징 추출을 위해 PointNet++를 사용하며, SA 및 FP레이어의 3개의 쌍으로 구성된다
- 영상의 시멘틱 정보를 보존하면서 여러 스케일 중 강력한 특징을 얻기 위해 마지막 FP레이어(Fp8)이 Ftotal과 함꼐 마지막 GFF모듈으 ㅣ입력이 되어 최종 융합 특징을 생성

  

### Image stream in RPN

- 카메라 영상을 입력으로 받아 CNN 백본을 통해 영상 특징을 추출
- EPNet과 마찬가지로 영상 특징을 추출하기 위해 3개의 컨볼루션 블록으로 구성된 아키텍쳐 사용
- 각 컨볼루션 블록은 stride2의 두번째 컨볼루션 계층을 갖는 2개의 3x3 컨볼루션 계층, batch normalization 및 ReLU활성화 함수로 구성된다

  

### Graph feature fusion module

- 점 단위 영상 특징과 점 특징을 입력으로 하는 GFF모듈을 통해 영상과 점 특징을 융합
- 두개의 Graph CNN을 사용