---
layout : single
title:  "[Paper] High Resolution processing and sigmoid fusion modules for efficient detection of small objects in an embedded system"
excerpt: "High Resolution processing and sigmoid fusion modules for efficient detection of small objects in an embedded system 논문 정리"

categories:
  - Paper
tags:
  - object detection

toc: true
toc_sticky: true

author_profile: true
sidebar_main: true

date: 2023-03-19
last_modified_at: 2023-03-19
---

## Open

본 논문에서는 2가지 플러그인 모듈을 사용한 경량 소형  물체 감지 모델을 제시

- 고해상도 처리 모듈(HRPM)
    - 감소된 계산비용을 사용하여 소형 물체의 다중 스케일 특징을 효율적으로 학습

- 시그모이드 융합 모듈(SFM)
    - 손실된 소형 물체 정보의 가중치를 조정하여 공간 노이즈로 인한 오분류 오류를 완화

  

경량 모델을 사용하여 소형 객체를 검출하기 어려운 이유

- 작은 객체를 나타내는 픽셀 수가 너무 작아 학습해야 할 feature가 충분하지 않다
- 작은 객체의 특징 정보는 큰 객체의 특징  정보로 상쇄될 수 있으며 딥러닝 기반 객체 탐지 네트워크는 컨볼루션 레이어를 통해 학습되므로 네트워크를 오버레이 하면서 feature 정보를 추출하는데 U-Net구조의 인코더 디코더는 feature map의 크기를 줄이고 채널 수를 늘리기 때문에 작은 객체의 특징 정보가 손실될 가능성이 높다.
- 일반적으로 경량모델은 계산 복잡도를 줄이기 위해 저해상도 입력 영상을 촬영하는데, 이는 작은 객체 검출을 어렵게 한다.
- 단일 네트워크에서 다양한 소형 객체 탐지 방법을 적용하는 것은 비현실적이고, 대부분의 임베디드 환경은 경량 네트워크를 필요로 한다.

  

본 논문에서는 이 문제들을 해결하여 작은 물체 검출의 성능을 향상시킬 수 있는 효율적인 방법을 제안

- 고해상도 영상을 처리하여 작은 물체의 검출 성능을 향상시키는 효율적인 방법을 제시한 후, 적은 화소 정보로 작은 물체를 학습하는 어려움을 극복하는 시그모이드 융합방법을 제시

  

## Related works

소형 물체 검출

충분히 높은 해상도의 이미지가 필요하다.

1stage,2stage 검출 방법들은 물체가 충분히 클 때 정확도를 보여주었기 때문에 다른 방법 제시

convolution 백본 네트워크를 사용하면 공간 정보를 포함하는 하위 수준 feature map을 손실하는 비용으로 작은 물체의 정보를 포함하는 상위 수준 feature map을 추출할 수 있다.(Trade-off)

  

## Proposed method

소형 객체 검출을 위한 경량 딥러닝 모델 제시

모델이 제한적인 임베디드 환경에서 구현될 수 있도록 2가지 플러그인 모듈 제시

  

### High Resolution Processing Module

- 고해상도 영상을 입력으로 취하면서 소형 객체에 대한 특징 정보를 최대한 학습하면서 연산을 최소화하는 모듈
- 소형 객체 검출 성능을 향상시키고 임베디드 환경에서 구현할 수 있도록 한 연산량이 작은 경량 모듈

  

- 시스템의 입력으로 고해상도 영상을 받는다.
- 객체의 정보를 효과적으로 학습하고 연산량이 크게 증가하는 것을 방지하기 위해 HRPM제시
- 입력 영상이 통과하는 stem 모듈 바로 뒤에 위치 → 많은 양의 텐서를 처리하는데 필요한 위치
- 백본 네트워크 앞에 위치하여 낮은 수준에서 학습한 영상의 edge와 색상 정보를 학습한다.
- HRPM은 고해상도 영역에서 얕은 수준으로 빠르고 효율적으로 학습할 수 있는 공간 정보 증폭
- 역할에 맞는 학습 경로인 3가지 경로로 구분되어 있다.

  

#### Learning Local context information using dilated convolution.

- 객체 자체에 대한 정보 부족으로 인해 주변 상황 정보를 필요로 한다.
- local context information → 검출 대상 객체 주변의 상황 정보
- 첫번째 경로에서 확장 컨볼루션을 통해 작은 객체의 로컬 컨텍스트 정보를 학습한다.
- 이미지 해상도가 높을수록 처리해야 하는 텐서의 양이 증가하기 때문에 확장 컨볼루션을 통해 HWC를 동시에 절반으로 줄이면서 로컬 컨텍스트 정보를 학습
- Min-Max Scaler를 통해 텐서의 값을 정규화 하여 전체 채널에 대한 평균값을 나타낸다.
- receptive field를 증가시켜 더 많은 공간 정보 학습
- pooling연산을 사용하지 않고 많은 양의 receptive field를 취할 수 있기 때문에 공간 차원의 손실이 완화되고 계산 효율이 높아진다.

  

#### Max pooling and max compression

- max pooling 과 max compression을 통해 강력한 edge 정보를 학습한다.
- 임베디드 환경을 고려할 때 계산 비용이 많은 global pooling을 채택하면 안된다.
- average pooling이 작은 객체 검출에 조금 더 적합하지만, 백본 네트워크 앞에 위치하여 고해상도 영상의 모든 객체뿐만 아니라 배경의 edge를 학습하긱 때문에 작은 객체의 edge와 함꼐 학습하고, 또한 임베디드 환경에서 구현하는 것을 목표로 하기 때문에 큰 크기(커널 크기)의 max pooling을 사용하지 않는다.
- 커널 크기가 작은 max pooling을  사용하여 엣지 정보를 효율적으로 학습하기 위해 사용
- max compression은 채널별 max pooling으로부터 feature map을 쌍으로 구성하며, max pooling은 두가지 역할을 수행한다.
    - 강력한 edge의 정보만을 통합
        - 채널별로 다양한 엣지를 학습하지만, 얕은 수준에서 학습한 특징정보는 시멘틱 정보보다 공간적 정보에 대해 더 강하기 때문에 각 채널에서 고유하게 학습한 정보도 edge 정보이다.
    - 전체 네트워크 경량화
        - max compression을 통해 필요한 정보가 포함된 채널을 선택하고 남은 채널을 제거하여 계산 효율과 속도를 도출

  

#### Multi‐scale integration and light‐weight module.

- 확장 컨볼루션은 로컬컨텍스트 정보를 학습하고, max pooling과 max compression은 강력한 엣지 정보를 학습하며 기본 컨볼루션은 객체의 특징 정보를 학습
- 로컬 컨텍스트 정보 → 객체의 특징 정보 학습하는 기본 컨볼루션
- 특징 정보 → 특징 맵에서 확장 컨볼루션을 수행
- 입력 출력 텐서의 크기 변화는 특징 맵의 너비와 높이이며, 채널의 개수는 동일하다.

  

### Sigmoid fusion module

- 객체의 공간적 정보에 초점을 맞추어 객체의 오분류를 완화하는 모듈을 제안
- PAFPN11과 HEAD사이에 위치
- PAFPN을 통과하는 3개의 OUTPL은 각각 다른 feature map 크기를 가지는데, 각각 소형, 중형, 대형 객체 검출에 활용된다.
- 높은 수준과 얕은 수준의 공간 정보만을 융합하여 작은 물체에 대해 손실된 공간 특징을 보상하고 배경에 대해 잘못 학습된 공간 특징을 처벌
- feature map이 평균/시그모이드 연산을 통과하면 attention map은 결과적으로 1개의 채널만 출력되며, 이를 입력 feature map에 곱하여 모든 채널에서 공간정보를 증폭시킨다.
- 객체에 대한 가중치를 부여하여 노이즈와 상대적인 차이를 만든다. → 결과적으로 객체에 집중
-