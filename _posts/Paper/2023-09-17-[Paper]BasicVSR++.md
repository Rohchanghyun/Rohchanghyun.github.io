---
layout : single
title:  "BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation ans Alignment"
excerpt: "BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation ans Alignment 논문 정리"

categories:
  - Paper
tags:
  - BasicVSR++
  - grid propagation
  - LLCV

toc: true
toc_sticky: true

author_profile: true
sidebar_main: true

date: 2023-09-17
last_modified_at: 2023-09-17
---


<p align="center"><img src="/assets/images/Paper/BasicVSR++/figure_1.png"></p>

## Abstract

recurrent 구조는 super resoluiton task에서 유명한 프레임워크다.

BasicVSR은 전체 입력 비디오의 정보를 효과적으로 활용하기 위해 feature alignment를 사용하는 양방향 전파를 사용한다.



본 논문에서는 2차 그리드 전파와 flow-guided deformable 정렬을 사용하여 BasicVSR을 재설계한다.

Enhanced propagatoin & alignment로 recurrent 프레임워크를 empwer 함으로써 잘못 정렬된 비디오 프레임 전반에 걸쳐 시공간 정보를 보다 효과적으로 활용할 수 있음을 보여준다.

Super resolution 비디오 외에도 압축 비디오 enhancement와 같은 다른 비디오 복원 작업에 대해 일반화가 잘 된다.



## Introduction

VSR은 복원을 위해 잘못 정렬된 비디오 프레임 전반에 걸쳐 보완 정보를 수집해야 한다는 어려움이 있다. 짧은 시간 윈도우 내의 프레임을 비교하여 비디오의 각 프레임이 복원되는 슬라이딩 윈도우 프레임 워크가 있지만, recurrent 프레임워크는 latent feature를 전파하여 장기 의존성(long-term dependency)를 이용하려고 시도한다. 하지만 장기 정보를 전송하고 recurrent 모델의 프레임 간 특징을 정렬하는 문제는 여전히 만만치 않다.

### BasicVSR

- 공통 VSR 파이프라인을 전파,정렬,aggregation 및 업샘플링의 4가지 구성 요소로 요약.
- 양방향 전파를 채택하여 재구성을 위해 전체 입력 비디오의 정보를 활용한다.
- 정렬의 경우, feature warping을 위해 optical flow를 사용한다.
- 하지만 전파 및 정렬의 기본 설계는 information aggregation 의 효율성을 제한한다.



결과적으로 네트워크는 폐색되고 복잡한 지역을 복원할 때 어려움을 겪는다. -> 전파 및 정렬의 정제된 설계를 요구한다.



본 논문에서는 <span style="color: #88c8ff">정보가 보다 효과적으로 전파되고 통합될 수 있도록 하는 2차 grid 전파 및 optical flow deformable 정렬을 고안하여 기본 VSR을 재설계한다.</span>

- 2차 grid 전파
  - grid 유사 방식으로 배열된 양방향 전파를 허용
  - 1차 마르코프 속성의 가정을 완화하고 2차 연결을 네트워크에 통합하여 서로 다른 시공간에서 정보를 집계할 수 있다.
- flow-guided deformable 정렬
  - DCN 오프셋을 직접 학습하는 대신 flow field의 residu에  의해 정제된 기본 오프셋으로 optical flow field를 사용하여 오프셋 학습의 부담을 줄인다.



이러한 보다 효과적인 설계의 장점으로 더 가벼운 backbone을 채택할 수 있다.



## Related works

### Grid connections

그리드와 유사한 설계는 object detection 및 semantic segmentation, frame interpolation 과 같은 다양한 비전 작업에서 볼 수 있다.

일반적으로 이러한 설계는 주어진 이미지/특징을 여러 해상도로 분해하고 미세 정보와 거친 정보를 모두 캡처하기 위해 해상도 전반에 그리드를 채택한다.

앞서 언급한 방법과 달리 BasicVSR++는 다중 스케일 설계를 채택하지 않지만, <span style="color: #88c8ff"> 그리드 구조가 양방향 방식으로 시간에 걸쳐 전파되도록 설계되어 서로 다른 프레임을 연결하여 특징을 반복적으로 다듬어 표현력을 향상시킨다.</span>



### Higher order propagation

고차 전파는 gradient flow를 개선하기 위해 연구되었다.

이러한 방법은 분류 및 언어 모델링을 포함한 다양한 task의 개선을 보여주지만, VSR task에서 중요한 시간적 정렬을 고려하지 않는다.

2차 전파에서 시간적 정렬을 허용하기 위해 flow guided deformable 정렬을 2차 설정으로 확장하여 정렬을 전파 체계에 통합한다.



### Deformable alignment

최근 연구는 deformable 정렬을 분석하고 flow vased 정렬에 대한 성능 이득이 오프셋 다양성에서 나온다는 것을 보여준다. 이에 deformable 정렬을 채택하지만 training instablity(훈련 불안정성)을 극복하기 위한 방법으로 채택.

본 논문은 optical flow를 기본 오프셋으로 모듈에 직접 통합하여 훈련 및 추론 중에 더 명확한 guide를 제공한다.

## Method

<p align="center"><img src="/assets/images/Paper/BasicVSR++/figure_2.png"></p>

BasicVSR++ 는 전파 및 정렬을 개선하기 위한 2가지 효과적인 수정으로 구성된다.

그림과 같이 입력 비디오가 주어지면 각 프레임에서 특징을 추출하기 위해 residual block이 먼저 적용된다.

그 다음 2차 grid 전파 방식으로 특징을 전파하고, flow-guided deformable 정렬에 의해 정렬을 수행한다.

전파 후, aggregate된 특징은 convolution 및 pixel shuffling을 통해 출력 이미지를 생성하는데 사용된다.



### Second-Order Grid Propagation

기존의 대부분의 방법들은 단방향 전파를 채택하고 있는데, 본 논문에서는 전파를 통해 반복적인 개선을 가능하게 하는 그리드 전파 체계를 고안한다.

중간 feature는 시간을 거슬러 앞뒤로 번갈아 가며 전파된다. 전파를 통해 서로 다른 프레임의 정보를 재검토 하여 feature refinement에서 채택한다.

한번만 특징을 전파하는 기존 작업에 비해 그리드 전파는 전체 시퀀스에서 반복적으로 정보를 추출하여 특징 표현력을 향상시킨다.



전파의 견고성을 더욱 강화하기 위해 BasicVSR에서 1차 마르코프 속성 가정을 완화하고 2차 연결을 채택하여 2차 마르코프 체인을 구현한다. 이러한 완화를 통해 서로 다른 시공간 위치에서 정보를 수집할 수 있어 폐색되고 미세한 영역에서 견고성과 효과를 향상시킬 수 있다.



두개의 특징을 합쳐 2차 그리드 전파를 고안한다. 

> x_i : input image
>
> g_i : feature extracted from x_i

<p align="center"><img src="/assets/images/Paper/BasicVSR++/figure_4.png"></p>

### Flow-Guided Deformable Alignment

<p align="center"><img src="/assets/images/Paper/BasicVSR++/figure_3.png"></p>

deformable 정렬은 가끔 오프셋 오버플로우를 초래하여 최종 성능을 저하시킨다.(training instability 때문)

이 불안정성을 극복하면서 오프셋 다양성을 활용하기 위해, deformable alignment와 flow-based alignment에서 고안하여 optical flow가 deformable alignment를 안내하도록 한다.



flow-guided deformable Alignment는 optical flow를 guide로 채택한다.

- CNN은 local receptive field를 가지고 있기 때문에 optical flow를 사용하여 feature를 사전 정렬함으로써 오프셋 학습을 지원할 수 있다.
- residu만 학습함으로써, 네트워크는 optical flow에서 작은 편차만 학습하면 되므로 일반적인 변형 정렬 모듈의 부담을 줄일 수 있다.
- 뒤틀린 feature를 직접 연결하는 대신 DCN의 변조 마스크는 서로 다른 픽셀의 기여도를 측정하는 attention map으로 작용하여 추가적인 유연성을 제공한다.

<p align="center"><img src="/assets/images/Paper/BasicVSR++/figure_5.png"></p>
<p align="center"><img src="/assets/images/Paper/BasicVSR++/figure_6.png"></p>