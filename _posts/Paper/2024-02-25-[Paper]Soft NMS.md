---
layout: single
title: "[Paper] Soft-NMS – Improving Object Detection With One Line of Code"
tags:
  - NMS
excerpt: "Soft-NMS – Improving Object Detection With One Line of Code 논문 정리"
categories:
  - Paper
toc: true
toc_sticky: true
author_profile: true
sidebar_main: true
date: 2024-02-25
last_modified_atlast_modified_at: 
cssclasses:
---


Soft-NMS – Improving Object Detection With One Line of Code

<mark style="background: #ADCCFFA6;">일정 비율 이상인 (iou > threshold) 겹치는 bbox 들의 confidence를 0 으로 만들어 없애지 말고, confidence 를 줄여서 최종 mAP 를 향상 시키자는 개념</mark>

# Abstract

NMS : Non-max suppression
- 모든 detection box M을 score에 따라 정렬
- maximum score의 box M이 선정되고 M과 중복된 영역을 가지는 다른 box는 suppressed 된다(사용하지 않음, threshold 사용)

<mark style="background: #ADCCFFA6;">Soft-NMS</mark> : M과의 중복된 정도에 따라 다른 객체의 detection score를 연속적인 함수로 감소 -> box를 제거하지 않음

# Introduction

Object detection task -> 많은 CV 분야에 사용
Computational bottleneck을 가져서는 안되고, re-train이 필요해서는 안된다(성능 향상도 크지 않을 때 re train까지 필요하면 비효율적)

Soft-NMS는 Object detection pipeline에서 NMS 알고리즘에 대한 대안.
<mark style="background: #ADCCFFA6;">모델의 re-train을 요구하지 않으며, 성능을 향상시키는 간단한 모듈
</mark>

Object detection은 multi-scale sliding window 기반의 방법을 사용하여 class에 대한 foreground/background score를 계산
그러나 가까이 있는 window는 종종 서로 연관된 score를 가지고 있음. -> NMS는 최종 detection을 얻기 위한 후처리 방식으로 사용

Deep learning이 나오고 이러한 방법들은 convolution net을 사용하여 생성된 category independent region proposal로 대체되었다

이러한 region proposal들은 class별 점수를 할당하는 classification sub-network와, proposal의 position을 정제하는 network에 입력된다.(Mask R-CNN)

이 과정에서 여러 proposal이 같은 위치로 regression 되어 여러 box가 겹치게 된다
따라서 NMS는 false positive를 줄이기 위해 사용

<p align="center"><img src="/assets/images/Paper/SoftNMS/20240225205500.png"></p>

B : list of initial detection boxes
S : corresponding detection scores
N_t : NMS threshold

- Maximum score M 을 가진 detection을 선택한 후 B에서 제거하고 D(final object detection set)에 추가
- 세트 B에서 M과 임계값 N_t 보다 큰 중복을 가지는 모든 box를 제거
- 남은 box B에 대해 반복

NMS는 인접한 detection의 score를 0으로 설정
이때 만약 중복된 box 안에 실제로 object가 존재했다면 객체가 누락되고 [[Average Precisoin]]이 감소한다.

이 대신 Soft NMS에서는 중복 정도가 증가함에 따라 detection score를 감소시킨다.

# Related work

# Background

Proposal network
- 앵커 박스에 대한 classification score와 regression offset을 생성
- 앵커 박스들을 순위를 매기고 상위 k(~~ 6000)개의 앵커를 선택하여 각 앵커에 대한 이미지 레벨 좌표를 얻기 위해 bbox regression offset을 추가

classificatoin network
- 각 proposal에 대한 classification 및 regression score를 생성
- 하나의 객체에 여러개의 proposal 생성 가능
- 첫번째 bbox 외에는 false positive 생성 -> 이를 위해 지정된 중복 threshold를 가진 각 class의 detection box에 대해 NMS가 독립적으로 수행

detection 수가 적고 작은 임계값 아래의 detection을 제거함으로써 computation 측면에서 효율적

# Soft-NMS

낮은 N_t 로 모든 인근의 detectoin box를 억제하면, 누락률이 증가


낮은 O_t, 높은 N_t : false positive 증가
<p align="center"><img src="/assets/images/Paper/SoftNMS/20240225215156.png"></p>

<mark style="background: #ADCCFFA6;">Pruning function</mark>

기존의 nms 함수
<p align="center"><img src="/assets/images/Paper/SoftNMS/20240225215859.png"></p>

고려해야할 점
- 인접한 detection score는 false positive 비율을 증가시킬 가능성이 적어지는 정도까지 감소되어야 하며 명백한 false positive보다는 높아야 한다
- 낮은 NMS 임계값으로 인접 detection을 전부 제거하는 것은 최적이 아니다. (높은 중복 threshold에서 누락률 증가)
- 높은 NMS threshold가 사용될 떄 다양한 중복 threshold에 걸쳐 AP가 떨어질 것

첫번쨰 soft nms 함수
<p align="center"><img src="/assets/images/Paper/SoftNMS/20240226020527.png"></p>

M과 겹치는 box 들의 score를 감소시키는 방법
많이 겹칠수록 더 많이 감소 (FP 될 가능성 하락)
원래라면 많이 겹쳐야 FP인 박스가 사라졌는데, 그걸 낮춰줌으로써 많이 겹칠수록 detect evaluate threshold에서 걸러질 확률 올라감


M,bi의 IoU 가 일정 값 이상일 때, 0으로 억제하는 것이 아닌 score 감소
bi는 동일한 클래스 내의 박스

두번째 soft nms 함수
<p align="center"><img src="/assets/images/Paper/SoftNMS/20240226022449.png"></p>

pruning step with gaussian function

It would be ideal if the penalty function was continuous, otherwise it could lead to abrupt changes to the ranked list of detections

when the over- lap is low, it should increase the penalty gradually, as M should not affect the scores of boxes which have a very low overlap with it.

연속적인 함수를 사용하여 penalty를 gradually 하게 줌

<p align="center"><img src="/assets/images/Paper/SoftNMS/20240226033912.png"></p>


