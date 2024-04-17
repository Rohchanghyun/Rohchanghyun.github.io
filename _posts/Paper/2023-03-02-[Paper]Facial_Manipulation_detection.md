---
layout : single
title:  "[Paper] Generalized Facial Manipulation Detection with Edge Region Feature Extraction"
excerpt: "Generalized Facial Manipulation Detection with Edge Region Feature Extraction 논문 정리"

categories:
  - Paper
tags:
  - detection

toc: true
toc_sticky: true

author_profile: true
sidebar_main: true

date: 2023-03-02
last_modified_at: 2023-03-02
---

## Abstract

* * *

  

본 논문은 이미지에 나타나는 edge 영역 feature를 기반으로 일반화되고 강력한 face manipulation detection방법을 제시

대부분의 얼굴 합성 프로세스는 color awkwardness reduction을 포함하지만 이는 edge 영역에서 natural fingerprint 손상시킨다.

이러한 color awkwardness reduction은 배경 영역에서 진행되지 않고 합성 프로세스는 시간 영역에 나타나는 이미지의 자연적 특성을 고려하지 않는다.

- 전체 이미지의 edge 영역에 나타나는 픽셀 수준의 색 특징을 활용하는 face forensic framework 제안.
- 추출된 색 특징을 공간적 및 시간적으로 해석하는 3d - cnn분류 모델을 포함
- 하나의 비디오 내에서 여러 프레임에서 추출된 모든 특징을 고려하여 진위 판단을 수행한다.

  

## Introduction

* * *

  

현재 나오는 얼굴 합성 방법의 공통적인 특징

1. 후처리는 얼굴 부분에만 수행된다.
2. 얼굴 합성 과정은 시간 개념을 포함하지 않는다.

  

이러한 관찰을 바탕으로 합성 이미지의 일반화된 검출 모델을 제안.

- 얼굴 합성 과정에서 색상 분포 변화를 통한 강력한 얼굴 이미지 포렌식 방법에 중점을 둔다.
- 시간 개념을 포함하는 분류 모델이 도입되어 탐지 성능이 향상된다.

  

본 논문의 프레임 작업은 전체 이미지에 걸쳐 나타나는 엣지 속성을 반영하고 이러한 기능을 시공간적으로 해석한다는 점에서 독특하다.

  

## Methods

* * *

  

### Edge Region Feature

얼굴 부분과 가장자리 영역의 배경 영역 모두에서 색상 차이 특징을 추출하여 이점에 주의

  

#### Facial Edge Region Feature

- 비디오에서 추출한 이미지 프레임 I를 사용하여 얼굴 영역을 dilb의 정면 얼굴 검출기로 검출
- 그 다음 dilb에서 68개의 얼굴 랜드마크 모양 예측기로 검출된 얼굴에서 68개의 특징점을 추출
- 이 중 얼굴 윤곽을 나타내는 처음 17갸의 특징점을 얼굴 경계점으로 정의
- 얼굴 가장자리 영역을 따라 색상 특징을 얻기 위해 얼굴 경계선에 수직인 선을 생성
- 모아진 선으로 경계 영역 특징을 공간적으로 반영하기 위해 window모양 특징을 만든다.
- 창모양의 20\*20 대응 좌표는 20개의 수평선의 집합에 의해 지정
- 이러한 400개의 좌표 특징을 엣지 윈도우 w 라 한다.
- 얼굴 경계점을 갖는 단일 이미지 I당 16개의 윈도우 생성
- 이후 엣지 윈도우의 RGB값이 얼굴 안쪽에서 바깥쪽으로 어떻게 변화하는지 알아보기 위해 파라미터 t에 따라 RGB픽셀값의 절대적인 차이를 계산 → 이 RGB값 차이는 얼굴 경계 특징인 F

  

#### Background Edge Region Feature

- 배경 edge point의 특징을 추출하기 전에 이미지 I에서 배경 엣지 이미지 EI를 생성하고 먼저 5\*5 가우시안 필터로 이미지I를 블러 처리하여 배경의 noise를 제거하고 독특한 경계 부분을 추출
- canny엣지 디텍터를 사용하여 얼굴의 엣지 포인트를 포함하는 러프 엣지 영상 EI가 생성된다.
- 엣지 이미지를 사용하여 전체 엣지 포인트의 10%를 랜덤 샘플링하여 배경 엣지를 따라 나타나는 통계 정보를 얻는다.
- 얼굴 경계 특징과 동일한 방법으로 샘플링된 엣지 포인트로 배경 윈도우를 추출
-