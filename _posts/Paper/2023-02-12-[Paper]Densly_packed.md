---
layout : single
title:  "[Paper] Densely-packed Object Detection via Hard Negative-Aware Anchor Attention"
excerpt: "Densely-packed Object Detection via Hard Negative-Aware Anchor Attention 논문 정리"

categories:
  - Paper
tags:
  - object detection
  - attention

toc: true
toc_sticky: true

author_profile: true
sidebar_main: true

date: 2023-02-12
last_modified_at: 2023-02-12
---

## Abstract

densely-packed object detection → 많은 물체를 다루어야 하고 물체의 크기가 매우 작음.

때문에 작은 크기의 물체에서 충분한 feature를 추출할 수 없음

물체가 서로 가까이 존재하기 때문에 많은 중복 영역이 존재 → object detection 성능에 영향

  

densly-packed object를 감지하기 위해 이미지에서 영역의 많은 부분을 덮을 수 있는 조밀한 앵커를 사용하여 물체 영역도 찾을 수 있었다.

그러나 이런 앵커는 중복될 수 있으며 계산 비용이 늘어났다.

  

기존 앵커의 문제점 

- positive 앵커들 사이의 연관 있는 중요성을 신경쓰지 않았다.
- 훈련중 IOU값이 작은 positive 앵커를 hard negative 앵커로 간주하지 않았다.

  

본 논문에서는 이를 해결하기 위해

- Advanced weighted Hausdorff distance(AWHD): 물체의 중심영역 지도를 정확하게 추정
- Hard Negative-Aware Anchor(HNAA) : 각 앵커의 중요성을 결정하고 hard negative anchor를 중요하게 고려하는 HNAAattention 제시

2가지 방법을 기반으로 하는 새로운 densely-packed object detection 제안.

  

## Related Work

### Hausendorff distance

미터법 공간에서 두 점 집함 사이의 가장 긴 거리

앵커 attention을 위한 objectiveness map을 추정.

정확한 탐지 결과를 제공할 수 있는 advanced distance function을 도입

  

## Proposed Method 

이미지에서 객체 중심 영역을 추출하고 중심 영역에 해당하는 앵커에 초점을 맞춘다.

이를 위해 이 방법은 각 개체의 정확하고 구별 간으한 중심 영역을 정확하게 추출

### Advanced Weighted Hausendorff Distance

-