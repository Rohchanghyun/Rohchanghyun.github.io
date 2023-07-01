---

layout : single

title: "[Paper]Modeling Semantic Correlation and Hierarchy for Real-world Wildlife Recognition"

excerpt: "Modeling Semantic Correlation and Hierarchy for Real-world Wildlife Recognition 논문 리뷰"



categories:

- Paper

tags:

- Paper




toc: true

toc_sticky: true



author_profile: true

sidebar_main: true



date: 2022-12-19

last_modified_at: 2022-12-19

---
# Modeling Semantic Correlation and Hierarchy for Real-world Wildlife Recognition

  

## Abstract

- neural nework로 야생 동물 인식 데이터 세트에 레이블을 지정
- 야생동물 이미지에서 인간의 annotation을 지원하는데 해결해야할 과제는 2개
    - training data set은 일반적으로 불균형 → 모델의 suggestion이 편향된다
    - class의 복잡한 분류
- 이러한 문제를 해결하기 위해 편향 손실 함수 및 쌍곡선 네트워크 구조를 포함 → 간단한 기준 설정
- training시에 모델에 co-occurence layer를 추가하여 모델을 보다 효과적으로 훈련
- 의미 상관 관계를 사용

  

## Introduction

- 야생 생물 개체군의 실제 데이터 수동 처리 → 많은 비용과 시간
- deep neural network를 활용하여 인간 annotator와 협력하여 실제 야생동물 데이터 세트를 효율적으로 처리하는 human in the loop 방법 제안
- 신경망을 이미지 분류 모델로 훈련시켜 레이블이 지정되지 않은 이미지의 클래스를 제안함으로써 인간 annotator가 야생 동물 인식 데이터 세트에 레이블을 지정하는것을 적극적으로 지원
- 네트워크 training시 2가지 주요 구별 지점

1. 불균형 데이터 분포
2. 클래스 계층 구조

- 이를 위해 이전의 클래스 분포를 활용하여 간단하지만 효과적인 편향 제거 loss를 활용
- 쌍곡선 신경망 → 레이블으 ㅣ계층 구조를 더욱 효과적으로 학습하기 위해
- 클래스 간의 공존을 모델링하여 의미적 상관관계 학습
- 모델의 최종 layer위에 학습 가능한 co-occurence matrix를 추가하여 클래스 예측 확률 개선

  

## Modeling Semantic Correlation and Hierarchy

- 입력 이미지 x와 레이블 y를 고려할 떄, 우리의 목표는 classifier를 학습하는것

1. 손실 편향 제거
- logit adjustment라는 널리 사용되는 이미지 분류방법
- 훈련중 debiasing loss function으로 logit adjustment적용
- logit adjustment loss → 
3. Hyperbolic Models
- 계층 구조를 효과적으로 학습하기 위해 classifier에 쌍곡선 모듈 추가 → 쌍곡선 공간을 활용하여 계층 구조로 데이터를 내장하는 쌍곡선 신경망을 만든다
- 유클리드 특징을 분류를 위해 쌍곡선 공간으로 끌어올린다
5. Learning with semantic Correlations
- 자연적인 상관관계 사용
- 모델이 희귀하지 않은 클래스의 높은 가능성을 제공하는 경우 모델은 희귀한 클래스의 유사한 가능성도 제공