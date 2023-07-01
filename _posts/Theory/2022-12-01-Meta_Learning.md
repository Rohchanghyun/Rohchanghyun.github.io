---
layout : single
title:  "Meta Learning"
excerpt: "Meta Learning 공부"

categories:
  - Theory
tags:
  - Theory
  - Meta Learning

toc: true
toc_sticky: true

author_profile: true
sidebar_main: true

date: 2022-12-01
last_modified_at: 2022-12-01
---
# 메타 러닝

- 데이터의 패턴을 정해진 프로세스로 학습하는 것이 아니라 데이터의 특성에 맞춰 모델 네트워크 구조를 변화시키면서 학습
- 하이퍼 파라미터 최적화(fine-tuning), 자동 신경망 네트워크 설계

  

## few shot learning 

- 적은 데이터를 효율적으로 학습하는 task
- 메타러닝 -> few shot learning 해결하기 위한 기반 알고리즘

  

- 서포트 셋 : 학습에 사용하는 데이터
- 쿼리 셋 : 테스트에 사용하는 데이터
- n-way,k-shot : n개의 카테고리, k개의 카테고리 당 이미지

  

## MAML 알고리즘 (Model Agnostic Meta Learning)

- 각 task에서 계산했던 그래디언트를 합산하여 모델을 업데이트