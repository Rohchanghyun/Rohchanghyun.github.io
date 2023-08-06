---
layout : single
title:  "Image Perturbation-Based Deep Learning for Face Recognition Utilizing Discrete Cosine Transform"
excerpt: "Image Perturbation-Based Deep Learning for Face Recognition Utilizing Discrete Cosine Transform 논문 정리"

categories:
  - Paper
tags:
  - face recognition

toc: true
toc_sticky: true

author_profile: true
sidebar_main: true

date: 2023-03-27
last_modified_at: 2023-03-27
---

## Abstract

감정 분류 및 얼굴 속성 분류를 포함한 얼굴 인식은 많은 사용자로부터 수집된 대규모 데이터 덕분에 많이 발전하였다. 하지만 민감한 생체 정보와 연결될 경우 개인정보 유출을 일으킬 수 있다.

이를 위해 본 논문에서는 이산 코사인 변환 계수 절단 방법을 사용

- DCT와 픽셀화를 결합

  

## Introductoin

프라이버시 보존하기위한 다른 연구들 

- 암호학 기반 딥 러닝 : 모델 정확도를 손상시키지 않고 민감한 내용을 암호화하여 프라이버시 위험정보를 보호

- 연합학습 : 집합 서버가 훈련된 모델만 보기 때문에 중앙 집중식 모델보다 프라이버시에 이점을 제공

  

그러나 이 방법들은 서버가 필요하기 때문에 본 논문에서는 이미지 기반 프라이버시 보존 방법에 초점을 두고 있다.

이미지 배포 단계에서 이미지를 변환하여 눈이 원본 이미지를 인식할 수 없도록 할 수 있다.

  

- 픽셀화 및 이산 코사인 변환. DCC를 기반으로 한 개인정보 보호 이미지 perturbation 방법 제안

  

## Methods

전체 프로세스

1. 이산 코사인 변환 및 픽셀화
2. 계수 절단 방법의 세부 정보
3. 역 DCT 설명하고 얼굴 이미지에 적용된 DCC의 결과를 제시

  

### Discrete Cosine Transform (DCT)

- DCT → 역 DCT를 위해 이미지를 공간 도메인에서 주파수 도메인으로 변환
- 1차원 DCT는 신호 처리에 사용되고 2차원 DCT는 이미지 처리에 사용된다
- 영상을 주파수 영역으로 변환하여 DCT 계수 행렬 생성
- 변환 결과에서 흰색 픽셀이 왼쪽 상단에 집중된다.
- 픽셀이 흰색일수록 DCT 계수가 크다.(절대값)
- 낮은 주파수와 관련된 DCT값이 클수록 원래 영상에서 공간 영역으로의 변환에 필수적인 부분을 나타낸다 → 인간의 눈이 영상에서 저주파 성분을 더 잘 감지하는 경향이 있기 때문이다.

  

### Coefficient Cutting(CUT)

- 주요 아이디어는 고주파수를 대부분 생략하더라도 민감한 정보는 숨기는 동안 영상의 주요 특징은 그대로 유지된다는 것.
- 각 블록에 대해 가장 큰 DCT계수를 선택하여 DCC계수 행렬에 저장하여 각 블록에 대해 적어도 하나의 DCT 계수를 유지하도록 하였다.
- 선택된 계수를 제외한 나머지 블록이 아닌 전체 영상에 대해 상위 DCT계수를 선택하여 DCC 계수 행렬에 저장하고 나머지 계수는 폐기
- 값 r은 프라이버시 강도를 조절하는 나머지 DCT계수의 갯수
- r값이 클수록 프라이버시 강도가 낮음을 의미

  

### Inverse Discrete cosine Transform(I-DCT)

- 주파수 영역을 공간 영역으로 변경