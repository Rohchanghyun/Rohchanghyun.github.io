---

layout : single

title: "[Paper]Signing outside the studio : Benchmarking background Robutness for Continuous Sign Language Recognition"

excerpt: "Signing outside the studio : Benchmarking background Robutness for Continuous Sign Language Recognition 논문 리뷰"



categories:

- Paper

tags:




toc: true

toc_sticky: true



author_profile: true

sidebar_main: true



date: 2023-01-16

last_modified_at: 2023-01-16

---

  

## Abstract  
    - background에 robust한 continuous sign language recognition모델을 만드는것이 목표.
    - 현존하는 CSLR 벤치마크는 대부분 배경이 고정되어 있으며 정적이고 단색 배경의 스튜디오에서 촬영된다.
    - 하지만 background가 움직이는 환경에서 CSLR 모델의 견고성을 분석하기 위해 다양한 배경에서 기존의 최첨단 CSLR모델을 평가한다.
    - 다양한 배경을 가진 sign비디오를 합성하기 위해 기존 cslr벤치마크를 활용하여 벤치마크 데이터 세트를 자동으로 생성하는 파이프라인을 제안.
    - 이러한 방법으로 새로 구상된 데이터 세트는 실제 환경을 시뮬레이션 하기 위한 다양한 장면으로 구성.
        - 배경 무작위화
        - CSLR모델을 위한 feature 분리

## Introduction  
    - 배경 이미지가 고정되고 단색인 데이터셋 -> CSLR모델의 실용성을 제한
    - 스튜디오 외부에서 새로운 데이터 세트를 구성하는것은 annotation관련 비용적으로 어려움이 있다.
    - 이를 해결하기 위해 다양한 배경을 가진 데이터셋을 합성하는 파이프라인을 제안
    - 이 과정에서 배경 데이터 세트에서 자연 배경을 선택하고 기존 벤치마크의 테스트 세트에 합친다.
    - 이렇게 생긴 데이터 세트의 이름: Scene-PHEONIX
    - 이 데이터 세트를 기반으로 현재 CSLR접근방식이 배경이 바뀌는 것에 강하지 않다는 것을 알게되었다.
    - 해결방안
        - 배경 무작위화 -> 훈련을 위한 사인 비디오가 믹스업을 통해 장면 이미지와 결합되어 배경 이동(배경이 바뀌는것)을 시뮬레이션한다.
        - latent space에서 signer의 feature와 배경 feature를 분리하는 것을 목표로 하는 disentangleing Auto-Encoder(DAE)를 제안

## Benchmarking Background RObustness  
    - > Background bias  
        - CSLR 비디오는 일기예보나 studio recordings에서 얻어지기 때문에 배경이 고정되어 있다.
        - 때문에 CSLR모델들은 같은 배경 분포에서 학습되고 배경을 가진 비디오에 대해 일반화되지 않는다.
    - > Robust Dataset Construction  
        - 기존 데이터 세트를 활용하는 자동 CSLR 벤치마크 데이터 세트 생성 알고리즘 제안

## Background Agnostic Framework  
    - 배경 무작위화
    - DAE
    - 시퀀스 모델은 분리된 disentangled signer와 함께 제공된다.
    - 우리는 meta-architecture를 CTC loss와 함꼐 훈련한다.
    -  Background Randomization  
        - 추가적은 배경 이미지를 활용하여 무작위 배경을 가진 비디오를 만들 것을 제안
        - 훈련 중 테스트때 사용되는 배경을 추가하는것은 견고성을 향상시키는 사소한 방법으로 볼 수 있다.
        - 따라서 견고성을 더 잘 테스트하고 잠재적 비용을 줄이기 위해 훈련 중에 사용할 수 있는 배경 이미지 수를 제한한다.
        - k : 우리가 샘플링하는 이미지의 수
        - 무작위 배경 이미지로 대상 비디오의 convex sum을 얻는다.
    -  Disentangling Auto-Encoder  
        - 배경 무작위화는 CSLR모델의 배경에 관해 robustness를 향상시키지만 이를 더욱 향상시키는 DAE를 추가로 제안한다.
        - 이를 설계할 때, 입력 비디오가 임베딩 공간에서 signer feature와 background feature로 분리될 수 있다고 가정한다.
        - 프레임워크는 teacher와 student로 나뉘어져 있다.
        - teacher : 원본 sign 비디오, student : 배경 무작위 sign비디오 를 입력으로 사용
        - 각 입력 비디오는 2d CNN을 통과한 다음 average pooling 없이 flatten 되어 d차원 벡터 fk 및 f9를 얻는다.
        - 순차적으로 키 인코더와 쿼리 인코더는 각각 fk와 f9를 hk와 h9에 내장한다.
        - 여기서 물리적으로 각 잠재적 특징(hk,h)를 두 부분으로 나누거 분할된 잠재적 특징은 signer특징과 배경 특징 hb로 구성된다고 가정
        - 차별적인 잠재 기능을 내장하기 signer feature가 서로 당겨지고 배경 feature가 서로 밀어지도록 추가 훈련 목표 제공
        - 잠재적 특징(hq,hk)이 수화 특징(hs)과 배경 특징(hb)으로 완벽하게 분리되는 경우 signer feature hq와 hk사이에 차이가 없어야 한다.
        - 마지막으로 오직 hq만이 gloss시퀀스를 예측
        - 네트워크가 배경에 구애받지 않는 방식으로 signer에게 더 집중할 수 있도록 CTC loss 전파.
    -  Objective Function