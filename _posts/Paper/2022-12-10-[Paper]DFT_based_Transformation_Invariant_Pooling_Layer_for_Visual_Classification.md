---

layout : single

title: "[Paper]DFT-based Transformation Invariant Pooling Layer for Visual Classification"

excerpt: "DFT-based Transformation Invariant Pooling Layer for Visual Classification 논문 리뷰"



categories:

- Paper

tags:




toc: true

toc_sticky: true



author_profile: true

sidebar_main: true



date: 2022-12-10

last_modified_at: 2022-12-10

---

## abstract

- CNN을 위한 이산 푸리에 변환 기반 풀링 레이어를 제안한다.
- 이산 푸리에 변환 기반 풀링은 컨볼루션과 완전히 연결된 레이어 사이의 전통적인 최대 평균 풀링 레이어를 대체하여 푸리에 변환의 이동 정리에 기초한 변환 불변성 및 형상 보존 특성을 유지한다.
- 풀링 단계에서 중요한 구조 정보를 유지하면서 이미지 정렬 오류를 처리할 수 있는 능력 -> 분류 정확도 크게 향상

  

중간 컨벌루션 레이어 출력을 사용하는 앙상블 네트워크를 위한 DFT + 방법을 제안  
  

## Introduction

- convolution response는 이미지 내용에 따라 결정될 뿐만 아니라 이미지에서 대상 개체의 위치 크기 및 방향에 따라 영향을 받는다.
- 이러한 비정렬 문제를 해결하기 위해 여러 cnn 모델이 사용하는것이 pooling layer이다.
- 컨벌루션 레이어와 fc layer 사이에 배치되어 각 채널의 컨볼루션 출력을 평균화 하여 다중 채널 2d 응답 맵을 1d 기능 벡터로 변환한다.

  
채널별 평균은 입력 형상 맵에서 활성화된 뉴런의 위치를 무시한다.  
모델이 정렬 오류에 덜 민감해지는 반면, 컨벌루션 출력의 공간 분포는 fc-layer로 전달되지 않는다.  
  
average pooling 에 관계 없이, 변환 뷸번성과 모양 보존 특성이 동시에 보존되지 않는데, 본 논문에서는 DFT 크기 풀링이 두가지 특성을 모두 유지하고 결과적으로 분류 성능을 크게 향상시킨다는 것을 보여주었다.  
  
입력 피처 맵의 각 채널에 2d-dft가 적용되며, 크기는 완전히 연결된 레이어에 대한 입력으로 사용된다.  
  
또한 고주파 계수를 폐기함으로써 중요한 형상 정보를 유지하고 노이즈의 영향을 최소화 하여 완전히 연결된 다음 계층에서 매개 변수의 수를 줄일 수 있다.  
  
중간 컨벌루션 레이엉의 응답을 앙상블하는 DFT + 방법까지 제안  
  
중간 계층의 출력 크기는 마지막 컨볼루션 계층의 출력 크기보다 훨씬 크지만 DFT 는 최종 출력의 유사한 해상도와 일치하는 경우에만 상당한 푸리에 계수를 선택할 수 있다.  
  
  

### robustness to certain object deformations

### proposed algorithm

### 2d shift Theorem of DFT

### DFT Magnitude Pooling Layer

  
컨벌루션 레이어는 M x M x C 피처 맵을 생성한다. 이 feature map은 각 채널의 뉴런 활성화를 나타내며, 모양과 위치를 포함한 시각적 특성을 인코딩하여 서로 다른 객체 클래스를 구별하는데 사용할 수 있다.  
  
average / max pooling은 위치 종속성을 제거하지만 동시에 중요한 모영 정보를 삭제한다  
  
DFT pooling layer에서 2d-dft는 각 채널에 적용된다.  
  
입력 feature map과 결과 푸리에 계수는 고주파 성분을 차단하여 N x N 으로 자른다.(N : 하이퍼 파라미터)  
  
나머지 저주파 계수는 완전히 연결된 다음 계층으로 공급  
3.1 에서 볼 수 있듯이 DFT pooling된 계수의 크기는 바뀌지 않으며 제안된 방법은 DFT 의 더 많은 pooling 된 계수를 사용함으로써 입력에 더 많은 형상 정보를 전파할 수 있다.  
또한 신호의 평균이 DFT 에 포함되기 때문에 average pooling을 완벽히 대체한다.  
  
푸리에 계수의 저주파 부분만 선택함으로써 공간 정보를 많이 손실하지 않고 feature 크기를 줄여 fclayer에서 매개변수를 많이 줄인다.  
  

### Late Fusion in DFT+

일반적인 CNN에서는 최종 컨벌루션 레이어의 출력만 분류에 사용된다. 그러나 중간 컨볼루션 레이어는 최종 레이어의 출력과 함께 사용될 수 있는 풍부한 시각적 정보를 포함한다.  
  
융합 계층에서는 중간 계층과 최종 계층의 모든 확률론적 추정치가 벡터화 되고 연결되어, SVM이 최종 결정한다.  
더 많은 시각적 정보를 통합하기 위해 중간 계층 그룹을 사용한다.  
  
각 레이어 그룹은 도일한 크기의 둘 이상의 컨벌루션 레이어로 구성되며 융합 수준에 따라 훈련 및 테스트에 사용되는 그룹의 수가 다르다.  
  
[http://cvlab.hanyang.ac.kr/](http://cvlab.hanyang.ac.kr/) <- 코드 구현