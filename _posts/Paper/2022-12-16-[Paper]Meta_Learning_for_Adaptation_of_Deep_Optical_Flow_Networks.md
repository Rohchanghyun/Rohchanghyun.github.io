---

layout : single

title: "[Paper]Meta-Learning for Adaptation of Deep Optical Flow Networks"

excerpt: "Meta-Learning for Adaptation of Deep Optical Flow Networks 논문 리뷰"



categories:

- Paper

tags:




toc: true

toc_sticky: true



author_profile: true

sidebar_main: true



date: 2022-12-16

last_modified_at: 2022-12-16

---

  

## abstract

  

- training -> 특정 도메인의 데이터 세트에 크게 의존 -> 다른 도메인에서 성능이 약함
- 일부 training 데이터를 사용하여 대상 도메인의 input data에 더 민감하게 적응하는 방법을 학습

  

논문의 알고리즘이 각 input image에 적용되기 때문에, 기존의 unsupervised loss들을 통합한다.

  

이러한 방법은 동일한 양의 데이터를 사용하는 fine tuning에서 지속적으로 더 나은 성능을 보여주었고 오류가 많은 이미지에 대해 더 정확한 결과를 보여준다.

  

## introduction

  

광학 흐름은 한 쌍의 이미지 사이의 명백한 2D 모션 필드를 정의한다.(비디오에서 인접한 프레임 간의 픽셀 대응을 나타낸다.)

빠르게 움직이는 물체와 폐색과 같은 일반적인 가시성 문제로 인해 광학 흐름 추정이 어렵다.

  

매우 정확한 광학 흐름은 비디오에서 픽셀 대응의 성공적인 예측을 가능하게 하므로, 높은 잠재적 가치를 가지고 있으며 모션 추정, 객체 추적, 객체 추적 등과 같은 광범위한 응용 분야에 활용될 수 있다,

비디오 초고해상도, 모션 세분화.

  

기존 연구 성과가 현장에서 사용되는 실제 데이터에 완전히 적용되지 않을 수 있다는 우려가 제기될 수 있다.

  

 대부분의 실제 환경에 대한 일반 네트워크를 훈련하지 못하게 한 것은 광학 흐름 지상 실측 자료의 부족이었다. 따라서 테스트 도메인에서 풍부한 레이블이 지정된 데이터를 가정하는 것은 적절치 않다.

  

  

 데이터 세트의 작은 부분만을 사용하여 미세 조정하면 보이지 않는 나머지 데이터에서 적절한 성능을 얻을 수 없을 것이다.

  

 반면, 전체 테스트 데이터 세트에 대한 감독되지 않은 훈련도 좋은 성능을 보장하지 않으며 엄청나게 느릴 수 있다.

  

  limited number of labeled data in the test domain and a strictly restricted number of gradient descent iterations를 통해 테스트 타임 adaption을 가능하게 한다.

  

- 개별 테스트 input의 고유한 특성을 활용한다. -> 이를 위해 적응 단계에 unsupervised loss를 사용한다.

  

- 본 논문의 접근 방법은 네트워크가 대상 도메인의 입력에 더 민감해질 수 있도록 돕기 때문에 테스트 시간에 gt를 요구하지 않는다.

  

## proposed method

  

두개의 연속적인 비디오 프레임을 사용하여 기존의 flow estimation Networks로 optical flow Vt를 계산할 수 있다.

  

 식에서 f : 파라미터 θ를 갖는 flow estimation network

 Vt의 각 요소는 픽셀 위치에서 움직임 변위를 나타내는 2차원 벡터

  

기존 flow estimation network는 도메인이 같지 않을 때 입력 프레임을 처리할 때 어려움이 있다.

  

본 논문에서는 일반적인 fine tuning과는 달리 주어진 특정 입력에 대한 test time adaption을 허용하는 새로운 적응 기술을 제안.

  

먼저 문제 설정을 정의.

그 후 동기에 대한 정당성을 제공하고 unsupervised optical flow loss에 대한 배경을 구축.

마짐가을 알고리즘과 독창성을 설명

  

### 3.1 test time adaptation of flow networks

  

특정 데이터 세트의 모션 분포를 통해 training 된 기존의 flow estimation network는 모션 분포가 다른 입력 프레임을 처리하는데에 있어 어려움이 있다.

\-> pre training된 네트워크의 매개 변수들을 새 테스트 도메인에 적용해야 한다.

  

 내부 동작 통계를 활용하여 테스트 단계에서 주어진 특정 입력에 흐름 네트워크를 적응시키는 것을 목표로 한다.

그러나 테스트 단계에서 지상 실측 정보를 사용할 수 없기 때문에 테스트 시간 적응을 위해 네트워크를 비지도 방식으로 훈련시킬 수 있는 기존의 비지도 손실을 사용한다.

  

### 3.2. Meta-learning for test-time adaptation

  

unspervised loss function을 최소화 함으로써 pre-trained model을 새 도메인의 각 테스트 입력에서 개별적으로 업데이트 할 수 있다.

  

단일 도메인의 optical flow 는 일반적으로 많은 모션을 포함하기 때문에 적은 수의 주석이 달린 데이터 세트만으로 flow network를 메타 훈련하고, 네트워크 매개변수를 unsuservised 방식으로 테스트 시간의 입력에 적용한다.

  

### 3.2.1 Meta train for new domains 

논문의 메타 트레인 단계는 두가지 업데이트 단계로 구성된다

내부 업데이트 단계에서 unsupervised loss 를 사용하여 네트워크 매개변수를 조정하고, 외부 업데이트 단계를 통해 레이블이 지정된 메타셋을 거의 사용하지 않고 메타 최적화를 수행한다.

  

논문에서 해결하려는 task는 두개의 연속적인 비디오 프레임과 이에 해당하는 ground truth optical flow Vt 로 구성된다.

  

각 업데이트 단계에서 균일한 분포에서 n개의 작업을 무작위로 샘플링하고 unspervised 방식으로 각 작업에 대한 네트워크 매개변수를 조정한다

  

수렴할때까지 이 절차를 반복하고 제안된 메타 트레인 알고리즘을 통해 특정 도메인의 유사한 작업에 걸쳐 매개변수를 일반화 할 수 있다.