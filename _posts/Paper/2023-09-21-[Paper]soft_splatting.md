Softmax Splatting for Video Frame interpolation

## Abstract

역방향 warping 형태의 미분 가능한 이미지 샘플링은 깊이 추정 및 optical flow 예측과 같은 작업에서 광범위하게 사용되었다. 반대로 순방향 warping은 부분적으로 미분 가능한 방식으로 여러 픽셀을 동일한 대상 위치에 mapping하는 충돌을 해결해야 하느 ㄴ등의 문제로 인해 관심이 적었다.

본 논문에서는 이러한 패러다임 변화와 반대로 softmax splatting을 제안하고, frame interpolation으로의 적용이 효과적이라는 것을 보여준다.

- 두 개의 입력 프레임이 주어지면 softmax splatting을 사용하여 optical flow estimation을 기반으로 프레임과 feature pyramid representation을 순방향 warping한다.
- 이로 인해 softmax splattin은 여러 소스 픽셀이 동일한 대상 위치에 mapping되는 경우를 원활하게 처리한다.
- 이후 합성 네트워크를 사용하여 warping된 representation의 보간 결과를 예측한다. softmax splatting을 통해 임의의 시간의 프레임을 보간할 수 있을 뿐만 아니라 feature pyramid와 optical flow를 미세 조정할 수 있다.



## Introduction

frame interpolation 에 대한 접근법은 flow base, kernel base의 2가지 방법으로 나뉠 수 있다.

본 논문에서는 flow based 패러다임이 벤치마크에서 잘 작동하는 것으로 입증되었기 때문에 flow based 패러다임을 채택한다.

flow base 방법의 한가지 일반적인 접근법은 합성해야 하는 프레임의 관점에서 두 입력 프레임 I_0과 I_1 사이의 optical flow F_t0 및 F_t1을 추정하는 것이다. 보간 결과는 F_t0 및 I_1에 따라 역방향 warping으로 얻을 수 있다.

이 방법은 기성의 optical flow를 사용하기 어렵게 하고 임의의 t에서 프레임을 자연스러운 방식으로 합성하는 것을 막는다.

이를 해결하기 위해 F_01 및 F_10에서 F_t0 및 F_t1을 근사화한다.

다른 접근방법으로는 역방향 warping과는 달리 직접 순방향 warping을 수행했는데, 이 방법은 여러 소스 픽셀이 동일한 대상 위치에 mapping되는 경우를 처리하기 위해 z버퍼링과 같은 것을 사용한다. 따라서 z 버퍼링으로 인해 완전히 구별하는 방법은 명확하지 않다.



이러한 문제를 해결하기 위해, 본 논문에서는 softmax splatting을 제안하여 이를 통해 순방향 warping에 대한 모든 입력을 공동으로 감독할 수 있다.

결과적으로 일반 context map을 학습하고 task별 feature pyramid를 warping하는 아이디어를 확장할 수 있다.

또한 optical flow estimator 뿐만 아니라 동일한 위치로 warping될 때 다른 픽셀의 중요도를 가중하는 메트릭을 감독할 수 있다.



### key ideas

- 차별화 가능한(충돌 픽셀을) 순방향 warping을 수행하고 프레임 보간의 적용에 대한 효과를 보여주기 위해 softmax splatting을 제안.
- 주 연구 문제는 동일한 대상 위치에 mapping되는 다른 소스 픽셀을 차별화 가능한 방법으로 처리하는것.
- 이미지 합성을 위해 task별 feature pyramid를 훈련하고 사용할 수 있게 해준다.
- 비디오 프레임 보간을 위한 기성 optical flow estimator를 미세 조정할 수 있고 여러 소스 픽셀이 동일한 순방향 warping된 위치에 mapping되는 경우를 명확하게 나타내는 데 사용되는 메트릭을 감독할 수 있다.



## Method

### Softmax Splatting for Frame Interpolation

프레임 보간은 2개의 프레임(I_0,I_1)이 주어지면 중간 프레임 합성을 목표로 하며, t는 원하는 시간적 위치를 정의한다.

이 문제를 해결하기 위해, 먼저 기성의 optical flow방법을 사용하여 입력 프레임 사이의 optical flow 를 양방향으로 측정하고, 그 다음 softmax splatting 형태의 순방향 warping을 사용하여 I_0를 warping한다.

이는 F_t0 및 F_t1이 필요하지만, F01 및 F10에서 이러한 t 중심 광학 흐름을 계산하는 것이 복잡하고 근사치에 영향을 받는 역방향 warping w와는 대조적이다.

이후 합성 네트워크를 사용하여 이를 얻기 위해 이러한 중간 결과를 결합한다. 구체적으로 입력 프레임을 color로 warping할 뿐만 아니라 합성 네트워크가 더 나은 예측을 할 수 있도록 여러 해상도에 걸친 feature space도 제공한다.

<p align="center"><img src="/assets/images/Paper/SoftSplatting/figure_1.png"></p>

#### Forward Warping via Softmax Splatting

mapping 시 동일한 target pixel에 위치하는 충돌을 해결하기 위한 softmax splatting을 설명한다. 



- **Summation splatting**

<p align="center"><img src="/assets/images/Paper/SoftSplatting/figure_2.png"></p>

<p align="center"><img src="/assets/images/Paper/SoftSplatting/figure_3.png"></p>

그림에서 보는 바와 같이, summation splatting은 자동차의 라이트같은 중첩된 영역에서 밝기 불일치를 유발하고, 이중선형 커널 b는 I_0의 픽셀들로부터 부분적인 contribution만을 받는 I의 픽셀들로 이어지게 하고, 이는 다시 밝기 불일치를 가져온다.

그러나 본 논문에서는 이 summation splatting을 모든 forward warping 접근의 기초로 사용한다.

<p align="center"><img src="/assets/images/Paper/SoftSplatting/figure_4.png"></p>



- **Average splatting**

밝기 불일치를 해결하기 위해, 이를 정규화할 필요가 있다.

떄문에 본 논문에서는 Σ의 정의를 재사용하여 average splatting Φ 를 결정한다.

<p align="center"><img src="/assets/images/Paper/SoftSplatting/figure_5.png"></p>

이 방법은 위의 그림과 같이 밝기 불일치를 처리하지만, 잔디와 같은 겹치는 지역까지 더하여 평균한다.



- **Linear splatting**

중첩 영역을 더 잘 분리하기 위해, importance mask z에 의해 I_0에 선형 가중치를 부여하고 다음과 같이 Linear splatting을 정의

<p align="center"><img src="/assets/images/Paper/SoftSplatting/figure_6.png"></p>

이 방법은 배경을 더 잘 분리할 수는 있지만, z에 대한 translation에 대해 불변하지는 않다. 예를들어, 만약 z가 inverse depth를 나타낸다면, 자동차는 z = 1/1이고 배경은 z = 1/10이라면 잘 분리될 것이다. 하지만 자동차는 z = 1/101이고 배경은 z = 1/110이라면, 깊이 면에서 비슷하게 떨어져 있음에도 불구하고 다시 평군이 될 것이다.



- **Softmax splatting**

translational invariance를 갖는 중요도 마스크 z에 따라 중첩 영역을 명확하게 분리하기 위해, 다음과 같이 softmax splatting을 제안한다.

<p align="center"><img src="/assets/images/Paper/SoftSplatting/figure_7.png"></p>

예를 들어 z가 픽셀의 깊이가 관련이 있다 가정하자. 이 접근법은 풀의 흔적 없이 자동차의 앞과 뒤를 명확히 분리할 수 있다. 게다가 softmax 함수와 resemblance(유사성)을 공유한다. 

따라서 여러 픽셀을 동일한 위치에 mapping할 때 중요한 특성인 z에 대한 변환 β 에는 불변성을 가진다.

scale에는 불변하지 않지만 z에 α를 곱하면 중복된 지역이 얼마나 잘 분리되는지에 영향을 줄 것이다.

α가 작으면 평균이 되고, α가 크면 z-buffering이 된다. 이 매개변수는 end-to-end training 을 통해 학습할 수 있다.



- **Importance metric**

아래와 같이 역방향 warping을 통해 얻을 수 있는 occlusion의 척도로 밝기 일정성을  사용한다. 제안한 softmax splatting은 완전히 미분 가능하기 때문에  α를 학습할 수 있을 뿐만 아니라 작은 신경망 v 를 사용하여 이 메트릭을 더욱 정제할 수 있다.



<p align="center"><img src="/assets/images/Paper/SoftSplatting/figure_8.png"></p>

v에서 직접 z 를 얻을 수도 있지만 이 v를 수렴할 수는 없다.

마지막으로 프레임 보간과 다른 작업에 softmax splatting을 적용할 때 중요도 메트릭을 그에 따라 조정할 수 있다.



#### Feature Pyramids for Image synthesis

두 개의 입력 프레임이 주어지면, 먼저 기성의 optical flow 방법을 사용하여 프레임간 flow를 추정하고, 그 후 미리 정의된 필터 ψ을 사용하여 입력 영상에서 일반적인 컨텍스트 정보를 추출하고 합성 네트워크 φ를 사용한 식에 따라 컨텍스트 맵과 함께 이미지를 forward warp 하여 보간 결과를 얻는다.

<p align="center"><img src="/assets/images/Paper/SoftSplatting/figure_9.png"></p>

본 논문에서 제안한 softmax splatting은 ψ를 감독할 수 있게 하여 이미지 합성에 중요한 특징을 추출하는 방법을 학습할 수 있게 해준다.

또한 feature pyramid의 형태로 여러 척도에서 feature를 추출하고, warping 함으로써 이 아이디어를 확장한다.이를 통해 네트워크 ψ는 예측을 더 향상시킬 수 있다.



- **optical flow estimator**

기성 optical flow 방법을 사용한다. 구체적으로 PWC-Net을 사용하고 FlowNet과 Lite-FlowNet이 동일하게 잘 수행됨을 보인다. 프레임 보간을 위해 PWC-Net을 추가로 fine-tune 한다.



- **feature Pyramid extractor**

<p align="center"><img src="/assets/images/Paper/SoftSplatting/figure_10.png"></p>

본 논문에서 제안한 softmax splatting은 이 feature pyramid extractor를 end to end 방식으로 감독할 수 있게 하여, 뒤에 이미지 합성에 유용한 특징을 추출하는것을 학습할 수 있게 한다.



- **Image synthesis network**

warped image와 그에 대응하는 feature pyramid에 의해 guide되는 보간 결과를 생성한다.3개의 행과 6개의 열을 가진 GridNet을 사용한다.



- **Importance metric**

제안한 softmax splatting은 여러 픽셀이 동일한 대상 위치로 forward-warp하는 경우를 해결하는데 사용되는 importance metric을 사용한다. 이 메트릭을 계산하기 위해 밝기 일정성을 사용한다.

또한 pyramid extractor와 image synthesis network로 end to end간 훈련된 세가지 레벨로 구성된 작은 U-Net을 사용하여 occlusoin 추정치를 미세화한다.