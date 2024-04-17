---
layout: single
title: "[Paper] CornerNet"
excerpt: "CornerNet 논문 리뷰"
categories:
  - Paper
tags:
  - object_detection
  - gaussian
  - Focal_loss
toc: true
toc_sticky: true
author_profile: true
sidebar_main: true
date: 2024-02-29
---
>CornerNet: Detecting Objects as Paired Keypoints
>ECCV 2018
>Hei Law, Jia Deng

# Abstract
CornerNet -> object <span style="color: #88c8ff"> bbox를 keypoint(top-left,right-bottom)의 쌍으로</span>single convolution neural network로 예측

이 방법을 통해 원래 single-stage에서 사용하던 anchor box를 design할 필요 사라짐

<span style="color: #88c8ff"> corner pooling</span> -> corner localize를 도와주는 새로운 pooling layer

# Introduction

이전의 object detection 은 CNN을 활용한 모델들이 좋은 성능
이중에서 SOTA 모델이 주로 사용한게 Anchor box
Anchor box : boxes of various sizes and aspect ratios that serve as detection candidates
- one-stage detector 에서 two-stage detector 못지 않게 성능을 낼 수 있고 효율적이라 자주 사용
- <span style="color: #ed6663"> 매우 많은 set의 anchor box가 필요 (DSSD 에서는 40k, RetinaNet에서는 100k) -> detector가 각 anchor box가 충분히 gt와 겹치는지를 classify 하기 때문. 정답을 확정짓기 위해서는 많은 anchor box 필요</span>
- <span style="color: #ed6663"> 많은 hyperparameter와 design choice들이 존재 -> 이 두개를 결정함에 있어 매우 복잡한 문제</span>

<span style="color: #88c8ff">CornerNet : new one-stage approach t object detection that does away with anchor boxes</span>
object의 bbox를 top-left corner 와 bottom-right corner의 keypoint의 쌍으로 표현

같은 object category의 top-left corner, bottom-right corner의 heatmap을 예측하기 위해 single convolutional layer 사용 + 각 코너의 embedding vector

embedding은 한 쌍의 corner들을 그룹화 하기 위해 사용 -> 네트워크는 비슷한 embedding을 예측하도록 훈련

이러한 방법을 통해 기존의 anchor box를 대체/출력을 간소화

<p align="center"><img src="/assets/images/Paper/CornerNet/20240229201852.png"></p>

<span style="color: #88c8ff">corner pooling</span>

<p align="center"><img src="/assets/images/Paper/CornerNet/20240229202302.png"></p>
convolutional network가 bbox corner를 더 잘 찾아내도록 도와주는 새로 제안한 pooling layer

그림에서 보면 종종 corner keypoint가 있는 부분이 location을 특정하기 위한 evidence가 없을 때가 있다.

<p align="center"><img src="/assets/images/Paper/CornerNet/20240229202742.png"></p>
이를 위해 top-left corner를 특정하기 위해서는 객체의 맨 위 경계에서 오른쪽으로 끝까지, 아래쪽으로 끝까지 이동 

이 개념으로 motivate

두개의 feature map 사용
첫번째 feature map에서는 각 채널마다 오른쪽으로 feature vector를 maxpool
두번째 feature map에서는 각 채널마다 아래쪽으로 있는 모든 feature vector에 대해 maxpool

2개의 pooling된 결과를 함께 추가

Anchor box보다 corner detecting이 좋은 2가지 이유
1. anchor의 중심이 객체의 4개의 방면과 연관되어 있기 때문에 corner는 2개 측면에서 보기 때문에 더 쉬움 + corner pooling을 통해 모서리에 대한 지식 encoding
2. box의 space를 빼곡히 이산화 하는데 효율적
just need $O(wh)$ corners to represent $O(w^2h^2)$ possible anchor boxes.

Ablation study를 통해 corner pooling이 매우 중요하다는 것을 보여줌

# Related works

## Two-stage object detectors

R-CNN에 의해 처음 나온 개념
RoI 를 생성하고 각각을 network에서 classification
이를 low-level vision algorithm을 통해 RoI 생성(selective search, Locating object proposals from edges)
이후 각자가 ConvNet에서 processed -> 많은 computation

SPPnet, Faster R-CNN 이 R-CNN 기반으로 feature map에서 region을 뽑는 새로운 pooling layer를 사용하여 성능 향상
그러나 두 방법 다 분리된 proposal algorithm에 의존 -> end-to-end로 학습할 수 없음

Faster R-CNN -> RPN 사용하여 low-level proposal algorithm 사용
이를 통해 detector가 더 효율적이게 되고 end-to-end로 train 가능

R-FCN 은 Faster R-CNN에서 fully-connected sub-detection network를 fully-convolutional sub-detection network로 교체하여 성능 향상

other works
- [[Subcategory-aware convolutional neural networks for object proposals and detection]] : 하위 category 정보 통합
- multi scale proposal
- [[Feature selective networks for object detection]] : select better feature
- [[Light-head R-CNN]] : 속도 향상
- [[Cascade R-CNN]] : cascade procedure
- [[An analysis of scale invariance in object detection-snip.]] : better training procedure

## One-stage object detectors

RoI pooling step을 줄인 one-stage 방식의 YOLO, SSD 방식이 주로 사용되었다
two-stage 방식에 비해 computationally efficient, 어려운 dataset에서 유사한 성능을 보여주었다

- SSD: multiple scale에서 anchor box들을 빼곡히 배치 -> 이를 통해 각 anchor box를 refine, classify
- YOLO: bbox를 image에서 뽑지만 이후 YOLO 9000에서 anchor box 사용
- DSSD,RON: hourglass network와 유사한 network 채택 -> low-level과 high-level feature를 skip connection을 통해 combine, 더 정확한 bbox 예측

이러한 one-stage 방법들은 RetinaNet 덕분에 two-stage를 이김
RetinaNet에서는 빼곡한 anchor box가 positive와 negative 사이에 큰 imbalance를 가져온다고 주장 -> training 비효율적
+anchor box의 weight를 dynamic하게 조정하는 Focal loss 사용

RefineDet: filtering을 통해 negative anchor box의 수를 줄이고 coarse하게 조정

DeNet: anchor box 없이 RoI 생성하는 two-stage detector 
top-left,top-right,bottom-left,bottom-right 에 대해 4개의 location이 여기에 속할 확률을 결정
이후 모든 가능한 corner 조합을 생성하여 이후 two-stage 방식을 따른다

본 논문의 방식은 이와 매우 다른 방법
1. 
- DeNet: 두 corner가 동일한 객체인지 식별하지 못하고 subnetwork에 의존하여 좋지 않은 RoI 거부
- CornerNet: 단일 Convnet을 사용하여 corner 탐지, 그룹화하는 one-stage detector

2. 
- DeNet: region과 비교하여 수동으로 결정된 위치에서 feature 선택
- CornerNet: feature selection step 없음

3. corner pooling

corner를 embedding vector를 통해 그룹화
3가지 특징
- corner pooling layer
- hourglass architecture
- focal loss

# CornerNet
## Overview

- 객체를 top-left, right-bottom 두개의 keypoint 쌍으로 detect
- conv layer는 다른 object category의 corner 위치를 나타내는 두 세트의 히트맵 예측(left-top, right-bottom)
- 감지된 corner에 대해 임베딩 벡터 예측하여 같은 객체의 두 모서리 사이의 임베딩 거리가 작도록 만든다
- bbox를 맞추기 위해 corner 위치를 조정해주는 offset도 예측
- 예측한 값들을 사용해 post processing 알고리즘을 적용하여 최종 bbox 생성

<p align="center"><img src="/assets/images/Paper/CornerNet/20240301101305.png"></p>

- hourglass network를 backbone으로 사용
- 이후 두개의 예측 모듈(top-left, right-bottom)
- 각 모듈은 히트맵, 임베딩, offset을 예측하는데 이 전 단계에서 corner pooling 사용
- multi-scale feature 사용하지 않는다

## Detecting corners

C x H x W
- C: 각각의 히트맵의 채널(카테고리 수), 백그라운드 채널 없음
- 각각의 채널은 class의 corner의 위치를 나타내는 binary mask
<p align="center"><img src="/assets/images/Paper/CornerNet/20240301102039.png"></p>
positive location을 통한 bbox도 gt annotation과 많이 겹침

- 각각의 corner에 대해 하나의 gt positive location이 존재(이외의 장소는 negative)
- training 시에 negative에 penalty를 동일하게 부과하지 않고 positive 위치의 반경 내에 있는 negative에 주어진 penalty를 줄인다
	- false corner detect pair도 만약 positive location과 가깝다면 충분히 gt annotation과 많이 겹치는 bbox 생성
- 생성한 bbox가 gt와 t(threshold,0.7)의 IoU 이상인 것을 가지고 radius를 pair of point 결정
- radius가 주어지면 $e^{-\frac{x^2 + y^2}{2\sigma^2}}$ (unnormalized 2D gaussian)을 사용하여 penalty 부여
	- center가 positive에 있고 σ가 반경의 1/3에 있는 negative에 대해

<span style="color: #88c8ff">Focal loss</span>
$
L_{\text{det}} = -\frac{1}{N} \sum_{c=1}^{C} \sum_{i=1}^{H} \sum_{j=1}^{W} \left\{ 
  \begin{array}{ll} 
  (1 - p_{cij})^\alpha \log(p_{cij}) & \text{if } y_{cij} = 1 \\
  (1 - y_{cij})^\beta (p_{cij})^\alpha \log(1 - p_{cij}) & \text{otherwise}
  \end{array} 
\right.
$
- $p_{cij}$: 예측한 heatmap의 클래스 c의 (i,j) 위치의 score
- $y_{cij}$: unnormalized gaussian에 의해 augment 된 ground truth
- N: image 내의 object 갯수
- α,β: 각 point의 기여도를 조정하는 hyperparameter(α = 2,β = 4)

<span style="color: #88c8ff">Unnormalized 2D Gaussian</span>
<p align="center"><img src="/assets/images/Paper/CornerNet/20240301104802.png"></p>
- 객체의 위치에 대한 gt 주변에서 penalty를 줄이는데 사용
- 예측 위치 주변에 가우시안 분포를 적용하여 실제 모서리 위치 근처의 예측이 정확이 일치하지 않더라도 페널티가 감소하도록 
- gt 위치를 중심으로 하여 가우시안 덩어리(bump)를 feature map에 인코딩

Fully convolutional layer 
- output이 image보다 작게 나오는 경우가 많다 $(x,y)$ -> $\left\lfloor \frac{x}{n} \right\rfloor, \left\lfloor \frac{y}{n} \right\rfloor$(heatmap location)
- n: downsampling factor
- 이때 원래의 image 위치로 mapping 시 부정확하게 위치 mapping -> IoU에 큰 영향
- 이를 해결하기 위해 location offset 예측

<span style="color: #88c8ff">location offset</span>
- corner location을 input resolution으로 mapping 하기 전에 미세 조정

$
o_k = \left( \left( \frac{x_k}{n} \right) - \left\lfloor \frac{x_k}{n} \right\rfloor, \left( \frac{y_k}{n} \right) - \left\lfloor \frac{y_k}{n} \right\rfloor \right)
$
- $O_k$: offset
- $x_k,y_k$: corner k의 x,y 좌표
- top-left, right-bottom 각각에 의해 공유되는 offset 집합 2개 예측
- gt corner location에 smooth L1 loss를 적용

$
L_{\text{off}} = \frac{1}{N} \sum_{k=1}^{N} \text{SmoothL1Loss} (o_k, \hat{o}_k)
$

## Grouping Corners

하나의 image 내의 여러개의 객체가 있을 수 있고, 이때는 여러개의 top-left, right-bottom corner 쌍이 detect된다
이를 위해 top-left, right-bottom corner가 같은 bbox의 쌍인지 결정해야한다

- 네트워크가 detected corner에 대해 embedding vector 예측(같은 bbox에 속하는지)
- 두 embedding 사이의 거리는 작아야 한다
- 거리를 통해 corner를 그룹화
- embedding의 실제 값들은 중요하지 않음

$
L_{\text{pull}} = \frac{1}{N} \sum_{k=1}^{N} \left[ (e_{t_k} - \bar{e}_k)^2 + (e_{b_k} - \bar{e}_k)^2 \right],
$

$
L_{\text{push}} = \frac{1}{N(N - 1)} \sum_{k=1}^{N} \sum_{\substack{j=1 \\ j \neq k}}^{N} \max (0, \Delta - |e_k - e_j|),
$

- 1차원 임베딩 사용
- $e_{t_k}$: top-left corner의 embedding
- $e_{b_k}$: bottom-right corner의 embedding
- pull loss: network가 corner를 그룹짓도록 사용
- push loss: corner를 분리하도록 사용
- $e_k$: $e_{t_k}$의 평균
- ∆: 1
- offset loss와 비슷하게 gt corner location에만 loss 적용

## Corner Pooling

<p align="center"><img src="/assets/images/Paper/CornerNet/20240229202302.png"></p>
그림에서는 corner를 특정할 수 있는 지역적인 증거가 없음
pixel이 top-left corner인지 결정하기 위해서는 오른쪽으로, 아래로 pixel을 살펴볼 필요가 있음
corner를 잘 localize하기 위해 명시적인 사전 지식을 encoding하는 corner pooling 제안

<p align="center"><img src="/assets/images/Paper/CornerNet/20240301113207.png"></p>
- $f_t$,$f_l$: top-left pooling layer의 input feature maps
- $f_{t_{ij}},f_{l_{ij}}$: vectors at location (i,j) in $f_t,f_l$ 
<p align="center"><img src="/assets/images/Paper/CornerNet/20240301131424.png"></p>
- top-left 는 오른쪽과 아래쪽 pixel을 보며 pixel 마다 max pool 진행
- left-corner의 방향은 아래 -> 위, 오른쪽 -> 왼쪽 순서 (fig는 잘못나온거같음)

<p align="center"><img src="/assets/images/Paper/CornerNet/20240301133811.png"></p>
- 모듈의 첫 부분은 residual block의 변형된 버전
- 3x3 conv module을 corner pooling module 로 대체
- 첫 3x3 conv 의 channel은 128 
- 3x3 Conv-ReLU- 1x1 Conv 통해 heatmap, embedding, offset 생성

## Hourglass Network

CornerNet은 hourglass network를 Backbone으로 사용
- 처음 human pose estimation task를 소개한 논문
- fully convolutional neural network (+여러개의 hourglass module 포함)

- hourglass network
	- 먼저 conv layer와 pooling layer 사용하여 input feature를 downsample
	- 이후 upsampling과 conv layer 사용하여 원래의 resolutionㅇ로 upsample
	- 이때 detail은 maxpooling layer에서 조금 손실되지만, skip connection을 통해upsampling 시 더해준다
	- 이를 통해 하나의 통합된 구조에서 local과 global feature를 잡아준다
이러한 hourglass module이 여러개 쌓이면 higher-level의 정보를 찾는데 도움이 된다

**논문의 hourglass network -> <span style="color: #88c8ff">2개의 modified hourglass 사용</span>**
- max pooling 대신 stride 2 사용하여 downsample
- resolution은 5번 줄이고 채널은 (256,384,384,384,512)의 순서로 증가
- upsample 시 2개의 residual module 사용(nn upsample)
- 모든 skip connection -> 2개의 residual module 사용
- 중간의 512 channel에는 4개의 residual module 존재
- hourglass module 전에 7x7conv, stride 2 모듈을 사용하여 resolution 4번 줄임 -> 128 channel

- 첫번째 hourglass module의 입력과 출력에 3x3 conv-BN 모듈 적용
- 이를 element-wise 덧셈을 수행한 후 ReLU와 256의 residual block을 거쳐 두번째 hourglass module의 입력으로 사용 
- 마지막 layer의 feature만을 사용

- 중간 감독일 추가했지만, 예측 결과는 네트워크에 다시 추가하지 않음

## Total loss

$
L = L_{\text{det}} + \alpha L_{\text{pull}} + \beta L_{\text{push}} + \gamma L_{\text{off}}
$

- α,β,γ: 각각의 weight 
# Experiments
## Training details

- Pytorch의 기본 initialize, pre-train, external dataset 사용하지 않음
- focal loss 사용 conv layer의 bias는 [[RetinaNet]] 을 참고
- Input resolution: 511 x 511
- Output resolution: 128 x 128
- random horizontal flipping, random scaling, random cropping and random color jittering adjusting the brightness, saturation and contrast of an image의 data augmentation 사용
- input image에 [[PCA]] 적용
- Adam optimizer
- α,β: 0.1 ,γ: 1
- Batch size: 49
- Iteration: 250k
- Learning rate: $2.5 * 10^-4$
- Compare
	- 500k iteration
	-  $2.5 * 10^-5$ for last 50k

## Testing details
- simple-post processing to generate heatmap, embedding, offset
- NMS with 3x3 max pooling layer on corner heatmap
- 100개의 top-left, 100개의 bottom-right
- offset에 의해 corner location 조정
- embedding사이의 거리를 L1 distance로 계산
- 거리 0.5 이상 혹은 달느 카테고리의 corner를 포함하는 pair 제거
- top-left, bottom-right의 average score가 detection score로 사용
- Image를 resize 하지 않고 0으로 padding
- original, flipped image 를 test에 사용
- original과 flipped image의 detection을 합친 뒤 soft-nms 적용 -> 100개의 detectoin

## MS COCO

- 80k images for training
- 40k images for validation
- 20k images for testing

training 시에는 training set 전체, 35k validation set 사용
나머지 5k validation set은 hyper parameter tuning과 ablation study에 사용

AP 사용

## Ablation study

- Corner pooling
	- CornerNet의 주요 구성요소
<p align="center"><img src="/assets/images/Paper/CornerNet/20240301151023.png"></p>
	- 중간, 큰 크기의 object에 대해 성능 향상
	- 이전까지의 경계들이 모서리 위치에서 멀리 떨어져 있었을 가능성 높음
	

- Reducing penalty to negative locations
	- positive location 주변의 penalty를 gaussian 을 사용하여 줄였음
	- 이를 penalty 적용 없이, 고정된 radius에 대해, 객체마다의 반경에 따른 penalty(CornerNet)으로 실험
	- 이 또한 중간 크기와 큰 크기의 성능 향상

<p align="center"><img src="/assets/images/Paper/CornerNet/20240301151519.png"></p>

- Error Analysis
	- heatmap, offset, embedding을 동시에 출력
	- 맨 위가 예측한 heatmap, offset 사용
	- 아래는 gt heatmap + 예측한 offset
	- 아래는 gt heatmap + gt offset
	- heatmap이 더 많이 증가 -> corner를 감지하고 그룹화하는데 있어 개선의 여지가 충분히 있음.

<p align="center"><img src="/assets/images/Paper/CornerNet/20240301151802.png"></p>

<p align="center"><img src="/assets/images/Paper/CornerNet/20240301152029.png"></p>
- 왼쪽이 top-left
- 오른쪽이 bottom-right

- comparisons with SOTA
<p align="center"><img src="/assets/images/Paper/CornerNet/20240301152203.png"></p>

- one-stage 이기고 two-stage와 경쟁력 있는 성능

