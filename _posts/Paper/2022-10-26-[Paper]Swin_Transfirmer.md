---

layout : single

title: "[Paper]Swin Transformer"

excerpt: "Swin Transformer 논문 리뷰"



categories:

- Paper

tags:

- transformer



toc: true

toc_sticky: true



author_profile: true

sidebar_main: true



date: 2022-10-26

last_modified_at: 2022-10-26

---

## Abstract

기존 transformer는 NLP 분야에서 CV분야로 사용하기 위해 구조를 바꾸면서 2가지의 문제점을 가지고 있었다.

- 피사체의 규모의 큰 변화
- 글자와 비교했을때 높은 해상도(high resolution)

이를 해결하기 위해 representation이 shifted windows를 통해 계산되는 계층형 transformer를 제안

이 shifted windowing 시스템은 cross-window connection을 허용하는 동시에 self attention 계산을 non-overlapping local window로 제한하여 효율성을 높인다.
이러한 계층 구조는 다양한 규모로 모델링 할 수 있는 flexibility를 가지고 있으며 이미지 크기와 관련하여 linear computational complexity를 가지고 있다.

이를 통해 image classification, object detection, semantic segmentation과 같은 vision task에서 경쟁력을 높였다.

게다가 이러한 계층형 디자인은 모든 MLP 구조에서도 효과적이라는 것을 증명했다.



## Introduction

<p align="center"><img src="/assets/images/Paper/Swin_Transformer/figure_1.png"></p>

NLP 분야와의 차이점

- scale
  - language transformer 과정의 basic element인 word token과는 다르게, 시각적 요소는 크기가 상당히 다양할 수 있고, 이는 object detection에서 의미가 있다.
  - 지금까지의 transformer base 모델들에서 token들은 고정된 크기였고 이러한 속성은 vision application에 적합하지 않다.

- much higher resolution compared to words
  - 픽셀 수준에서의 세밀한 prediction을 필요로 하는 semantic segmentation같은 vision task가 많이 존재하고, 이는 고해상도 이미지에서 self attention의 계산 복잡도가 높기 떄문에 다루기 어렵다.



이러한 문제들을 극복하기 위해 이 논문에서는 image size에 대해 linear computational complexity를 가지게 하는 계층형 feature map을 설계하는 Swin Transformer를 제안한다.

linear computational complexity는 이미지를 분할하는 중복되지 않는 window 내에서 self-attention을 계산함으로써 달성된다.

각 window의 patch수는 고정되어 있으므로 complexity는 이미지 크기에 따라 선형이 된다.

<p align="center"><img src="/assets/images/Paper/Swin_Transformer/figure_2.png"></p>

그림에 표시된 것처럼 Swin Transformer는 작은 크기의 patch에서 시작하여 더 깊은 Transformer 계층에서 인접한 patch를 점진적으로 병합하여 계층적 representation을 구성한다.

이러한 계층적 feature map을 통해 Swin Transformer 모델은 FPN 또는 U-Net과 같은 dense preditction기술들을 편히라게 활용할 수 있다.



### shift of the window

- 연속적인 self-attention 계층 간에 window paartition을 이동시킨다.

- 이러한 shifted window는 이전 계층의 window를 연결하여 모델링 능력을 크게 향상시킨다.

## Architecture

<p align="center"><img src="/assets/images/Paper/Swin_Transformer/figure_3.png"></p>

- input RGB image를 patch splitting module을 통해 겹치지 않도록 patch로 나눈다. 각각의 patch는 token이라 하고, 이 token의 feature는 raw pixelRGB value들의 concatenation이다.

- 본 논문에서는 4X4의 patch size를 사용하기 때문에 각 patch의 feature dimension은 4X4X3 이다.

- linear embedding layer가 이 raw-valued feature에 적용되어 C의 임의 차원으로 투영된다.

- # stage 1

  - 이 patch token에 Swin Transformer Blocks들이 적용된다.
  - Transformer block은 H/4 X W/4의 token 갯수를 유지하고, linear embedding과 함께 'stage1'이라 한다.

- # stage2

  - 계층적 representation을 생성하기 위해 네트워크가 깊어질수록 layer를 patch merging하여 token의 수를 줄인다.
  - 첫번째 patch merging layer는 2X2의 인접한 patch group의 각 feature를 concat하고 4C 차원의 concat된 feature에 linear layer를 적용한다.
  - 토큰의 수가 2X2 =4(해상도의 2배 downsampling)의 배수로 줄어들고 output dimension이 2C로 설정된다.
  - Swin Transgormer block은 feature transformation을 위해 적용되며, 해상도는 H/8 X W/8로 유지된다.
  - 이 절차가 3,4단계에서 2번 반복되며 해상도가 H/16 X W/16, H/32 X W/32로 차례대로 바뀐다.

이러한 단계는 VGG,ResNet과 같은 convolution network와 같은 featuremap resolution과 계층적 representation을 생성한다.

이 구조는 다양한 vision task를 위해 backbone network를 편리하게 대체할 수 있다.

### Swin Transformer Block

<p align="center"><img src="/assets/images/Paper/Swin_Transformer/figure_4.png"></p>

- Transformer의 multi-head attention을 shifted window로 대체하여 구성하였다.
- shifted window 기반 MSA 모듈과 그 사이에 GELU non-linearity 가 있는 2layer MLP로 구성된다.
- 각 MSA 모듈 및 MLP 이전에는 Layer Normalization이 적용되고, 각 모듈 이후에는 residual connection이 적용된다.

### Shifted Wondow based Self-Attention

- image classification을 위한 표준 Transformer 구조는 global self-attention을 수행하는데, 이러한 계산은 token 수와 관련하여 2차 복잡성으로 이어지므로 엄청난 token set을 요구하거나 고해상도 이미지를 나타내는 많은 vision task에 적합하지 않다.

#### Self-attention in non-overlapped windows

- 효율적인 모델링을 위해, self-attention을 local window 내에서 계산한다.
- window는 겹치지 않는 방식으로 균등하게 분할하도록 배열된다.

<p align="center"><img src="/assets/images/Paper/Swin_Transformer/figure_5.png"></p>

- 각 window가 M X M 의 patch를 가지고 있을 때 각각 global MSA 모듈과 h X w patch 이미지의 window based 모듈에 대한 계산 복잡도
- MSA 모듈의 식은 hw에 대해 제곱이고, window base모듈의 식은 M이 고정되어 있다는 가정 하에 선형이다.(논문에서는 7로 고정)



#### Shifted window partitioning in successive blocks

- window 기반의 self-attention 모듈은 window간 connection이 부족하여 모델링 능력이 제한된다.
- 효율적인 계산을 유지하면서 cross-window connection을 도입하기 위해, 이 논문에서는 연속되는 Swin Transformer block에서 2개의 partitioning 구성을 번갈아 사용하는 방식을 제안한다.
- 첫번째 모듈은 왼쪽 상단에서 시작하는 일반적인 window partitioning 전략을 사용하며, 8X8 feature map은 4X4 크기의 2X2(M=4)의 window로 고르게 분할된다.
- 다음 모듈은 규칙적으로 partitioning 된 window에서 ([M/2],[M/2])픽셀로 window를 이동하여 이전 계층의 window에서 전환된 window 구성을 채택한다.
- 이러한 방법을 사용한 연속적인 Swin Transformer block은 다음과 같이 계산된다

<p align="center"><img src="/assets/images/Paper/Swin_Transformer/figure_6.png"></p>

- z^l:  W_MSA 에서의 block i에 대한 output feature
- zl: MLP에서의 block i에 대한 output feature

#### Efficient batch computation for shifted configuration

<p align="center"><img src="/assets/images/Paper/Swin_Transformer/figure_7.png"></p>

- Shifted window partitioning의 문제점은 shift된 구성에서 [h/M] X [w/M] 에서 ([h/M] + 1) X ([w/M] + 1)으로 더 많은 window가 발생하고, 일부 window는 M X M보다 작다는 것이다.
-  이 논문에서는 그림과 같이 왼쪽 상단 방향으로 cyclic shift하여 더 효율적인 batch 계산을 만들어낸다.
- shift 후에 배치된 window는 feature map에서 인접하지 않은 여러 하위 window로 구성될 수 있으므로 masking 메커니즘을 사용하여 각 하위 window 내에서 self-attention을 계산한다.
- cyclic shift를 사용하면 배치된 window 수가 일반 window partitioning의 window 수와 동일하므로 효율적이다.



#### Relative position bias

<p align="center"><img src="/assets/images/Paper/Swin_Transformer/figure_8.png"></p>

- self-attention 계산에서 각 헤드에 대한 relative position bias(B ∈ R^(M2×M2))를 포함시킨다.
- Q,K,V : query,key,value matrices
- d: query/key dimension
- M^2 : window 내에 있는 patch의 수
- 각 axis를 따르는 relative position이 [-M + 1,M. -1] 범위 내에 있기 때문에 더 작은 크기의 bias matrix Bhat (∈ R^((2M-1)×(2M-1))) 을 매개변수화 하고 B의 값은 Bhat에서 가져온다.

### Architecture Variants

- Swin-T: C = 96, layer numbers = {2, 2, 6, 2}
- Swin-S: C = 96, layer numbers ={2, 2, 18, 2} 
- Swin-B: C = 128, layer numbers ={2, 2, 18, 2}
- Swin-L: C = 192, layer numbers ={2, 2, 18, 2}
