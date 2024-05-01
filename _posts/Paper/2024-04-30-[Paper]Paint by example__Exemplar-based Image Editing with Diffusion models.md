---
layout: single
title: "[Paper] Paint by Example: Exemplar-based Image Editing with Diffusion Models"
excerpt: "[Paper] Paint by Example: Exemplar-based Image Editing with Diffusion Models 논문 리뷰"
categories:
  - Paper
tags:
  - Diffusion
toc: true
toc_sticky: true
author_profile: true
sidebar_main: true
date: 2024-04-30
---

>CVPR 2023

>Binxin yang, Shuyang Gu, Bo Zhang, Ting Zhang, Xuejin Chen, Xiaoyan Sn, Dong Chen, Fang Wen

>University of Science and Technology of China, Microsoft Reasearch Asia

# **<span style="color: #a6acec">Abstract</span>**

이 논문에서는 Language-guided image editing task를 위한 exemplar-guided image editing을 제안하여 정확한 control을 목표로 한다. 이를 위해 논문에서는 <span style="color: #88c8ff">disentangle을 위한 self-supervised training</span>과 <span style="color: #88c8ff">source image와 exemplar와의 re-organize</span>를 사용한다.

그러나 단순히 source image와 exemplar image를 합성하느 것은 합성 시 여러 artifacr가 생길 수 있다. 이를 위해 <span style="color: #88c8ff">content bottleneck과 strong augmentation</span>을 사용하여 exemplar image를 그대로 복사해오는 문제를 해결하였다.

이에 더해 editing process에서 controllability 특성을 더해주기 위해 exmaplar image에 임의의 모양의 mask를 사용하여 이를 classifier-free guidance를 통해 exampler image와 유사하게 만들어주었다.

전체 framework에는 diffusion model의 single forward process만 들어가고 어떠한 iteraive optimization도 들어가 있지 않다.

# **<span style="color: #a6acec">Introduction</span>**

semantic image editing task는 이전에 있었던 inpainting, composition, colorization과 같은 task 보다는 좀 더 어려운 task다. 
이전에는 생성 모델의 semantic latent space를 사용하는 방법을 제시했지만, 대부분의 이런 방법들은 특정 image 장르에 국한되어있다.

최근 diffusion 혹은 auto-regressive model을 기반으로 한 Large-scale language-image model이 연구되면서, 복잡한 image를 표현하는 성능이 많이 발전하였다. 이러한 모델은 text를 사용해 생성모델을 guide 해주는 구조를 채용하는데, text prompt로 image를 표현할 때 애매한 부분이 있거나 사용자가 원하는 특징을 잘 표현하지 못한다는 단점이 있었다. 

때문에 논문에서는 text 대신 image를 통해 guide를 해주는 시도를 하였고, 그게 이 논문에서 제안한 exemplar-based editing approach다. 이 방법은 exemplar image를 통해 정확한 semantic 정보를 가져온다는 특징이 있다.

이전에 있던 task 중 harmonization 이라는 task가 있는데, 이 task와 여기서 사용하는 task와의 차이점은 image harmonization은 주로 color, lightening correction에 더 집중한다는 점이다.

결과적으로 논문에서는 classifier-free guidance diffusion 모델을 사용하여 exemplar image에 condition을 준다. 

이 논문이 text-guided model과 다른점은 training을 위한 충분한 triplet pair를 모으기가 어렵다는 점이다. 직접 합성 이미지 라벨 y를 만든다 하더라도, 사람이 만든 결과는 많이 차이가 날 수도 있기 때문이다.

때문에 논문에서는 이를 해결하기 위해 reference image 하나를 가지고 학습을 수행하는 self-reference setting을 사용하여 학습한다. 하지만 이 방법 또한 모델이 reference object를 단순히 복사하여 붙여넣는 방법을 학습할 수 있기 때문에 실제 상황에 일반적이지 못하다는 단점이 있다.

이러한 문제들을 해결하기 위해 논문에서는 content bottle neck을 사용하여 self-reference condition 시에 spatial token을 버리고 class token만 사용하여 global image embedding을 condition으로써 사용한다. 이를 통해 네트워크가 high-level semantic 정보를 이해하고 source image의 context를 학습할 수 있게 해준다. 

추가로 strong augmentation을 사용하여 self-supervised training 시 생기는 train과 test set 간의 gap을 줄였다.
# **<span style="color: #a6acec">Related works</span>**

## Image composition

이전 방법들은 foreground와 background가 어느정도 유사한 스타일이라고 가정하고, structure를 유지한 채 low-level color space에서 이미지를 조정한다. 

이 논문에서는 semantic image composition을 목표로 한다.

## Semantic image editing

여러 작업들이 GAN의 latent space를 분석하여 semantic적으로 분리된 latent factor들을 찾고자 한다.

또다른 방법으로는 semantic mask를 사용하는 방법인데, 이 방법들은 특정 image genre에 국한되어있는 경향이 있다.

## Text-driven image editing

GAN과 text encoder를 사용하여 image를 text에 맞게 최적화 하는 방법을 사용하였었다. 하지만 GAN 기반 방법들은 GAN의 modeling 능력의 한계로 인해 복잡한 장면을 editing하는데 어려움이 있었다.

또한 위에서 말한 것 처럼, text 기반으로 guide해주는 것은 정확한 control이 어렵고, image가 훨씬 더 좋은 표현을 전달할 수 있다고 말한다.
# **<span style="color: #a6acec">Method</span>**

논문의 목표는 reference image를 source image로 현실적으로 합성해주는 exemplar-based image editing이다. 비록 text 기반의 image editing 방법들이 좋은 성능을 보여주었지만, 언어 표현으로 복잡하고 여러가지 idea가 들어있는 image를 표현하기 어렵다. 하지만 image는 text보다 더 많은 정보와 사람의 의도를 정확히 담을 수 있다.

- $X_s$: source image ($\in \mathbb{R}^{H \times W \times 3}$)
- $X_r$: reference image($\in \mathbb{R}^{H' \times W' \times 3}$)
- $m$: binary mask ($\in\{0,1\}^{H\times W}$). 1->editable position
이때 mask는 사각형의 형태가 될 수도 있고 연결되어있는 한 어떤 형태든 취할 수 있다.


이 3가지를 활용하여 합성 이미지 y를 만드는게 목적이다.

하지만 이는 challenging한 task인데, 이유는 먼저 모델이 reference image의 객체를 이해해야 한다. 이때 background의 noise를 무시하고 객체의 모양과 texture를 학습해야 한다.
두번째로는 mask의 형태에 맞게 변형된 객체의 image를 만드는게 까다롭다. 
세번째는 객체의 주변까지 source image에 맞게 inpaint해야 한다는 점이다.
마지막으로 reference와 source image 간의 resolution 차이도 있을 수 있고, 이러한 문제들로 인해 까다로운 task라고 한다.

## Preliminaries

### Self-supervised training

{($X_s,X_r,m$),y}는 이 task에서 필요한 input과 label의 쌍인데, training을 위해 이러한 paired data를 얻어야 하는데 이는 사실상 불가능하다. annotate 비용이 비싸다는 점도 있고, 사람마다 label을 만들어낼 때 각자 다른 결과를 낼 수 있다.

때문에 논문에서는 self-supervised training을 사용한다. Image와 object의 bbox가 들어오면, object의 bbox를 mask로 사용한다. bbox 안의 source image를 reference image로 사용한다. 
$
X_r = m \circ X_s
$
이렇게 만들어낸 reference image와 source image를 합성하면, 결과 image는 원래의 source image가 나올 것이다.
$
\{(\bar{m} \circ x_s, x_r, \bar{m}), x_s\}
$
$
\bar{m} = 1 - m
$

### Naive solution
exemplar-based image editing을 위한 단순한 접근 방법은 text condition을 reference image condition으로 대체하여 diffusion model을 사용하는 방법이 있다. 
$
\mathcal{L} = \mathbb{E}_{t, y_o, \epsilon} \left\| \theta(\hat{y}_t, \bar{m} \circ x_s, c, t) - \epsilon \right\|_2^2
$

이 때 c(condition)은 text 사용 시 CLIP 모델의 text encoder를 사용하는데, 논문에서는 image를 사용하기 때문에 CLIP 모델의 image embedding을 사용한다. (257개의 token 사용. 256 patch token + 1 class token)

$
c=CLIP_{all}(X_r)
$

이러한 naive 한 방법을 사용했을 때, image를 단순히 복사 붙여넣기 한 결과가 나오는 경우도 있고, 생성된 부분이 부자연스러운 경향이 있었다. 이는 모델이 단순한 mapping 함수를 학습하기 때문이라고 한다.(trivial mapping function)
이는 network가 reference image의 content와 source image와의 connection을 이해하는 것을 방해한다.

때문에 논문에서는 3가지를 개선 목표로 둔다.
1. <span style="color: #88c8ff">content bottleneck</span>을 사용하여 network가 reference image의 content를 복사 붙여넣기 하지 않고 이해하여 regenerate 하도록 한다.
2. <span style="color: #88c8ff">strong augmentation</span>을 적용하여 object 뿐만 아니라 background의 transformation도 학습하도록 한다.
3. <span style="color: #88c8ff">edit region의 shape</span>과 edit region과 reference image 사이의 similarity degree를 조절할 수 있게 만든다.

## Model design

<p align="center"><img src="/assets/images/Paper/PaintExample/20240430101813.png"></p>

### Content Bottleneck

#### Compressed representation

reference image의 정보를 압축하여 mask 영역을 복원하는 난이도를 증가시킨다. 257개의 token을 출력하는 CLIP image embedding에서, class token만을 가져오고 이를 $224\times224\times3$ 에서 차원이 1024인 1차원 벡터로 압축한다.

이러한 압축된 representation은 고주파수 detail을 무시하며 semantic 정보를 유지하는 경향이 있다.

이 compressed representation을 사용해 모델이 reference content를 이해하고 generator가 좋은 결과를 낼 수 있도록 한다.

#### Image prior

reference image를 기억하는 것을 방지하기 위해 pre-train 된 Stable Diffusion model과 CLIP model을 가져와 사용한다.

### Strong Augmentation

또다른 문제점으로는 self-supervised learning의 training과 testing의 domain gap이 있다. 이 문제는 2가지의 측면에서 비롯된다.

#### Reference image augmentation

첫번째 문제는, 학습 시에는 source image에서 reference image를 가져오는데, 정작 test 시에는 reference image가 따로 존재한다는 점이다. 

이러한 차이를 줄이기 위해 reference image에 여러가지의 data augmentation을 적용한다. (flip, rotation, blur, elastic transform) 

이러한 data augmentation을 $A$라고 표현한다. 결과적으로 아래와 같은 condition이 diffusion model에 들어가게 된다.
$
c=MLP(CLIP(A(X_r))
$

#### Mask shape augmentation

bbox에서부터 만들어진 mask region은 reference image 내의 전체 객체를 가져오는데, 이는 실제 상황에서는 유용하지 않을 수 있다.

이 점을 보완하기 위해 bbox를 기반으로 임의의 모양의 mask를 생성하고 이를 training에 사용한다.

- 각 bbox의 모서리에 대해 먼저 Bessel curve로,먼저 곡선 위에서 20개의 점을 sampling한 후 무작위로 1~5 pixel의 offset을 좌표에 추가한다.
- 이점들을 순차적으로 연결하여 임의의 형태의 mask를 생성한다.

이러한 방법은 사용자가 mask 형태를 직접 정할 수 있도록 하는 확장성을 가지게 되었다.
### Control the similarity degree

edited image와 reference image 사이의 similarity degree를 조절하기 위해 제안했다. 
앞선 연구에서 classifier-free model이 사실상 prior, posterior constraint들이 들어있다는 것을 찾아냈고 논문에서도 이를 사용하였다.
$
\log p(\hat{y}_t \mid c) + (s - 1) \log p(c \mid \hat{y}_t) \propto \log p(\hat{y}_t) + s (\log p(\hat{y}_t \mid c) - \log p(\hat{y}_t))
$

- $s$: classifier-free guidance scale
이는 결과에 conditional reference image의 영향을 얼마나 줄지를 결정하는 factor라고 볼 수 있다.

또한 실험적으로, reference condition의 20%를 learnable vector v로 바꾸어 학습하였다. 이 term은 모델이 v에 의해 fixed condition input의 도움을 받도록 한다.

결과적으로 inference 시에 denoising step이 다음과 같은 prediction 식을 사용한다.

$
\tilde{\epsilon}_\theta (y_t, c) = \epsilon_\theta (y_t, v) + s(\epsilon_\theta (y_t, c) - \epsilon_\theta (y_t, v))
$


# **<span style="color: #a6acec">Experiments</span>**

## Implementation Details and Evaluation

### Implementation detail

- Stable Diffusion 사용
- training dataset: OpenImages
- 64개의 V100으로 7일 학습

### Test benchmark

이전에는 이런 시도가 없었기 때문에 새로운 test benchmark를 만들었다. 
MSCOCO로부터 3500개의 source image를 직접 골랐다. 각각의 image는 하나의 bbox만 가지고 있고 mask region이 전체 이미지 크기의 반을 넘지 읺도록 했다.
이후 직접 reference image patch를 training set에서 선택했다. 
이를 COCO Exemplar-based image Editing benchmark(COCOEE)라고 이름지었다.

### Evaluation metrics

1. FID score
	일반화된 image를 평가하는 metric
2. Quality Score
	진정성(authenticity) 평가
3. CLIP score

### Qualitative analysis

<p align="center"><img src="/assets/images/Paper/PaintExample/20240501112819.png"></p>

blended diffusion -> 원하는 곳에 객체를 넣을 수는 있지만 사실적이지 않다.
Stable diffusion -> 더 사실적인 image를 만들 수 있지만 reference image의 특성을 유지하는데에 실패한다.
image-guided blended diffusion -> 같은 이유의 어려움이 있다.

이러한 결과를 봤을 때 gradient guidance 방법이 content information을 보존하지 못한다고 생각한다. 

DCCF(image harmonization) -> 거의 exemplar image와 같은 결과를 만들어내지만 appearance가 source image와 맞지 않는 문제가 있다.

### Quantitative analysis

<p align="center"><img src="/assets/images/Paper/PaintExample/20240501113011.png"></p>

- image 기반의 방법들이 높은 CLIP score를 보여주었다. -> condition image의 정보를 잘 보존하였지만 결과 이미지의 품질은 떨어진다.
- stable diffusion은 FID와 QS를 봤을 때 더 설득력 있는 결과를 보여주지만 이미지의 추가적인 정보를 거의 사용하지 못한다.
- 이 논문의 방법은 세가지 지표 모두에서 좋은 성능을 달성한다.

### User study

<p align="center"><img src="/assets/images/Paper/PaintExample/20240501144744.png"></p>

- 50명의 학생에게 user study 진행하였다.
- 30개의 그룹(2개의 input과 5개의 output으로 이루어짐)
- 점수를 1에서 5까지 매기는 방법(낮을수록 좋은 결과)
- DCCF가 reference image를 그대로 가져와 만들었기에 consistency는 DCCF가 가장 높지만 전체 quality로는 논문의 방법이 가장 좋았다고 한다.

### Ablation study

<p align="center"><img src="/assets/images/Paper/PaintExample/20240501150145.png"></p>

<p align="center"><img src="/assets/images/Paper/PaintExample/20240501145106.png"></p>

- 논문에서 제안한 4가지 방법에 대한 ablation study
- baseline -> naive study
- prior: pre-trained text to image model 사용

<p align="center"><img src="/assets/images/Paper/PaintExample/20240501154246.png"></p>
- classifier-free guidance의 scale을 바꿔가며 실험한 결과
- scale이 커질수록 생성된 영역은 reference image와 점점 더 비슷해진다.

<p align="center"><img src="/assets/images/Paper/PaintExample/20240501154542.png"></p>
- text guide를 추가하며 실험한 결과
- text가 추가될수록 점점 더 비슷해지지만 image base보다는 좋지 않은 성능을 보여준다.

<p align="center"><img src="/assets/images/Paper/PaintExample/20240501154636.png"></p>

<p align="center"><img src="/assets/images/Paper/PaintExample/20240501155008.png"></p>

- diffusion의 무작위성으로 인해 동일한 입력에서 여러 결과를 낼 수 있다.(figure.9)
- 생성 이미지가 다양하긴 하지만 객체의 특징은 전부 가지고 있다. 

# **<span style="color: #a6acec">Conclusion</span>**

## <span style="color: #88c8ff">+</span>
- image 기반으로 condition을 넣어주고, compressed representation을 통해 guide 해줌으로써 경계 아티팩트까지 제거해주었기 때문에 좋은 성능을 보여준다.
- 
## <span style="color: #ed6663">-</span>


