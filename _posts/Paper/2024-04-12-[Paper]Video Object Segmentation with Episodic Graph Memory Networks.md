---

layout : single

title: "[Paper] Video Object Segmentation with Episodic Graph Memory Networks"

excerpt: "Video Object Segmentation with Episodic Graph Memory Networks 논문 리뷰"



categories:

- Paper

tags:

- video_segmentation


toc: true

toc_sticky: true



author_profile: true

sidebar_main: true



date: 2024-04-12

---


>ECCV 2020   
>Xiankai Lu1, Wenguan Wang2, Martin Danelljan2 Tianfei Zhou1, Jianbing Shen1, and Luc Van Gool2 1Inception Institute of Artificial Intelligence 2ETH Zurich

# Abstract

논문에서는 learning to update the segmentation model 이라는 표현을 사용하며 제안한 graph memory network가 이 아이디어를 구현하기 위해 제안되었다고 설명한다.

fully connected graph로 구성된 episodic memory network 제안하였다.
episodic memory network: frame을 node로 저장하고, frame 간의 correlation을 edge로 표현.

추가로 memory read/write를 용이하게 하고 고정된 memory scale을 유지하게 하는 learnable controller가 포함되어있다.

이러한 structured external memory design은 모델이 제한된 시각 정보로도 새로운 지식을 탐색하고 빠르게 저장할 수 있도록 하고, memory controller는 memory에 유용한 representation을 저장하는 추상적인(abstract) 방법과 prediction에 어떻게 사용할지를 gradient descent를 통해 배우게 된다.

제안한 모델은 one-shot 혹은 few-shot video object segmentation task에 잘 일반화된다.

# Introduction

**Video Object Segmentation**(VOS)
- video 내에서 target을 pixel level로 예측
- 첫번째 frame에 annotation이 주어졌는지 아닌지에 따라 O-VOS, Z-VOS로 나뉠 수 있다

- O-VOS
application 장면에 방해 객체가 포함될 수 있어 어려운 task이다. 
이를 해결하기 위해 초기 방법들은 일반적으로 annotation된 객체에 대해 finetuning 수행. -> 시간이 많이 소요된다.

최근 방법들은 효율적인 matching 기반framework를 기반으로 하고, support set(first labeled frame)과 query set(current frame)간의 차별화 가능한 matching 절차로 구성되어있다.

annotated first frame 과 previous processed frame간의 픽셀별 유사성을 사용해 query frame에 직접 라벨 할당

하지만 여기에는 몇가지 문제점이 있다.
- 일반적으로 generic matching network를 학습한 다음 test video에 적용하여 첫 frame의 target-specific 정보를 활용하지 못한다.
- 외형 변화가 생길 수 있다.
- pixel별 유사성만 사용하기 떄문에 context 정보 무시

이러한 문제를 해결하기 위해 graph memory network를 개발하고 이를 통해 특정 대상에 대한 segmentation model을 online으로 적응시킨다.

오프라인 훈련 데이터에서 high-level 지식을 천천히 학습하고, 첫 프레임 주석에서 얻은 미처리 정보를 외부 메모리 모듈을 통해 테스트 비디오로 신속하게 통합하는 능력을 갖추게 된다.

광범위한 파라미터 최적화 없이도 online model update를 쉽게 구현할 수 있다. 

메모리 소비를 증가시키지 않고 고정 크기의 graph memory에서 message passing을 수행한다.

# Related work

## Online Leaerning
- data를 순차적으로 받아 모델을 지속적으로 update하는 방법.
- 모델이 data를 받을 때마다 즉시 학습을 수행하고 기존에 학습한 정보를 기반으로 조정.
- 새로운 패턴이나 변화하는 data 분포에 빠르게 적응할 수 있지만 잘못된 data나 outlier에 민감하게 반응할 수 있다.

## One-shot Video Object Segmentation(O-VOS)
 - 첫 frame에서 주어진 객체의 annotation(하나의 example)을 바탕으로 비디오 내 객체를 구분하고 tracking하는 task.
- 첫번째 frame에서 객체에 대한 정보를 학습하고 이후 프레임에서 해당 객체를 자동으로 식별하고 분할하는데 초점을 맞춘다.
## Zero-shot Video Object Segmentation(Z-VOS)
- 첫 프레임에서 객체 정보가 주어지지 않음
- message passing 단계를 통해 object 파악하고 그에 따라 segmentation

# Method

## Proposed Algorithm

### Preliminary: Episodic Memory Networks

**Backgrounds**

**Memory network**
- neural network에 external memory component를 추가한 형태로, network가 과거의 experience들에 명시적으로 접근할 수 있게 한다.

---

**<span style="color: #88c8ff">Episodic Memory network</span>**
- visual questioning answering과 visual dialog task에서 reasoning 문제를 해결하기 위해 사용하였다. train 가능한 read/write operation을 통해 memory로부터 질문에 답하기 위해 필요한 정보를 검색하는 것이 기본 개념.

1. input representation의 set이 주어지면 episodic memory module은 neural attention을 통해 입력의 어떤 부분에 초점을 맞출지 선택한다.
2. 이후 query와 memory를 고려하여 memory summarization representation을 생성한다.

이러한 과정을 반복하여 입력에 대한 관련 정보를 memory module에 제공하고, 결과적으로 memory module이 새로운 정보를 검색할 수 있는 능력을 가지고 input에 대한 새로운 representation을 얻게 된다.

### Learning to Update

O-VOS를 위해 이전 방법들은 network를 세밀하게 조정하고 각 video에 대해 online learning을 수행했다. 
하지만 본 논문에서는 training task(video)들의 분포로부터 sampling된 다양한 task에 대해 episodic memory based learner를 구축하여 학습된 모델이 새로운 task(test video)에서 성능을 내도록 한다.

이에 대해 O-VOS에 <span style="color: #88c8ff">"learning to update"</span> 과정을 사용한다.
<p align="center"><img src="/assets/images/Paper/VOS_episodic_graph_memory/20240411161646.png"></p>
**<span style="color: #88c8ff">Learning to update</span>**
1. one-shot support set에서 task representation 추출
2. representation이 주어진 query에 대해 segmentation network를 update

그림처럼 episodic memory 구조에 graph 구조를 사용하였다.
- 많은 finetuning 과정을 수행하는 대신 특정 object에 segmentation network를 즉시 적응
- video sequence 내의 context를 충분히 활용한다는 장점이 있다.

이를 통해 논문의 network는 2가지의 장점을 가지게 된다.
- model initialization 과정에서 one-shot support set로부터 segmentation network를 학습하는 능력
- frame processing 과정에서 segment된 frame을 활용하여 segmentation을 update 하는 능력

case별로 효율적으로 적응하고 한번의 feed forward 과정 내에서 online-update 를 수행할 수 있다.

> online learning vs. learning to update

- Online learning
	- data가 순차적으로 도착. 각 data point는 독립적으로 처리되며 이전 상태의 모델을 기반으로 update
	- 간단한 parameter 조정이나 gradient descent를 사용하여 모델을 지속적으로 update
- Learning to update
	- 특정 task를 기반으로 한 적응성 학습
	- 초기 데이터에서 출발하여 추가적인 정보를 통해 모델이 스스로 update 하는 방법을 배운다
	- task의 특성을 반영. 모델이 새로운 정보를 통합하고 예전 데이터로부터 어떤 정보를 기억해야할지 결정하는 과정을 포함.
	- 논문에서 말하고자 하는 바는 episodic graph memory를 통해 frame간의 연관성을 찾아 edge로 연결하기 때문에 online learning과는 다르다는 점이다.(어떤 정보를 기억 해야할지 결정)

### Graph Memory Network
<p align="center"><img src="/assets/images/Paper/VOS_episodic_graph_memory/20240411170315.png"></p>
graph memory network는 external graph memory와 learnable controller 로 이루어져 있다.

<span style="color: #88c8ff">Graph Memory</span>
memory는 새로운 지식(frame)에 대한 encoding을 위한 short-term storge를 가지고 있다.(파란 박스)

<span style="color: #88c8ff">Controller</span>
graph 구조는 controller를 통해 context를 전부 탐색할 수 있게 해준다.
controller는 graph memory와 read / write를 통해 상호작용하며, slow weight update를 위한 long term storage가 가능하다.
controller를 통해 memory에 어떤 종류의 representation을 넣어야 할지, 나중에 이러한 representation을 segmentation에 어떻게 사용해야 할지에 대한 strategy를 학습한다.

Graph Memory Network의 핵심 아이디어
- episodic reasoning을 K번 수행하여 memory 안의 구조들을 효율적으로 탐사
- target-specific 정보를 잘 포착

memory는 고정된 크기의 fully-connected graph로 구성(파란박스 위쪽 초록부분 과정)
$G = (M, E)$
i번째 memory cell
$
m_i \in M
$
i번째 cell과 j번째 cell 사이의 관계
$
e_{i,j} = (m_i, m_j) \in E
$
support set은 첫번째 annotated frame과 이전에 segment된 frame들의 조합
이때 graph memory는 N(=|M|)개의 frame에서 초기화되고, 이 frame들은 support sampling된다. 
각 메모리 노드 $m_i$는 해당하는 support frame에 fc layer mempry encoder를 적용하여 생성된 초기 임베딩 $m_i^0$를 가진다. 

<p align="center"><img src="/assets/images/Paper/VOS_episodic_graph_memory/20240411205029.png"></p>
### Graph Memory Reading

memory 내에서 가장 관련성이 높은 정보를 검색하고 이를 현재 작업에 활용하기 위한 과정
새로운 쿼리를 통해 현재 memory cell을 update.
위의 그림의 controller의 read 단계
1. query frame으로부터 visual feature $q$를 추출 $q \in \mathbb{R}^{W \times H \times C}$
2. 먼저 read controller가 q를 input으로 받아 initial state $h^0$ 생성
$$
h^0 = f_P(q) \in \mathbb{R}^{W \times H \times C}
$$
여기서 $f_p$는 projection function

3. 각 episodic reasoning step $k \in \{1,...,K\}$에서, external graph memory들을 사용하여, initial state와 memory node 간의 similarity를 비교한다.
$$
s_i^k = \frac{h^{k-1} \cdot m_i^{k-1}}{\| h^{k-1} \| \| m_i^{k-1} \|} \in [-1, 1]
$$
4. 이후 read weight $w_i^k$를 softmax normalization function을 통해 계산
$$
w_i^k = \frac{\exp(s_i^k)}{\sum_j \exp(s_j^k)} \in [0,1]
$$
5. 이때 몇몇 node들이 카메라 움직임이나 객체가 밖으로 나간 장면일 수 있기 때문에(noise가 존재),$w_i^k$가 memory cell의 confidence를 계산한다. 이 가중치를 사용하여 mask를 update. 이를 가중치를 통해 memory를 요약하는 과정이라 한다.
$$
m^k = \sum_i w_i^k m_i^{k-1} \in \mathbb{R}^{W \times H \times C}
$$
6. 이전까지의 방법을 통해 memory module은 $h^k$(query state of k-th step)과 가장 유사한 memory cell을 검색하여 memory summarization $m^k$를 얻는다. 이 memory summarization을 사용해 read controller는 다음과 같이 state를 update한다.
$$
\tilde{h}^k = W_r^h * h^{k-1} + U_r^h * m^k \in \mathbb{R}^{W \times H \times C},
$$
$$
a_r^k = \sigma(W_r^a * h^{k-1} + U_r^a * m^k) \in [0,1]^{W \times H \times C},
$$
$$
h^k = a_r^k \circ \tilde{h}^k + (1 - a_r^k) \odot h^{k-1} \in \mathbb{R}^{W \times H \times C},
$$
- $W_r^h, U_r^h$:convolution kernel
- $*$: convolution
- $\sigma$: sigmoid 
- $\circ$: hadamard product
이때 update gate $a^k$는 이전 hidden state $h^{t-1}$이 얼마나 유지될지 를 결정한다.
이러한 방법으로 controller의 hidden state가 graph memory와 query representation을 encode한다. 이는 output 생성에 꼭 필요한 요소이다.

### Episodic Graph Memory Updating

sequence 내에서 이전 단계에서 얻은 정보와 현재 단계의 새로운 정보를 통합하는 과정.
memory의 전체를 보다 장기적인 관점에서 조정하고 update.

1. 각 k번째 step에서, 각 memory cell을($m_i$) 이전 state memory($m_i^{k-1}$), 현재 content $h^k$, 그리고 다른 cell들의 state($\{m_j^{k-1}\}_{j \neq i}$)를 사용하여 update한다.
2. $m_i$에서 $m_j$로의 relation $e_{i,j}^k$를 feature 행렬에 대한 inner-product similarity(내적 유사도)로 정의한다.
$$
e_{i,j}^k = m_i^{k-1} W_e m_j^{k-1 T} \in \mathbb{R}^{(WH) \times (WH)}
$$
- $W_e \in \mathbb{R}^{C \times C}$: learnable weight matrix
- $m_i^{k-1} \in \mathbb{R}^{(WH) \times C},m_j^{k-1} \in \mathbb{R}^{(WH) \times C}$ 가 matrix representation으로 flatten된다.
3. 이후 $m_i$에 대해 summarized된 information $c_i^k$ 를 다른 cell로부터 계산한다.
$$
c_i^k = \sum_{j \neq i} \text{softmax}(e_{i,j}^k) m_j^{k-1} \in \mathbb{R}^{W \times H \times C}
$$
- $softmax()$: input의 각 row를 normalize한다.

4. 3번 과정을 통해 이웃들의 information을 모은 뒤, state $m_i$를 memory write controller가 update 한다.
$$
\tilde{m}_i^k = W_u^m * h^k + U_u^m * m_i^{k-1} + V_u^m * c_i^k \in \mathbb{R}^{W \times H \times C},
$$
$$
a_u^k = \sigma(W_u^\alpha * h^k + U_u^\alpha * m_i^{k-1} + V_u^\alpha * c_i^k) \in [0,1]^{W \times H \times C},
$$
$$
m_i^k = a_u^k \circ \tilde{m}_i^k + (1 - a_u^k) \odot m_i^{k-1} \in \mathbb{R}^{W \times H \times C}.
$$

이 과정은 각 memory cell이 이웃 정보를 자신의 representation에 포함시킬 수 있도록 한다.
또한 그래프 구조를 통해 반복적으로 추론함으로써(reasoning) 각 memory cell이 새로운 query 정보를 인코딩하고 개선된 representation을 만들어낸다.

원래의 memory network와 비교하여 2가지의 장점을 가지고 있다.

- memory write 연산이 memory 크기 증가 없이 update 절차에 포함된다.(write시 memory cell이 늘어나지 않음)
- 복잡한 memory write 전략을 사용하지 않는다

### Final Segmentation Readout

episodic memory update의 K단계 이후, memory read controller로부터 나온 최종 state $h^K$를 활용하여 query에 대한 prediction을 생성한다.
$$
\hat{S} = f_R(h^K, q) \in [0,1]^{W \times H \times 2},
$$
- $f_R$: 최종 segmentation probability map을 제공한다.

## Network architecture

- fully convolutional한 end-to-end 모델.
- query encoder와 memory encoder는 동일한 구조.
- read,write controller는 1x1 convolution kernel을 사용하는 ConvGRU를 사용.
- $f_P$: projection 함수도 1x1 convolution으로 구현.
- $f_R$: ResNet50 block에 해당하는 skip connection을 가지는 4개의 block으로 구성된 decoder network
- decoder 내의 convolutional layer의 kernel은 마지막 1x1 을 제외하고 3x3 사용
- 최종 2channel의 segmentation prediction은 softmax 연산에 의해 얻어진다.
- query, memory encoder는 pre-train된 ResNet 50의 4개의 convolutional block으로 구성

# Experiments

## label shuffling

인스턴스와 one-hot label 간의 관계를 기억하지 못하도록 label shuffling을 사용한다.

<p align="center"><img src="/assets/images/Paper/VOS_episodic_graph_memory/20240414163115.png"></p>

segmentation 실행할때마다 instance label을 shuffle한다.
Z-VOS에서는 label이 주어지지 않기 때문에 사용하지 않는다.

## O-VOS
<p align="center"><img src="/assets/images/Paper/VOS_episodic_graph_memory/20240414185756.png"></p>

- online learning 방법들과 비교했을 때 더 높은 정확도를 보여준다.
- AGAME, FEELVOS, RGMP와 같은 matching based methods들은 빠른 inference 속도를 보여주지만 memory cost가 높고, 이에 비교하여 논문의 method는 빠른 속도와 낮은 memory usage를 보여준다.

<p align="center"><img src="/assets/images/Paper/VOS_episodic_graph_memory/20240414190450.png"></p>

- overall: 4개의 metric을 평균낸것

<p align="center"><img src="/assets/images/Paper/VOS_episodic_graph_memory/20240414191153.png"></p>


## Z-VOS
<p align="center"><img src="/assets/images/Paper/VOS_episodic_graph_memory/20240414191933.png"></p>

- $\tau$: stability

<p align="center"><img src="/assets/images/Paper/VOS_episodic_graph_memory/20240414192843.png"></p>

<p align="center"><img src="/assets/images/Paper/VOS_episodic_graph_memory/20240414192905.png"></p>

<p align="center"><img src="/assets/images/Paper/VOS_episodic_graph_memory/20240414193806.png"></p>

- Graph structre: node 3개를 쓰는게 최적이라고 설명. 성능은 올라가지만 메모리 사용량 측면에서 trade-off라고 생각
- state updating: K가 4일때까지 성능은 계속 올라가지만 이후로는 변화가 없다고 한다.
- label shuffling을 사용했을 때 성능이 더 좋았다.

# Conclusion

- 성능이 여러 부분에서 SOTA를 찍지 못했음. graphic memory model을 처음 사용한 것은 novelty가 있으나, 결과가 좋지 못했다.