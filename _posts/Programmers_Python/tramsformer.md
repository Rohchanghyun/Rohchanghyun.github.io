# VIT 논문



## abstract 

transformer 구조는 NLP 에서는 standard 로써 많이 쓰였지만 vision 분야 에서는 제한적이었다

vision 분야에서 attention 을 사용할 때 주로 cnn 과 혼합되어 사용하거나 몇가지 부분만 대체하는 방법으로 사용해왔다

이 논문에서는 transformer 구조가 image path 의 시퀀스에 직접적으로 적요오디어 이미지 분류 task 에 매우 잘 동작할 수 있다는 것을 밝힌다

많은 양의 데이터에 pre-trained 한 뒤 학습할 때 기존의 SOTA 모델에 비해 연산이 적고 성능은 좋았다고 한다

## Introduction

self-attention 기반 구조는 NLP 분야에서 단골로 선택되고 있으며. 대부분 corpus 에 대해 pre-trained 한 뒤 더 작은 데이터셋이 fine tuning 하는 방법으로 사용하고 있다

transformer 의 효율성과 확장성 덕분에 큰 사이즈의 모델을 훈련하는 것이 가능해졌다

모델과 데이터가 커져도 성능이 saturate 되지 않는다



그러나 CV 분야에서는 아직 CNN 구조가 많이 쓰이는데, 많은 연구에서 CNN 구조에 self-attention 을 혼합하여 실험하였다. 하지만 convolutional layer 를 전부 self-attention 으로 바꾼 모델은 현대의 하드웨어 문제로 인해 효율적으로 확장되지는 못하였다



이를 위해 이 논문에서는 이미지를 작은 patch 로 분할하여 이 patch 의 linear embedding 시퀀스를 입력으로 전달하였다

이때 하나의 patch 는 NLP 에서의 token 과 같은 개념으로 다뤄진다



하지만 중간 사이즈의 데이터셋에 대해 resnet과 비슷한 구조의 모델에 비해 성능이 떨어졌다

이는 transformer 에는 locality 와 같은 성질이 부족하기 때문에 적은 양의 데이터셋일 경우 모델의 일반화 성능이 좋지 않다고 가정하였다

(멘토님께서 말씀해 주신 말로 transformer 는 inductive bias 가 cnn 구조보다 적기 때문에 더 큰 데이터셋에서 학습할 시 성능이 좋다 라고 하셨다)

이때문에 trnasformer 를 적은 데이터셋의 모델에 사용 시 pre trained 모델을 가져와 fine tunung 하는 방법으로 사용한다



## Method

이 논문에서는 모델 설계 시에 original transformer 와 최대한 유사하게 하려고 하였다

### VIT

표준 트랜스포머는 1차원 토큰 임베딩 시퀀스를 입력으로 받지만, 모델 내부에서 2차원의 image 를 다루기 위해 3차원의 이미지를 2차원의 패치로 flatten 하였다

이때 p 는 패치들의 해상도이다

이때 N 은 패치의 수를 의미하는데 encoder 에 들어가는 유효 시퀀스 길이(하이퍼 파라미터) 로 볼 수 있다

트랜스포머는 내부의 모든 layer 에 흐르는 latent vector 의 크기가 D 로 통일되어 있기 때문에 이  논문에서는 다시 1차원으로 flatten 하여 d차원 벡터로 lineaer projection 을 통해 매핑 시켰다

또한 임베딩된 패치들의 맨 앞에 하나의 벡터를 추가했다(class 토큰 임베딩 벡터)

이 임베딩 벡터는 여러 encoder 를 거쳐 output 으로 나왔을 때, 이미지에 대한 reporesentation vector 역할을 한다

train, tuning 시에 모두 이 reporesentation vector 위에 classification head 가 부착된다

위치 정보를 유지하기 위해 patch embeddings 에 position 임베딩이 더해진다

이 논문에서는 2d 임베딩과 1d 임베딩을 비교했을 때 성능향상이 거의 없어 1d 를 사용했다고 한다

트랜스포머 인코더는 MSA layer 들과 MLP 블록들이 교차되어 구성된다

Layer normalization 이 모든 block 전에 적용되었고, residual connection 이 모든 블록 이후에 붙는다

MLP 는 GELU 를 사용하는 2개의 layer 를 포함하여 구성된다



hybrid architecture 

