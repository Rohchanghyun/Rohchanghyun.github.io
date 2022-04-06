---
layout : single
title:  "[Theory] Optimization and Regularization "
excerpt: "Optimization 과 Regularization의 기초 이론"

categories:
  - Theory
tags:
  - Optimization
  - Regularization

toc: true
toc_sticky: true

author_profile: true
sidebar_main: true

date: 2022-04-06
last_modified_at: 2022-04-06
---


# <span style="color: #f0b752">Optimization</span>

## <span style="color: #a6acec">Generalization</span>

우리가 사용하는 모델을 만들때의 목적은 일반화 성능을 높이는 것이 주된 목적이다. 

<p align="center"><img src="/assets/images/optimization/general.png"></p>

학습 시에는 iteration 증가함에 따라서 training error가 줄어들어 0에 수렴하게 된다

하지만 어느정도 시간이 지나면 내가 train을 수행한 train data에서의 training error와 다르게 test data에서의 test error 는 늘어나게 된다(<span style="color: #ed6663">overfitting</span> 이 원인)

**<span style="color: #88c8ff">generalization performance</span>** : <span style="color: #88c8ff">training error</span>와 <span style="color: #ed6663">test error</span> 사이의 차이

- generalization 성능이 좋으면 학습 데이터와 학습데이터가 아닌 데이터에서의 성능이 비슷하게 나올 것이라는 것을 의미
- 하지만 training error와 test error 둘다 낮을 수 있기 때문에 <span style="color: #ed6663">generalization 성능이 높다고 해서 model 의 성능이 높다고 할 수는 없다. </span>

<p align="center"><img src="/assets/images/optimization/overfit.png"></p>

## <span style="color: #a6acec">Cross validation</span>

<p align="center"><img src="/assets/images/optimization/cross.png"></p>

- <span style="color: #88c8ff">K-fold validation</span> 이라고도 불린다
- train dataset과 validation dataset 의 비율의 문제를 해결하고자 하였다
- 전체 데이터를 k개(fold)로 나누어 <span style="color: #88c8ff">k-1 개의 fold로 학습을 시키고 나머지 1개로 validation</span>을 해보는 방법
- 최적의 <span style="color: #88c8ff">hyperparameter</span> 을 찾기 위해 cross validation을 사용하여 찾은 후, hyperparameter 를 고정시킨 뒤 더 많은 데이터를 사용하여 학습을 진행한다
- <span style="color: #ed6663">test data는 학습에 어떠한 관여도 하면 안된다</span>

## <span style="color: #a6acec">Bias and variance</span>

<p align="center"><img src="/assets/images/optimization/bias.png"></p>

- <span style="color: #88c8ff">Bias</span> : 비슷한 입력을 넣었을 때 출력이 분산되더라도 평균적으로 ground truth 에서 얼마나 벗어났는지
- <span style="color: #88c8ff">Variance</span> :  비슷한 입력을 넣었을 때 출력이 얼마나 일관적으로 나오는지

### <span style="color: #b1cf89">Bias and Variance Traodeoff</span>

내 학습 데이터에 noise가 껴 있다고 가정 했을 때, 2norm 기준으로 noise가 낀 target data를 최소화 하는 것은 3가지 part(bias,variance,noise)로 나누어진다.

minimize 하는 것은 하나의 target data지만 3가지의 conference로 이루어져 있어 각각을 minimize 하는것이 아니라 하나를 줄이면 다른것들이 커지는 tradeoff 를 설명한다

<p align="center"><img src="/assets/images/optimization/trade.png"></p>

- t : target
- f hat : neural network 출력값

여기서는 t에 noise가 꼈다고 가정

<p align="center"><img src="/assets/images/optimization/off.png"></p>

bias , variance , noise 3개를 같이 줄이기는 힘들다는 fundamental한 limitation을 나타낸다

## <span style="color: #a6acec">Bootstrapping</span>

학습 데이터가 고정되어있을 때 subsampling 을 통해 여러 model 을 만들어 관찰

<p align="center"><img src="/assets/images/optimization/boost.png"></p>

### <span style="color: #b1cf89">Bagging(ensemble)</span>

**Bootstrapping aggregating**

학습 데이터를 여러개를 만들어 <span style="color: #88c8ff">여러 model을 만들고, output을 평균을 내는 방법</span>

### <span style="color: #b1cf89">Boosting</span>

학습 데이터를 sequential 하게 바라보고 학습을 진행 한 뒤, 학습이 잘 되지 않은 data를 학습하는 2번쨰 model(weak learner)을 만드는 방법으로 반복하여, 합치는 방법

독립적인 model로 보고 n개의 결과를 뽑는게 아니라. <span style="color: #88c8ff">weak learner 를 여러개를 만들어 sequential 하게 합치는 방법으로 하나의 strong learner 를 만드는 방법</span>

## <span style="color: #a6acec">Practical Gradient Descent Methods</span>

- Stochastic gradient descent
	- 여러개의 sample 이 아닌 하나의 sample 을 가지고 gradient 를 update 하는 방법

- Mini-batch gradient descent
	- batch size 만큼의 data를 보고 gradient 를 update 하는 방법
- Batch gradient descent
	- 한번에 data 전부 사용하여 평균을 가지고 gradient를 update 하는 방법

### <span style="color: #b1cf89">Batch siae matters</span>

> On large-batch Training for Deep Learning : Generalization Gap and Sharp Minima(2017)

- 큰 batch size 를 활용하면 sharp minimizer 라는 것에 도달하게 된다

- 작은 batch size 를 활용하면 flat minimizer 라는 것에 도달하게 된다

이 논문에선는 실험적 결과로 sharp minimizer 보다는 flat minimizer 라는 것이 더 좋다, 즉 작은 batch size 를 쓰는것이 더 좋다고 설명하고 있다

<p align="center"><img src="/assets/images/optimization/minima.png"></p>

model 의 목적은 Test function 에서의 minimum 을 찾고 싶은 거지, Training function 에서의 minimun을 찾고싶은게 아니다. 

**<span style="color: #88c8ff">Flat minimum</span>**

training function 에서 조금 멀어져도, test function 에서도 적당히 낮은값이 나온다 : <span style="color: #ed6663">Generalization performance 좋다</span>

**<span style="color: #88c8ff">Sharp minimum</span>**

Training function 에서는 낮지만, Testing function 에서는 제대로 일을 하지 못한다 : <span style="color: #ed6663">Generalization performance 나쁘다</span>

### <span style="color: #b1cf89">Gradient Descent Methods</span>

#### <span style="color: #88c8ff">Gradient descent</span>

<p align="center"><img src="/assets/images/optimization/gd.png"></p>

미분한 gradient 에 learning rate 를 곱하여 차이를 가중치에 update

Problems 

- Learning rate를 잡는것이 어렵다

#### <span style="color: #88c8ff">Momentum</span>

<p align="center"><img src="/assets/images/optimization/moment.png"></p>

한번 gradient 흐른 방향대로 다음 gradient에서 방향 정보를 이용한 방법

problems

- local minimum 에 잘 못빠지기 떄문에 converge가 잘 안되는 경향이 있다

#### <span style="color: #88c8ff">Nesterov Accelerate</span>

<p align="center"><img src="/assets/images/optimization/NAG.png"></p>

<p align="center"><img src="/assets/images/optimization/nag2.png"></p>

gradient 계산 시 <span style="color: #ed6663">Lookahead gradient</span> 를 계산하게 된다

현재 주어진 parameter 에서 gradient 계산해서 다음 정보로 한번 이동하여 한번 더 gradient 계산 뒤 accumulation

momentum 처럼 gradient 계산 시 방향성을 가지고 흘러가는 것이 아니라 한번 지나간 곳에서 gradient 계산하기 때문에 빨리 converge 하는 효과가 생기게 된다

#### <span style="color: #88c8ff">Adagrad</span>

<p align="center"><img src="/assets/images/optimization/adag.png"></p>

train 동안 gradient 가 얼마나 변해왔는지를 보게 된다

G : 지금까지 gradient 가 얼마나 많이 변했는지를 제곱하여 더한 값

problems 

- G 가 무한대로 커지기 떄문에 학습이 진행될수록 훕나에는 학습이 잘 되지 않는다

#### <span style="color: #88c8ff">Adadelta</span>

<p align="center"><img src="/assets/images/optimization/adag.png"></p>

Adagrad 의 문제를 해결하기 위해 window size 를 설정하여 시간축으로 얼마만큼의 gradient 를 볼지 정하고, <span style="color: #ed6663">exponential moving average</span> 를 통해 parameter 가 많이 쌓였을때의 해결책을 제시하였다

problems

- earning rate 가 없기 때문에 많이 활용되지 않는다

#### <span style="color: #88c8ff">RMSprop</span>

<p align="center"><img src="/assets/images/optimization/prop.png"></p>

gradient squares 를 그냥 더하지 않고 EMA를 더해준다. 이를 분모에 넣고 stepsize 를 넣어주었다

#### <span style="color: #88c8ff">Adam</span>

<p align="center"><img src="/assets/images/optimization/Adam.png"></p>

가장 무난하게 사용

gradient square 에 exponential moving average 를 가져감과 동시에, momentum 을 같이 활용하는 방법

<span style="color: #ed6663">division by zero 를 막아주는 epsilon parameter 를 잘 설정해주는것이 중요하다</span>

## <span style="color: #a6acec">Regularization Methods</span>

#### <span style="color: #88c8ff">Early stopping</span>

<p align="center"><img src="/assets/images/optimization/early.png"></p>

generalization 이 잘되었을 때 학습을 일찍 종료



#### <span style="color: #88c8ff">Parameter Norm Penalty(weight decay)</span>

<p align="center"><img src="/assets/images/optimization/penalty.png"></p>

Neural Network parameter 가 너무 커지지 않게 하는 방법

parameter 를 전부 더한 값을 같이 줄인다

<span style="color: #ed6663">function space 에서 함수를 최대한 부드럽게 해준다(특정 값만 크게 하지 않음)</span>

#### <span style="color: #88c8ff">Data Augmentation</span>

<p align="center"><img src="/assets/images/optimization/aug.png"></p>

data 적을 떄는 random forest, SVM 같은 방식이 더 잘 되었었다

data가 많아지면 ML에서 활용하는 방법들이 많은 수의 data를 표현할만한 표현력이 부족해진다

하지만 data 가 한정적이므로 <span style="color: #ed6663">data를 변환하여 늘려준다</span>

#### <span style="color: #88c8ff">Noise Robustness</span>

<p align="center"><img src="/assets/images/optimization/noise.png"></p>

입력 data와 weight에 noise 를 추가하는 방법

#### <span style="color: #88c8ff">Label Smoothing</span>

<p align="center"><img src="/assets/images/optimization/smooth1.png"></p>

<p align="center"><img src="/assets/images/optimization/smooth2.png"></p>

data 2개를 뽑아 섞어준다

classification 는 이미지 공간에서 <span style="color: #ed6663">decision boundary</span> 를 찾는 task인데 이 <span style="color: #ed6663">decision boundary를 부드럽게 해준다</span>

data, label을 함께 섞어준다



#### <span style="color: #88c8ff">Dropout</span>

<p align="center"><img src="/assets/images/optimization/drop.png"></p>

몇개의 Neural network 의 weight을 확률적(p)으로 0으로 만드는 방법



#### <span style="color: #88c8ff">Batch Normalization</span>

<p align="center"><img src="/assets/images/optimization/batch1.png"></p>

<p align="center"><img src="/assets/images/optimization/batch2.png"></p>

layer 의 statistics 를 평균을 빼고 표준편차로 나누어 정규화 시키는 방법



## <span style="color: #a6acec">BN vs Dropout</span>

보통 model 을 만드 때 batch_size 가 32 이하이면 BN 대신 Dropout 을 쓰는 것이 효과적이라고 한다.

<p align="center"><img src="/assets/images/optimization/bndr.png"></p>

small batch-size 의 경우 보는 데이터가 적기 때문에, layer 의 weight 를 정규화 하는 방법 보다는 아예 layer 를 없애는 방법이 더욱 효과적이다 

large batch-size 의 경우에는 보는 데이터가 많기 때문에, dropout 의 효과 보다는 batch normalization 의 효과가 더 크기 때문에 batch-size 가 작으면 Dropout 이 더욱 효과적이다