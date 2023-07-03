---

layout : single

title: "[Paper]Generative Bias for Robust Visual Question Answering"

excerpt: "Generative Bias for Robust Visual Question Answering 논문 리뷰"



categories:

- Paper

tags:

- Generative bias



toc: true

toc_sticky: true



author_profile: true

sidebar_main: true



date: 2022-12-12

last_modified_at: 2022-12-12

---

## Abstract  
    - VQA작업은 최종 예측을 위해 데이터 세트 내의 bias를 사용하는 문제로 인해 어려움이 있다.
    - 이를 위해 모델이 의도적으로 biased되도록 훈련하는 앙상블 기반 debias기법이 제안되었었다.
        - 하지만 이는 훈련 데이터의 레이블 통계 또는 단일 모달 분기에서 모델에 대한 bias 계산.
    - 해당 논문에서는 Generative bias라고 하는 대상 모델에서 직접 편향 모델을 훈련시키는 생성 방법을 제안
        - 이는 adversarial objective 와 knowledge distillation을 조합하여 대상 모델의 bias를 학습하기 위해 generative network를 사용한다.

## Introduction  
    - VQA : 입력된 이미지와 질문 쌍이 주어진 답변을 올바르게 이해하고 예측해야 하는 까다로은 멀티모달 task
    - 많은 연구에서 VQA는 데이터 세트 내에서 편향되기 쉽고 데이터 세트 내에 존재하는 언어 bias에 크게 의존.
    - 이에 대응하여 최근의 연구들은 다양한 debiasing 기법 개발(ex 앙상블 기반 debiasing)
    - 앙상블 모델의 주요 목적은 주어진 입력으로 형성된 bias를 포착하는것.
    - 만약 이 앙상블 편향 모델이 편향을 잘 나타낸다면, 이러한 편향된 대답을 회피하기 위해 학습시 사용 가능
    - 기존의 앙상블 방법 -> 훈련 데이터의 계산된 레이블 통계를 사용 or 질문 또는 이미지에서 답을 계산하는 single modal branch를 사용
        - 이러한 방법은 모델의 용량이 제한되어 있기 때문에 얻을 수 있는 편향 표현에 한계가 있다고 추측
        - 사전 계산한 레이블 통계 -> 편향의 일부만 나타낸다
    - 때문에 대상 모델에서 직접 편향을 학습하는 새로운 확률적 편향 모델 제안
    - 대상 모델의 편향 모델을 으로 모델링하여 무작위 노이즈 벡터 도입
        - 동일한 질문 입력에 주어지면 대상 모델의 답변 분포를 확률적으로 모방
        - 대부분의 bias는 question에 있기 떄문에 question을 주요 bias양식으로 사용
    - 또한, 적대적 훈련 외에 지식 확산을 활용하여 편향 모델이 대상 모델에 최대한 근접하도록 하여 모델이 편향 모델로부터 더 어려운 부정적 감독으로 학습하도록
    - 마지막으로 generative bias model을 사용하여 수정된 편향 제거 손실 함수를 사용하여 대상 모델을 훈련
    - 앙상블 기반 편향 제거를 위한 새로운 편향 모델 제안
    - GAN 과 knowledge distillation 사용
    - updn 기준선 사용
### Methodology  
    -  VIsual Question Answering Baseline  
        - 이미지와 질문을 입력 쌍으로 사용하여 전체 답변 집합 A에서 정답을 정확하게 예측하는 방법을 학습
        - 이 연구에서 우리는 VQA 편향 제거 연구에서 널리 사용되는 UpDn중 하나를 채택
    -  Ensembling with Bias Models  
        - 기존 바이어스 모델의 목표 : 최대한 바이어스에 과적합
        - 과적합한 바이어스 모델이 주어지면 편향 손실 함수로 대상 모델을 훈련(대상 모델: 훈련 시키려는 모델)
        - 결과적으로 대상 모델은 바이어스 모델에서 편향된 답변을 피함으로써 편향되지 않은 답변을 예측하는 방법을 배운다.
        - 앞서 있었던 연구들이 개별 양식의 바이어스를 활용하려고 하지만, 이것이 바이어스를 나타내는 모델의 능력을 제한할 것이라고 생각.
        - 따라서 대상 모델과 유사한 편향을 나타내기 위해 구조를 동일하게 설정하고 UpDn 모델을 사용
    -  Generative Bias  
        - 확률적 편향 표현을 생성할 수 있는 바이어스 모델을 훈련시기키 위해 무작위 노이즈 벡터를 활용하여 대상 모델이 나타낼 수 있는 데이터 세트 편향과 편향을 모두 학습
        - 질문이 편향되기 쉽기 때문에 질문 양식을 유지하고 편향 모델에 대한 입력으로 사용
        - 이미지 기능도 사용하기 때문에 무작위 노이즈 도입
    -  Training the bias model  
        - 편향 모델 genB가 편향을 학습하기 위해 VQA loss 사용(Binary Cross Entropy Loss)
        - 편향 모델이 대상 모델의 편향도 포착하기를 원한다.
        - 대상 모델의 편향을 답변의 무작위 분포로 모방하기 위해 adversarial training을 제안하여 편향 모델을 훈련
        - real/fake구분하는 판별기 도입
        - 생성자는 real/fake 구분하기 어렵게 생성
        - 생성자,판별자의 훈련을 위해 편향 모델의 답변 벡터 분포는 목표 모델의 분포에 가까워야 한다
        - 편향 모델을 더욱 강화하기 위해 유사한 지식 증류 목표를 추가하여 모델 편향 모델이 q만 주어진 대상 모델의 동작을 따를 수 있도록 한다.
    -  Debiasing the Target Model