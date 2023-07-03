---
layout : single
title:  "[Theory] Monitoring"
excerpt: "Monitoring"

categories:
  - Theory
tags:
  - Monitoring


toc: true
toc_sticky: true

author_profile: true
sidebar_main: true

date: 2022-06-17
last_modified_at: 2022-06-17
---

# Monitoring tools for Pytorch

- 학습 동안 loss,성능 확인 가능한 도구



## Tensorboard

- 주로 시각화 기능을 제공하지만 그 외에도 다양한 편의 기능 제공



### 사용법

- 디렉터리 세팅
  - 텐서플로우 설치 시 자동으로 설치
  - 시각화하고자 하는 log 데이터를 모아야 한다
  - Tensorboard가 log데이터가 쌓이는 디렉터리를 모니터링하고, 자동으로 변경된 부분을 읽어 그래프를 업데이트

```python
import os
logs_base_dir = "logs"
os.makedirs(logs_base_dir, exist_ok=True)# Tensorboard 기록을 위한 directory 생성

from torch.utils.tensorboard import SummaryWriter # 기록 생성 객체 SummaryWriter 생성
import numpy as np

writer = SummaryWriter(logs_base_dir)
for n_iter in range(100):
    writer.add_scalar('Loss/train', np.random.random(), n_iter)
    writer.add_scalar('Loss/test', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
writer.flush()# 값 기록

%load_ext tensorboard # jupyter 상에서 Tensorboard 실행
%tensorboard --logdir {logs_base_dir}
```

- `add_scalar` : scalar 값을 기록
- `Loss/train` : loss category 에 train 값
- `n_iter` : x축의 값



## weight and biases

- 머신러닝 실험을 원활히 지원하기 위한 상용도구
- 협업,code versioning, 실험 결과 기록 등 제공
- MLOps 의 대표적인 툴

### 사용법

- 가입 후 API 확인

<p align="center"><img src="/assets/images/Theory/monitoring/figure_1.png"></p>



- 새로운 프로젝트 생성

<p align="center"><img src="/assets/images/Theory/autograd/figure_2.png"></p>



```python
!pip install wandb -q

#config 설정
config={"epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate" : LEARNING_RATE}
wandb.init(project="my-test-project", config=config)
# wandb.config.batch_size = BATCH_SIZE
# wandb.config.learning_rate = LEARNING_RATE

for e in range(1, EPOCHS+1):
	epoch_loss = 0
	epoch_acc = 0
	for X_batch, y_batch in train_dataset:
		X_batch, y_batch = X_batch.to(device), y_batch.to(device).type(torch.cuda.FloatTensor)
# …
optimizer.step()
# …
#기록 add_scalar 함수와 동일
wandb.log({'accuracy': train_acc, 'loss': train_loss})
```

