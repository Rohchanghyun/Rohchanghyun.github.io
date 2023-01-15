---
layout : single
title:  "[Python] numpy seed"
excerpt: "numpy seed 설명"

categories:
  - Python
tags:
  - numpy seed

toc: true
toc_sticky: true

author_profile: true
sidebar_main: true

date: 2022-06-17
last_modified_at: 2022-06-17
---

# Numpy Seed

- 컴퓨터 프로그램에서 발생한 random 값은 무작위 수가 아니라 특정 시작 숫자값을 정해주면 알고리즘에 따라 수열을 생성하는 방식으로 만들어진다
- 이떄의 특정 숫자가 바로 Seed
- 보통 자동으로 정해지기도 하지만 사람이 설정 가능



## 사용법

```python
np.random.seed(seed = 1)
```

이때 인수값으로 0 이상의 정수 값을 넣어준다



