---
layout : single
title:  "[Theory] Bayesian mathematics "
excerpt: "Bayesian mathematics 정리"

categories:
  - Theory
tags:
  - bayesian mathematics

toc: true
toc_sticky: true

author_profile: true
sidebar_main: true

date: 2022-04-04
last_modified_at: 2022-04-04
---

# <span style="color: #f0b752">Bayesian statistics</span>

## <span style="color: #a6acec">조건부 확률</span>

<p align="center"><img src="/assets/images/bayesian/option.png"></p>

이때 <span style="color: #88c8ff">P(A|B)</span> 는 사건 <span style="color: #ed6663">B</span>가 일어난 상황에서 사건 <span style="color: #88c8ff">A</span>가 발생할 확률을 의미한다.

- Bayesian statistics 는 조건부 확률을 이용하여 정보를 갱신하는 방법을 알려준다

<p align="center"><img src="/assets/images/bayesian/how.png"></p>

<span style="color: #88c8ff">A</span>라는 새로운 정보가 주어졌을 때 <span style="color: #ed6663">P(B)</span> 로부터 <span style="color: #88c8ff">P(A|B)</span> 를 계산하는 방법을 제공

<p align="center"><img src="/assets/images/bayesian/ex.png"></p>

### <span style="color: #b1cf89">Example</span>

<p align="center"><img src="/assets/images/bayesian/problem.png"></p>

<details> <summary><span style="color: #ffffff">Answer</span></summary> <div markdown="1"><p align="center"><img src="/assets/images/bayesian/answer.png"></p></div> </details>

### <span style="color: #b1cf89">조건부 확률의 시각화</span>

<p align="center"><img src="/assets/images/bayesian/visual.png"></p>

<p align="center"><img src="/assets/images/bayesian/excel.png"></p>

### <span style="color: #b1cf89">베이즈 정리를 통한 정보의 갱신</span>

<p align="center"><img src="/assets/images/bayesian/update.png"></p>

- 갱신된 사후확률을 구하기 위해 이전 사후확률을 P(θ)에 대입한다.
- 이때 <span style="color: #88c8ff">evidence</span>도 갱신해주어야 한다

<p align="center"><img src="/assets/images/bayesian/update2.png"></p>