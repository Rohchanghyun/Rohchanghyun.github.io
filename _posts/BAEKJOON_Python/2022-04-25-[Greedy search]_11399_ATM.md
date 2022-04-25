---
layout : single
title:  "[Greedy search] 11399_ATM"
excerpt: "ATM 문제풀이 및 회고"

categories:
  - BAEKJOON_Python
tags:
  - BAEKJOON
  - Python

toc: true
toc_sticky: true

author_profile: true
sidebar_main: true

date: 2022-04-25
last_modified_at: 2022-04-25
---

# <span style="color: #f0b752">ATM</span>

## <span style="color: #a6acec">내 풀이</span>

```python
import sys

N = int(sys.stdin.readline())
answer = 0
time_accum = 0
time = list(map(int,sys.stdin.readline().split()))
time.sort()

for i in time:
    time_accum += i
    answer += time_accum

print(answer)
```

- `time`: 인출시간 입력을 받아와 int 형 list 로 저장
- `time_accum`: answer 에 더해줄 누적시간 

입력을 받아 map 함수를 사용하여 int 형으로 list 에 저장한다.

오름차순으로 정렬한 뒤 누적시간에 time 내의 인자를 더해주고, 그 값을 answer 에 더해준다



## <span style="color: #a6acec">회고</span>



