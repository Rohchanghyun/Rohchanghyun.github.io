---
layout : single
title:  "[2022 KAKAO BLIND RECRUITMENT] 신고결과받기"
excerpt: "신고 결과 받기 문제풀이 및 회고"

categories:
  - Programmers_Python
tags:
  - Python

toc: true
toc_sticky: true

author_profile: true
sidebar_main: true

date: 2022-04-12
last_modified_at: 2022-04-12
---

# <span style="color: #f0b752">신고 접수</span>

## <span style="color: #a6acec">내 풀이</span>

```python
from collections import defaultdict
def solution(id_list, report, k):
    report_count = defaultdict(list)
    answer_count = {}
    answer = []
    
    report = set(report)
    for s in id_list:
        answer_count[s] = 0
        
    for r in report:
        report_from,report_to = r.split(' ')
        report_count[report_to].append(report_from)
            
    for key,val in report_count.items():
        if len(val) >= k:
            for i in val:
                answer_count[i] += 1
                
    for key,val in answer_count.items():
        answer.append(val)
        
    return answer
```

- `report = set(report)` : 중복된 report 를 줄임으로써 시간적으로 큰 여유를 가져왔다
- `report_count` : dict 타입
	- <span style="color: #88c8ff">key</span> = 신고당한 사람
	- <span style="color: #ed6663">value</span> = 신고한사람 목록 type(list)

- `answer_count` : dict 타입
	- <span style="color: #88c8ff">key</span> = 메일을 받을 사람
	- <span style="color: #ed6663">value</span> = 메일 갯수

먼저 report = set(report)로 중복된 report 를 제거해준다

report_from,report_to 로 공백을 기준으로 신고한사람과 신고당한사람을 나누어 준다. 

그 후 `report_count` 에 신고한 사람 목록을 list 형태로 저장해준다

report_count 내에서 iter를 통해 value 의 list 길이가 k 이상이면 answer_count의 value 를 +1 해준다

최종적으로 answer 리스트에 저장해준다

## <span style="color: #a6acec">정답</span>

```python
def solution(id_list, report, k):
    answer = [0] * len(id_list)    
    reports = {x : 0 for x in id_list}

    for r in set(report):
        reports[r.split()[1]] += 1

    for r in set(report):
        if reports[r.split()[1]] >= k:
            answer[id_list.index(r.split()[0])] += 1

    return answer
```





## <span style="color: #a6acec">회고</span>

밑의 코드는 내가 처음 실행했을 때의 코드로, 몇개의 task 에서 시간초과가 나왔다.

```python
def solution(id_list, report, k):
    report_list = []
    report_original = []
    report_count = {}
    answer_count = {}
    answer = []
    for i in range(len(report)):
        if report[i] in report_original:
            pass
        else:
            report_list.append(report[i].split(' ')[-1])
            report_original.append(report[i])
    for s in id_list:
        report_count[s] = report_list.count(s)
        answer_count[s] = 0
    for key,val in report_count.items():
        if val >= k:
            for s in report_original:
                if s.split(' ')[-1] == key:
                    answer_count[s.split(' ')[0]] += 1
    for key,val in answer_count.items():
        answer.append(val)
    return answer
```

내가 생각한 이 코드의 문제점은 

```python
for key,val in report_count.items():
        if val >= k:
            for s in report_original:
                if s.split(' ')[-1] == key:
                    answer_count[s.split(' ')[0]] += 1
```

이 부분에서 조건문 및 반복문을 너무 사용하여 시간초과가 났다고 생각하였고, 이러한 최종 제출 이전 코드를 만들게 되었다

```python
from collections import defaultdict
def solution(id_list, report, k):
    report_list = []
    report_original = []
    report_count = defaultdict(list)
    answer_count = {}
    answer = []
    
    for s in id_list:
        answer_count[s] = 0
        
    for r in report:
        report_from,report_to = r.split(' ')
        if r in report_original:
            pass
        else:
            report_original.append(r)
            report_count[report_to].append(report_from)
            
    for key,val in report_count.items():
        if len(val) >= k:
            for i in val:
                answer_count[i] += 1
                
    for key,val in answer_count.items():
        answer.append(val)
        
    return answer
```

하지만 같은 test case 에서 시간초과가 나서, 이번에는 `report_original` (중복된 report 를 없애는 코드) 관련 코드를 제거하고, `report = set(report)`를 사용하였더니, 시간 초과를 잡을 수 있었다

