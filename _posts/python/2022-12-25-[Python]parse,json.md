---

layout : single

title: "[Python]parse,json"

excerpt: "parse,json 공부"



categories:

- Python

tags:

- Python
- parse
- json




toc: true

toc_sticky: true



author_profile: true

sidebar_main: true



date: 2022-12-25

last_modified_at: 2022-12-25

---
# parse,json

  

## json : Java script Object Notation

단순한 데이터 포맷. 데이터를 표시하는 방법

### 사용 이유 : 

   - 가지고 있는 데이터를 받아 객체나 변수에 할당해서 사용하기 위해

   - 다른 포맷에 비해 경량화된 데이터 포맷

  

### 구조:

   1. object(객체)

      - name/value의 순서쌍으로 set

      - {}으로 정의된다.

   2. Array(배열)

  

## json parsing

   json 파일 내에 특정 data만 추출하는 것을 json parsing 이러고 한다.

  

  

파이썬 argument parser 사용법

  

사용 이유: 파이썬 실행 시 커맨드 라인 인수를 다룰 때 ArgumentParser를 사용하면 편리하다. 다양한 형식으로 인수를 지정하는 것이 가능

  

  

### 사용 방법

1\. argparse import

2\. parser 생성

3\. 인수 설정

4\. 분석

  

```python
import argparse
parser = argparse.ArgumentParser(description = 'parse test')# 이 프로그램의 설명

parser.add_argument('arg1',help = '해당 인수의 설명')# 필요한 인수 추가

parser.add_argument('arg2',default = 'hello')# 디폴트 설정

parser.add_argument('arg3',type = int,default = 24)# 타입 설정

parser.add_argument('--fruit', choices=['apple', 'banana', 'orange'])# 선택지 설정

parser.add_argument('--colors', args='*')# 여러개의 데이터 

parser.add_argument("-i", required=True)# 필수 지정

tp = lambda x:list(map(int, x.split('.')))
parser.add_argument('--address', type=tp, help='IP address')# lambda 함수를 사용하여 문자형으로부터 원하는 형태로 변환

parser.add_argument('-a','-arg4')# 커맨드라인 약칭 사용
```

  

사용 예

`python argparse.py -a 1234` 

  

인수 정보 확인 예

`python argparse.py -h`