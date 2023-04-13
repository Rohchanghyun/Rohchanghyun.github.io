---
layout : single
title:  "[Python]검색 알고리즘"
excerpt: "Python search algorithm"

categories:
  - Python
tags:
  - search
  - Python

toc: true
toc_sticky: true

author_profile: true
sidebar_main: true

date: 2022-09-19
last_modified_at: 2022-09-19
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


# Search Algorithm


## 선형 검색


- 선형으로 늘어선 배열에서 검색하는 경우에 원하는 키값을 가진 원소를 찾을 때까지 맨 앞부터 스캔하여 순서대로 검색하는 알고리즘



```python
from typing import Any,Sequence

#index 반환
def seq_search(a: Sequence, key: Any) -> int:
    for i,element in enumerate(a):
        if i == len(a):
            return "no answer"
        elif element == key:
            return i 
        
```


```python
a = [1,2,3,4,5]
key = 3
print(seq_search(a,key))
```

<pre>
2
</pre>
### 보초법


- 2가지 종료 조건을 체크하는 대신 배열의 마지막에 검색하려는 값을 추가하여 비용을 반으로 줄이는 방법



```python
import copy
def seq_search(a: Sequence,key : Any) -> int:
    seq = copy.deepcopy(a)
    seq.append(key)
    answer = 0

    for i,element in enumerate(seq):
        if element == key:
            answer = i
            break
    
    if answer == len(a):
        raise RuntimeError('no answer')
    else:
        print('answer : ',answer)
```


```python
import time

start = time.time()

a = [1,2,3,4,5]
key = 6
seq_search(a,key)

end = time.time()
print("runtime = ",(end - start))
```

## 이진 검색



```python
from typing import Any,Sequence

def bin_search(a: Sequence,key: Any) -> int:
    pl = 0
    pr = len(a) - 1
    pc = len(a) // 2

    while True:
        if a[pc] > key:
            pr = pc - 1
            pc = (pl + pr) // 2

        elif a[pc] < key:
            pl = pc + 1
            pc = (pl + pr) // 2

        else: 
            return pc
        
        if pl > pr:
            raise RuntimeError('no answer')
    

    
```


```python
import time

start = time.time()

a = [1,2,3,4,5]
key = 3
print('answer : ',bin_search(a,key))

end = time.time()
print("runtime = ",(end - start))
```

<pre>
answer :  2
runtime =  0.0005581378936767578
</pre>

```python
import time

start = time.time()

a = [1,2,3,4,5]
key = 7
print('answer : ',bin_search(a,key))

end = time.time()
print("runtime = ",(end - start))
```

## 해시법


- 데이터를 저장할 위치 = 인덱스 를 간단한 연산으로 구하는 방법

- 특정한 해시값을 구하여 이를 인덱스로 사용하는 해시 테이블 생성

- 해시값을 구하는 과정: 해시 함수

- 해시 테이블에서 만들어진 원소: 버킷

- 하지만 이러한 과정에서 새로운 원소 추가 시 버켓이 중복되는 현상을 `해시 충돌` 이라 한다


2가지 해시 충돌 해결법

- 체인법: 해시값이 같은 원소를 연결 리스트로 관리

- 오픈 주소법: 빈 버켓을 찾을 때까지 해시를 반복


### 체인법



```python
from __future__ import annotations
from typing import Any,Type
import hashlib

#해시를 구성하는 노드
class NODE:
    def __init__(self,key : Any,value : Any, next : NODE) -> None:
        self.key = key
        self.value = value
        self.next = next

class ChainedHash:
    def __init__(self,capacity: int) -> None:
        self.capacity = capacity
        self.table = [None] * self.capacity

    def hash_value(self,key : Any) -> int:
        if isinstance(key,int):
            return key % self.capacity
        return(int(hashlib.sha256(str(key).encode()).hexdigest(),16) % self.capacity)
    
    def search(self,key) -> int:
        hash = self.hash_value(key)
        p = self.table[hash]

        while p is not None:
            if p.value == key:
                return p.value
            p = p.next

        return None
    
    def add(self,key: Any,value : Any) -> bool:
        hash = self.hash_value(key)
        p = self.table[hash]

        while p is not None:
            if p.key == key:
                return False
            p = p.next
        
        temp = NODE(key,value,self.table[hash])
        self.table[hash] = temp
        return True
    
    def remove(self,key: Any,value : Any) -> bool:
        hash = self.hash_value(key)
        p = self.table[hash]
        pre_node = None

        while p is not None:
            if p.key == key:
                if pre_node is None:
                    self.table[hash] = p.next
                else:
                    pre_node.next = p.next
                return True
            pre_node = p
            p = p.next
        
        return False
    
    def dump(self) -> None:
        for i in range(self.capacity):
            p = self.table[i]
            print(i,end = '')
            while p is not None:
                print(f' -> {p.key}({p.value})',end = '')
                p = p.next
            print()
                

        
```

- `capacity` : 해시 테이블의 크기(배열 table의 원소 수)

- `table` : 해시 테이블을 저장하는 list형 배열



- `__init__()` : 원소 수가 capacity인 list형 배열 table을 생성하고 모든 원소를 `None`으로 한다.

- `hash_value()` : 인수 key에 대응하는 <span style="color: #88c8ff"> 해시값 </span>을 구한다.

    - key가 int형인 경우

        - key를 capacity로 나눈 나머지를 해시값으로 설정

    - key가 int형이 아닌경우

        - key가 정수가 아닌 경우는 그 값으로 바로 나누 수 없다.

        - 표준 라이브러리로 형 변환을 해야 해시값을 얻을 수 있다.

- 표준 알고리즘

    - sha256 알고리즘 : 주어진 바이트 문자열의 해시값을 구하는 해시 알고리즘의 생성자.(hashlib 모듈에서 제공, RSA의 FIPS알고리즘을 바탕으로 한다.)

    - encode() 함수 : `hashlib.sha256`에는 바이트 문자열의 인수를 전달해야 한다. key를 str문자열로 변환한 뒤 그 문자열을 이 함수에 전달하여 바이트 문자열을 생성.

    - hexdigest() 함수 : sha256 알고리즘에서 해시값을 16진수 문자열로 꺼낸다.

    - int() 함수 : hexdigest() 함수로 꺼낸 문자열을 16진수 문자열로 하는 int형으로 변환



### 오픈 주소법(선형 탐사법)



```python
from __future__ import annotations
from typing import Any,Type
from enum import Enum
import hashlib

class Status(Enum):
    OCCUPIED = 0
    EMPTY = 1
    DELETED = 2

class Bucket:
    def __init__(self,key: Any = None, value: Any = None, stat: Status = Status.EMPTY) -> None:
        self.key = key
        self.value = value
        self.stat = stat

    def set(self,key: Any, value: Any, stat: Status) -> None:
        self.key = key
        self.value = value
        self.status = stat
        
    def set_status(self,stat: Status) -> None:
        self.stat = stat

class OpenHash:
    def __init__(self,capacity: int) -> None:
        self.capacity = capacity
        self.table = [Bucket()] * self.capacity

    def hash_value(self,key : Any) -> int:
        if isinstance(key,int):
            return key % self.capacity
        return(int(hashlib.sha256(str(key).encode()).hexdigest(),16) % self.capacity)
    
    def rehash_value(self,key: Any) -> int:
        return (self.hash_value(key + 1)) % self.capacity
    
    def search_node(self,key: Any) -> Any:
        hash = self.hash_value(key)
        p = self.table[hash]

        for i in range(self.capacity):
            if p.stat == Status.EMPTY:
                break
            elif p.stat == Status.OCCUPIED and p.key == key:
                return p
            hash = self.rehash_value(hash)
            p = self.table[hash]
        return None
    
    def search(self,key: Any) -> Any:
        p = self.search_node(key)
        if p is not None:
            return p.value
        else:
            return None
        
    def add(self,key: Any,value: Any) -> bool:
        if self.search(key) is not None:
            return False
        
        hash = self.hash_value(key)
        p = self.table[hash]
        for i in range(self.capacity):
            if p.stat == Status.EMPTY or p.stat == Status.DELETED:
                self.table[hash] = Bucket(key,value,Status.OCCUPIED)
                return True
            hash = self.rehash_value(hash)
            p = self.table[hash]
        return False

    def remove(self,key: Any) -> int:
        p = self.search_node(key)
        if p is None:
            return False
        p.set_status(Status.DELETED)
        return True
    
    def dump(self) -> None:
        for i in range(self.capacity):
            print(f'{i:2} ',end = '')
            if self.table[i].stat == Status.OCCUPIED:
                print(f'{self.table[i].key} ({self.table[i].value})')
            elif self.table[i].stat == Status.EMPTY:
                print('--미등록--')
            elif self.table[i].stat == Status.DELETED:
                print('--삭제완료--')
```


```python
```
