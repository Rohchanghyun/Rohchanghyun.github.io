---
layout : single
title:  "[Python]스택,큐 공부"
excerpt: "Python stack,queue 공부"

categories:
  - Python
tags:
  - stack
  - queue
  - Python

toc: true
toc_sticky: true

author_profile: true
sidebar_main: true

date: 2022-10-09
last_modified_at: 2022-10-09
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


# Stack


- 가장 나중에 넣은 데이터를 가장 먼저 꺼낸다는 LIFO(Last in First out) 방식

- 스택에 데이터를 넣는 작업: `push`

- 스택에서 데이터를 꺼내는 작업: `pop`


## 배열로 Stack 구현하기


- 스택 배열: `stk`

- 스택 크기: `capacity`

- 스택 포인터: `ptr`



```python
from typing import Any
class FixedStack:
    class Empty(Exception):
        pass

    class Full(Exception):
        pass

    def __init__(self,capacity: int = 256) -> None:
        self.capacity = capacity
        self.stk = [None] * self.capacity
        self.ptr = 0

    def __len__(self) -> int:
        return self.ptr

    def is_empty(self) -> None:
        return self.ptr <= 0
    
    def is_full(self) -> None:
        return self.ptr >= self.capacity
    
    def push(self,value: Any) -> None:
        if self.is_full:
            raise FixedStack.Full
        self.stk[self.ptr] = value
        self.ptr += 1
        
    def pop(self) -> Any:
        if self.is_empty:
            raise FixedStack.Empty
        self.ptr -= 1
        return self.stk[self.ptr]
    
    def peek(self) -> Any:
        if self.is_empty:
            raise FixedStack.Empty
        return self.stk[self.ptr - 1]
    
    def clear(self) -> None:
        self.ptr = 0
    
    def find(self,value: Any) -> Any:
        for i in range(self.ptr-1, -1, -1):
            if self.stk[i] == value:
                return i
            
        return -1
    
    def count(self,value: Any) -> int:
        count = 0
        for i in range(self.ptr):
            if self.stk[i] == value:
                count += 1

        return count
    
    def __contains__(self,value: Any) -> bool:
        return self.count(value) >= 0
    
    def dump(self) -> None:
        if self.is_empty():
            print('stack is empty')
        else:
            print(self.stk[:self.ptr])
```

## collections.deque로 Stack 구현하기



```python
from typing import Any
from collections import deque

class Stack:
    def __init__(self,maxlen: int = 256) -> None:
        self.capacity = maxlen
        self.__stk = deque([],self.capacity)

    def __len__(self) -> int:
        return len(self.__stk)

    def is_empty(self) -> bool:
        return not self.__stk
    
    def is_full(self) -> bool:
        return len(self.__stk) == self.__stk.maxlen
    
    def push(self,value: Any) -> None:
        self.__stk.append(value)

    def pop(self) -> Any:
        self.__stk.pop()

    def peek(self) -> Any:
        self.__stk[-1]

    def claer(self) -> None:
        self.__stk.clear()

    def find(self,value: Any) -> Any:
        try:
            return self.__stk.index(value)
        except ValueError:
            return -1
        
    def count(self,value: Any) -> int:
        return self.__stk.count(value)
    
    def __contains__(self,value: Any) -> bool:
        return self.count(value)
    
    def dump(self) -> None:
        print(list(self.__stk))
```

# Queue


- 가장 먼저 넣은 데이터를 먼저 꺼내는 FIFO(First in First out) 방식

- 큐에 데이터를 추가하는 작업: `enqueue`

- 큐에서 데이터를 꺼내는 작업: `dequeue`

- 데이터를 꺼내는 쪽(맨 앞의 원소를 가리킴): `front`

- 데이터를 넣는 쪽(맨 끝의 원소를 가리킴): `rear`


## 배열로 Queue 구현하기


- 링 버퍼를 사용하여 구현



```python
from typing import Any

class FixedQueue:
    class Empty(Exception):
        pass

    class Full(Exception):
        pass

    def __init__(self,capacity: int) -> None:
        self.capacity = capacity
        self.no = 0
        self.front = 0
        self.rear = 0
        self.que = [None] * self.capacity

    def __len__(self) -> int:
        return self.no

    def is_empty(self) -> bool:
        return self.no <= 0

    def is_full(self) -> bool:
        return self.no >= self.capacity

    def enque(self,value: Any) -> None:
        if self.is_full():
            raise FixedQueue.FUll
        self.que[self.rear] = value
        self.no += 1
        self.rear += 1
        if self.rear == self.capacity:
            self.rear = 0

    def deque(self) -> Any:
        if self.is_empty():
            raise FixedQueue.Empty
        temp = self.que[self.front]
        self.front += 1
        self.no -= 1
        if self.front == self.capacity:
            self.front = 0
        return temp
    
    def peek(self) -> Any:
        if self.is_empty():
            raise FixedStack.Empty
        return self.que[self.front]
    
    def find(self,value: Any) -> Any:
        for i in range(self.no):
            idx = (i + self.front) & self.capacity
            if self.que[idx] == value:
                return idx
            
        return -1
    
    def count(self,value: Any) -> int:
        count = 0
        for i in range(self.no):
            idx = (i + self.front) & self.capacity
            if self.que[idx] == value:
                count += 1

        return count
    
    def __contains__(self,value: Any) -> bool:
        return self.count(value)
    
    def clear(self) -> None:
        self.no = self.front = self.rear = 0

    def dump(self) -> None:
        if self.is_empty():
            raise FixedQueue.Empty
        else:
            for i in range(self.no):
                print(self.que[(i + self.front & self.capacity)], end ='')
            print()

    
```


```python

```
