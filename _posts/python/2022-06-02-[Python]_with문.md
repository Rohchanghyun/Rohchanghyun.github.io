---
layout : single
title:  "[Python] with문"
excerpt: "with문 사용법"

categories:
  - Python
tags:

toc: true
toc_sticky: true

author_profile: true
sidebar_main: true

date: 2022-06-02
last_modified_at: 2022-06-02
---

# with 

- 해당 코드를 벗어날 때 자동으로 close 함수를 호출



## 사용법

as 와 같이 사용

```python
with open('text.txt') as f:
	f.readlines()
```



## 동작 방법

1. `__init__` 호출
2. with 문에 진입할 때 객체의 `__enter__` 호출
3. `__exit__` 호출



```python
class Test:
    def __init__(self):
        print('init')
        
    def __enter__(self):
        print('enter')
        
    def __exit__(self,exc_type,exc_val,exc_tb):
        print('exit')
        
with Test() as f:
    print('with 문 실행중')
      
```



결과

```
init
enter
with 문 실행중
exit
```







