---
layout : single
title:  "[Python]Iterator 속성 공부"
excerpt: "Python iterator 관련 공부"

categories:
  - Python
tags:
  - Iterator

toc: true
toc_sticky: true

author_profile: true
sidebar_main: true

date: 2022-08-30
last_modified_at: 2022-08-30
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


# Iterator


- `__iter__` 는 반복자 개체를 반환

- 하나의 원소를 꺼내서 사용하려면 `__next__`사용

- `StopIterator`: 모든 원소를 처리하면 반복자 객체의 원소를 다 처리했다는 의미로 이 예외 발생

- 반복자 class를 정의할 때는 `__iter__`, `__next__` 2개를 반드시 구현



```python
class Iterator:
    def __init__(self,iterable):
        self.iterable = iterable

    def __iter__(self):
        return self
    
    def __next__(self):
        if not self.iterable:
            raise StopIteration("데이터가 없음")
        return self.iterable.pop(0)
```


```python
it = Iterator([1,2,3,4,5])
```


```python
it.iterable
```

<pre>
[1, 2, 3, 4, 5]
</pre>

```python
next(it)
```

<pre>
1
</pre>

```python
next(it)
```

<pre>
2
</pre>

```python
it.iterable
```

<pre>
[3, 4, 5]
</pre>
- 다시 사용하려면 새로운 반복자 개체를 만든다

- 순환문에서 처리 시 모든 원소 다 소진 후에 예외가 발생하지 않는다



```python
it_1 = Iterator([1,2,3,4,5])
for i in it_1:
    print(i)
```

<pre>
1
2
3
4
5
</pre>