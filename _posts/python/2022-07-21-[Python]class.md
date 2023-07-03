---
layout : single
title:  "[Python]Class 속성 공부"
excerpt: "Python class 관련 공부"

categories:
  - Python
tags:
  - class

toc: true
toc_sticky: true

author_profile: true
sidebar_main: true

date: 2022-07-21
last_modified_at: 2022-07-21
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


# Class


## class 내부 속성


- `object.__getattribute__`는 점 연산에 해당하는 스페셜 메소드

- 인자로 클래스, 속성, 메소드를 문자열로 넣고 조회하면 이름공간 내의 속성과 메소드를 조회할 수 있다



```python
ll = [str,int]
for i in dir(object):
    if (type(object.__getattribute__(object,i)) in ll):
        print(i)
```

<pre>
__doc__
</pre>