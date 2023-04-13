---
layout : single
title:  "[Python]Comprehension 공부"
excerpt: "Python Comprehension 관련 공부"

categories:
  - Python
tags:
  - comprehension
  - Python

toc: true
toc_sticky: true

author_profile: true
sidebar_main: true

date: 2022-08-14
last_modified_at: 2022-08-14
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


# List comprehension



```python
ll = [1,2,3,4,5]
```


```python
h_1 = [x**2 for x in ll]
h_1
```

<pre>
[1, 4, 9, 16, 25]
</pre>
## 제어문 추가



```python
h_2 = [x**2 for x in ll if x % 2 == 0]
h_2
```

<pre>
[4, 16]
</pre>
# Dict,Set comprehension



```python
s = {1,2,3,4,5}
```


```python
s_1 = [x**2 for x in s]
s_1
```

<pre>
[1, 4, 9, 16, 25]
</pre>

```python
t = [('a' , 1 ),('b' , 2 )]
```


```python
d = {x:y**2 for x,y in t}
d
```

<pre>
{'a': 1, 'b': 4}
</pre>