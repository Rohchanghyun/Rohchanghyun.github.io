---
layout : single
published: true
title:  "[Python] Data structure"
excerpt: "Python 의 여러가지 자료형과 사용법 "

categories:
  - Python
tags:
  - data structure

toc: true
toc_sticky: true

author_profile: true
sidebar_main: true

date: 2022-03-09
last_modified_at: 2022-03-09
---
# 내장 자료구조

## tuple

1차원의 고정된 **변경 불가능한** 순차 자료형

### tuple 생성

쉼표로 대입


```python
tup = 4,5,6
type(tup)
```




    tuple



괄호 사용

중첩 tuple


```python
tup = (4,5,6),(7,8)
print(type(tup))
print(tup)
```

    <class 'tuple'>
    ((4, 5, 6), (7, 8))
    

순차 자료형, iterator 


```python
tuple([4,0,2])
```




    (4, 0, 2)




```python
tup = tuple('string')
print(type(tup))
print(tup)
```

    <class 'tuple'>
    ('s', 't', 'r', 'i', 'n', 'g')
    

### tuple 접근

인덱스를 통해 접근


```python
print(tup[0])
```

    s
    

**tuple이 저장된 `객체 자체`는 변경 가능 하지만 각 슬롯에 저장된 `객체`는 변경 불가능하다**


```python
tup = tuple(['foo',[1,2],True])
tup[2] = False
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    ~\AppData\Local\Temp/ipykernel_15936/3351164410.py in <module>
          1 tup = tuple(['foo',[1,2],True])
    ----> 2 tup[2] = False
    

    TypeError: 'tuple' object does not support item assignment


tuple 내에 저장된 객체는 그 위치에서 변경 가능


```python
tup[1].append(3)
print(tup)
```

    ('foo', [1, 2, 3], True)
    

### tuple 이어붙이기    


```python
tup_a = 1,2,3
tup_b = 4,5,6

print(tup_a + tup_b)
print(tup_a * 2)
```

    (1, 2, 3, 4, 5, 6)
    (1, 2, 3, 1, 2, 3)
    

이때 객체는 복사하지 않고, **객체에 대한 참조만 복사**한다

### tuple 값 분리하기


```python
a,b,c = tup_a
print('a = {0}, b = {1}, c = {2}'.format(a,b,c))
```

    a = 1, b = 2, c = 3
    


```python
tup = (4,5,(6,7))
a,b,(c,d) = tup
print('a = {0}, b = {1}, c = {2}, d = {3}'.format(a,b,c,d))
```

    a = 4, b = 5, c = 6, d = 7
    

이러한 unpack 을 통해 `b,a = a,b`처럼 쉽게 swap 가능<br>
또한 **함수에서 여러개의 값 반환할 때 자주 사용한다**


```python
values = 1,2,3,4,5
a,b,*rest = values
print('a = {0}, b = {1}, rest = {2}'.format(a,b,rest))
```

    a = 1, b = 2, rest = [3, 4, 5]
    

이처럼 나머지 값들을 버릴 때 사용할 수도 있다

## list

1차원 순차 자료형<br>
iterator 에서 실제 값을 모두 담기 위해 사용

### list 생성


```python
a_list = [2,3,7]
b_list = list(tup)

print(type(a_list))
print(type(b_list))
```

    <class 'list'>
    <class 'list'>
    

### list 원소 추가, 삭제


```python
a_list.append(4)
print(a_list)
```

    [2, 3, 7, 4, 4]
    


```python
a_list.extend([4,5,(7,8)])
print(a_list)
```

    [2, 3, 7, 4, 4, 4, 5, (7, 8)]
    


```python
a_list.insert(0,1)
print(a_list)
```

    [1, 1, 2, 3, 7, 4, 4, 4, 5, (7, 8)]
    

insert 는 append 에 비해 연산비용이 많다. 시작과 끝에 추가하고 싶으면 양방향 큐인 `collections.deque`를 사용하는 것을 추천한다

### list 원소 삭제


```python
a_list.pop(0)
print(a_list)
```

    [1, 2, 3, 7, 4, 4, 4, 5, (7, 8)]
    


```python
a_list.remove(7)
print(a_list)
```

    [1, 2, 3, 4, 4, 4, 5, (7, 8)]
    

### 값 존재 여부 검사


```python
6 in a_list
```




    False



### list 이어붙이기


```python
print(a_list + b_list)
```

    [1, 2, 3, 4, 4, 4, 5, (7, 8), 4, 5, (6, 7)]
    

### 정렬


```python
a_list.sort()
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    ~\AppData\Local\Temp/ipykernel_15936/1042120378.py in <module>
    ----> 1 a_list.sort()
    

    TypeError: '<' not supported between instances of 'tuple' and 'int'


이처럼 sort 시에 다른 자료형이 있으면 에러가 난다(대소 비교 불가능)


```python
a_list = [2,3,1,5]
a_list.sort()
print(a_list)
```

    [1, 2, 3, 5]
    


```python
b_list = ['a','22','333','4']
b_list.sort(key = len)
print(b_list)
```

    ['a', '4', '22', '333']
    

### 이진탐색


```python
import bisect
```


```python
c = [1,2,2,2,3,4,7]
bisect.bisect(c,2)
```




    4



bisect는 list 가 정렬된 상태인지는 모르기 떄문에 연산 비용이 높을 수 있고, 정확하지 않은 값을 반환할 수 있다<br>
이때 반환되는 것은 2 인자가 들어갈 index를 반환해준다


```python
bisect.insort(c,6)
print(c)
```

    [1, 2, 2, 2, 3, 4, 6, 7]
    

### slicing


```python
a_list = [1,2,3,4,5]
print(a_list[1:3])
```

    [2, 3]
    

시작 위치는 포함하지만, 마지막 위치는 포함하지 않기 때문에 slicing 의 결과 갯수는 `stop - start` 

step 지정


```python
print(a_list[::2])
```

    [1, 3, 5]
    

## 내장 순차 자료형

### enumerate

순차 자료형에서 현재 아이템의 index를 함께 처리


```python
a_list = [1,2,3,4,5]
for i,values in enumerate(a_list):
    print('index = {0}, values = {1}'.format(i,values))
```

    index = 0, values = 1
    index = 1, values = 2
    index = 2, values = 3
    index = 3, values = 4
    index = 4, values = 5
    

enumerate 를 사용하여 dict에 저장


```python
some_list = ['a','b','c','d']
mapping = {}

for i,v in enumerate(some_list):
    mapping[v] = i

print(mapping)
```

    {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    

### sorted

정렬된 새로운 숫자 자료형을 반환<br>
`sort` 메서드와 인자가 같다

### zip

여러 숫자 자료형을 index 끼리 묶어준다


```python
seq1 = [1,2,3]
seq2 = ['a','b','c']
for i,(a,b) in enumerate(zip(seq1,seq2)):
    print('index {0}: {1},{2}'.format(i,a,b))
```

    index 0: 1,a
    index 1: 2,b
    index 2: 3,c
    

여러 개의 순차 자료형을 받을 수 있으며 반환되는 리스트의 크기는 넘겨받은 순차 자료형 중 가장 짧은 크기로 정해진다


```python
seq3 = [True,False]

list(zip(seq1,seq2,seq3))
```




    [(1, 'a', True), (2, 'b', False)]




```python
pitchers = [('Nolan','Ryan'),('Roger','Clemens'),('Schilling','Curt')]
first_names,last_names = zip(*pitchers)
```


```python
first_names
```




    ('Nolan', 'Roger', 'Schilling')




```python
last_names
```




    ('Ryan', 'Clemens', 'Curt')




```python
first_names,last_names,*rest = zip(pitchers)
```


```python
first_names
```




    (('Nolan', 'Ryan'),)




```python
last_names
```




    (('Roger', 'Clemens'),)




```python
rest
```




    [(('Schilling', 'Curt'),)]



이처럼 `*`로 넘겨줘야 제대로 unpack이 된다

### reversed

순차 자료형을 역순으로 순회한다


```python
list(reversed(range(10)))
```




    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]



'reversed' 는 **generator** 이기 때문에 모든 값을 다 받아오기 전에는 순차 자료형을 생성하지 않는다

## dict

dict는 **해시맵** 또는 **연관 배열**이라고 알려져 있다. `유연한 크기를 가지는 키-값 쌍`으로 키와 값은 모두 파이썬 객체이다

사전의 `value`로는 어떤 파이썬 객체라도 가능하지만 `key`는 tuple, 스칼라형 처럼 **값이 바뀌지 않는 객체만 가능**하다.(해시 가능)

### dict 생성


```python
empty_dict = {}
```


```python
d1 = {'a' : 'some_vlaue','b':[1,2,3,4]}
d1
```




    {'a': 'some_vlaue', 'b': [1, 2, 3, 4]}



### dict 원소 추가


```python
d1[7] = 'an integer'
```


```python
d1
```




    {'a': 'some_vlaue', 'b': [1, 2, 3, 4], 7: 'an integer'}



### dict 원소 불러오기


```python
d1['b']
```




    [1, 2, 3, 4]



### dict 내에 존재 여부


```python
'b' in d1
```




    True



### dict 원소 삭제


```python
d1[5] = 'some_value'
d1['dummy'] = 'another value'

del d1[5]
```


```python
d1
```




    {'a': 'some_vlaue',
     'b': [1, 2, 3, 4],
     7: 'an integer',
     'dummy': 'another value'}




```python
ret = d1.pop('dummy')
ret
```




    'another value'




```python
d1
```




    {'a': 'some_vlaue', 'b': [1, 2, 3, 4], 7: 'an integer'}



pop은 원소를 dict 에서 삭제하고 반환한다

### update

dict 2개를 합칠 수 있다


```python
d1.update({'b':'foo','c':12})

d1
```




    {'a': 'some_vlaue', 'b': 'foo', 7: 'an integer', 'c': 12}



### 순차 자료형에서 사전 생성

[여기](#enumerate-를-사용하여-dict에-저장)

### hash 가능 여부

key는 해시 가능한 자료형만 들어올 수 있기 때문에 해시 가능 여부를 판단해야한다


```python
hash('string')
```




    -5203080771432362467




```python
hash((1,2,[2,3]))
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    ~\AppData\Local\Temp/ipykernel_22600/3257896050.py in <module>
    ----> 1 hash((1,2,[2,3]))
    

    TypeError: unhashable type: 'list'


이러한 list를 key로 사용하기 위해서는 tuple 로 변환해주면 된다

## set

유일한 원소만 담는 정렬되지 않는 자료형이다

### set 생성


```python
set([2,2,2,1,3,3,3,3,4,4,5])
```




    {1, 2, 3, 4, 5}



### 집합 연산

set 은 **집합 연산**을 제공한다


```python
a = {1,2,3,4,5}
b = {3,4,5,6,7,8}
```


```python
a.union(b)
```




    {1, 2, 3, 4, 5, 6, 7, 8}




```python
a | b
```




    {1, 2, 3, 4, 5, 6, 7, 8}



dict처럼 set 원소들도 변경 불가능 해야한다

### 부분집합 확대집합 검사


```python
a_set = {1,2,3,4,5}

{1,2,3}.issubset(a_set)
```




    True




```python
a_set.issuperset({1,2,7})
```




    False



## 사전 표기법

간결한 표현으로 새로운 리스트를 만들 수 있다

### list

`[expr for val in collection if condition]`

이를 반복문으로 구현하면 

```python
result = []
for val i ncollection:
    if condition:
        result.append(expr)
```


ex)


```python
strings = ['a','as','bat','car','dove','python']

[x.upper() for x in strings if len(x) > 2]
```




    ['BAT', 'CAR', 'DOVE', 'PYTHON']



### dict

```python
dict_comp = {key-expr : value-expr for value in collection if condition}
```

### set

```python
set_comp = {expr for value in collection if condition}
```

## generator

### iterator

iterator protocol 을 통해 순회 가능한 객체를 만들 수 있다

dict 를 순회하면 key 값이 반환된다


```python
some_dict = {'a':1,'b':2,"c":3}

for key in some_dict:
    print(key)
```

    a
    b
    c
    

이때 `for key in some_Dict` 라고 작성하면 파이썬 인터프리터는 some_dict에서 iterator 를 생성한다


```python
dict_iterator = iter(some_dict)
dict_iterator
```




    <dict_keyiterator at 0x1655d133db0>



이터레이터는 for문 같은 context 에서 사용될 경우 객체를 반환한다. 

### generator

순회 가능한 객체는 간단한 방법이다

일반 함수는 실행되면 단일 값을 반환하는 반면 제너레이터는 순차적인 값을 매 요청 시마다 하나씩 반환해준다<br>

생성하려면 `yield` 사용해야 한다


```python
def squares(n = 10):
    print('Generator squares from 1 to {0}'.format(n**2))
    for i in range(1,n+1):
        yield i**2
```


```python
gen = squares()
gen
```




    <generator object squares at 0x000001655D1490B0>




```python
for x in gen:
    print(x,end=' ')
    
```

    Generator squares from 1 to 100
    1 4 9 16 25 36 49 64 81 100 

#### 제너레이터 표현식


```python
gen = (x ** 2 for x in range(100))
```


```python
sum(gen)
```




    328350



이 코드는 아래의 코드와 동일한 코드다<br>
```python
def _make_gen():
    for x in range(100):
        yield x ** 2
gen = _make_gen()


```python

```
