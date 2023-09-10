---
layout : single
title:  "[Python]python interview question repo 공부"
excerpt: "python interview question repo 읽고 정리"

categories:
  - Python
tags:

toc: true
toc_sticky: true

author_profile: true
sidebar_main: true

date: 2023-09-09
last_modified_at: 2023-09-09
---

출처 : [link](https://github.com/boost-devs/ai-tech-interview/blob/main/answers/4-python.md)

# What is the difference between list and tuples in Python?

- 리스트
  - mutuable
- 튜플
  - immutable



# What are the key features of Python?

파이썬의 주요 특징 

- 인터프리터 언어
  - 실행 전 컴파일 필요 없다
- 동적 타이핑
  - 실행시간에 자료형을 검사하여 선언시 변수 유형을 명시할 필요 없다.
  - 변수 선언시 타입을 명시하는 C,C++같은 언어는 정적 타입 언어라고 한다.
- 객체 지향 언어
  - 클래스와 구성 및 상속을 함꼐 정의할 수 있다는 점에서 객체지향 프로그래밍에 적합하다.
- 일급 객체
  - 파이썬에서 함수와 클래스는 일급 객체.
  - 변수나 데이터 구조 안에 담을 수 있고, 매개변수로 전달이 가능하며 리턴값으로 사용될 수 있다.
- 스크립트 언어이므로 다른 컴파일 언어에 비해 다소 느리다.



# What type of language is python? Programming or scripting?

파이썬은 정확하게는 스크립트 언어이다. 모든 스크립트 언어가 프로그래밍 언어에 속하지만 모든 프로그래밍 언어가 스크립트 언어는 아니다. 

일반적인 경우에 파이썬을 프로그래밍 언어의 목적으로 분류하고, 많이 사용한다.



# Python an interpreted language. Explain.

- 인터프리터 언어: 고급 언어로 작성된 원시코드 명령어들을 한줄씩 읽어들여 실행하는 프로그램

- 실행시간 전에 기계 레벨 코드를 만드는 컴파일 언어와 다르게 소스코드를 바로 실행한다.



# What is pep 8?

- Python Enhancement proposal
- 파이썬 코드를 포맷하는 방법을 지정하는 규칙 집합
- 파이썬 코드ㅡㄹ 어떻게 구성할 지 알려주는 스타일 가이드로서의 역할을 한다.
- `black`,`flake8`,`autopep8`,`yamf`등이 있다.



# How is memory managed in Python?

파이썬은 객체로 모든것을 관리한다. 객체가 더이상 필요하지 않으면 파이썬 메모리 관리자가 자동으로 객체에서 메모리를 회수하는 방식을 사용하므로, 파이썬은 동적 메모리 할당 방식을 사용한다고 말할 수 있다.

`heap`은 동적할당을 구현하는데 사용된다. 힙을 사용하여 동적으로 메모리를 관리하면 필요하지 않은 메모리를 비우고 재사용할 수 있다는 장점이 있다.



# What is `namespace` in Python?

naming conflict를 피하기 위해 이름이 고유한지 확인하는데 사용되는 이름 지정 시스템이다.

`namespace`는 프로그래밍 언어에서 특정한 객체를 이름에 따라 구분할 수 있는 범위를 의미한다.

파이썬 내부의 모든것은 객체로 구성되며 이들 각각은 특정 이름과의 매핑 관계를 갖게 되는데 이 매핑을 포함하고 있는 공간을 `namespace` 라고 한다.



# What is PYTHONPATH?

모듈을 import할 때 사용되는 환경변수이다.

import할 때마다 pythonpath를 조회하여 가져온 모듈이 디렉토리에 있는지 확인한다. 인터프리터는 이를 사용하여 로드할 모듈을 가져온다.



환경변수에 경로를 추가함녀, 파이썬은 이 경로들을 `sys.path`에 추가한다.

이를 통해 파이썬 코드 내부에서뿐만아니라 파이썬 코드 밖에서도 `sys.path`를 조작할 수 있다.



# What are python modules? Name some commonly used built-in modules in Python?

파이썬 코드를 포함하는 파일로써, 함수나 변수 또는 클래스를 모아놓은 파일이다. 다른 파이썬 프로그램에서 불러와 사용할 수 있게끔 만든 파이썬 파일이라고도 할 수 있다.

실행 가능한 코드를 퐇마하는 파이썬 확장자로 만든 파이썬 파일은 모두 모듈이다.



# Is python case sensitive?

파이썬은 대소문자를 구분하는 언어이다.



# What is type conversion in Python?

type conversion은 타입 캐스팅과 동일한 의미를 가지며, 이는 어떤 데이터 타입을 다른 데이터 타입으로 변환하는것을 말한다.

ex) `int()`,`float()` ...



# What is self in Python?

`self`는 인스턴스 메서드의 첫번쨰 인자이다.

메서드가 호출될 때, 파이썬은 `self`에 인스턴스를 넣고 이 인스턴스를 참조하여 인스턴스 메서드를 실행할 수 있게된다.



# What’s the difference between iterator and iterable?

iterable 객체는 `iter` 함수에 인자로 전달 가능한 반복 가능한 객체를 말한다.

iterable 객체를 `iter` 함수의 인자로 넣음녀 iterable 객체를 순회할 수 있는 객체를 반환하는데, 이것이 iterator객체이다.



# What is pickling and unpickling?

- 직렬화(Serialization) : 객체를 바이트 스트림으로 변환하여 디스크에 저장하거나 네트워크로 보낼 수 있도록 만들어주는 것을 말한다.
- 역 직렬화(Deserialization) : 바이트 스트림을 파이썬 객체로 변환하는 것



pickle 모듈은 파이썬 객체의 직렬화와 역 직렬화를 수행하는 모듈이다.  이 때 파이썬 객체를 직렬화할 때를 `pickling`이라고 하며, 바이트 스트림을 역 직렬화를 할 때를 `unpickling`이라고 한다.



# What are the generators in python?

iterator 객체를 간단히 만들 수 있는 함수를 말한다. 제너레이터는 다음과 같이 만들 수 있다.

```python
def generator_list(value):
    for i in range(value):
        # 값을 반환하고 여기를 기억
        yield i
```

```python
value = 2
gen = (i for i in range(value))
print(next(gen))    # 0
print(next(gen))    # 1
print(next(gen))    # StopIteration 에러 발생
```



# Whenever Python exits, why isn’t all the memory de-allocated?

다른 객체나 전역 네임스페이스에서 참조되는 객체를 순환 참조하는 파이썬 모듈은 항상 해제되지는 않는다.

또한 C 라이브러리가 예약한 메모리의 해당 부분을 해제하는 것은 불가능하다. 

그러므로 파이썬 종료 시 모든 메모리가 해제되지 않는다.



# What does this mean: `*args`, `**kwargs`? And why would we use it?

`*args`는 함수에 전달되는 argument의 수를 알 수 없거나, list나 tuple의 argument들을 함수에 전달하고자 할때 사용한다.

파이썬에서는 어디서부터 어디까지 `*arge`에 담아야 할지 알수 없기 때문에, 일반 변수를 앞에 두고 그 뒤에 `*args`를 지정해주어야 한다.



`**kwargs`는 함수에 전달되는 keyword argument의 수를 모르거나, dictionary의 keyword argument들을 함수에 전달하고자 할 때 사용한다.



# What is the process of compilation and linking in python?

파이썬 파일을 실행하면, 소스 코드는 바이트 코드로 변환되며, `.pyc`,`.pyo` 형식으로 저장된다.

이 떄 소스 코드를 바이트 코드로 변환하는 과정을 컴파일 단계라고 한다.

파이썬 가상머신이 바이트 코드를 기계어로 변환하여 어떤 운영체제든 실행할 수 있도록 한다.

이 떄 우리의 코드와 인터프리터가 필요한 라이브러리를 연결시키는 과정이 있는데, 이를 링크 단계라고 한다.



# What is Polymorphism in Python?

다형성 : 객체지향의 주요 개념으로 여러가지 형태를 가질 수 있는 능력



# How do you do data abstraction in Python?

데이터 추상화는 객체지향의 주요 개념으로 사용자에게 데이터의 주요 정보만 제공하여 구체적인 구현은 몰라도 사용할 수 있게 만드는 방법이다.



# What are Decorators in Python?

함수를 인자로 받고 내부 함수에서 인자로 받은 함수를 사용하는 클래스나 함수가 있을 때, 인자로 사용할 함수를 간편하게 지정해주는 역할을 하는 것이 `Decorator`이다

```python
import time


def make_time_checker(func):
      def new_func(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            print('실행시간:', end_time - start_time)
            return result
      return new_func

@make_time_checker
def big_number(n):
      return n ** n ** n

@make_time_checker
def big_number2(n):
      return (n+1) ** (n+1) ** (n+1)
```





# Does python make use of access specifiers?

변수명을 통해 접근 제어

_ : protected

__ : private

접미사 __ or 없음 : public



# What is object interning?

파이썬에는 모든 것들이 객체이므로 변수들은 값을 바로 가지지 않고 값을 가진 주소를 참조하게된다.

`object interning`은 재활용될 object에 대해 매번 새로운 주소를 할당하는 것은 비효율적이므로, 하나의 주소에 값을 주고 그 주소를 재활용하는 작업을 말한다.

