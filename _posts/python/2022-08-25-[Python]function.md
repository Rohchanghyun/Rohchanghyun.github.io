---
layout : single
title:  "[Python]Function 공부"
excerpt: "Python function 관련 공부"

categories:
  - Python
tags:
  - function
  - Python

toc: true
toc_sticky: true

author_profile: true
sidebar_main: true

date: 2022-08-25
last_modified_at: 2022-08-25
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


# Python function


## 함수 생성


- 예약어 def를 먼저 쓰고 함수명과 매개변수를 작성하고 콜론을 붙여서 완성한다

- 함수의 기능을 작성하고 결과를 return과 함께 반환한다

- 아무런 기능이 없어도 블록 문장을 구성하도록 pass를 사용한다






```python
def func():
    pass
```


```python
f = func()
print(f)
```

<pre>
None
</pre>
- 함수 객체가 만들어지면 함수 이름은 속성 `__name__` 에 저장



```python
func.__name__
```

<pre>
'func'
</pre>

```python
def return_():
    return "return"
```


```python
return_()
```

<pre>
'return'
</pre>

```python
def func_():
    return 1,2,3
```


```python
f_ = func_()
f_,type(f_)
```

<pre>
((1, 2, 3), tuple)
</pre>
## doc


- 도움말 `help`도 하나의 class이다

- 하나의 인자를 전달하면 그 인자가 관리하는 문서화 속성 `__doc__`를 조회하여 출력

- 함수의 이름과 문서화 정보를 같이 보여준다



```python
help(int)
```

<pre>
Help on class int in module builtins:

class int(object)
 |  int([x]) -> integer
 |  int(x, base=10) -> integer
 |  
 |  Convert a number or string to an integer, or return 0 if no arguments
 |  are given.  If x is a number, return x.__int__().  For floating point
 |  numbers, this truncates towards zero.
 |  
 |  If x is not a number or if base is given, then x must be a string,
 |  bytes, or bytearray instance representing an integer literal in the
 |  given base.  The literal can be preceded by '+' or '-' and be surrounded
 |  by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
 |  Base 0 means to interpret the base from the string as an integer literal.
 |  >>> int('0b100', base=0)
 |  4
 |  
 |  Built-in subclasses:
 |      bool
 |  
 |  Methods defined here:
 |  
 |  __abs__(self, /)
 |      abs(self)
 |  
 |  __add__(self, value, /)
 |      Return self+value.
 |  
 |  __and__(self, value, /)
 |      Return self&value.
 |  
 |  __bool__(self, /)
 |      self != 0
 |  
 |  __ceil__(...)
 |      Ceiling of an Integral returns itself.
 |  
 |  __divmod__(self, value, /)
 |      Return divmod(self, value).
 |  
 |  __eq__(self, value, /)
 |      Return self==value.
 |  
 |  __float__(self, /)
 |      float(self)
 |  
 |  __floor__(...)
 |      Flooring an Integral returns itself.
 |  
 |  __floordiv__(self, value, /)
 |      Return self//value.
 |  
 |  __format__(self, format_spec, /)
 |      Default object formatter.
 |  
 |  __ge__(self, value, /)
 |      Return self>=value.
 |  
 |  __getattribute__(self, name, /)
 |      Return getattr(self, name).
 |  
 |  __getnewargs__(self, /)
 |  
 |  __gt__(self, value, /)
 |      Return self>value.
 |  
 |  __hash__(self, /)
 |      Return hash(self).
 |  
 |  __index__(self, /)
 |      Return self converted to an integer, if self is suitable for use as an index into a list.
 |  
 |  __int__(self, /)
 |      int(self)
 |  
 |  __invert__(self, /)
 |      ~self
 |  
 |  __le__(self, value, /)
 |      Return self<=value.
 |  
 |  __lshift__(self, value, /)
 |      Return self<<value.
 |  
 |  __lt__(self, value, /)
 |      Return self<value.
 |  
 |  __mod__(self, value, /)
 |      Return self%value.
 |  
 |  __mul__(self, value, /)
 |      Return self*value.
 |  
 |  __ne__(self, value, /)
 |      Return self!=value.
 |  
 |  __neg__(self, /)
 |      -self
 |  
 |  __or__(self, value, /)
 |      Return self|value.
 |  
 |  __pos__(self, /)
 |      +self
 |  
 |  __pow__(self, value, mod=None, /)
 |      Return pow(self, value, mod).
 |  
 |  __radd__(self, value, /)
 |      Return value+self.
 |  
 |  __rand__(self, value, /)
 |      Return value&self.
 |  
 |  __rdivmod__(self, value, /)
 |      Return divmod(value, self).
 |  
 |  __repr__(self, /)
 |      Return repr(self).
 |  
 |  __rfloordiv__(self, value, /)
 |      Return value//self.
 |  
 |  __rlshift__(self, value, /)
 |      Return value<<self.
 |  
 |  __rmod__(self, value, /)
 |      Return value%self.
 |  
 |  __rmul__(self, value, /)
 |      Return value*self.
 |  
 |  __ror__(self, value, /)
 |      Return value|self.
 |  
 |  __round__(...)
 |      Rounding an Integral returns itself.
 |      Rounding with an ndigits argument also returns an integer.
 |  
 |  __rpow__(self, value, mod=None, /)
 |      Return pow(value, self, mod).
 |  
 |  __rrshift__(self, value, /)
 |      Return value>>self.
 |  
 |  __rshift__(self, value, /)
 |      Return self>>value.
 |  
 |  __rsub__(self, value, /)
 |      Return value-self.
 |  
 |  __rtruediv__(self, value, /)
 |      Return value/self.
 |  
 |  __rxor__(self, value, /)
 |      Return value^self.
 |  
 |  __sizeof__(self, /)
 |      Returns size in memory, in bytes.
 |  
 |  __sub__(self, value, /)
 |      Return self-value.
 |  
 |  __truediv__(self, value, /)
 |      Return self/value.
 |  
 |  __trunc__(...)
 |      Truncating an Integral returns itself.
 |  
 |  __xor__(self, value, /)
 |      Return self^value.
 |  
 |  as_integer_ratio(self, /)
 |      Return integer ratio.
 |      
 |      Return a pair of integers, whose ratio is exactly equal to the original int
 |      and with a positive denominator.
 |      
 |      >>> (10).as_integer_ratio()
 |      (10, 1)
 |      >>> (-10).as_integer_ratio()
 |      (-10, 1)
 |      >>> (0).as_integer_ratio()
 |      (0, 1)
 |  
 |  bit_length(self, /)
 |      Number of bits necessary to represent self in binary.
 |      
 |      >>> bin(37)
 |      '0b100101'
 |      >>> (37).bit_length()
 |      6
 |  
 |  conjugate(...)
 |      Returns self, the complex conjugate of any int.
 |  
 |  to_bytes(self, /, length, byteorder, *, signed=False)
 |      Return an array of bytes representing an integer.
 |      
 |      length
 |        Length of bytes object to use.  An OverflowError is raised if the
 |        integer is not representable with the given number of bytes.
 |      byteorder
 |        The byte order used to represent the integer.  If byteorder is 'big',
 |        the most significant byte is at the beginning of the byte array.  If
 |        byteorder is 'little', the most significant byte is at the end of the
 |        byte array.  To request the native byte order of the host system, use
 |        `sys.byteorder' as the byte order value.
 |      signed
 |        Determines whether two's complement is used to represent the integer.
 |        If signed is False and a negative integer is given, an OverflowError
 |        is raised.
 |  
 |  ----------------------------------------------------------------------
 |  Class methods defined here:
 |  
 |  from_bytes(bytes, byteorder, *, signed=False) from builtins.type
 |      Return the integer represented by the given array of bytes.
 |      
 |      bytes
 |        Holds the array of bytes to convert.  The argument must either
 |        support the buffer protocol or be an iterable object producing bytes.
 |        Bytes and bytearray are examples of built-in objects that support the
 |        buffer protocol.
 |      byteorder
 |        The byte order used to represent the integer.  If byteorder is 'big',
 |        the most significant byte is at the beginning of the byte array.  If
 |        byteorder is 'little', the most significant byte is at the end of the
 |        byte array.  To request the native byte order of the host system, use
 |        `sys.byteorder' as the byte order value.
 |      signed
 |        Indicates whether two's complement is used to represent the integer.
 |  
 |  ----------------------------------------------------------------------
 |  Static methods defined here:
 |  
 |  __new__(*args, **kwargs) from builtins.type
 |      Create and return a new object.  See help(type) for accurate signature.
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |  
 |  denominator
 |      the denominator of a rational number in lowest terms
 |  
 |  imag
 |      the imaginary part of a complex number
 |  
 |  numerator
 |      the numerator of a rational number in lowest terms
 |  
 |  real
 |      the real part of a complex number

</pre>

```python
def add_(x,y):
    '''
    this function returns sum of x + y
    '''
    return x+y
```


```python
help(add_)
```

<pre>
Help on function add_ in module __main__:

add_(x, y)
    this function returns sum of x + y

</pre>
## 함수의 정의


- 함수도 1급 객체이기 때문에 변수,매개변수,반환값 등에 정의해서 사용할 수 있다



```python
def func(x,y):
    return x + y

def higher_order(func,*args):
    return func(*args)
```


```python
func
```

<pre>
<function __main__.func(x, y)>
</pre>
- 함수 클래스 이름 다음 함수와 매개변수를 출력



```python
higher_order(func,1,3)
```

<pre>
4
</pre>
- 함수를 호출할 때 매개변수 앞에 *를 붙인 이유는 args에 튜플이 보관되어 있어 이를 다시 각 원소로 분리해서 인자로 전달하라는 의미


## `__get__`


- 내부에 정의된 메소드 객체를 확인할 수 있다

- 함수의 이름으로 조회하면 함수 객체를 자동으로 가져오기 위해 이 `__get__` 메소드가 자동으로 실행된다



```python
def add(x,y):
    return x+y

add.__get__
```

<pre>
<method-wrapper '__get__' of function object at 0x7fab7cc03280>
</pre>
- 이 스페셜 메소드에 정수를 넣으면 메소드로 변환



```python
add.__get__(1)
```

<pre>
<bound method add of 1>
</pre>

```python
b = add.__get__(1)
```


```python
b(2)
```

<pre>
3
</pre>
## 함수 입력 데이터 처리


### 고정 매개변수 초기값 지정



```python
def func_d(x = 1,y = 1,z = 1):
    print('locals',locals())
    return x + y + z
```


```python
func_d()
```

<pre>
locals {'x': 1, 'y': 1, 'z': 1}
</pre>
<pre>
3
</pre>
### 초기값 속성 확인



```python
func_d.__defaults__
```

<pre>
(1, 1, 1)
</pre>
- 가변인자 할당

- 관행적으로 키워드 인자를 받는 매개변수의 이름으로는 kwargs를 사용한다

- 가변 키워드 인자를 받을 시에는 *를 2개 붙인다



```python
def func_a(*args):
    print('locals',locals())
    return args
```


```python
def func_k(**kwargs):
    print('locals',locals())
    result = 0
    for k,v in kwargs.items():
        result += v
    return result
```


```python
func_k(x = 1,y = 2,z = 3)
```

<pre>
locals {'kwargs': {'x': 1, 'y': 2, 'z': 3}}
</pre>
<pre>
6
</pre>