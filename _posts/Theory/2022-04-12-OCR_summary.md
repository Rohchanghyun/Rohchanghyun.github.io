---
layout : single
title:  "[Theory]OCR_summary "
excerpt: "OCR 개요"

categories:
  - Theory
tags:
  - OCR

toc: true
toc_sticky: true

author_profile: true
sidebar_main: true

date: 2022-04-12
last_modified_at: 2022-04-12
---


# <span style="color: #f0b752">OCR Technology and Services</span>

## <span style="color: #a6acec">OCR Technology</span>

OCR : Optical Character Recognition(글자 인식 시스템)

### <span style="color: #b1cf89">사람이 글자를 읽을 때</span>

- 글자를 찾고
- 찾은 글자를 인식



### <span style="color: #b1cf89">OCR Process</span>

- 글자 영역 찾기 
- 영역 내 글자 인식



### <span style="color: #b1cf89">Offline handwriting VS Online handwriting</span>

<p align="center"><img src="/assets/images/ocr/offline.png"></p>

- offline handwriting 
	- 입력 : 이미지

- online handwriting
	- 입력 : 좌표 시퀀스



### <span style="color: #b1cf89">Object Detection VS Text Detection</span>

#### <span style="color: #88c8ff">Object detection</span>

<p align="center"><img src="/assets/images/ocr/single.png"></p>

- 단일 객체 검출 : 이미지 내에 객체가 하나 있다고 가정하고 객체의 위치를 찾아낸다

<p align="center"><img src="/assets/images/ocr/double.png"></p>

- 다수 객체 검출 : 이미지 내에 여러개의 객체의 위치 찾아낸다

#### <span style="color: #88c8ff">Text Detection</span>

<p align="center"><img src="/assets/images/ocr/text.png"></p>

- 글자 영역 다수 검출 

영역이 있으면 class 는 글자이기 때문에 class 가 무엇인지는 구분할 필요 없다 -> 단일 클래스



### <span style="color: #b1cf89">글자 검출기</span>

이미지가 입력으로 들어왔을 떄 글자 영역 위치를 출력하는 모델

<p align="center"><img src="/assets/images/ocr/model.png"></p>

객체 검출과의 차이점

<p align="center"><img src="/assets/images/ocr/reason1.png"></p>

- 영역의 종횡비
	- 글자이기 떄문에 글자의 특성상 가로 or 세로가 긴 영역이 많다

<p align="center"><img src="/assets/images/ocr/reason2.png"></p>

- 객체 밀도 
	- 밀도가 높다

#### <span style="color: #88c8ff">detector</span>

글자 영역 감지

#### <span style="color: #88c8ff">recognizer</span>

영역 내에서 글자를 인식한다

#### <span style="color: #88c8ff">serializer </span>

recognizer 결과를 사람이 읽는 순서로 정렬해 최종 문자열을 만든다

<p align="center"><img src="/assets/images/ocr/serial.png"></p>

이 serializer 를 사용하면 모듈 뒤에 자연어 처리 모듈을 쉽게 사용할 수 있다

#### <span style="color: #88c8ff">text parser</span>

정의된 key 들에 대한 value 추출

ex) 회사명, 이름, 전화번호 등등

- BIO 태깅

	자연어 처리에서 많이 사용하는 기술

	<p align="center"><img src="/assets/images/ocr/parser.png"></p>

	- 입력을 글자단위로 쪼개어 토큰화

	- BIO  태깅 : Begin / Inside / Outside

		​	Begin : 객체명의 시작

		​	Inside : 객체의 중간에 해당되는 토큰

		​	Outside : 정의된 객체와 상관없는 토큰

	- 후처리



## <span style="color: #a6acec">OCR Services</span>

### <span style="color: #b1cf89">text extractor</span>

- copy and paste

<p align="center"><img src="/assets/images/ocr/copy1.png"></p>



### <span style="color: #b1cf89">text extractor + NLP</span>

- google photo( + search )

<p align="center"><img src="/assets/images/ocr/search.png"></p>

- 번역
- 금칙어 처리



### <span style="color: #b1cf89">key-value extractor</span>

- 신용카드
- 신분증
- 명함(수기 입력 대체)