---

layout : single

title: "optical flow 관련 지식"

excerpt: "optical flow 관련 지식 공부"



categories:

- Theory

tags:

- optical flow




toc: true

toc_sticky: true



author_profile: true

sidebar_main: true



date: 2022-12-20

last_modified_at: 2022-12-20

---

  

## 연속 영상

- 일반적으로 연속된 frame이 들어올 때, 영상은 영상 일관서(coherence) 라는 성질을 가진다.

    -> 주변 픽셀도 유사한 성질을 가질 수 있다

  

- 다른 성질중 하나는 시간 일관성
    - t 순간의 픽셀값 f(y,x,t) 는 다음 순간 f(y,x,t+1) 과 비슷할 가능성이 높다

  

- 가장 직관적인 방법(두개의 frame의 픽셀 값차이 비교) -> 간단하지만 한계점이 명확
    - 위치가 고정된 카메라에서만 사용 가능 
    - 배경과 물체의 색상이나 명암에 큰 변화가 있으면 사용 불가

  

이를 개선하기 위해 실제 영상에서 움직이는 물체를 찾을 필요 있음

  

## motion field(optical field)

- 움직임이 발생한 모든 점의 모션 벡터로 얻어낸 2차원 모션맵

  

어려운 점

- 구체 회전 : texture 가 단조로우면 모션 벡터가 발생함에도 불구하고 영상에는 아무런 변화가 발생하지 않음
- 광원 회전 : 구가 회전하지 않았지만 광원이 움직이는 경우에는 모션벡터가 0이어야 하지만 영상에는 변화가 발생

  

## temporal feature

- 시간의 흐름에 따라 변하는 특징

  

## optical flow

- optical field 를 구하기 위하여 이전 프레임과 현재 프레임의 차이를 이용하고 픽셀값과 주변 픽셀들과의 관계를 통해 각 픽셀의 이동을 계산하여 추출
- optical flow 의 문제를 정의하면 각 픽셀별로 I(y,x,y) -> I(y,x,t+1) 를 어떻게 추정하는지 구하는 문제
- 핵심은 t frame의 어떤 픽셀이 t+1 frame에대응이 되는지 알아야 motion vector를 구할 수 있다.
- 이는 카메라가 고정이 되지 않은 상황에서도 적용할 수 있어야 한다.

  

- 카메라가 고정되어있는 상황 -> optical flow에 등장하는 움직임은 모두 물체의 움직임
- 물체가 고정된 상황 -> optical flow 를 카메라 움직임으로 해석

  

이를 추정하기 위해 가정을 세운다

- color/brightness constancy : 어떤 픽셀과 그 픽셀의 주변 픽셀의 색/밝기는 같음을 가정으로 세운다 -> 조명의 변화가 없어야 한다
- small motion : frame간 움직임이 작아 어떤 픽셀 점은 멀리 움직이지 않는것은 가정으로 세운다