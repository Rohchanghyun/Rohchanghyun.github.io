---

layout : single

title: "[Paper] DASO : Distribution-aware Senmantics Oriented Pseudo-label for Imbalanced Semi-Supervised Learning"

excerpt: "DASO: Distribution-aware Senmantics Oriented Pseudo-label for Imbalanced Semi-Supervised Learning 논문 리뷰"



categories:

- Paper

tags:

- Semi-supervised



toc: true

toc_sticky: true



author_profile: true

sidebar_main: true



date: 2022-10-26

last_modified_at: 2022-10-26

---

  

##  Abstract  
    - 기존의 semi-supervised learning은 클래스 불균형 및 레이블이 지정된 데이터와 지정되지 않은 데이터 간의 클래스 분포 불일치로 인해 실제 적용과는 거리가 멀었다.
        - 두 유형의 유사 레이블이 모두 편향 측면에서 보완적 특성을 가진다는 관찰을 한 후, 유사성 기반 분류기에서 선형 분류기로 의미론적 유사 레이블을 클래스 적응적으로 혼합하는 일반적은 유사 레이블링 프레임워크를 제안
        - classifier에서 편향된 예측을 줄이기 위해 균형 잡힌 특징 표현을 설정하기 위해 새로운 의미 정렬 손실을 도입
    - 이 전체 프레임 워크를 DASO 유사 레이블 이라고 한다.
##  Introduction  
    - semi-supervised learning은 레이블이 지정되지 않은 데이터를 활용하여 레이블이 지정된 데이터를 구성하는 비용을 줄인다.
    - 일반적으로 모델의 예측을 기반으로 레이블이 지정되지 않은 데이터에 대한 유사 레이블을 생성하고 모델을 정규화 하는데 사용
    - 하지만 종종 분포가 편향될 때가 있다.
    - 유사 레이블의 편향은 레이블이 지정된 데이터와 지정되지 않은 데이터 사이의 클래스 분포 불일치에도 의존하며 레이블이 지정되지 않은 데이터에 대한 부정확한 추정치 또는 잘못된 가정을 사용하는 것은 불균형 SSL에서 도움이 될 수 없다.
    - 논문에서는 레이블이 없는 데이터의 클래스 분포가 레이블의 분포와 같다는 일반적인 가정을 버리는 동시에 클래스 불균형 데이터에서 유사 레이블의 편향을 완화하기 위해 특별히 조정된 새로운 불균형 SSL방법을 제시
        - 유사성 기반 분류기에서 얻은 의미론적 유사 레이블이 헤드 클래스에 편향된 것과 반대로 소수 클래스에 편향되어 있음을 관찰
        - 새로운 의사 레이블링 체계를 개발하기 위해 두 가지 다른 유형의 의사 레이블링의 보완적 특성에서 핵심 영감을 얻는다.
        - DASO라는 불균형 SSL 프레임워크 소개
            - 기존 SSL학습자를 기반으로, 전체 편향을 줄이기 위해 각 클래스에 대해 서로 다른 비율로 선형 및 유사 레이블을 혼합할 것을 제안.

    - 주요 효과
        - 의사 레이블의 현재 클래스 분포를 관찰하는 두가지 보완적 유형의 의사 레이블을 클래스 적응적으로 혼합하여 의사 레이블을 제거하기 위한 새로운 의사 레이블링 프레임워크인 DASO를 제안
        - DASO는 레이블링 되지 않은 각 예제를 유사한 프로토타입에 정렬하여 고품질 기능 표현의 편향을 더욱 완화하기 위해 의미 정렬 손실을 도입
        - 다양한 불균형 SSL 설정에서 상당한 성능 향상을 보여주기 위해 다른 프레임워크와 쉽게 통합된다.

## Proposed Method  
###  Preliminaries  
        - Problem setup
        - Imbalanced semi-supervised learning
###  Motivation  
        - Linear and semantic pseudo-label
            - 선형 분류기로 생성한 유사 레이블 -> 선형 유사 레이블
        - Trade-offs bet