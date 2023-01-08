---

layout : single

title:  "[Theory] torch.no_grad() 와 model.eval() 의 차이점"

excerpt: "torch.no_grad() 와 model.eval() 의 차이점 비교"



categories:

 - Theory

tags:

 - Pytorch
 - torch.no_grad()



toc: true

toc_sticky: true



author_profile: true

sidebar_main: true



date: 2022-05-22

last_modified_at: 2022-05-22

---

# torch.no_grad() 와 model.eval() 의 차이점

## with torch.no_grad():

`no_grad()` 를 사용하게 되면 Pytorch가 autograd engine 을 꺼버려 더이상 gradient를 추적하지 않는다. 

일반적으로 inference 시에 `with torch.no_grad():` 를 사용한다



## model.eval()

모델링 시에 training 과 inference 시에 다르게 작용하는 layer 들이 있다. Dropout layer 과 BatchNorm layer 가 여기에 포함되는데, 두 layer 모두 학습 시에는 동작하지만 inference 시에는 동작하지 않는다. 

이러한 layer 들의 동작을 inference mode 로 바꿔주는 목적으로 사용한다.



때문에 모델의 product serving 시에는 이 두가지를 모두 사용해야 한다