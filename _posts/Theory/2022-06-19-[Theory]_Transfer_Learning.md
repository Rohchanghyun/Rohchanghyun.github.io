---
layout : single
title:  "[Theory] Transfer Learning"
excerpt: "Transfer Learning"

categories:
  - Theory
tags:
  - Theory
  - Transfer Learning


toc: true
toc_sticky: true

author_profile: true
sidebar_main: true

date: 2022-06-19
last_modified_at: 2022-06-19
---

# Transfer Learning

## model.save()

- 학습의 결과를 저장
- 모델 architecture 과 parameter 저장
- 모델 학습 중간 결과를 저장해 최선의 결과 모델 선택

`torch.save(object,path)` : 전체 모델을 저장하고나 state_dict 를 저장할 때 사용

- object : 저장할 모델 객체
- path: 저장할 위치 + 파일명



`torch.load(path)` : 전체 모델을 불러오거나, 모델의 state_dict 를 불러올 떄 사용

- path : 불러올 위치 + 파일명



`torch.nn.Module.loadstatedict(dict)` : state_dict 를 이용하여 모델 객체 내의 매개 변수 값을 초기화

- dict: 불러올 매개 변수 값들이 담겨있는 sate_dict 객체



```python
# Print model's state_dict
print("Model's state_dict:") 
for param_tensor in model.state_dict():
	print(param_tensor,"\t", model.state_dict()[param_tensor].size())
    # state_dict: 모델의 파라미터 표시
    
torch.save(model.state_dict(),os.path.join(MODEL_PATH, "model.pt"))# 모델의 파라미터 저장

new_model = TheModelClass()
new_model.load_state_dict(torch.load(os.path.join(MODEL_PATH, "model.pt")))
# 모델에서 parameter load

torch.save(model, os.path.join(MODEL_PATH, "model.pt"))# 모델 저장
model = torch.load(os.path.join(MODEL_PATH, "model.pt"))# 모델 load

```



## checkpoints

- 학습의 중간 결과를 저장하여 최선의 결과를 선택
- earlystopping 기법 사용시 이전 학습의 결과물을 저장
- loss 와 metric 값을 지속적으로 확인해야함
- 일반적으로 epoch,loss,metric 을 함께 저장하여 확인



```python
torch.save({
	'epoch': e,
	'model_state_dict': model.state_dict(),
	'optimizer_state_dict': optimizer.state_dict(),
	'loss':epoch_loss,},
f"saved/checkpoint_model_{e}_{epoch_loss/len(dataloader)}_{epoch_acc/len(dataloader)}.pt")
# 모델의 정보를 epoch과 함께 저장

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```



## Transfer Learning

- 다른 데이터셋으로 만든 모델을 현재 데이터에 적용
- 일반적으로 대용량 데이터셋으로 만들어진 모델의 성능이 더 높다
- backbone architecture가 잘 학습된 모델에서 일부분만 변경하여 학습 수행

- TorchVision -> 다양한 기본 모델 제공



## Freezing

- pretrained model 을 활용 시 모델의 일부분을 frozen 시킴(학습하지 않음)
- required_grad 를 통해 적용

```python
vgg = models.vgg16(pretrained=True).to(device) #vgg 16 모델을 vgg에 할당하기
class MyNewNet(nn.Module):
	def __init__(self):
        super(MyNewNet, self).__init__()
        self.vgg19 = models.vgg19(pretrained=True)
        self.linear_layers = nn.Linear(1000, 1)# 모델의 마지막 Linear layer 추가(num_class 변경하여 원하는 task 에 맞도록)
        
# Defining the forward pass
	def forward(self, x):
		x = self.vgg19(x)
		return self.linear_layers(x)

# 마지막 layer 를 제외하고 frozen
for param in my_model.parameters():
param.requires_grad = False
for param in my_model.linear_layers.parameters():
param.requires_grad = True

```

