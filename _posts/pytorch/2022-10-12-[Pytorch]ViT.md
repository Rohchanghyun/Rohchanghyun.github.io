---
layout : single
title:  "[Pytorch] ViT code 공부"
excerpt: "ViT code 보며 구현"

categories:
  - pytorch
tags:
  - pytorch
  - transformer

toc: true
toc_sticky: true

author_profile: true
sidebar_main: true

date: 2022-10-12
last_modified_at: 2022-10-12

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


ViT



```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary
```

# Data size


batch size = 8 <br/>

image = (3,224,224)



```python
x = torch.randn(8,3,224,224)
x.shape
```

<pre>
torch.Size([8, 3, 224, 224])
</pre>
# Patch Embedding


Batch * C * H * W -> Batch * N * (P * P * C)로 임베딩



```python
patch_size = 16
print('x : ',x.shape)
patches_shape = x.reshape((8,-1,16*16*3))
patches = rearrange(x, 'b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size)
print('patch : ',patches.shape)

print(torch.eq(patches_shape,patches))

```

<pre>
x :  torch.Size([8, 3, 224, 224])
patch :  torch.Size([8, 196, 768])
tensor([[[ True, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         ...,
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False,  True]],

        [[ True, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         ...,
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False,  True]],
    
        [[ True, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         ...,
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False,  True]],
    
        ...,
    
        [[ True, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         ...,
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False,  True]],
    
        [[ True, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         ...,
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False,  True]],
    
        [[ True, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         ...,
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False,  True]]])
</pre>
embedding 과정을 rearrange 대신 reshape로 진행하려 했는데, 두 함수의 결과 tensor를 비교해보니 결과가 다르다. 




```python
shape_test = torch.tensor([[1,2,3],
                           [4,5,6]])
print('original_shape : ',shape_test.shape)

tensor_reshape = shape_test.reshape(3,2)
tensor_rearrange = rearrange(shape_test,'a b -> b a ')

print('reshape_tensor : ',tensor_reshape)
print('rearrange_tensor : ',tensor_rearrange)
```

<pre>
original_shape :  torch.Size([2, 3])
reshape_tensor :  tensor([[1, 2],
        [3, 4],
        [5, 6]])
rearrange_tensor :  tensor([[1, 4],
        [2, 5],
        [3, 6]])
</pre>
이 2,3의 tensor를 각각 reshape와 rearrange로 transpose한 결과를 보면, 두 tensor는 사이즈는 같지만 tensor의 정렬이 다르다<br/>

reshape는 원하는 사이즈에 맞춰 데이터를 원래의 형태에 관계 없이 순서대로 채우지만, rearrange는 데이터의 정렬 방법이 다르다.


ViT에서는 einops 대신 kernel_size와 stride_size를 patch_size로 갖는 Convolutional 2D layer를 사용한 후 flatten 해준다.<br/>

이 방법을 사용할 시 성능이 향상되었다고 한다.



```python
patch_size = 16
emb_size = 768
in_channels = 3

projection = nn.Sequential(
    nn.Conv2d(in_channels,emb_size,kernel_size = patch_size,stride = patch_size),
    Rearrange('b e (h) (w) -> b (h w) e'))

projection(x).shape
```

<pre>
torch.Size([8, 196, 768])
</pre>

```python
img_size = 224

projected_x = projection(x)
print('Projected shape : ',projected_x.shape)

cls_token = nn.Parameter(torch.randn(1,1,emb_size))
positions = nn.Parameter(torch.randn((img_size//patch_size) ** 2 + 1,emb_size))
print('Cls Shape : ',cls_token.shape,', Pos shape : ',positions.shape)

batch_size = 8
cls_tokens = repeat(cls_token,'() n e -> b n e',b = batch_size)
print('Repeated Cls shape : ',cls_tokens.shape)

cat_x = torch.cat([cls_tokens,projected_x],dim = 1)

cat_x += positions
print('output : ',cat_x.shape)

```

<pre>
Projected shape :  torch.Size([8, 196, 768])
Cls Shape :  torch.Size([1, 1, 768]) , Pos shape :  torch.Size([197, 768])
Repeated Cls shape :  torch.Size([8, 1, 768])
output :  torch.Size([8, 197, 768])
</pre>
torch.cat 으로 두 tensor를 concat



```python
a = torch.zeros(2,3,3)
a
```

<pre>
tensor([[[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]],

        [[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]]])
</pre>

```python
b = torch.ones(2,1,3)
b
```

<pre>
tensor([[[1., 1., 1.]],

        [[1., 1., 1.]]])
</pre>

```python
cat = torch.cat([b,a],dim = 1)
print(cat)
print('shape : ',cat.shape)
```

<pre>
tensor([[[1., 1., 1.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]],

        [[1., 1., 1.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]]])
shape :  torch.Size([2, 4, 3])
</pre>
하나의 클래스로 구현한 결과



```python
class PatchEmbedding(nn.Module): 
    def __init__(self,in_channels: int = 3,patch_size :int = 16,emb_size : int = 768,img_size : int = 224):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels,emb_size,kernel_size = patch_size,stride = patch_size),
            Rearrange('b e (h) (w) -> b (h w) e')
        )
        self.cls_token = nn.Parameter(torch.randn(1,1,emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1,emb_size))
        

    def forward(self,x: Tensor) -> Tensor:
        b,_,_,_ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token,'() n e -> b n e',b = b)
        x = torch.cat([cls_tokens,x],dim = 1)
        x += self.positions

        return x

```

# Multi-head Attention


ViT 에서는 q,k,v가 같은 tensor로 입력된다.<br/>

3개의 linear projection을 통해 embedding 된 후 각각 scaled dot-product attention 진행


## Linear Projection



```python
emb_size = 768
num_heads = 8
keys = nn.Linear(emb_size,emb_size)
queries = nn.Linear(emb_size,emb_size)
values = nn.Linear(emb_size,emb_size)
print(keys, queries, values)
```

<pre>
Linear(in_features=768, out_features=768, bias=True) Linear(in_features=768, out_features=768, bias=True) Linear(in_features=768, out_features=768, bias=True)
</pre>
## Multi-Head



```python
query = rearrange(queries(cat_x),'b n (h d) -> b h n d',h = num_heads)
key = rearrange(keys(cat_x),'b n (h d) -> b h n d',h = num_heads)
value = rearrange(values(cat_x),'b n (h d) -> b h n d',h = num_heads)

print('shape : ',query.shape,key.shape,value.shape)
```

<pre>
shape :  torch.Size([8, 8, 197, 96]) torch.Size([8, 8, 197, 96]) torch.Size([8, 8, 197, 96])
</pre>
# Scaled Dot Product Attention



```python
energy = torch.einsum('bhqd, bhkd -> bhqk',query,key)
print('energy : ',energy.shape)

scaling = emb_size ** (1/2)
att = F.softmax(energy / scaling,dim = -1)
print('att : ',att.shape)

out = torch.einsum('bhal, bhlv -> bhav',att,value)
print('out: ',out.shape)

out = rearrange(out,'b h n d -> b n (h d)')
print('out_2 : ',out.shape)
```

<pre>
energy :  torch.Size([8, 8, 197, 197])
att :  torch.Size([8, 8, 197, 197])
out:  torch.Size([8, 8, 197, 96])
out_2 :  torch.Size([8, 197, 768])
</pre>

```python
class MultiHeadAttention(nn.Module):
    def __init__(self,emb_size: int = 768,num_heads : int = 8, dropout : float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads

        self.qkv = nn.Linear(emb_size,emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size,emb_size)

    def forward(self,x : Tensor,mask : Tensor = None) -> Tensor:
        qkv = rearrange(self.qkv(x),'b n (h d qkv) -> (qkv) b h n d',h = self.num_heads,qkv = 3)
        queries,keys,values = qkv[0], qkv[1], qkv[2]
        energy = torch.einsum('bhqd, bhkd -> bhqk',queries,keys)

        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask,fill_value)

        scaling= self.emb_size ** (1/2)
        att = F.softmax(energy/ scaling,dim = -1)
        att = self.att_drop(att)

        out = torch.einsum('bhal,bhlv -> bhav',att,values)
        out = rearrange(out,'b h n d -> b n (h d)')
        out = self.projection(out)
        return out


        
```

## Residual Block



```python
class Residualblock(nn.Module):
    def __init__(self,fn):
        super().__init__()
        self.fn = fn

    def forward(self,x):
        res = x
        x = self.fn(x)
        output = x + res
        return output
```

## Feed Forward MLP


- expansion 후에 GELU 와 Dropout 진행 후 다시 원래의 emb_size로 축소



```python
class FeedForwardBlock(nn.Sequential):
    def __init__(self,emb_size : int,expansion : int = 4,drop_p : float = 0.):
        super().__init__(
            nn.Linear(emb_size,expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size,emb_size),
        )
```

# Transformer Encoder Block



```python
class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,emb_size :int = 768,
                 drop_p:float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p : float = 0.,
                 **kwargs):
        super().__init__(
                Residualblock(
                    nn.Sequential(
                        nn.LayerNorm(emb_size),
                        MultiHeadAttention(emb_size,**kwargs),
                        nn.Dropout(drop_p),
                                )
                            ),
                Residualblock(
                    nn.Sequential(
                        nn.LayerNorm(emb_size),
                        FeedForwardBlock(emb_size,expansion = forward_expansion,drop_p = forward_drop_p),
                        nn.Dropout(drop_p)
                                )
                            )
            )   
```

## test



```python
x = torch.randn(8,3,224,224)
patches_embedded = PatchEmbedding()(x)
TransformerEncoderBlock()(patches_embedded).shape
```

<pre>
cls: torch.Size([1, 1, 768])
x: torch.Size([8, 196, 768])
</pre>
<pre>
torch.Size([8, 197, 768])
</pre>
# Architecture



```python
class TransformerEncoder(nn.Sequential):
    def __init__(self,depth = 12,**kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])
        
```

Head



```python
class ClassificationHead(nn.Sequential):
    def __init__(self,emb_size:int = 768,n_classes: int = 1000):
        super().__init__(
            Reduce('b n e -> b e',reduction = 'mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size,n_classes)
        )
```

Summary



```python
class ViT(nn.Sequential):
    def __init__(self,
                 in_channels : int = 3,
                 patch_size : int = 16,
                 emb_size : int = 768,
                 img_size : int = 224,
                 depth : int = 12,
                 n_classes = 1000,
                 **kwargs):
        super().__init__(
            PatchEmbedding(in_channels,patch_size,emb_size,img_size),
            TransformerEncoder(depth,emb_size = emb_size,**kwargs),
            ClassificationHead(emb_size, n_classes)
        )
```


```python
summary(ViT(),(3,224,224),device = 'cpu')
```

<pre>
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1          [-1, 768, 14, 14]         590,592
         Rearrange-2             [-1, 196, 768]               0
    PatchEmbedding-3             [-1, 197, 768]               0
         LayerNorm-4             [-1, 197, 768]           1,536
            Linear-5            [-1, 197, 2304]       1,771,776
           Dropout-6          [-1, 8, 197, 197]               0
            Linear-7             [-1, 197, 768]         590,592
MultiHeadAttention-8             [-1, 197, 768]               0
           Dropout-9             [-1, 197, 768]               0
    Residualblock-10             [-1, 197, 768]               0
        LayerNorm-11             [-1, 197, 768]           1,536
           Linear-12            [-1, 197, 3072]       2,362,368
             GELU-13            [-1, 197, 3072]               0
          Dropout-14            [-1, 197, 3072]               0
           Linear-15             [-1, 197, 768]       2,360,064
          Dropout-16             [-1, 197, 768]               0
    Residualblock-17             [-1, 197, 768]               0
        LayerNorm-18             [-1, 197, 768]           1,536
           Linear-19            [-1, 197, 2304]       1,771,776
          Dropout-20          [-1, 8, 197, 197]               0
           Linear-21             [-1, 197, 768]         590,592
MultiHeadAttention-22             [-1, 197, 768]               0
          Dropout-23             [-1, 197, 768]               0
    Residualblock-24             [-1, 197, 768]               0
        LayerNorm-25             [-1, 197, 768]           1,536
           Linear-26            [-1, 197, 3072]       2,362,368
             GELU-27            [-1, 197, 3072]               0
          Dropout-28            [-1, 197, 3072]               0
           Linear-29             [-1, 197, 768]       2,360,064
          Dropout-30             [-1, 197, 768]               0
    Residualblock-31             [-1, 197, 768]               0
        LayerNorm-32             [-1, 197, 768]           1,536
           Linear-33            [-1, 197, 2304]       1,771,776
          Dropout-34          [-1, 8, 197, 197]               0
           Linear-35             [-1, 197, 768]         590,592
MultiHeadAttention-36             [-1, 197, 768]               0
          Dropout-37             [-1, 197, 768]               0
    Residualblock-38             [-1, 197, 768]               0
        LayerNorm-39             [-1, 197, 768]           1,536
           Linear-40            [-1, 197, 3072]       2,362,368
             GELU-41            [-1, 197, 3072]               0
          Dropout-42            [-1, 197, 3072]               0
           Linear-43             [-1, 197, 768]       2,360,064
          Dropout-44             [-1, 197, 768]               0
    Residualblock-45             [-1, 197, 768]               0
        LayerNorm-46             [-1, 197, 768]           1,536
           Linear-47            [-1, 197, 2304]       1,771,776
          Dropout-48          [-1, 8, 197, 197]               0
           Linear-49             [-1, 197, 768]         590,592
MultiHeadAttention-50             [-1, 197, 768]               0
          Dropout-51             [-1, 197, 768]               0
    Residualblock-52             [-1, 197, 768]               0
        LayerNorm-53             [-1, 197, 768]           1,536
           Linear-54            [-1, 197, 3072]       2,362,368
             GELU-55            [-1, 197, 3072]               0
          Dropout-56            [-1, 197, 3072]               0
           Linear-57             [-1, 197, 768]       2,360,064
          Dropout-58             [-1, 197, 768]               0
    Residualblock-59             [-1, 197, 768]               0
        LayerNorm-60             [-1, 197, 768]           1,536
           Linear-61            [-1, 197, 2304]       1,771,776
          Dropout-62          [-1, 8, 197, 197]               0
           Linear-63             [-1, 197, 768]         590,592
MultiHeadAttention-64             [-1, 197, 768]               0
          Dropout-65             [-1, 197, 768]               0
    Residualblock-66             [-1, 197, 768]               0
        LayerNorm-67             [-1, 197, 768]           1,536
           Linear-68            [-1, 197, 3072]       2,362,368
             GELU-69            [-1, 197, 3072]               0
          Dropout-70            [-1, 197, 3072]               0
           Linear-71             [-1, 197, 768]       2,360,064
          Dropout-72             [-1, 197, 768]               0
    Residualblock-73             [-1, 197, 768]               0
        LayerNorm-74             [-1, 197, 768]           1,536
           Linear-75            [-1, 197, 2304]       1,771,776
          Dropout-76          [-1, 8, 197, 197]               0
           Linear-77             [-1, 197, 768]         590,592
MultiHeadAttention-78             [-1, 197, 768]               0
          Dropout-79             [-1, 197, 768]               0
    Residualblock-80             [-1, 197, 768]               0
        LayerNorm-81             [-1, 197, 768]           1,536
           Linear-82            [-1, 197, 3072]       2,362,368
             GELU-83            [-1, 197, 3072]               0
          Dropout-84            [-1, 197, 3072]               0
           Linear-85             [-1, 197, 768]       2,360,064
          Dropout-86             [-1, 197, 768]               0
    Residualblock-87             [-1, 197, 768]               0
        LayerNorm-88             [-1, 197, 768]           1,536
           Linear-89            [-1, 197, 2304]       1,771,776
          Dropout-90          [-1, 8, 197, 197]               0
           Linear-91             [-1, 197, 768]         590,592
MultiHeadAttention-92             [-1, 197, 768]               0
          Dropout-93             [-1, 197, 768]               0
    Residualblock-94             [-1, 197, 768]               0
        LayerNorm-95             [-1, 197, 768]           1,536
           Linear-96            [-1, 197, 3072]       2,362,368
             GELU-97            [-1, 197, 3072]               0
          Dropout-98            [-1, 197, 3072]               0
           Linear-99             [-1, 197, 768]       2,360,064
         Dropout-100             [-1, 197, 768]               0
   Residualblock-101             [-1, 197, 768]               0
       LayerNorm-102             [-1, 197, 768]           1,536
          Linear-103            [-1, 197, 2304]       1,771,776
         Dropout-104          [-1, 8, 197, 197]               0
          Linear-105             [-1, 197, 768]         590,592
MultiHeadAttention-106             [-1, 197, 768]               0
         Dropout-107             [-1, 197, 768]               0
   Residualblock-108             [-1, 197, 768]               0
       LayerNorm-109             [-1, 197, 768]           1,536
          Linear-110            [-1, 197, 3072]       2,362,368
            GELU-111            [-1, 197, 3072]               0
         Dropout-112            [-1, 197, 3072]               0
          Linear-113             [-1, 197, 768]       2,360,064
         Dropout-114             [-1, 197, 768]               0
   Residualblock-115             [-1, 197, 768]               0
       LayerNorm-116             [-1, 197, 768]           1,536
          Linear-117            [-1, 197, 2304]       1,771,776
         Dropout-118          [-1, 8, 197, 197]               0
          Linear-119             [-1, 197, 768]         590,592
MultiHeadAttention-120             [-1, 197, 768]               0
         Dropout-121             [-1, 197, 768]               0
   Residualblock-122             [-1, 197, 768]               0
       LayerNorm-123             [-1, 197, 768]           1,536
          Linear-124            [-1, 197, 3072]       2,362,368
            GELU-125            [-1, 197, 3072]               0
         Dropout-126            [-1, 197, 3072]               0
          Linear-127             [-1, 197, 768]       2,360,064
         Dropout-128             [-1, 197, 768]               0
   Residualblock-129             [-1, 197, 768]               0
       LayerNorm-130             [-1, 197, 768]           1,536
          Linear-131            [-1, 197, 2304]       1,771,776
         Dropout-132          [-1, 8, 197, 197]               0
          Linear-133             [-1, 197, 768]         590,592
MultiHeadAttention-134             [-1, 197, 768]               0
         Dropout-135             [-1, 197, 768]               0
   Residualblock-136             [-1, 197, 768]               0
       LayerNorm-137             [-1, 197, 768]           1,536
          Linear-138            [-1, 197, 3072]       2,362,368
            GELU-139            [-1, 197, 3072]               0
         Dropout-140            [-1, 197, 3072]               0
          Linear-141             [-1, 197, 768]       2,360,064
         Dropout-142             [-1, 197, 768]               0
   Residualblock-143             [-1, 197, 768]               0
       LayerNorm-144             [-1, 197, 768]           1,536
          Linear-145            [-1, 197, 2304]       1,771,776
         Dropout-146          [-1, 8, 197, 197]               0
          Linear-147             [-1, 197, 768]         590,592
MultiHeadAttention-148             [-1, 197, 768]               0
         Dropout-149             [-1, 197, 768]               0
   Residualblock-150             [-1, 197, 768]               0
       LayerNorm-151             [-1, 197, 768]           1,536
          Linear-152            [-1, 197, 3072]       2,362,368
            GELU-153            [-1, 197, 3072]               0
         Dropout-154            [-1, 197, 3072]               0
          Linear-155             [-1, 197, 768]       2,360,064
         Dropout-156             [-1, 197, 768]               0
   Residualblock-157             [-1, 197, 768]               0
       LayerNorm-158             [-1, 197, 768]           1,536
          Linear-159            [-1, 197, 2304]       1,771,776
         Dropout-160          [-1, 8, 197, 197]               0
          Linear-161             [-1, 197, 768]         590,592
MultiHeadAttention-162             [-1, 197, 768]               0
         Dropout-163             [-1, 197, 768]               0
   Residualblock-164             [-1, 197, 768]               0
       LayerNorm-165             [-1, 197, 768]           1,536
          Linear-166            [-1, 197, 3072]       2,362,368
            GELU-167            [-1, 197, 3072]               0
         Dropout-168            [-1, 197, 3072]               0
          Linear-169             [-1, 197, 768]       2,360,064
         Dropout-170             [-1, 197, 768]               0
   Residualblock-171             [-1, 197, 768]               0
          Reduce-172                  [-1, 768]               0
       LayerNorm-173                  [-1, 768]           1,536
          Linear-174                 [-1, 1000]         769,000
================================================================
Total params: 86,415,592
Trainable params: 86,415,592
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.57
Forward/backward pass size (MB): 364.33
Params size (MB): 329.65
Estimated Total Size (MB): 694.56
----------------------------------------------------------------
</pre>

```python
```
