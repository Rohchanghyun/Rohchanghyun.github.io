---
title:  "[Pytorch] Imagenet Practice"
excerpt: "pre-trained ResNet34 를 통한 Imagenet의 데이터 훈련 "

categories:
  - pytorch
tags:
  - ResNet
  - drive_mount
  - kaggle_dataset

toc: true
toc_sticky: true
 
date: 2022-02-19
last_modified_at: 2022-02-19
---

```python
!pip install --upgrade --force-reinstall --no-deps kaggle
```

    Collecting kaggle
      Downloading kaggle-1.5.12.tar.gz (58 kB)
    [K     |████████████████████████████████| 58 kB 1.8 MB/s 
    [?25hBuilding wheels for collected packages: kaggle
      Building wheel for kaggle (setup.py) ... [?25l[?25hdone
      Created wheel for kaggle: filename=kaggle-1.5.12-py3-none-any.whl size=73051 sha256=17d6294af68d8ebe30614dc0b532c54f553b59c13d570451b8c5cfe2c0584bbb
      Stored in directory: /root/.cache/pip/wheels/62/d6/58/5853130f941e75b2177d281eb7e44b4a98ed46dd155f556dc5
    Successfully built kaggle
    Installing collected packages: kaggle
      Attempting uninstall: kaggle
        Found existing installation: kaggle 1.5.12
        Uninstalling kaggle-1.5.12:
          Successfully uninstalled kaggle-1.5.12
    Successfully installed kaggle-1.5.12
    


```python
!pip install kaggle
from google.colab import files
files.upload()
```

    Requirement already satisfied: kaggle in /usr/local/lib/python3.7/dist-packages (1.5.12)
    Requirement already satisfied: six>=1.10 in /usr/local/lib/python3.7/dist-packages (from kaggle) (1.15.0)
    Requirement already satisfied: certifi in /usr/local/lib/python3.7/dist-packages (from kaggle) (2021.10.8)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.7/dist-packages (from kaggle) (4.62.3)
    Requirement already satisfied: urllib3 in /usr/local/lib/python3.7/dist-packages (from kaggle) (1.24.3)
    Requirement already satisfied: python-slugify in /usr/local/lib/python3.7/dist-packages (from kaggle) (5.0.2)
    Requirement already satisfied: python-dateutil in /usr/local/lib/python3.7/dist-packages (from kaggle) (2.8.2)
    Requirement already satisfied: requests in /usr/local/lib/python3.7/dist-packages (from kaggle) (2.23.0)
    Requirement already satisfied: text-unidecode>=1.3 in /usr/local/lib/python3.7/dist-packages (from python-slugify->kaggle) (1.3)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests->kaggle) (2.10)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests->kaggle) (3.0.4)
    



<input type="file" id="files-9777bff1-c81b-408d-889c-3c25307ad697" name="files[]" multiple disabled
   style="border:none" />
<output id="result-9777bff1-c81b-408d-889c-3c25307ad697">
 Upload widget is only available when the cell has been executed in the
 current browser session. Please rerun this cell to enable.
 </output>
 <script src="/nbextensions/google.colab/files.js"></script> 



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-6-8090cb07c344> in <module>()
          1 get_ipython().system('pip install kaggle')
          2 from google.colab import files
    ----> 3 files.upload()
    

    /usr/local/lib/python3.7/dist-packages/google/colab/files.py in upload()
         61   result = _output.eval_js(
         62       'google.colab._files._uploadFiles("{input_id}", "{output_id}")'.format(
    ---> 63           input_id=input_id, output_id=output_id))
         64   files = _collections.defaultdict(_six.binary_type)
         65   # Mapping from original filename to filename as saved locally.
    

    /usr/local/lib/python3.7/dist-packages/google/colab/output/_js.py in eval_js(script, ignore_result, timeout_sec)
         38   if ignore_result:
         39     return
    ---> 40   return _message.read_reply_from_input(request_id, timeout_sec)
         41 
         42 
    

    /usr/local/lib/python3.7/dist-packages/google/colab/_message.py in read_reply_from_input(message_id, timeout_sec)
         99     reply = _read_next_input_message()
        100     if reply == _NOT_READY or not isinstance(reply, dict):
    --> 101       time.sleep(0.025)
        102       continue
        103     if (reply.get('type') == 'colab_reply' and
    

    KeyboardInterrupt: 



```python
ls -1ha kaggle.json
```

    kaggle.json
    

![image.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAALgAAAAoCAYAAABNVTCEAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAAG5SURBVHhe7dwNaoNAEIZh7alyn5hTGc9qGXDIMKw/EbdrP94HlmrGXSP7abaFtH88HnMHiPpZfgKSCDikEXBII+CQRsAhjYBDGgGHNAIOaQQc0gg4pP15wKdp6vq+X/basfdRU+3xcQxPcEgj4JU8n89lCy3dIuD549z2va2J9dJxXrflUKm+xfvGMb61dc48fsla7f1+7/bFR/OA2yTFp53veytNYjxmrz6O4/LqMfn8PsZVvzeUxs/XEI/JNTMMw2YdH00D7hMZ5f0s98nH57qF4e7i+7UndL4+Qnxes4DniYxsQr21EM9v7cq/+nhgraG+JgG3cK+FxibeQuCthXh+b/N83ReffEyCXl+TgPsa8szk5n55jFy3m+muYtBRR9M1+N7krtW8nzXbzmL99Xotr55X+sRZe2979vrZzR+Pse3SNeKYpgE3OeSl8OYJN1YrTXyuf7u0sD5+fm92k1y1RCmNn68jHpNr+M6//Fa9TXyUQ7BXr8nPTTDvgX8bAWnNlyhATQQc0gg4pBFwSCPgkEbAIY2AQxoBhzQCDmkEHNIIOKQRcEgj4JBGwCGNgEMaAYewrvsFBlcUV2GPRsUAAAAASUVORK5CYII=)


```python
!pip install -q kaggle
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/

# Permission Warning 방지
!chmod 600 ~/.kaggle/kaggle.json
```


```python
#!kaggle datasets download -d ambityga/imagenet100
```

    404 - Not Found
    


```python
!ls
```

     imagenet100.zip   imagenet_resnet.ipynb  'kaggle (1).json'   kaggle.json
    


```python
!unzip imagenet100.zip
```


```python
cd drive/My\Drive/imagenet

```

    /content/drive/MyDrive/imagenet
    


```python
mkdir drive/My\Drive/imagenet
```

    mkdir: cannot create directory ‘drive/MyDrive/imagenet’: No such file or directory
    


```python
cd drive/My\Drive/imagenet
```

    [Errno 2] No such file or directory: 'drive/MyDrive/imagenet'
    /content/drive/MyDrive/imagenet
    


```python
from torchvision import models
import torch

resnet34_pretrained = models.resnet34(pretrained=True)

print(resnet34_pretrained)
```

    Downloading: "https://download.pytorch.org/models/resnet34-b627a593.pth" to /root/.cache/torch/hub/checkpoints/resnet34-b627a593.pth
    


      0%|          | 0.00/83.3M [00:00<?, ?B/s]


    ResNet(
      (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      (layer1): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): BasicBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (2): BasicBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (layer2): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (downsample): Sequential(
            (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (2): BasicBlock(
          (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (3): BasicBlock(
          (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (layer3): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (downsample): Sequential(
            (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (2): BasicBlock(
          (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (3): BasicBlock(
          (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (4): BasicBlock(
          (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (5): BasicBlock(
          (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (layer4): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (downsample): Sequential(
            (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (2): BasicBlock(
          (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
      (fc): Linear(in_features=512, out_features=1000, bias=True)
    )
    


```python
num_classes = 5
num_ftrs = resnet34_pretrained.fc.in_features
resnet34_pretrained.fc = torch.nn.Linear(num_ftrs, num_classes)

device = torch.device('cuda:0')
resnet34_pretrained.to(device)
```




    ResNet(
      (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      (layer1): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): BasicBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (2): BasicBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (layer2): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (downsample): Sequential(
            (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (2): BasicBlock(
          (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (3): BasicBlock(
          (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (layer3): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (downsample): Sequential(
            (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (2): BasicBlock(
          (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (3): BasicBlock(
          (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (4): BasicBlock(
          (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (5): BasicBlock(
          (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (layer4): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (downsample): Sequential(
            (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (2): BasicBlock(
          (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
      (fc): Linear(in_features=512, out_features=5, bias=True)
    )




```python
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split,DataLoader
import os
import os.path
```


```python
trans = transforms.Compose([transforms.ToTensor(),
                            transforms.Resize((224,224))])

dataset = ImageFolder(root = '/content/drive/MyDrive/imagenet/train_ray',transform = trans)
```


```python

val_num = int(len(dataset) * 0.1)
train_num = int(len(dataset)) - val_num

print(train_num,val_num)
```

    5850 650
    


```python
train_data,val_data = random_split(dataset,[train_num,val_num])
```


```python
train_data.__getitem__(2)
```




    (tensor([[[4.7149e-02, 2.1258e-04, 0.0000e+00,  ..., 1.1765e-02,
               1.5091e-02, 1.3270e-02],
              [7.6706e-03, 1.6582e-03, 0.0000e+00,  ..., 8.4384e-03,
               1.1765e-02, 1.3270e-02],
              [1.1630e-03, 2.6135e-03, 0.0000e+00,  ..., 1.4666e-02,
               1.2232e-02, 1.5169e-02],
              ...,
              [2.7794e-02, 3.1370e-02, 2.3777e-02,  ..., 2.0306e-02,
               3.0660e-02, 3.4454e-02],
              [2.3529e-02, 2.3529e-02, 2.3529e-02,  ..., 3.0893e-02,
               2.3852e-02, 2.7731e-02],
              [2.7556e-02, 2.3529e-02, 2.6050e-02,  ..., 2.5763e-02,
               2.6050e-02, 2.7451e-02]],
     
             [[2.7363e-01, 2.4608e-01, 2.4794e-01,  ..., 1.9216e-01,
               1.9548e-01, 1.9366e-01],
              [2.4766e-01, 2.4032e-01, 2.5096e-01,  ..., 1.8883e-01,
               1.9608e-01, 1.9758e-01],
              [2.5005e-01, 2.4491e-01, 2.4500e-01,  ..., 1.9555e-01,
               1.9346e-01, 1.9758e-01],
              ...,
              [1.6897e-01, 1.6471e-01, 1.6495e-01,  ..., 9.4185e-02,
               9.3405e-02, 9.7199e-02],
              [1.6471e-01, 1.6471e-01, 1.6471e-01,  ..., 9.8074e-02,
               9.0239e-02, 9.4118e-02],
              [1.6873e-01, 1.6471e-01, 1.6723e-01,  ..., 9.2024e-02,
               9.2717e-02, 9.4118e-02]],
     
             [[5.9554e-01, 6.0284e-01, 5.8964e-01,  ..., 4.9349e-01,
               4.9464e-01, 4.9282e-01],
              [6.0667e-01, 6.0621e-01, 5.9245e-01,  ..., 4.8295e-01,
               4.7843e-01, 4.7994e-01],
              [6.0519e-01, 5.9017e-01, 5.8649e-01,  ..., 4.8942e-01,
               4.8506e-01, 4.8563e-01],
              ...,
              [3.7289e-01, 3.7124e-01, 3.6888e-01,  ..., 2.1183e-01,
               1.9537e-01, 1.9916e-01],
              [3.6863e-01, 3.6863e-01, 3.6804e-01,  ..., 2.1572e-01,
               1.9948e-01, 2.0336e-01],
              [3.7265e-01, 3.6863e-01, 3.7052e-01,  ..., 2.0987e-01,
               2.0756e-01, 2.0896e-01]]]), 1)




```python
trainloader = DataLoader(train_data,batch_size = 64,shuffle = True)
valloader = DataLoader(val_data,batch_size = 64,shuffle = True)
```


```python
dataiter = iter(trainloader)
images,labels = dataiter.next()
print(labels)
```

    tensor([1, 0, 3, 0, 3, 0, 2, 1, 1, 2, 1, 4, 0, 0, 0, 2, 3, 2, 4, 2, 2, 4, 1, 0,
            4, 0, 2, 2, 2, 2, 0, 3, 0, 0, 0, 4, 4, 2, 3, 3, 4, 2, 1, 3, 2, 2, 1, 0,
            3, 4, 3, 1, 1, 4, 4, 0, 1, 0, 0, 3, 3, 2, 1, 4])
    


```python
learning_rate = 0.001
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adadelta(resnet34_pretrained.parameters(),lr = learning_rate)
```


```python
total_batch = len(trainloader)
print('총 배치의 수 : {}'.format(total_batch))
```

    총 배치의 수 : 92
    


```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
```


```python
nb_epochs = 10
for epoch in range(nb_epochs):
    avg_cost = 0
    for x,y in trainloader:
        x_train = x
        y_train = y

        x_train = x_train.cuda()
        y_train = y_train.cuda()

        prediction = resnet34_pretrained(x_train)

        loss = criterion(prediction,y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_cost += loss / total_batch

    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))
```

    [Epoch:    1] cost = 1.47325099
    [Epoch:    2] cost = 0.964924395
    [Epoch:    3] cost = 0.678268194
    [Epoch:    4] cost = 0.520831168
    [Epoch:    5] cost = 0.41871202
    [Epoch:    6] cost = 0.355509788
    [Epoch:    7] cost = 0.312933207
    [Epoch:    8] cost = 0.277055025
    [Epoch:    9] cost = 0.252626389
    [Epoch:   10] cost = 0.229041934
    


```python

for x,y in valloader:
    correct = 0
    
    x_val = x
    y_val = y

    x_val = x_val.cuda()
    y_val = y_val.cuda()

    prediction = resnet34_pretrained(x_val)
    _,predicted = torch.max(prediction,1)
    batch_total = len(predicted)
    #print('pred :',predicted )
    #print('y_val : ',y_val)
    for i in range(len(predicted)):
        if y_val[i] == predicted[i]:
            correct += 1

    print("accuracy:",correct/batch_total * 100)



```

    accuracy: 93.75
    accuracy: 95.3125
    accuracy: 90.625
    accuracy: 93.75
    accuracy: 96.875
    accuracy: 95.3125
    accuracy: 90.625
    accuracy: 95.3125
    accuracy: 93.75
    accuracy: 93.75
    accuracy: 90.0
    


```python

```
