# Semantic Segmentation CV-11

## 대회 개요
>바야흐로 대량 생산, 대량 소비의 시대. 우리는 많은 물건이 대량으로 생산되고, 소비되는 시대를 살고 있습니다. 하지만 이러한 문화는 '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제를 낳고 있습니다. 분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나입니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문입니다. 따라서 우리는 **사진에서 쓰레기를 Segmentation하는 모델을 만들어** 이러한 문제점을 해결해보고자 합니다.
> - Input : 배경, 일반 쓰레기, 플라스틱, 종이, 유리 등 11 종류의 쓰레기가 찍힌 이미지\
(annotation은 COCO format으로 제공)
> - Output : 모델은 pixel 좌표에 따라 카테고리 값을 리턴합니다. 이를 submission 양식에 맞게 csv 파일을 만들어 제출합니다.
> - 평가방법 : Test set의 mIoU


## Members

| [류건](https://github.com/jerry-ryu) | [심건희](https://github.com/jane79) | [윤태준](https://github.com/ta1231) | [이강희](https://github.com/ganghe74) | [이예라](https://github.com/Yera10) |
| :-: | :-: | :-: | :-: | :-: | 
| <img src="https://avatars.githubusercontent.com/u/62556539?v=4" width="200"> | <img src="https://avatars.githubusercontent.com/u/48004826?v=4" width="200"> | <img src="https://avatars.githubusercontent.com/u/54363784?v=4"  width="200"> | <img src="https://avatars.githubusercontent.com/u/30896956?v=4" width="200"> | <img src="https://avatars.githubusercontent.com/u/57178359?v=4" width="200"> |  
|[Blog](https://kkwong-guin.tistory.com/)  |[Blog](https://velog.io/@goodheart50)|[Blog](https://velog.io/@ta1231)| [Blog](https://dddd.ac/blog) | [Blog](https://yedoong.tistory.com/) |

<div align="center">

![python](http://img.shields.io/badge/Python-000000?style=flat-square&logo=Python)
![pytorch](http://img.shields.io/badge/PyTorch-000000?style=flat-square&logo=PyTorch)
![ubuntu](http://img.shields.io/badge/Ubuntu-000000?style=flat-square&logo=Ubuntu)
![git](http://img.shields.io/badge/Git-000000?style=flat-square&logo=Git)
![github](http://img.shields.io/badge/Github-000000?style=flat-square&logo=Github)

</div align="center">

## Environments
> - Ubuntu 18.04.5 LTS
> - Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz
> - NVIDIA Tesla V100-PCIE-32GB
> - mmsegmentation
> - pytorch ..

## 개발 환경 설정
### git
```CMDs
git clone add origin https://github.com/boostcampaitech4lv23cv2/level2_semanticsegmentation_cv-level2-cv-11.git
```

### mmsegmentation
```CMDs
conda create -n openmmlab --clone base 
conda activate mmopenlab 
pip install -r mmsegmentation/requirements.txt
pip install -U openmim
mim install mmcv-full
mim install mmcv-full==1.7.0
```

버전 확인
```CMD
python -c 'import torch;print(torch.__version__);print(torch.version.cuda)'
pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7/index.html
```

잘 설치되었는지 확인하기
```CMD
cd mmsegmentation
pip install -e .
apt-get update && apt-get install libgl1
mim download mmsegmentation --config pspnet_r50-d8_512x1024_40k_cityscapes --dest .
python demo/image_demo.py demo/demo.png configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth --device cuda:0 --out-file result.jpg
```

### mmsegmentation train
```CMD
cd mmsegmentation
python tools/train.py <config_file>
```

### mmsegmentation inference
config file, work_dir, epoch 변경 후
```CMD
cd mmsegmentation
python tools/inference.py
```

### smp. train
```CMD
cd smp
python train.py
```

### smp. inference
input/code/baseline_train_smp.ipynb 실행


## Communication
>### Github
> - 코드 공유를 통해 협업
> - Issue로 todo 관리
> - PR로 코드리뷰
>### Notion
> - 실험보드, 회의록, 컨벤션 공유
>### WandB
> - 실험기록을 공유하여 hyperparameter 튜닝 및 모델 평가

## Directory Structure
```CMDs
.
|-- README.md
|-- input
|   |-- code
|   `-- data
|-- mmsegmentation
|   |-- configs
|   |   |-- _base_
|   |   |-- _teajun_
|   |   |   |-- _base_
|   |   |   |-- beit
|   |   |   |-- convnext
|   |   |   |-- knet
|   |   |   |-- mobilenet_v3
|   |   |   |-- segformer
|   |   |   `-- swin
|   |-- demo
|   |-- docker
|   |-- docs
|   |-- mmseg
|   |-- resources
|   |-- setup.py
|   |-- tests
|   `-- tools
|-- notebooks
|   |-- copy_paste
|   |-- ensemble.ipynb
|   |-- ensemble_refactor.ipynb
|   |-- model_soup.ipynb
|   `-- stratified_kfold.ipynb
`-- smp
    |-- __pycache__
    |-- data_loader.py
    |-- saved
    |-- train.py
    |-- trainer.py
    |-- utils.py
    `-- wandb
```

## Project Implementation Procedures and Methods
[![image](https://user-images.githubusercontent.com/62556539/200262300-3765b3e4-0050-4760-b008-f218d079a770.png)](https://excessive-help-ce8.notion.site/8c0240de8c394e5184de03dbdb9aac79)

## Team Roles
구미호: 심건희, 류건, 윤태준, 이강희, 이예라

- 심건희: 모델 (smp. / efficientnet, scheduler) 실험, 튜닝
- 류건: 모델(mmseg. / segformer, ConvNext) 실험, 튜닝, model soup, copy & paste 구현, wandb 연결
- 윤태준: 모델실험(mmseg. / KNet, ConvNext), 앙상블, 추론코드 구현
- 이강희: mmsegmentation 데이터셋 구현, 모델 실험(mmseg. / BEIT)
- 이예라: 모델(smp. / fpn, encoder, augmentation) 실험, wandb 관련 구현, smp-template 관련 구현

## Timeline
![Untitled (7)](https://user-images.githubusercontent.com/62556539/210492522-1cc7b7aa-fa1e-45f3-b5f1-767063abcce4.png)

## Results
<img src="https://user-images.githubusercontent.com/62556539/210496223-bdf8e426-85c3-4abc-ada4-508b3c691e67.png"  width="75%" height="75%"/>

**smp. library**

- ensemble (deeplabV3++, pan, fpn)  ⇒ 0.5861

 **mmseg. library**

- model soup (ConvNext, KNet, BEiT)
- ensemble (ConvNext,KNet,BEiT) ⇒ 0.7454

 **ensemble(smp. + mmseg.)** ⇒ 0.7431


## 자체 평가 의견
**잘한 점**
- Model Soup를 게시판에 올려 다른 사람들이 사용해볼 수 있도록 기여했다.
- Pull Request를 통해 한번씩 검사를 받고 merge하여 오류가 줄었다.
- 힘든 일정에도 서로를 격려하고 팀 분위기를 긍정적으로 유지하였다.
- 다양한 SOTA segmentation에 대해 공부하고 실험했다.
- 함께 디버깅하여 빠른 문제 대응을 할 수 있었다.
- Wandb를 사용하여 실시간 모니터링 및 팀원들과의 결과 공유가 용이했다.
- 일일 PM을 통해서 코드 리뷰와 공유를 더 용이하게 하였다.

**아쉬운 점:**
- PM의 역할에 대해 명확히 정해두지 못했다

**개선할 점:**
- PM 역할 명확하게 해두기


---
