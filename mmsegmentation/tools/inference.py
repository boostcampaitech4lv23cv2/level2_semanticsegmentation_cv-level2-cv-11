import os

from mmcv import Config
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import single_gpu_test
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
import torch
import torchvision
import pandas as pd
import numpy as np
import json

################여기########################
# config파일 경로복사해주세요
cfg = Config.fromfile('/opt/ml/mmsegmentation/configs/_teajun_/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K.py')
################여기########################
# print(cfg)
root = '/opt/ml/data'
epoch = 'latest'

# work_dir 설정해주세요
cfg.work_dir = '/opt/ml/mmsegmentation/work_dirs/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K'
checkpoint_path = os.path.join(cfg.work_dir, f'{epoch}.pth')

# print('-'*10)
# print(checkpoint_path)
# print('-'*10)

dataset = build_dataset(cfg.data.test)
data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)


model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')

model.CLASSES = dataset.CLASSES
model = MMDataParallel(model.cuda(), device_ids=[0])


output = single_gpu_test(model, data_loader)

# print('-'*10)
# print(len(output[0]))
# print(len(output))
# print('-'*10)

submission = pd.read_csv('/opt/ml/input/code/submission/sample_submission.csv', index_col=None)
json_dir = os.path.join("/opt/ml/input/data/test.json")
with open(json_dir, "r", encoding="utf8") as f:
    test_json = json.load(f)

output = torch.Tensor(output)
# print(output.shape)
for image_id, predict in enumerate(output):
    image_id = test_json["images"][image_id]
    file_name = image_id["file_name"]
    temp_mask = []
    # resize
    # print(predict.shape)
    predict = predict.reshape(1, 512, 512)
    mask = torchvision.transforms.Resize(256)(predict)
    # print(mask.shape)
    temp_mask.append(np.array(mask))
    oms = np.array(temp_mask)
    # print(oms.shape)
    oms = oms.flatten()
    submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in oms.tolist())}, 
                                   ignore_index=True)

# output 경로
submission.to_csv(os.path.join(cfg.work_dir, f'submission_{epoch}.csv'), index=False)