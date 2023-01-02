# Library
import os
import json
import warnings 
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from utils import seed_fix
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import wandb
import segmentation_models_pytorch as smp

from data_loader import CustomDataLoader, collate_fn
from trainer import Train

# GPU, seed fix
print('GPU 사용 가능 여부: {}'.format(torch.cuda.is_available()))

cfg = dict(
    model = dict(
        name = 'FPN',
        arch = dict(
            encoder_name='efficientnet-b4',
            encoder_depth = 5, 
            encoder_weights='imagenet',
            in_channels=3,
            classes=11
        )
    ), 
    hp = dict(
        seed = 42,
        num_epochs = 20,
        batch_size = 4,
        criterion = nn.CrossEntropyLoss(),
        optimizer = dict(
            name = 'SGD',
            param = dict(
                lr = 1e-4,
                momentum = 0.9,
                weight_decay = 1e-6
            )
        ),
        scheduler = dict(
            name = 'CyclicLR',
            param = dict(
                base_lr = 1e-5, 
                max_lr = 1e-3, 
                step_size_up = 4, 
                step_size_down = 4,
                mode = 'exp_range', 
                gamma = 0.9 
            )
        ),
    ),
)
wandb_setting = dict(
    name = "0101_yr_fpn_effb4_cycliclr",
    notes = "smp - FPN scheduler 실험",
    tags = ['smp', 'adam', 'fpn', 'efficientnet-b4', 'cycliclr']
)
    
def main(cfg):
    seed_fix(cfg['hp']['seed'])
    
    # DATA PATHS
    dataset_path  = '../input/data'
    anns_file_path = dataset_path + '/' + 'train_all.json'
    with open(anns_file_path, 'r') as f:
        dataset = json.loads(f.read())
    category_names = ['Backgroud'] + list(pd.DataFrame(dataset['categories'])['name'])
    train_path = dataset_path + '/train.json'
    val_path = dataset_path + '/val.json'

    # augmentation
    train_transform = A.Compose([
                                ToTensorV2()
                                ])

    val_transform = A.Compose([
                            ToTensorV2()
                            ])
    # DATASET
    train_dataset = CustomDataLoader(data_dir=train_path, category_names=category_names, mode='train', transform=train_transform)
    val_dataset = CustomDataLoader(data_dir=val_path, category_names=category_names, mode='val', transform=val_transform)

    # MODEL
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = getattr(smp, cfg['model']['name'])
    model = model(**cfg['model']['arch'])
    model = model.to(device)
    
    # model check
    x = torch.randn([2, 3, 512, 512]).cuda()
    out = model(x)
    print(f"MODEL CHECK! -- input shape : {x.shape}, output shape : {out.size()}")

    # 모델 저장을 위한 변수
    val_every = 1
    saved_dir = './saved'
    if not os.path.isdir(saved_dir):                                                           
        os.mkdir(saved_dir)
        
    # wandb setting
    wandb.init(
        entity = 'miho',
        project = 'segmentation',
        **wandb_setting
    )
        # name = exp_name,
        # notes = notes,
        # tags = tags)
    wandb.config.update(cfg)

    # DATALOADER
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=cfg['hp']['batch_size'],
                                            shuffle=True,
                                            num_workers=4,
                                            collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                            batch_size=cfg['hp']['batch_size'],
                                            shuffle=False,
                                            num_workers=4,
                                            collate_fn=collate_fn)

    # train
    optimizer = getattr(torch.optim, cfg['hp']['optimizer']['name'])
    optimizer = optimizer(params = model.parameters(), **cfg['hp']['optimizer']['param'])
    scheduler = getattr(torch.optim.lr_scheduler, cfg['hp']['scheduler']['name'])
    scheduler = scheduler(optimizer, **cfg['hp']['scheduler']['param'])
    Train(cfg['hp']['num_epochs'], model, train_loader, val_loader, cfg['hp']['criterion'], optimizer, scheduler, saved_dir, val_every, device, wandb_setting['name'], category_names=category_names)
    wandb.finish()

if __name__ == '__main__':
    main(cfg)