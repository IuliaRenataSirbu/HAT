# flake8: noqa
import os.path as osp

import hat.archs
import hat.data
import hat.models
from basicsr.train import train_pipeline


# import torch
# print('MY_PRINT', torch.load('./experiments/pretrained_models/HAT_SRx4_ImageNet-pretrain.pth', map_location='cpu').keys())

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
