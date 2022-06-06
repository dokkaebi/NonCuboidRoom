import numpy as np
import torch
import yaml
from easydict import EasyDict
import argparse
import cv2
import copy
from collections import OrderedDict

from datasets import NYU303, CustomDataset, Structured3D
from models import (ConvertLayout, Detector, DisplayLayout, display2Dseg, Loss,
                    Reconstruction, _validate_colormap, post_process)
from models.detector import MobileNetDeeplabDetector, MobileViTDeeplabDetector
from scipy.optimize import linear_sum_assignment



if __name__ == '__main__':
    # create network
    from options.opts import get_training_arguments
    from options.utils import load_config_file

    opts = argparse.Namespace()

    setattr(opts, 'common.config_file', 'NonCuboidRoom/config/deeplabv3_mobilevit_small.yaml')
    opts = load_config_file(opts)

    model = MobileViTDeeplabDetector(opts)

    params = list(model.parameters())
    total = sum(p.numel() for p in params)
    print(f'total: {total}')

    print(f'encoder: {sum(p.numel() for p in model.encoder.parameters())}')
    print(f'decoder: {sum(p.numel() for p in model.decoder.parameters())}')
    
    # print(f'backbone: {sum(p.numel() for p in model.backbone.parameters())}')
    # print(f'merge: {sum(p.numel() for p in model.merge.parameters())}')
    # print(f'heads: {sum(p.numel() for p in model.heads.parameters())}')

    # print(f'heads.plane_center: {sum(p.numel() for p in model.heads.plane_center.parameters())}')
    # print(f'heads.plane_xy: {sum(p.numel() for p in model.heads.plane_xy.parameters())}')
    # print(f'heads.plane_wh: {sum(p.numel() for p in model.heads.plane_wh.parameters())}')
    # print(f'heads.plane_params_pixelwise: {sum(p.numel() for p in model.heads.plane_params_pixelwise.parameters())}')
    # print(f'heads.plane_params_instance: {sum(p.numel() for p in model.heads.plane_params_instance.parameters())}')
    # print(f'heads.line_region: {sum(p.numel() for p in model.heads.line_region.parameters())}')
    # print(f'heads.line_params: {sum(p.numel() for p in model.heads.line_params.parameters())}')

# MobileNetDeeplabDetector
# total: 10368718
# encoder: 3520424
# decoder: 2713108

# MobileViTDeeplabDetector
# total: 11244606
# encoder: 5584152
# decoder: 1525268

