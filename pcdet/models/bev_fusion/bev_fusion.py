import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import yaml
from easydict import EasyDict
from pathlib import Path


class BEV_FUSION(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        
        self.model_cfg = model_cfg

        cur_layers = [
                    nn.Conv2d(512, 256, kernel_size=(3,3), stride=(1,1), bias=False, padding=1),
                    nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
                    nn.ReLU(),
                    # nn.Dropout2d(p=0.2),
                    nn.MaxPool2d((2,2), stride=2),
                    
                ]

        self.blocks = nn.ModuleList()
        self.blocks.append(nn.Sequential(*cur_layers))


    def forward(self, data_dict):

        range_f_1 = data_dict['range_bev_conv1'] # B*128*400*352
        range_f_2 = data_dict['range_bev_conv2'] # B*128*200*176

        spatial_features = data_dict['spatial_features']# B*256*200*176

        temp_cat_f = torch.cat((range_f_2,spatial_features), dim=1) # B*(256+128)*200*176
        up_temp_cat_f = nn.functional.interpolate(temp_cat_f, scale_factor=(2,2), mode='bilinear', align_corners=True) # 双线性插值上采样 B*(256+128)*400*352
        cat_f = torch.cat((range_f_1,up_temp_cat_f), dim=1) # B*512*400*352

        out = self.blocks[0](cat_f)

        data_dict['bev_fusion_out'] = out

        return data_dict