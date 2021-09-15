import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import yaml
from easydict import EasyDict
from pathlib import Path




class Range2D(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        
        self.model_cfg = model_cfg

        input_c = self.model_cfg['INPUT_C']

        if self.model_cfg.get('LAYER_NUM', None) is not None:
            assert len(self.model_cfg.LAYER_NUM) == len(self.model_cfg.NUM_FILTERS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_PADDING)
            # print('True')
            layer_nums = self.model_cfg.LAYER_NUM
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
            num_pad = self.model_cfg.NUM_PADDING
            c_in_list = [input_c, *num_filters]
        else:
            layer_nums = layer_strides = num_filters = []


        num_levels = len(layer_nums)

        self.blocks = nn.ModuleList()

        for idx in range(num_levels):

            cin = c_in_list[idx]
            cout = c_in_list[idx+1]
            # kernel = down_layer_strides[idx]
            cur_layers = [
                    nn.Conv2d(cin, cout, kernel_size=layer_strides[idx], stride=(1,1), bias=False, padding=num_pad[idx]),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU(),
                    # nn.Dropout2d(p=0.2),
                    nn.MaxPool2d((2,2), stride=2),
                ]

            self.blocks.append(nn.Sequential(*cur_layers))
        

    def forward(self, batch_dict):

        x = batch_dict['range_bev']

        out = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            out.append(x)

        
        batch_dict['range_bev_conv1'] = out[0]
        batch_dict['range_bev_conv2'] = out[1]

        return batch_dict