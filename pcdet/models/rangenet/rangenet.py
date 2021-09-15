
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import yaml
from easydict import EasyDict
from pathlib import Path




class DRB(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()

        # self.cfg = cfg

        self.dil_c1 = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=(3, 3), stride=1, bias=False, dilation=1, padding=1),
            nn.BatchNorm2d(cout, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )

        self.dil_c2 = nn.Sequential(
            nn.Conv2d(cout, cout, kernel_size=(3, 3), stride=1, bias=False, dilation=3, padding=3),
            nn.BatchNorm2d(cout, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )

        self.dil_c3 = nn.Sequential(
            nn.Conv2d(cout, cout, kernel_size=(3, 3), stride=1, bias=False, dilation=5, padding=5),
            nn.BatchNorm2d(cout, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )
        self.bn = nn.BatchNorm2d(cout, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU()
        self.c1_1 = nn.Conv2d(3*cout,cout,kernel_size=(1,1),stride=1)

    def forward(self, i):
        d1 = self.dil_c1(i)
        d2 = self.dil_c2(d1)
        d3 = self.dil_c3(d2)
        conta = torch.cat((d1,d2,d3), 1)
        con_cin = self.c1_1(conta)
        # con_cin_bn = self.bn(con_cin)
        out = i + con_cin
        # out = self.bn(out)
        # out = self.relu(out)
        return out



class RangeNet(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        
        self.model_cfg = model_cfg

        if self.model_cfg.get('DownSample_LAYER_NUM', None) is not None:
            assert len(self.model_cfg.DownSample_LAYER_NUM) == len(self.model_cfg.DOWN_NUM_FILTERS)
            # print('True')
            down_layer_nums = self.model_cfg.DownSample_LAYER_NUM
            # down_layer_strides = self.model_cfg.DOWN_LAYER_STRIDES
            down_num_filters = self.model_cfg.DOWN_NUM_FILTERS
            down_c_in_list = [input_channels, *down_num_filters]
        else:
            down_layer_nums = down_layer_strides = down_num_filters = []

        if self.model_cfg.get('UpSample_LAYER_NUM', None) is not None:
            assert len(self.model_cfg.UpSample_LAYER_NUM) == len(self.model_cfg.UP_NUM_FILTERS)
            up_layer_nums = self.model_cfg.UpSample_LAYER_NUM
            # up_layer_strides = self.model_cfg.UPSAMPLE_STRIDES
            up_num_filters = self.model_cfg.UP_NUM_FILTERS
            cin = down_num_filters[len(down_num_filters)-1]
            up_c_in_list = [cin, *up_num_filters]
        else:
            up_layer_nums = up_layer_strides = up_num_filters = []


        down_num_levels = len(down_layer_nums)
        up_num_levels = len(up_layer_nums)
        self.blocks = nn.ModuleList()
        self.upblocks = nn.ModuleList()

        for idx in range(down_num_levels):

            cin = down_c_in_list[idx]
            cout = down_c_in_list[idx+1]
            # kernel = down_layer_strides[idx]
            if idx < 2:
                cur_layers = [
                    nn.Conv2d(cin, cout, kernel_size=(1,1)),
                    DRB(cout, cout),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    
                ]
            else:
                cur_layers = [
                    nn.Conv2d(cin, cout, kernel_size=(1,1)),
                    DRB(cout, cout),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.MaxPool2d((2,2), stride=2),
                    
                ]

            self.blocks.append(nn.Sequential(*cur_layers))

        for idx in range(up_num_levels):

            cin = up_c_in_list[idx]
            cout = up_c_in_list[idx+1]

            cur_layers = [
                nn.Conv2d(2*cin,cout,kernel_size=(1,1),stride=1),
                DRB(cout, cout),
                nn.BatchNorm2d(cout, eps=1e-3, momentum=0.01),
                nn.ReLU(),
                # nn.Dropout2d(p=0.2)
            ]

            self.upblocks.append(nn.Sequential(*cur_layers))

    def forward(self, batch_dict):

        # pcdet里是字典，batch_dict['rangeview']
        data = batch_dict['range_in']
        down_out = []
        up_out = []

        # downsample
        x = data
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            down_out.append(x)
        '''
        batch_dict.update({
            'range_down_f': {
                'x_dn1': down_out[0],
                'x_dn2': down_out[1],
                'x_dn3': down_out[2],
                'x_dn4': down_out[3],
                'x_dn5': down_out[4],
                'x_dn6': down_out[5],
            }
        })
        '''

        # upsample
        index = len(down_out) - 1
        in_f = down_out[index]

        for i in range(len(self.upblocks)):
            up_in = nn.functional.interpolate(in_f, scale_factor=(2,2), mode='bilinear', align_corners=True) # 双线性插值上采样
            index = index-1
            drb_in = torch.cat((up_in,down_out[index]), 1)
            drb_in = self.upblocks[i](drb_in)
            in_f = drb_in
            up_out.append(drb_in)
        '''
        batch_dict.update({
            'range_up_f': {
                'x_up1': up_out[0],
                'x_up2': up_out[1],
                'x_up3': up_out[2],
                'x_up4': up_out[3],
            }
        })
        '''

        '''
        ？ 输出：上采样每一曾继续上采样，和点拼接
        '''

        
        batch_dict['range_res'] = up_out[3]

        # bev = gen_bev_map(batch_dict)

        return batch_dict