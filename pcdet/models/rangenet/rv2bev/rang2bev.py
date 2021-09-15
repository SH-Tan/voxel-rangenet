"""
to generate bird eye view, we have to filter point cloud
first. which means we have to limit coordinates


"""
import numpy as np

import os
import torch
import sys
import torch.nn as nn

if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
  sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

# POINT_CLOUD_RANGE: [0, -40, -3, 70.4, 40, 1] res: 0.1  800*704  stride: 4 200*176

#[0,69.12] [-39.68,39.68]  bev : 496*432 y*x

# image size would be 800*704


class RANGE2BEV(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg  # range, res
        self.x_range = self.model_cfg['X_RANGE']
        self.y_range = self.model_cfg['Y_RANGE']
        self.z_range = self.model_cfg['Z_RANGE']
        self.resolution = self.model_cfg['RESOLUTION']

    def forward(self,batch_dict):

        bc = batch_dict['batch_size']
        range_up = batch_dict['range_res']
        b,c,_,_ = range_up.shape

        w = int((self.y_range[1] - self.y_range[0])/self.resolution) # 800
        h = int((self.x_range[1] - self.x_range[0])/self.resolution) # 704

        _,c2,_ = batch_dict['range_scan'].shape
        c = c + c2
        bev = torch.Tensor(bc,h,w,c) # list
        p_y = batch_dict['range_y'].long()
        p_x = batch_dict['range_x'].long()

        for i in range(bc):
            scan = batch_dict['range_scan'][i].permute(1,0).contiguous()

            range_f = range_up[i,:,p_y[i],p_x[i]].permute(1,0).contiguous()

            point = torch.cat((scan,range_f),1) # N*C  N*(64+4)

            # res = res.permute(1,0).contiguous()

            # point = res.permute(1,0).contiguous()
            # p = point.cpu().detach().numpy()

            x = point[:, 0]
            y = point[:, 1]
            z = point[:, 2]

            im = torch.zeros((h, w, c), dtype=torch.float32)

            '''
            # filter point cloud
            f_filt = np.logical_and((x>self.x_range[0]), (x<self.x_range[1]))
            s_filt = np.logical_and((y>-self.y_range[1]), (y<-self.y_range[0]))
            filt = np.logical_and(f_filt, s_filt)
            indices = np.argwhere(filt).flatten()

            x = x[indices]
            y = y[indices]
            z = z[indices]
            point = p[indices]

            print(point.shape)
            print(p.shape)
            input()
            '''

            # convert coordinates to 
            x_img = (-y/self.resolution).type(torch.long)
            y_img = (-x/self.resolution).type(torch.long)
            # shifting image, make min pixel is 0,0
            x_img -= int(np.floor(self.y_range[0]/self.resolution))
            y_img += int(np.ceil(self.x_range[1]/self.resolution))

            x_max = int((self.y_range[1]-self.y_range[0])/self.resolution-1) # 799
            y_max = int((self.x_range[1]-self.x_range[0])/self.resolution-1) # 703

            x_img_c = torch.clamp(x_img, 0, x_max)
            y_img_c = torch.clamp(y_img, 0, y_max)



            # crop z to make it not bigger than 255
            height_range = self.z_range
            z_c = torch.clamp(z, height_range[0], height_range[1])

            '''
            def scale_to_255(a, min, max, dtype=np.uint8):

                return (((a - min) / float(max - min)) * 255).astype(dtype)

            pixel_values = scale_to_255(pixel_values, min=height_range[0], max=height_range[1])
            '''

            # according to width and height generate image
            z_c = z_c.reshape(-1,1)
            
            if point.is_cuda:
                im = im.cuda()
                bev = bev.cuda()
                # z_c = z_c.cuda()


            im[y_img_c, x_img_c] = z_c*point

            bev[i] = im

        bev = bev.permute(0,3,2,1).contiguous() # B*C*H*W B*68*800*704

        # print(bev.shape)

        batch_dict['range_bev'] =  bev

        return batch_dict
