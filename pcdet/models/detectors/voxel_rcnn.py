from .detector3d_template import Detector3DTemplate
import torch


class VoxelRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):

        batch_dict['range_scan'] = batch_dict['range_scan'].permute(0,2,1).contiguous() # B*C*N   1*4*N
        batch_dict['range_xyz'] = batch_dict['range_xyz'].permute(0,3,1,2).contiguous()
        batch_dict['range'] = batch_dict['range'].unsqueeze(1).contiguous()
        batch_dict['range_r'] = batch_dict['range_r'].unsqueeze(1).contiguous()

        # print(batch_dict['range_xyz'].shape)
        # print(batch_dict['range'].shape)
        # print(batch_dict['range_r'].shape)
        # input()


        batch_dict.update({'range_in': torch.cat((batch_dict['range_xyz'],batch_dict['range'],batch_dict['range_r']),1)})

        del batch_dict['range_xyz']
        del batch_dict['range']
        del batch_dict['range_r']

        for cur_module in self.module_list:

            # print('='*100)
            # print(cur_module)
            
            batch_dict = cur_module(batch_dict)

            # print(batch_dict)
            # print('='*100)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss = 0
        
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss + loss_rpn + loss_rcnn
        return loss, tb_dict, disp_dict
