from .detector3d_template import Detector3DTemplate
from .detector3d_template_multi_db import Detector3DTemplate_M_DB
from pcdet.utils import common_utils
import numpy as np
import torch

class ADPT(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()


    def roi(self, batch_dict, model_cfg): 
        
        features = batch_dict['spatial_features_2d'].detach()
        rois = batch_dict['gt_boxes']
        # once dataset cfg
        min_x = -75.2
        min_y = -75.2
        voxel_size_x = 0.1
        voxel_size_y = 0.1
        down_sample_ratio = 8

        batch_size, channels, height, width = features.shape
        roi_size = rois.size(1)

        x1 = (rois[:, :, 0] - rois[:, :, 3] / 2 - min_x) / (voxel_size_x * down_sample_ratio)
        x2 = (rois[:, :, 0] + rois[:, :, 3] / 2 - min_x) / (voxel_size_x * down_sample_ratio)
        y1 = (rois[:, :, 1] - rois[:, :, 4] / 2 - min_y) / (voxel_size_y * down_sample_ratio)
        y2 = (rois[:, :, 1] + rois[:, :, 4] / 2 - min_y) / (voxel_size_y * down_sample_ratio)

        mask = torch.zeros(batch_size, roi_size, height, width).bool().cuda()
        grid_y, grid_x = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
        grid_y = grid_y[None, None].repeat(batch_size, roi_size, 1, 1).cuda()
        grid_x = grid_x[None, None].repeat(batch_size, roi_size, 1, 1).cuda()

        mask_y = (grid_y >= y1[:, :, None, None]) * (grid_y <= y2[:, :, None, None])
        mask_x = (grid_x >= x1[:, :, None, None]) * (grid_x <= x2[:, :, None, None])
        mask = (mask_y * mask_x).float()
        roi_features = []
        for k in range(roi_size): 
            _, _, i, j = torch.where(mask[:, k, :, :].unsqueeze(1))
            feature = features[:, :, i, j] 
            roi_features.append(feature)
        if len(roi_features) == 0: 
            return roi_features
        roi_features  = torch.cat(roi_features, dim = -1).permute(0, 2, 1)
        B, N, C = roi_features.shape
        roi_features = roi_features.contiguous().view(B*N, C)

        return roi_features

    def forward(self, batch_dict):
        batch_dict = self.vfe(batch_dict)
        if self.unetscn:
            batch_dict = self.unetscn(batch_dict)
        batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.map_to_bev_module(batch_dict)
        batch_dict = self.backbone_2d(batch_dict)
        rois = self.roi(batch_dict, self.model_cfg)
        # if self.training:
        #     loss, tb_dict, disp_dict = self.get_training_loss()

        #     ret_dict = {
        #         'loss': loss
        #     }
        #     return ret_dict, tb_dict, disp_dict
        # else:
        #     pred_dicts, recall_dicts = self.post_processing(batch_dict)
        return rois

    def get_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        if self.point_head is not None:
            loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        else:
            loss_point = 0
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_point + loss_rcnn
        return loss, tb_dict, disp_dict

