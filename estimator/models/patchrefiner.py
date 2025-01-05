
# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Zhenyu Li

import itertools

import math
import copy
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mmengine import print_log
from torchvision.ops import roi_align as torch_roi_align
from huggingface_hub import PyTorchModelHubMixin
from mmengine.config import ConfigDict
from transformers import PretrainedConfig

from estimator.registry import MODELS
from estimator.models import build_model
from estimator.models.utils import get_activation, generatemask, RunningAverageMap
from estimator.models.baseline_pretrain import BaselinePretrain
from estimator.models.utils import HookTool

from zoedepth.models.zoedepth import ZoeDepth
from zoedepth.models.base_models.midas import Resize as ResizeZoe
from depth_anything.transform import Resize as ResizeDA

@MODELS.register_module()
class PatchRefiner(BaselinePretrain, PyTorchModelHubMixin):
    def __init__(
        self,
        config):
        """ZoeDepth model
        """
        nn.Module.__init__(self)
        
        if isinstance(config, ConfigDict):
            # convert a ConfigDict to a PretrainedConfig for hf saving
            config = PretrainedConfig.from_dict(config.to_dict())
        else:
            # used when loading patchfusion from hf model space
            config = PretrainedConfig.from_dict(ConfigDict(**config).to_dict())
            config.base_depth_pretrain_model = None
        
        self.config = config
        self.min_depth = config.min_depth
        self.max_depth = config.max_depth

        self.patch_process_shape = config.patch_process_shape
        self.tile_cfg = self.prepare_tile_cfg(config.image_raw_shape, config.patch_split_num)
        
        # process coarse model
        if config.coarse_branch.type == 'ZoeDepth':
            self.coarse_branch = ZoeDepth.build(**config.coarse_branch)
            print_log("Current zoedepth.core.prep.resizer is {}".format(type(self.coarse_branch.core.prep.resizer)), logger='current')
            if config.pretrain_coarse_model is not None:
                print_log("Loading coarse_branch from {}".format(config.pretrain_coarse_model), logger='current')
                print_log(self.coarse_branch.load_state_dict(torch.load(config.pretrain_coarse_model, map_location='cpu')['model_state_dict'], strict=True), logger='current') # coarse ckp
            self.resizer = ResizeZoe(self.patch_process_shape[1], self.patch_process_shape[0], keep_aspect_ratio=False, ensure_multiple_of=32, resize_method="minimal")
            
        elif config.coarse_branch.type == 'DA-ZoeDepth':
            self.coarse_branch = ZoeDepth.build(**config.coarse_branch)
            print_log("Current zoedepth.core.prep.resizer is {}".format(type(self.coarse_branch.core.prep.resizer)), logger='current')
            if config.pretrain_coarse_model is not None:
                print_log("Loading coarse_branch from {}".format(config.pretrain_coarse_model), logger='current')
                print_log(self.coarse_branch.load_state_dict(torch.load(config.pretrain_coarse_model, map_location='cpu')['model_state_dict'], strict=True), logger='current') # coarse ckp
            self.resizer = ResizeDA(self.patch_process_shape[1], self.patch_process_shape[0], keep_aspect_ratio=False, ensure_multiple_of=14, resize_method="minimal")
            
        for param in self.coarse_branch.parameters():
            param.requires_grad = False
        
        # process fine model
        if config.refiner.fine_branch.type == 'ZoeDepth':
            self.refiner_fine_branch = ZoeDepth.build(**config.refiner.fine_branch)
            print_log("Current zoedepth.core.prep.resizer is {}".format(type(self.refiner_fine_branch.core.prep.resizer)), logger='current')
            print_log("Hacking refiner_fine_branch (copy weights from the coarse one)", logger='current')
            self.refiner_fine_branch.load_state_dict(self.coarse_branch.state_dict(), strict=True) # overload model
        elif config.refiner.fine_branch.type == 'DA-ZoeDepth':
            self.refiner_fine_branch = ZoeDepth.build(**config.refiner.fine_branch)
            print_log("Current zoedepth.core.prep.resizer is {}".format(type(self.refiner_fine_branch.core.prep.resizer)), logger='current')
            print_log("Hacking refiner_fine_branch (copy weights from the coarse one)", logger='current')
            self.refiner_fine_branch.load_state_dict(self.coarse_branch.state_dict(), strict=True) # overload model
        
        self.sigloss = build_model(config.sigloss) # here

        self.fusion_feat_level = config.fusion_feat_level
        self.refiner_fusion_model = build_model(config.refiner.fusion_model) # FusionUnet(input_chl=self.update_feat_chl, temp_chl=[32, 256, 256], dec_chl=[256, 32])
        self.strategy_refiner_target = config.strategy_refiner_target

        if config.pretrained is not None:
            pretrained_dict = torch.load(config.pretrained, map_location='cpu')['model_state_dict']
            updated_dict = {}
            for k, v in pretrained_dict.items():
                if config.load_whole:
                    # load_everthing
                    updated_dict[k] = v
                else:
                    # only load refiner
                    if 'coarse_branch' in k:
                        continue
                    else:
                        updated_dict[k] = v
            
            if config.load_whole:
                print_log("Loading the whole patchrefiner from {} (Overload everything)".format(config.pretrained), logger='current')
            else:
                print_log("Loading the refiner part in patchrefiner from {}".format(config.pretrained), logger='current')
            print_log(self.load_state_dict(updated_dict, strict=False), logger='current') # coarse ckp
        
        # pre-norm bbox: if setting it as True, no need to norm the bbox in roi_align during the forward pass in training stage
        # For test stage: we norm the bbox during the forward pass
        # Reason: we merge datasets to train the model, and we norm the bbox in the dataloader
        # No need to norm the bbox during the test stage (single domain)
        self.pre_norm_bbox = config.pre_norm_bbox # here
            
    def load_dict(self, dict):
        return self.load_state_dict(dict, strict=False)
        
    def get_save_dict(self):
        model_state_dict = {}
        default_dict = self.state_dict()
        for k, v in default_dict.items():
            if 'coarse_branch.' in k:
                continue
            else:
                model_state_dict[k] = v
        return model_state_dict
    
    def coarse_forward(self, image_lr):
        with torch.no_grad():
            if self.coarse_branch.training:
                self.coarse_branch.eval()
                    
            deep_model_output_dict = self.coarse_branch(image_lr, return_final_centers=True)
            deep_features = deep_model_output_dict['temp_features'] # x_d0 1/128, x_blocks_feat_0 1/64, x_blocks_feat_1 1/32, x_blocks_feat_2 1/16, x_blocks_feat_3 1/8, midas_final_feat 1/4 [based on 384x4, 512x4]
            coarse_prediction = deep_model_output_dict['metric_depth']
            
            coarse_features = [
                deep_features['x_d0'],
                deep_features['x_blocks_feat_0'],
                deep_features['x_blocks_feat_1'],
                deep_features['x_blocks_feat_2'],
                deep_features['x_blocks_feat_3'],
                deep_features['midas_final_feat']] # bs, c, h, w

            return coarse_features, coarse_prediction
    
    def coarse_postprocess_train(self, coarse_prediction, coarse_features, bboxs, bboxs_feat):
        coarse_features_patch_area = []
        for idx, feat in enumerate(coarse_features):
            bs, _, h, w = feat.shape
            cur_lvl_feat = torch_roi_align(feat, bboxs_feat, (h, w), h/self.patch_process_shape[0], aligned=True)
            coarse_features_patch_area.append(cur_lvl_feat)
        
        coarse_prediction_roi = torch_roi_align(coarse_prediction, bboxs_feat, coarse_prediction.shape[-2:], coarse_prediction.shape[-2]/self.patch_process_shape[0], aligned=True)
        
        # lvl, -> bs, c, h, w; bs, 1, h, w
        return coarse_features_patch_area, coarse_prediction_roi
    
    def coarse_postprocess_test(self, coarse_prediction, coarse_features, bboxs, bboxs_feat):
        patch_num = bboxs_feat.shape[0]

        coarse_features_patch_area = []
        for idx, feat in enumerate(coarse_features):
            bs, _, h, w = feat.shape
            feat_extend = feat.repeat(patch_num, 1, 1, 1)
            cur_lvl_feat = torch_roi_align(feat_extend, bboxs_feat, (h, w), h/self.patch_process_shape[0], aligned=True)
            coarse_features_patch_area.append(cur_lvl_feat)
        
        coarse_prediction = coarse_prediction.repeat(patch_num, 1, 1, 1)
        coarse_prediction_roi = torch_roi_align(coarse_prediction, bboxs_feat, coarse_prediction.shape[-2:], coarse_prediction.shape[-2]/self.patch_process_shape[0], aligned=True)
        # coarse_prediction_roi = None
        
        return_dict = {
            'coarse_depth_roi': coarse_prediction_roi,
            'coarse_feats_roi': coarse_features_patch_area}
        
        return return_dict
        
    def refiner_fine_forward(self, image_hr):
        refiner_out_put_dict = self.refiner_fine_branch(image_hr, return_final_centers=True)

        refiner_features = [
            refiner_out_put_dict['temp_features']['x_d0'],
            refiner_out_put_dict['temp_features']['x_blocks_feat_0'],
            refiner_out_put_dict['temp_features']['x_blocks_feat_1'],
            refiner_out_put_dict['temp_features']['x_blocks_feat_2'],
            refiner_out_put_dict['temp_features']['x_blocks_feat_3'],
            refiner_out_put_dict['temp_features']['midas_final_feat']] # bs, c, h, w

        refiner_continous_depth = refiner_out_put_dict['metric_depth'] 
        
        return refiner_features, refiner_continous_depth
    
    def refiner_fusion_forward(
        self, 
        coarse_features_patch, 
        coarse_predicton_patch, 
        refiner_features, 
        refiner_prediction, 
        update_base=None):

        c_feat_list = []
        r_feat_list = []

        for idx, (c_feat, r_feat) in enumerate(zip(coarse_features_patch[-self.fusion_feat_level:], refiner_features[-self.fusion_feat_level:])):
            c_feat_list.append(c_feat)
            r_feat_list.append(r_feat)
        
        offset_pred = self.refiner_fusion_model(
            c_feat=c_feat_list[::-1], 
            f_feat=r_feat_list[::-1], 
            pred1=coarse_predicton_patch,
            pred2=refiner_prediction,
            update_base=update_base)
        
        return offset_pred
    
    def infer_forward(self, imgs_crop, bbox_feat_forward, tile_temp, coarse_temp_dict):
        # tile_temp = {
        #     'coarse_prediction': coarse_prediction,
        #     'coarse_features': coarse_features,}
        
        # coarse_temp_dict = {
        #     'coarse_depth_roi': coarse_prediction_roi,
        #     'coarse_feats_roi': coarse_features_patch_area}
        
        refiner_features, refiner_continous_depth = self.refiner_fine_forward(imgs_crop)
        # return refiner_continous_depth
        # return coarse_temp_dict['coarse_depth_roi']
        
        # update
        if self.strategy_refiner_target == 'offset_fine':
            update_base = refiner_continous_depth
        elif self.strategy_refiner_target == 'offset_coarse':
            update_base = coarse_temp_dict['coarse_depth_roi']
        else:
            update_base = None
            
        depth_prediction = self.refiner_fusion_forward(coarse_temp_dict['coarse_feats_roi'], coarse_temp_dict['coarse_depth_roi'], refiner_features, refiner_continous_depth, update_base=update_base)
        if self.strategy_refiner_target == 'direct':
            depth_prediction = F.sigmoid(depth_prediction) * self.max_depth
  
        return depth_prediction
    
    
    def forward(
        self,
        mode=None,
        image_lr=None,
        image_hr=None,
        crops_image_hr=None,
        depth_gt=None,
        crop_depths=None,
        bboxs=None,
        tile_cfg=None,
        cai_mode='m1',
        process_num=4,
        **kwargs):
        
        if mode == 'train':
            if self.pre_norm_bbox:
                bboxs_feat = bboxs
            else:
                bboxs_feat_factor = torch.tensor([
                    1 / self.tile_cfg['image_raw_shape'][1] * self.patch_process_shape[1], 
                    1 / self.tile_cfg['image_raw_shape'][0] * self.patch_process_shape[0], 
                    1 / self.tile_cfg['image_raw_shape'][1] * self.patch_process_shape[1], 
                    1 / self.tile_cfg['image_raw_shape'][0] * self.patch_process_shape[0]], device=bboxs.device).unsqueeze(dim=0)
                bboxs_feat = bboxs * bboxs_feat_factor
            inds = torch.arange(bboxs.shape[0]).to(bboxs.device).unsqueeze(dim=-1)
            bboxs_feat = torch.cat((inds, bboxs_feat), dim=-1)
            
            # all of them are at whole-image level
            coarse_features, coarse_prediction = self.coarse_forward(image_lr) 
            coarse_features_patch, coarse_prediction_roi = self.coarse_postprocess_train(coarse_prediction, coarse_features, bboxs, bboxs_feat)
            # patch refiner features 
            refiner_features, refiner_continous_depth = self.refiner_fine_forward(crops_image_hr)

            # update
            if self.strategy_refiner_target == 'offset_fine':
                update_base = refiner_continous_depth
            elif self.strategy_refiner_target == 'offset_coarse':
                update_base = coarse_prediction_roi
            else:
                update_base = None
                
            depth_prediction = self.refiner_fusion_forward(coarse_features_patch, coarse_prediction_roi, refiner_features, refiner_continous_depth, update_base=update_base)
            if self.strategy_refiner_target == 'direct':
                depth_prediction = F.sigmoid(depth_prediction) * self.max_depth

            loss_dict = {}
            
            sig_loss = self.sigloss(depth_prediction, crop_depths, self.min_depth, self.max_depth)
            
            loss_dict['sig_loss'] = sig_loss
            loss_dict['total_loss'] = sig_loss
            
            return loss_dict, {'rgb': crops_image_hr, 'depth_pred': depth_prediction, 'depth_gt': crop_depths}
            
            
        else:
            
            if tile_cfg is None:
                tile_cfg = self.tile_cfg
            else:
                tile_cfg = self.prepare_tile_cfg(tile_cfg['image_raw_shape'], tile_cfg['patch_split_num'])
            
            assert image_hr.shape[0] == 1
            
            coarse_features, coarse_prediction = self.coarse_forward(image_lr) 
            # return coarse_prediction, {'rgb': image_lr, 'depth_pred': coarse_prediction, 'depth_gt': depth_gt}
            
            tile_temp = {
                'coarse_prediction': coarse_prediction,
                'coarse_features': coarse_features,}

            blur_mask = generatemask((self.patch_process_shape[0], self.patch_process_shape[1]), border=0.15)
            blur_mask = torch.tensor(blur_mask, device=image_hr.device)
            avg_depth_map = self.regular_tile(
                offset=[0, 0], 
                offset_process=[0, 0], 
                image_hr=image_hr[0], 
                init_flag=True, 
                tile_temp=tile_temp, 
                blur_mask=blur_mask,
                tile_cfg=tile_cfg,
                process_num=process_num)

            if cai_mode == 'm2' or cai_mode[0] == 'r':
                avg_depth_map = self.regular_tile(
                    offset=[0, tile_cfg['patch_raw_shape'][1]//2], 
                    offset_process=[0, self.patch_process_shape[1]//2], 
                    image_hr=image_hr[0], init_flag=False, tile_temp=tile_temp, blur_mask=blur_mask, avg_depth_map=avg_depth_map, tile_cfg=tile_cfg, process_num=process_num)
                avg_depth_map = self.regular_tile(
                    offset=[tile_cfg['patch_raw_shape'][0]//2, 0],
                    offset_process=[self.patch_process_shape[0]//2, 0], 
                    image_hr=image_hr[0], init_flag=False, tile_temp=tile_temp, blur_mask=blur_mask, avg_depth_map=avg_depth_map, tile_cfg=tile_cfg, process_num=process_num)
                avg_depth_map = self.regular_tile(
                    offset=[tile_cfg['patch_raw_shape'][0]//2, tile_cfg['patch_raw_shape'][1]//2],
                    offset_process=[self.patch_process_shape[0]//2, self.patch_process_shape[1]//2], 
                    init_flag=False, image_hr=image_hr[0], tile_temp=tile_temp, blur_mask=blur_mask, avg_depth_map=avg_depth_map, tile_cfg=tile_cfg, process_num=process_num)
                
            if cai_mode[0] == 'r':
                blur_mask = generatemask((tile_cfg['patch_raw_shape'][0], tile_cfg['patch_raw_shape'][1]), border=0.15) + 1e-3
                blur_mask = torch.tensor(blur_mask, device=image_hr.device)
                avg_depth_map.resize(tile_cfg['image_raw_shape'])
                patch_num = int(cai_mode[1:]) // process_num
                for i in range(patch_num):
                    avg_depth_map = self.random_tile(
                        image_hr=image_hr[0], tile_temp=tile_temp, blur_mask=blur_mask, avg_depth_map=avg_depth_map, tile_cfg=tile_cfg, process_num=process_num)

            # depth = avg_depth_map.average_map
            depth = avg_depth_map.get_avg_map()
            depth = depth.unsqueeze(dim=0).unsqueeze(dim=0)

            return depth, \
                {'rgb': image_lr, 
                 'depth_pred': depth, 
                 'depth_gt': depth_gt, }