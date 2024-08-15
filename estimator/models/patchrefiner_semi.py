
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
import kornia
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mmengine import print_log

from estimator.registry import MODELS
from estimator.models import build_model
from estimator.models.utils import HookTool
import matplotlib.pyplot as plt

from zoedepth.models.zoedepth import ZoeDepth

@MODELS.register_module()
class PatchRefinerSemi(nn.Module):
    def __init__(
        self, 
        model_cfg_student,
        teacher_pretrain=None,
        sigloss=None,
        edgeloss=None,
        model_cfg_teacher=None,
        edge_loss_weight=1,
        edge_thr=0.08,
        mix_loss=False,
        ranking_weight=0.1,
        ssi_weight=0.1,
        edgeloss_ranking=None,
        edgeloss_ssi=None,
        distill=False,
        distill_loss_weight=1,
        distill_loss=None,
        last_feat=True,
        **kwargs):
        """ZoeDepth model
        """
        super().__init__()
        
        self.edge_loss_weight = edge_loss_weight
        self.mix_loss = mix_loss
        self.edge_thr = edge_thr
        if self.mix_loss:
            self.edgeloss_cfg = copy.deepcopy(edgeloss_ranking)
            self.edgeloss_cfg.type = '' # hack
            self.edgeloss_ranking = build_model(edgeloss_ranking)
            self.edgeloss_ssi = build_model(edgeloss_ssi)
            self.ranking_weight = ranking_weight
            self.ssi_weight = ssi_weight
        else:
            self.edgeloss_cfg = edgeloss
            self.edgeloss = build_model(edgeloss)
            
            
        self.model_cfg_teacher = model_cfg_teacher
        if self.model_cfg_teacher is not None:
            self.teacher_model = build_model(model_cfg_teacher)
            if teacher_pretrain is not None:
                print_log("Loading teacher_model's CKP from {}".format(teacher_pretrain), logger='current')
                self.teacher_model.load_dict(torch.load(teacher_pretrain)['model_state_dict'])
            for param in self.teacher_model.parameters():
                param.requires_grad = False
        
        self.student_model = build_model(model_cfg_student)
        
        self.distill = distill
        self.last_feat = last_feat
        if self.distill is True:
            self.feat_stu_hook = HookTool()
            self.feat_tea_hook = HookTool()
            if self.last_feat is True:
                self.student_model.refiner_fusion_model.final_conv.register_forward_hook(self.feat_stu_hook.hook_in_fun)
                self.teacher_model.refiner_fusion_model.final_conv.register_forward_hook(self.feat_tea_hook.hook_in_fun)
            else:
                self.student_model.refiner_fusion_model.decoder_layers[0].conv.double_conv[0].register_forward_hook(self.feat_stu_hook.hook_in_fun)
                self.teacher_model.refiner_fusion_model.decoder_layers[0].conv.double_conv[0].register_forward_hook(self.feat_tea_hook.hook_in_fun)
            self.distill_loss_weight = distill_loss_weight
            self.distill_loss = build_model(distill_loss)
    
    def load_dict(self, dict):
        if 'student_model.coarse_branch.core.core.pretrained.model.cls_token' in dict.keys():
            print_log('loading from an old ckp (with both teacher and student)', logger='current')
            return self.load_state_dict(dict, strict=True)
        else:
            print_log('loading from an new ckp (only with student)', logger='current')
            return self.student_model.load_state_dict(dict, strict=False)
        
    def get_save_dict(self):
        model_state_dict = {}
        model_state_dict.update(self.student_model.get_save_dict())
        return model_state_dict 
    
    def forward(
        self,
        mode=None,
        image_lr=None,
        image_hr=None,
        crops_image_hr=None,
        depth_gt=None,
        crop_depths=None,
        bboxs=None,
        image_mid_tensor=None,
        center_mask=None,
        pseudo_label=None,
        camera_info=None,
        pseudo_uncert=None,
        cai_mode='m1',
        **kwargs):
        
        if mode == 'train':
            if self.model_cfg_teacher is not None:
                if self.teacher_model.training:
                    self.teacher_model.eval()
            
                loss_dict_t, output_dict_t = \
                    self.teacher_model(mode=mode, image_lr=image_lr, image_hr=image_hr, crops_image_hr=crops_image_hr, depth_gt=depth_gt, crop_depths=crop_depths, bboxs=bboxs, image_mid_tensor=image_mid_tensor)
                    
                _, pseudo_label, _ = output_dict_t['rgb'], output_dict_t['depth_pred'], output_dict_t['depth_gt']

            else:
                # sometimes we have saved the pseudo label!
                pass
                
            loss_dict_s, output_dict_s = \
                self.student_model(mode=mode, image_lr=image_lr, image_hr=image_hr, crops_image_hr=crops_image_hr, depth_gt=depth_gt, crop_depths=crop_depths, bboxs=bboxs, image_mid_tensor=image_mid_tensor)
            
            _, prediction_s, _ = output_dict_s['rgb'], output_dict_s['depth_pred'], output_dict_s['depth_gt']
                
            output_dict_s['pseudo_gt'] = pseudo_label.detach().cpu().float().numpy()
            
            if self.edgeloss_cfg.type == 'ScaleAndShiftInvariantLoss': # midas implementation
                mask = torch.ones_like(pseudo_label).bool()
                edge_loss = self.edgeloss(prediction_s, pseudo_label, crop_depths, mask, self.student_model.min_depth, self.student_model.max_depth)
            elif self.edgeloss_cfg.type == 'ScaleAndShiftInvariantDALoss': # depth-anything implementation
                mask = torch.ones_like(pseudo_label).bool()
                edge_loss = self.edgeloss(prediction_s, pseudo_label, crop_depths, mask, self.student_model.min_depth, self.student_model.max_depth)
            elif self.edgeloss_cfg.type == 'ScaleAndShiftInvariantUncertLoss': # consider uncertainty!
                mask = torch.ones_like(pseudo_label).bool()
                edge_loss = self.edgeloss(prediction_s, pseudo_label, crop_depths, mask, self.student_model.min_depth, self.student_model.max_depth, pseudo_uncert)
            elif self.edgeloss_cfg.type == 'EdgeguidedRankingLoss': # ranking
                edge_loss, sample_num = self.edgeloss(prediction_s, pseudo_label, crops_image_hr, crop_depths)
                loss_dict_s['sample_num'] = torch.tensor(sample_num)
            # elif self.mix_loss: # ECCV Version
            #     mask = torch.ones_like(pseudo_label).bool()
            #     edge_loss_ranking, sample_num = self.edgeloss_ranking(prediction_s, pseudo_label, crops_image_hr, crop_depths)
            #     loss_dict_s['sample_num'] = torch.tensor(sample_num)
            #     edge_loss_ssi = self.edgeloss_ssi(prediction_s, pseudo_label, crop_depths, mask, self.student_model.min_depth, self.student_model.max_depth)
            #     edge_loss = self.ranking_weight * edge_loss_ranking + self.ssi_weight * edge_loss_ssi
            elif self.edgeloss_cfg.type == 'SILogLoss': # not good
                edge_loss = self.edgeloss(prediction_s, pseudo_label, min_depth=self.student_model.min_depth, max_depth=self.student_model.max_depth)
            else:
                raise NotImplementedError
            

            # avoid nan/inf
            loss_dict_s['edge_loss'] = edge_loss
            if torch.isnan(edge_loss) or torch.isinf(edge_loss):
                print_log("nan/inf edge loss", logger='current')
                edge_loss = torch.DoubleTensor([0.0]).cuda() * prediction_s[0, 0, 0, 0]
            if torch.isnan(loss_dict_s['total_loss']) or torch.isinf(loss_dict_s['total_loss']):
                print_log("nan/inf loss_dict_s['total_loss']", logger='current')
                loss_dict_s['total_loss'] = torch.DoubleTensor([0.0]).cuda() * prediction_s[0, 0, 0, 0]
            loss_dict_s['total_loss'] = loss_dict_s['total_loss'] + self.edge_loss_weight * edge_loss            
            
            
            # distill part
            if self.distill:
                distill_loss = self.distill_loss(self.feat_stu_hook.feat[0], self.feat_tea_hook.feat[0], crop_depths, self.student_model.min_depth, self.student_model.max_depth)
                loss_dict_s['distill_loss'] = distill_loss
                distill_weighted_loss = distill_loss * self.distill_loss_weight
                loss_dict_s['total_loss'] += distill_weighted_loss

            return loss_dict_s, output_dict_s
        
        else:
            # return self.teacher_model(mode=mode, image_lr=image_lr, image_hr=image_hr, crops_image_hr=crops_image_hr, depth_gt=depth_gt, crop_depths=crop_depths, bboxs=bboxs, image_mid_tensor=image_mid_tensor)
            return self.student_model(mode=mode, image_lr=image_lr, image_hr=image_hr, crops_image_hr=crops_image_hr, depth_gt=depth_gt, crop_depths=crop_depths, bboxs=bboxs, image_mid_tensor=image_mid_tensor, cai_mode=cai_mode)