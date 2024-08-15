import copy
import kornia

import torch
import torch.nn as nn
from mmengine import print_log
import torch.nn.functional as F
import random
import math

from estimator.registry import MODELS
from kornia.losses import dice_loss, focal_loss

import numpy as np
from estimator.utils import RandomBBoxQueries
import kornia
from estimator.utils import get_boundaries, compute_metrics, compute_boundary_metrics, extract_edges
from pytorch3d.loss import chamfer_distance


@MODELS.register_module()
class ExistLoss(nn.Module):
    """ExistLoss loss (pixel-wise)"""
    def __init__(self, reweight_target):
        super(ExistLoss, self).__init__()
        self.name = 'ExistLoss'
        self.reweight_target = reweight_target


    def forward(self, pred_grad, pl_grad, pseudo_edge_area):
        
        pred_grad_edge = pred_grad[pseudo_edge_area]
        pl_grad_edge = pl_grad[pseudo_edge_area]
        pl_grad_weight = torch.exp(pl_grad_edge)
        
        if self.reweight_target:
            loss = torch.exp(-pred_grad_edge / pl_grad_weight).mean()
        else:
            loss = torch.exp(-pred_grad_edge).mean()
        return loss

@MODELS.register_module()
class GeometryAwareDetailEnhancementLoss(nn.Module):
    def __init__(self,
                 region_num=200,
                 inv_projection=True, # ablation: if inv-projecting back to 3D
                 whole_region=True, #default no changes
                 window_size=[3, 7, 15, 31, 63], 
                 gamma_window=0.3,
                 process_h=256,
                 process_w=512,
                 loss_type='l1',
                 region=True, # ablation: if use region proposals
                 **kargs):
        super().__init__()
        self.name = "GADELoss"
        
        # NOTE: for ablation
        self.region_num = region_num
        self.inv_projection = inv_projection
        self.whole_region = whole_region
        self.region = region
        
        self.window_size = window_size 
        self.gamma_window = gamma_window
        self.anchor_generator = RandomBBoxQueries(4, process_h, process_w, self.window_size, N=region_num)
        self.loss_type = loss_type
        
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"region_num='{self.region_num}',"
        repr_str += f"inv_projection='{self.inv_projection}',"
        repr_str += f"whole_region='{self.whole_region}',"
        repr_str += f"window_size='{self.window_size}',"
        repr_str += f"gamma_window='{self.gamma_window}')"
        return repr_str
    
    def forward(self, prediction, pseudo_label, gt_depth, min_depth, max_depth, camera_info, bboxs):
        self.anchor_generator.to(prediction.device)
        
        # prepare
        bs, _, h_i, w_i = prediction.shape
        _, _, h_t, w_t = pseudo_label.shape
        _, _, h_g, w_g = gt_depth.shape
        
        if h_i != h_t or w_i != w_t:
            prediction = F.interpolate(prediction, (h_t, w_t), mode='bilinear', align_corners=True)
        if h_g != h_t or w_g != w_t:
            gt_depth = F.interpolate(gt_depth, (h_t, w_t), mode='bilinear', align_corners=True)

        missing_mask = gt_depth == 0.
        missing_mask_extend = kornia.filters.gaussian_blur2d(missing_mask.float(), kernel_size=(7, 7), sigma=(5., 5.), border_type='reflect', separable=True)
        missing_mask_extend = missing_mask_extend > 0
        missing_mask_extend = missing_mask_extend.squeeze()
        
        prediction, pseudo_label, gt_depth = prediction.squeeze(), pseudo_label.squeeze(), gt_depth.squeeze()
        
        valid_mask = torch.logical_and(gt_depth>min_depth, gt_depth<max_depth)
        sampling_mask = torch.logical_and(valid_mask, missing_mask_extend)
        
        if torch.sum(sampling_mask) <= 1:
            print_log("torch.sum(sampling_mask) <= 1, skip due to none sampling points", logger='current')
            return prediction * 0.0
        
        assert prediction.shape == pseudo_label.shape, f"Shape mismatch: Expected same shape but got {prediction.shape} and {pseudo_label.shape}."

        # scale, shift = compute_scale_and_shift(pseudo_label, gt_depth, valid_mask) # compute scale and shift with valid_value mask
        # pseudo_label = scale.view(-1, 1, 1) * pseudo_label + shift.view(-1, 1, 1) # scaled preditcion aligned with target
        scale, shift = compute_scale_and_shift(prediction, pseudo_label, torch.ones_like(pseudo_label).bool())
        prediction = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        # get 3D points
        fx = camera_info[:, 0]
        fy = camera_info[:, 1]
        u0 = camera_info[:, 2] - bboxs[:, 0]
        v0 = camera_info[:, 3] - bboxs[:, 1]
        
        intrinsics = torch.zeros((bs, 4, 4))
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 0, 2] = u0
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 1, 2] = v0
        intrinsics[:, 2, 2] = 1.
        intrinsics[:, 3, 3] = 1.

        pesudo_edge_list = []
        for idx in range(bs):
            pesudo_edge = torch.from_numpy(extract_edges(pseudo_label[idx].detach().cpu(), use_canny=True, preprocess='log')).cuda()
            pesudo_edge_list.append(pesudo_edge)
        pesudo_edges = torch.stack(pesudo_edge_list, dim=0).unsqueeze(dim=1)
        pesudo_edges_extend = kornia.filters.gaussian_blur2d(pesudo_edges.float(), kernel_size=(7, 7), sigma=(5., 5.), border_type='reflect', separable=True)
        pesudo_edges = pesudo_edges_extend > 0
        
        h, w = h_t, w_t
        
        # inv + process windows
        if self.inv_projection:
            meshgrid = np.meshgrid(range(w), range(h), indexing='xy')
            id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
            id_coords = pseudo_label.new_tensor(id_coords)
            pix_coords = torch.cat([id_coords[0].view(-1).unsqueeze(dim=0), id_coords[1].view(-1).unsqueeze(dim=0)], 0)
            ones = torch.ones(1, w * h, device=pseudo_label.device)
            pix_coords = torch.cat([pix_coords, ones], dim=0) # 3xHW

            inv_K_list = []
            for idx, intrinsics_bs in enumerate(intrinsics):
                inv_K = torch.linalg.inv(intrinsics_bs)
                inv_K_list.append(inv_K)
            
            inv_K = torch.stack(inv_K_list, dim=0).cuda()
            pix_coords = torch.stack([pix_coords] * len(inv_K_list), dim=0)
            cam_points = torch.bmm(inv_K[:, :3, :3], pix_coords)
            
            pc_pseudo_label = torch.einsum('bcn,bn->bcn', cam_points, pseudo_label.flatten(-2))
            pc_prediction = torch.einsum('bcn,bn->bcn', cam_points, prediction.flatten(-2))
            
            pc_prediction = pc_prediction.unsqueeze(dim=1).expand(-1, self.region_num, -1, -1).flatten(3)  # .shape B, N, 3, HxW
            pc_pseudo_label = pc_pseudo_label.unsqueeze(dim=1).expand(-1, self.region_num, -1, -1).flatten(3)  # .shape B, N, 3, HxW
        
        else:
            pc_prediction = prediction.unsqueeze(dim=1).expand(-1, self.region_num, -1, -1).flatten(-2)
            pc_pseudo_label = pseudo_label.unsqueeze(dim=1).expand(-1, self.region_num, -1, -1).flatten(-2)
            
        sampling_mask = sampling_mask.unsqueeze(dim=1).expand(-1, self.region_num, -1, -1).flatten(2)
        pesudo_edges = pesudo_edges.expand(-1, self.region_num, -1, -1).flatten(2)  # .shape B, N, HxW

        if self.region is not True:
            if self.loss_type == 'l1':
                if self.inv_projection:
                    loss = nn.functional.l1_loss(pc_prediction, pc_pseudo_label)
                    return loss
                else:
                    raise NotImplementedError # degrade to a simple ssi-midas loss
            else:
                raise NotImplementedError
            
        else:
            # NOTE: start point
            loss = torch.DoubleTensor([0.0]).cuda()
            w_window = 1.0
            w_window_sum = 0.0
            
            for idx, win_size in enumerate(self.window_size):
                if idx > 0 :
                    w_window = w_window * self.gamma_window
                
                abs_coords = self.anchor_generator.absolute[win_size]  # B, N, 2; babs[b,n] = [x,y]
                B, N, _two = abs_coords.shape
                k = win_size // 2
                x = torch.arange(-k, k+1)
                y = torch.arange(-k, k+1)
                Y, X = torch.meshgrid(y, x)
                base_coords = torch.stack((X, Y), dim=0)[None, None,...].to(prediction.device)  # .shape 1, 1, 2, k, k
                
                coords = abs_coords[...,None,None] + base_coords  # shape B, N, 2, k, k
                
                x = coords[:,:,0,:,:]
                y = coords[:,:,1,:,:]
                flatten_indices = y * w + x  # .shape B, N, k, k
                
                flatten_flatten_indices = flatten_indices.flatten(2)  # .shape B, N, kxk

                # .shape B, N, kxk
                sampling_mask_sample = torch.gather(sampling_mask, dim=-1, index=flatten_flatten_indices.long())
                pesudo_edges_sample = torch.gather(pesudo_edges, dim=-1, index=flatten_flatten_indices.long())
                if self.inv_projection:
                    pc_prediction_sample = torch.gather(pc_prediction, dim=-1, index=flatten_flatten_indices.long().unsqueeze(dim=-2).repeat(1, 1, 3, 1))
                    pc_pseudo_label_sample = torch.gather(pc_pseudo_label, dim=-1, index=flatten_flatten_indices.long().unsqueeze(dim=-2).repeat(1, 1, 3, 1))
                else:
                    pc_prediction_sample = torch.gather(pc_prediction, dim=-1, index=flatten_flatten_indices.long())
                    pc_pseudo_label_sample = torch.gather(pc_pseudo_label, dim=-1, index=flatten_flatten_indices.long())
                
                # merge N boxes into batch
                sampling_mask_sample = sampling_mask_sample.flatten(0, 1) 
                pesudo_edges_sample = pesudo_edges_sample.flatten(0, 1) 
                pc_prediction_sample = pc_prediction_sample.flatten(0, 1) 
                pc_pseudo_label_sample = pc_pseudo_label_sample.flatten(0, 1) 
                pc_pseudo_label_len = sampling_mask_sample.float().sum(dim=-1).long()
                
                if self.inv_projection:
                    if self.whole_region is not True:
                        # NOTE: sample points only in the sample-mask region
                        pc_pseudo_label_sample[~sampling_mask_sample.unsqueeze(dim=-2).repeat(1, 3, 1)] = float('inf')
                        values, indices = torch.sort(pc_pseudo_label_sample[:, 0, :], dim=-1)
                        expanded_indices = indices.unsqueeze(1).expand(-1, 3, -1)
                        pc_pseudo_label_sample = torch.gather(pc_pseudo_label_sample, 2, expanded_indices)
                        
                        # NOTE: sample points only in the sample-mask region
                        pc_prediction_sample[~sampling_mask_sample.unsqueeze(dim=-2).repeat(1, 3, 1)] = float('inf')
                        values, indices = torch.sort(pc_prediction_sample[:, 0, :], dim=-1)
                        expanded_indices = indices.unsqueeze(1).expand(-1, 3, -1)
                        pc_prediction_sample = torch.gather(pc_prediction_sample, 2, expanded_indices)
                else:
                    if self.whole_region is not True:
                        # NOTE: sample points only in the sample-mask region
                        pc_pseudo_label_sample[~sampling_mask_sample] = float('inf')
                        values, indices = torch.sort(pc_pseudo_label_sample, dim=-1)
                        pc_pseudo_label_sample = torch.gather(pc_pseudo_label_sample, 1, indices)
                        
                        # NOTE: sample points only in the sample-mask region
                        pc_prediction_sample[~sampling_mask_sample] = float('inf')
                        values, indices = torch.sort(pc_prediction_sample, dim=-1)
                        pc_prediction_sample = torch.gather(pc_prediction_sample, 1, indices)

                edge_portion = torch.sum(pesudo_edges_sample.long(), dim=-1) / float(pc_pseudo_label_sample.size(-1))
                value_portion = torch.sum(sampling_mask_sample.long(), dim=-1) / float(pc_pseudo_label_sample.size(-1))
                
                valids = torch.logical_and(edge_portion > 0.1, value_portion > 0.1) # will be BxN
                
                edge_percentage = torch.sum(edge_portion > 0.1).float() / valids.shape[0]
                value_percentage = torch.sum(value_portion > 0.1).float() / valids.shape[0]
                valid_percentage = torch.sum(valids).float() / valids.shape[0]
                # print("window_size: {}: edge_percentage: {:.2f}, value_percentage: {:.2f}, valid_percentage: {:.2f}".format(win_size, edge_percentage, value_percentage, valid_percentage))
                
                pc_prediction_sample_valid, pc_pseudo_label_sample_valid, pc_pseudo_label_len_valid = pc_prediction_sample[valids], pc_pseudo_label_sample[valids], pc_pseudo_label_len[valids]
                
                if pc_pseudo_label_len_valid.shape[0] == 0:
                    continue
                
                if self.loss_type == 'chamfer':
                    if self.inv_projection:
                        loss_win, _ = chamfer_distance(
                            pc_prediction_sample_valid.permute(0, 2, 1), 
                            pc_pseudo_label_sample_valid.permute(0, 2, 1), 
                            x_lengths=pc_pseudo_label_len_valid, 
                            y_lengths=pc_pseudo_label_len_valid,
                            single_directional=True,
                            norm=1)
                    else:
                        loss_win, _ = chamfer_distance(
                            pc_prediction_sample_valid.unsqueeze(dim=-1), 
                            pc_pseudo_label_sample_valid.unsqueeze(dim=-1), 
                            x_lengths=pc_pseudo_label_len_valid, 
                            y_lengths=pc_pseudo_label_len_valid,
                            single_directional=True,
                            norm=1)
                        
                elif self.loss_type == 'l1':
                    mask = pc_prediction_sample_valid != torch.nan
                    if self.inv_projection:
                        loss_win = nn.functional.l1_loss(pc_prediction_sample_valid[mask], pc_pseudo_label_sample_valid[mask])
                    else:
                        loss_win = nn.functional.l1_loss(pc_prediction_sample_valid[mask], pc_pseudo_label_sample_valid[mask])
                
                else:
                    raise NotImplementedError
                
                loss += loss_win * w_window
                w_window_sum += w_window
            
            loss = loss / w_window_sum
        return loss
    
@MODELS.register_module()
class ScaleAndShiftInvariantMidasGradLoss(nn.Module):
    def __init__(self, only_missing_area=False, **kargs):
        super().__init__()
        self.name = "SSIGradLoss"
        self.idx = 0
        self.only_missing_area = only_missing_area

    def forward(self, prediction, pseudo_label, gt_depth, mask, min_depth, max_depth):
        
        bs, _, h_i, w_i = prediction.shape
        _, _, h_t, w_t = pseudo_label.shape
    
        if h_i != h_t or w_i != w_t:
            prediction = F.interpolate(prediction, (h_t, w_t), mode='bilinear', align_corners=True)

        prediction, pseudo_label, mask = prediction.squeeze(), pseudo_label.squeeze(), mask.squeeze()
        
        if torch.sum(mask) <= 1:
            print_log("torch.sum(mask) <= 1, hack to skip avoiding bugs", logger='current')
            return input * 0.0
        
        assert prediction.shape == pseudo_label.shape, f"Shape mismatch: Expected same shape but got {prediction.shape} and {pseudo_label.shape}."

        scale, shift = compute_scale_and_shift(prediction, pseudo_label, mask)

        scaled_prediction = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        missing_mask = gt_depth == 0.
        missing_mask_extend = kornia.filters.gaussian_blur2d(missing_mask.float(), kernel_size=(7, 7), sigma=(5., 5.), border_type='reflect', separable=True)
        missing_mask_extend = missing_mask_extend > 0
        missing_mask_extend = missing_mask_extend.squeeze()
        
        prediction, pseudo_label, gt_depth = prediction.squeeze(), pseudo_label.squeeze(), gt_depth.squeeze()
        
        # compute mask, edges
        valid_mask = torch.logical_and(gt_depth>min_depth, gt_depth<max_depth)
        missing_value_mask = torch.logical_and(valid_mask, missing_mask_extend)
        
        # get edge
        pesudo_edge_list = []
        for idx in range(bs):
            pesudo_edge = torch.from_numpy(extract_edges(pseudo_label[idx].detach().cpu(), use_canny=True, preprocess='log')).cuda()
            pesudo_edge_list.append(pesudo_edge)
        pesudo_edges = torch.stack(pesudo_edge_list, dim=0).unsqueeze(dim=1)
        pesudo_edges_extend = kornia.filters.gaussian_blur2d(pesudo_edges.float(), kernel_size=(7, 7), sigma=(5., 5.), border_type='reflect', separable=True)
        pesudo_edges = pesudo_edges_extend > 0 # edge mask
        pesudo_edges = pesudo_edges.squeeze()
        
        if self.only_missing_area:
            sampling_mask = torch.logical_and(missing_value_mask, pesudo_edges)
        else:
            sampling_mask = torch.ones_like(pesudo_edges).bool()
        
        N = torch.sum(sampling_mask)
        d_diff = scaled_prediction - pseudo_label
        d_diff = torch.mul(d_diff, sampling_mask)

        v_gradient = torch.abs(d_diff[:, 0:-2, :] - d_diff[:, 2:, :])
        v_mask = torch.mul(sampling_mask[:, 0:-2, :], sampling_mask[:, 2:, :])
        v_gradient = torch.mul(v_gradient, v_mask)

        h_gradient = torch.abs(d_diff[:, :, 0:-2] - d_diff[:, :, 2:])
        h_mask = torch.mul(sampling_mask[:, :, 0:-2], sampling_mask[:, :, 2:])
        h_gradient = torch.mul(h_gradient, h_mask)

        gradient_loss = torch.sum(h_gradient) + torch.sum(v_gradient)
        loss = gradient_loss / N
        
        return loss


@MODELS.register_module()
class ScaleAndShiftInvariantDAGradLoss(nn.Module):
    def __init__(self, **kargs):
        super().__init__()
        self.name = "SSILoss"

    def forward(self, prediction, target, mask, interpolate=True, return_interpolated=False):
        
        _, _, h_i, w_i = prediction.shape
        _, _, h_t, w_t = target.shape
    
        if h_i != h_t or w_i != w_t:
            prediction = F.interpolate(prediction, (h_t, w_t), mode='bilinear', align_corners=True)

        prediction, target, mask = prediction.squeeze(), target.squeeze(), mask.squeeze()
        
        if torch.sum(mask) <= 1:
            print_log("torch.sum(mask) <= 1, hack to skip avoiding bugs", logger='current')
            return input * 0.0
        
        assert prediction.shape == target.shape, f"Shape mismatch: Expected same shape but got {prediction.shape} and {target.shape}."

        shift_pred = torch.mean(prediction[mask])
        shift_gt = torch.mean(target[mask])
        scale_pred = torch.std(prediction[mask])
        scale_gt = torch.std(target[mask])
        
        scaled_prediction = (prediction - shift_pred) / scale_pred
        scale_target = (target - shift_gt) / scale_gt
        
        sampling_mask = mask
        N = torch.sum(sampling_mask)
        d_diff = scaled_prediction - scale_target
        d_diff = torch.mul(d_diff, sampling_mask)

        v_gradient = torch.abs(d_diff[:, 0:-2, :] - d_diff[:, 2:, :])
        v_mask = torch.mul(sampling_mask[:, 0:-2, :], sampling_mask[:, 2:, :])
        v_gradient = torch.mul(v_gradient, v_mask)

        h_gradient = torch.abs(d_diff[:, :, 0:-2] - d_diff[:, :, 2:])
        h_mask = torch.mul(sampling_mask[:, :, 0:-2], sampling_mask[:, :, 2:])
        h_gradient = torch.mul(h_gradient, h_mask)

        gradient_loss = torch.sum(h_gradient) + torch.sum(v_gradient)
        loss = gradient_loss / N
        
        return loss