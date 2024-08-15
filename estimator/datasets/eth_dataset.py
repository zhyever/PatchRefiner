import torch
import random
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
from torchvision import transforms
from zoedepth.models.base_models.midas import Resize
from depth_anything.transform import Resize as ResizeDA
import os.path as osp
from collections import OrderedDict
from prettytable import PrettyTable
from mmengine import print_log
import copy
from estimator.datasets.transformers import aug_color, aug_flip, to_tensor, random_crop, aug_rotate
from estimator.registry import DATASETS
from estimator.utils import get_boundaries, compute_metrics
import cv2
from PIL import Image
import kornia

@DATASETS.register_module()
class ETHDataset(Dataset):
    def __init__(
        self,
        mode,
        split,
        transform_cfg,
        min_depth,
        max_depth,
        stitcher_stage=0,
        overlap=0,
        crop_strategy='random',
        resize_mode='zoe'):
        
        self.dataset_name = 'eth3d'
        
        self.mode = mode
        self.split = split
        self.data_infos = self.load_data_list()
        self.min_depth = min_depth
        self.max_depth = max_depth
        
        self.overlap = overlap # only for consistency evaluation
        
        # load transform info
        # not for zoedepth (also da-zoe): do resize, but no normalization. Consider putting normalization in model forward now
        net_h, net_w = transform_cfg.input_size_deep
        if resize_mode == 'zoe':
            self.resize = Resize(net_w, net_h, keep_aspect_ratio=False, ensure_multiple_of=32, resize_method="minimal")
            self.normalize = None
        elif resize_mode == 'depth-anything':
            self.resize = ResizeDA(net_w, net_h, keep_aspect_ratio=False, ensure_multiple_of=14, resize_method="minimal")
            self.normalize = None
        else:
            raise NotImplementedError
   
            
        self.transform_cfg = transform_cfg
        
        self.stitcher_stage = stitcher_stage
        self.crop_strategy = crop_strategy
        
        self.h_start_list = []
        self.w_start_list = []
        
        # 0, 540, 1080, 1620, (2160)
        # 0, 960, 1920, 2880, (3840)
        # NOTE: random
        if self.transform_cfg.get('random_crop', False):
            pass
        else:
            if self.crop_strategy == 'random_select_horizontal_2patches':
                for i in range(12):
                    if i * (384 / 12) * (2160 / 384) > 1620:
                        break
                    else:
                        self.h_start_list.append(int(i * (384 / 12) * (2160 / 384)))

                for i in range(16):
                    if i * (512 / 16) * (3840 / 512) > 1920:
                        break
                    else:
                        self.w_start_list.append(int(i * (512 / 16) * (3840 / 512)))
            elif self.crop_strategy == '16patches':
                self.h_start_list = [0, 540, 1080, 1620]
                self.w_start_list = [0, 960, 1920, 2880]
            else:
                raise NotImplementedError
            
        if self.mode == 'infer':
            self.h_start_list = [int(0 + 3 * self.overlap / 2), int(540 + self.overlap / 2), int(1080 - self.overlap / 2), int(1620 - 3 * self.overlap / 2)]
            self.w_start_list = [int(0 + 3 * self.overlap / 2), int(960 + self.overlap / 2), int(1920 - self.overlap / 2), int(2880 - 3 * self.overlap / 2)]
                
                             
    def load_data_list(self):
        """Load annotation from directory.
        Args:
            data_root (str): Data root for img_dir/ann_dir.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None
        Returns:
            list[dict]: All image info of dataset.
        """
        split = self.split
        
        self.invalid_depth_num = 0
        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_info = dict()
                    img, depth_map = line.strip().split(" ")

                    img_info['depth_map_path'] = depth_map
                    img_info['img_path'] = img

                    img_info['depth_fields'] = []
                    img_infos.append(img_info)
        else:
            raise NotImplementedError 

        # github issue:: make sure the same order
        img_infos = sorted(img_infos, key=lambda x: x['img_path'])
        return img_infos
    

    def __getitem__(self, idx):
        img_file_path = self.data_infos[idx]['img_path']
        depth_path = self.data_infos[idx]['depth_map_path']
        
        image = Image.open(img_file_path).convert("RGB")
        
        # load depth
        height, width = 4032, 6048
        depth = np.fromfile(depth_path, dtype=np.float32).reshape(height, width)
        depth = np.nan_to_num(depth, posinf=0., neginf=0., nan=0.)
        depth = depth.astype(np.float32)
        depth = Image.fromarray(depth)
        depth_gt = depth
        
  
        if self.mode == 'train':
            # Resize depth_gt to match the size of image only do it during training
            image, gt_info = aug_rotate(image, [depth_gt], self.transform_cfg.degree, input_format='PIL')
            depth_gt = gt_info[0]
        
        # convert to rgb, it's only for u4k
        image = np.asarray(image).astype(np.float32).copy()
        depth_gt = np.asarray(depth_gt).astype(np.float32).copy()
        disp_gt = depth_gt.copy()
        
        # div 255
        image = image / 255.0
        
        # resize
        if self.transform_cfg.get('input_size_shallow', None) is not None:
            image = torch.from_numpy(image).unsqueeze(dim=0).permute(0, 3, 1, 2)
            image = F.interpolate(image, self.transform_cfg.input_size_shallow, mode='bilinear', align_corners=True)
            image = image.squeeze().permute(1, 2, 0).numpy()
        
        if self.mode == 'train':
            # process the data
            image = aug_color(image)
            image, gt_info = aug_flip(image, [depth_gt])
            depth_gt = gt_info[0]
        
        
        # process for the coarse input
        image_tensor = to_tensor(image)
        if self.normalize is not None:
            image_tensor = self.normalize(image_tensor) # feed into light branch (hr)
            
        # image_hr_tensor = copy.deepcopy(image_tensor)
        image_lr_tensor = self.resize(image_tensor.unsqueeze(dim=0)).squeeze(dim=0) # feed into deep branch (lr, zoe)
        depth_gt_tensor = to_tensor(depth_gt)
        
        if self.transform_cfg.get('random_crop', False):
            depth_gt_tensor = F.interpolate(depth_gt_tensor.unsqueeze(dim=0), (2160, 3840)).squeeze(dim=0)
            
            h, w = 540, 960
            image_tensor, gt_info, crop_info = random_crop(image_tensor, [depth_gt_tensor], self.transform_cfg.random_crop_size)
            depth_gt_crop_tensor = gt_info[0]
            disp_gt_crop_tensor = depth_gt_crop_tensor
            
            crop_images = self.resize(image_tensor.unsqueeze(dim=0)).squeeze(dim=0)
            crop_depths = depth_gt_crop_tensor
            # bboxs = torch.tensor([crop_info[1] / (w * 4) * 512, crop_info[0] / (h * 4) * 384, (crop_info[1]+w) / (w * 4) * 512, (crop_info[0]+h) / (h * 4) * 384])
            bboxs = torch.tensor([crop_info[1], crop_info[0], crop_info[1]+w, crop_info[0]+h])
            
        else:
            # in validation or stitcher
            h_start_list = self.h_start_list
            w_start_list = self.w_start_list
            h, w = 540, 960
            
            if self.mode == 'train': 
                raise NotImplementedError
                
                
            else: # inference mode
                crop_images = []
                crop_depths = []
                bboxs = []
                for h_start in self.h_start_list:
                    for w_start in self.w_start_list:
                        crop_image = image_tensor[:, h_start: h_start+h, w_start: w_start+w]
                        crop_depth = depth_gt_tensor[:, h_start: h_start+h, w_start: w_start+w]
                        crop_image_resized = self.resize(crop_image.unsqueeze(dim=0)).squeeze(dim=0)
                        # bbox = torch.tensor([w_start / (w * 4) * 512, h_start / (h * 4) * 384, (w_start+2*w) / (w * 4) * 512, (h_start+h) / (h * 4) * 384])
                        bbox = torch.tensor([w_start, h_start, w_start+w, h_start+h])
                        
                        
                        crop_images.append(crop_image_resized)
                        crop_depths.append(crop_depth)
                        bboxs.append(bbox)
                         
                crop_images = torch.stack(crop_images, dim=0)
                crop_depths = torch.stack(crop_depths, dim=0)
                bboxs = torch.stack(bboxs, dim=0)
        
        if self.mode == 'train': 
            return_dict = \
                {'image_lr': image_lr_tensor, 
                 'image_hr': torch.tensor([2160, 3840]), 
                 'crops_image_hr': crop_images, 
                 'depth_gt': depth_gt_tensor, 
                 'crop_depths': crop_depths,
                #  'image_mid_tensor': image_mid_tensor,
                 'bboxs': bboxs}
            
            
        else:
            boundary = get_boundaries(disp_gt, th=1, dilation=0) # for eval maybe
            boundary_tensor = to_tensor(boundary)
            
            img_file_basename, _ = osp.splitext(img_file_path)
            img_file_basename = img_file_basename.replace('/', '_')[1:]
            return_dict = \
                {'image_lr': image_lr_tensor, 
                 'image_hr': image_tensor, 
                 'crops_image_hr': crop_images, 
                 'depth_gt': depth_gt_tensor, 
                 'boundary': boundary_tensor, 
                 'img_file_basename': img_file_basename, 
                 'crop_depths': crop_depths,
                #  'image_mid_tensor': image_mid_tensor,
                 'bboxs': bboxs}
            
            
        return return_dict
            

    def __len__(self):
        # return len(self.data_infos[:10])
        return len(self.data_infos)
    
    def get_metrics(self, depth_gt, result, disp_gt_edges, image_hr=None, **kwargs):
        
        image_grad = kornia.filters.spatial_gradient(image_hr)
        image_grad = (image_grad[:, :, 0, :, :] ** 2 + image_grad[:, :, 1, :, :] ** 2) ** (1/2)
        image_grad = image_grad.sum(dim=1, keepdim=True)
        image_grad_max = image_grad.max()
        edge_area = image_grad.ge(image_grad_max * 0.5)
        # import matplotlib.pyplot as plt
        # plt.subplot(1, 2, 1)
        # plt.imshow(edge_area[0].squeeze().cpu().numpy())
        edge_area = edge_area.float()
        edge_area_extend = kornia.filters.gaussian_blur2d(edge_area, kernel_size=(3, 3), sigma=(3., 3.), border_type='reflect', separable=True)
        edge_area_extend = F.interpolate(edge_area_extend, size=depth_gt.shape[-2:], mode='bilinear', align_corners=True)
        edge_area_extend = edge_area_extend > 0
        # plt.subplot(1, 2, 2)
        # plt.imshow(edge_area_extend[0].squeeze().cpu().numpy())
        # plt.savefig('debug.png')
            
        edge_metrics = compute_metrics(
            depth_gt, result, disp_gt_edges=disp_gt_edges, min_depth_eval=self.min_depth, max_depth_eval=self.max_depth, garg_crop=False, eigen_crop=False, dataset='', additional_mask=edge_area_extend)
        update_metric = {}
        for k, v in edge_metrics.items():
            update_metric['edge_{}'.format(k)] = v.item()
        noedge_metrics = compute_metrics(
            depth_gt, result, disp_gt_edges=disp_gt_edges, min_depth_eval=self.min_depth, max_depth_eval=self.max_depth, garg_crop=False, eigen_crop=False, dataset='', additional_mask=torch.logical_not(edge_area_extend))
        for k, v in noedge_metrics.items():
            update_metric['noedge_{}'.format(k)] = v.item()
        normal_metrics = compute_metrics(
            depth_gt, result, disp_gt_edges=disp_gt_edges, min_depth_eval=self.min_depth, max_depth_eval=self.max_depth, garg_crop=False, eigen_crop=False, dataset='')
        for k, v in normal_metrics.items():
            update_metric['{}'.format(k)] = v.item()
        return update_metric
    
    def pre_eval_to_metrics(self, pre_eval_results):
        aggregate = []
        for item in pre_eval_results:
            aggregate.append(item.values())
        pre_eval_results = aggregate
            
        # convert list of tuples to tuple of lists, e.g.
        # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
        # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
        pre_eval_results = tuple(zip(*pre_eval_results))
        ret_metrics = OrderedDict({})

        ret_metrics['edge_a1'] = np.nanmean(pre_eval_results[0])
        ret_metrics['edge_a2'] = np.nanmean(pre_eval_results[1])
        ret_metrics['edge_a3'] = np.nanmean(pre_eval_results[2])
        ret_metrics['edge_abs_rel'] = np.nanmean(pre_eval_results[3])
        ret_metrics['edge_rmse'] = np.nanmean(pre_eval_results[4])
        ret_metrics['edge_log_10'] = np.nanmean(pre_eval_results[5])
        ret_metrics['edge_rmse_log'] = np.nanmean(pre_eval_results[6])
        ret_metrics['edge_silog'] = np.nanmean(pre_eval_results[7])
        ret_metrics['edge_sq_rel'] = np.nanmean(pre_eval_results[8])
        ret_metrics['edge_see'] = np.nanmean(pre_eval_results[9])

        ret_metrics['noedge_a1'] = np.nanmean(pre_eval_results[10])
        ret_metrics['noedge_a2'] = np.nanmean(pre_eval_results[11])
        ret_metrics['noedge_a3'] = np.nanmean(pre_eval_results[12])
        ret_metrics['noedge_abs_rel'] = np.nanmean(pre_eval_results[13])
        ret_metrics['noedge_rmse'] = np.nanmean(pre_eval_results[14])
        ret_metrics['noedge_log_10'] = np.nanmean(pre_eval_results[15])
        ret_metrics['noedge_rmse_log'] = np.nanmean(pre_eval_results[16])
        ret_metrics['noedge_silog'] = np.nanmean(pre_eval_results[17])
        ret_metrics['noedge_sq_rel'] = np.nanmean(pre_eval_results[18])
        ret_metrics['noedge_see'] = np.nanmean(pre_eval_results[19])
        
        ret_metrics['a1'] = np.nanmean(pre_eval_results[20])
        ret_metrics['a2'] = np.nanmean(pre_eval_results[21])
        ret_metrics['a3'] = np.nanmean(pre_eval_results[22])
        ret_metrics['abs_rel'] = np.nanmean(pre_eval_results[23])
        ret_metrics['rmse'] = np.nanmean(pre_eval_results[24])
        ret_metrics['log_10'] = np.nanmean(pre_eval_results[25])
        ret_metrics['rmse_log'] = np.nanmean(pre_eval_results[26])
        ret_metrics['silog'] = np.nanmean(pre_eval_results[27])
        ret_metrics['sq_rel'] = np.nanmean(pre_eval_results[28])
        ret_metrics['see'] = np.nanmean(pre_eval_results[29])
        
        ret_metrics = {metric: value for metric, value in ret_metrics.items()}

        return ret_metrics

    def evaluate(self, results, **kwargs):
        """Evaluate the dataset.
        Args:
            results (list[tuple[torch.Tensor]] | list[str]): per image pre_eval
                 results or predict depth map for computing evaluation
                 metric.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
        Returns:
            dict[str, float]: Default metrics.
        """
        
        eval_results = {}
        # test a list of files
        ret_metrics = self.pre_eval_to_metrics(results)
        
        ret_metric_names = []
        ret_metric_values = []
        for ret_metric, ret_metric_value in ret_metrics.items():
            ret_metric_names.append(ret_metric)
            ret_metric_values.append(ret_metric_value)

        num_table = len(ret_metrics) // 30
        for i in range(num_table):
            names = ret_metric_names[i*30: i*30 + 30]
            values = ret_metric_values[i*30: i*30 + 30]

            # summary table
            ret_metrics_summary = OrderedDict({
                ret_metric: np.round(np.nanmean(ret_metric_value), 7)
                for ret_metric, ret_metric_value in zip(names, values)
            })

            # for logger
            summary_table_data = PrettyTable()
            for key, val in ret_metrics_summary.items():
                summary_table_data.add_column(key, [val])

            print_log('Evaluation Summary: \n' + summary_table_data.get_string(), logger='current')

        # each metric dict
        for key, value in ret_metrics.items():
            eval_results[key] = value

        return eval_results
    
        
    def pre_eval_to_metrics_consistency(self, pre_eval_results):
        aggregate = []
        for item in pre_eval_results:
            aggregate.append(item.values())
        pre_eval_results = aggregate
            
        # convert list of tuples to tuple of lists, e.g.
        # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
        # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
        pre_eval_results = tuple(zip(*pre_eval_results))
        ret_metrics = OrderedDict({})

        ret_metrics['consistency_error'] = np.nanmean(pre_eval_results[0])

        ret_metrics = {metric: value for metric, value in ret_metrics.items()}

        return ret_metrics