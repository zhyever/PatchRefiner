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
from PIL import Image
import kornia
import os
from estimator.registry import DATASETS
from estimator.utils import get_boundaries, compute_metrics, compute_boundary_metrics, extract_edges

@DATASETS.register_module()
class ScanNetDataset(Dataset):
    def __init__(
        self,
        mode,
        # data_root, 
        split,
        transform_cfg,
        min_depth,
        max_depth,
        with_pseudo_label=False,
        pseudo_label_path=None,
        patch_raw_shape=[720, 960],
        resize_mode='zoe',
        pre_norm_bbox=True):
        
        self.dataset_name = 'scannet'
        
        self.mode = mode
        # self.data_root = data_root
        self.split = split
        self.with_pseudo_label = with_pseudo_label
        self.pseudo_label_path = pseudo_label_path
        self.data_infos = self.load_data_list()
        self.min_depth = min_depth
        self.max_depth = max_depth
        
        # load transform info
        net_h, net_w = transform_cfg.network_process_size
        if resize_mode == 'zoe':
            self.resize = Resize(net_w, net_h, keep_aspect_ratio=False, ensure_multiple_of=32, resize_method="minimal")
            self.normalize = None
        elif resize_mode == 'depth-anything':
            self.resize = ResizeDA(net_w, net_h, keep_aspect_ratio=False, ensure_multiple_of=14, resize_method="minimal")
            self.normalize = None
        else:
            raise NotImplementedError
        
        self.patch_raw_shape = patch_raw_shape
        self.transform_cfg = transform_cfg
        self.pre_norm_bbox = pre_norm_bbox
        
                
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

                    if self.with_pseudo_label:
                        name_frag1, name_frag2 = depth_map.split('/')[-4], depth_map.split('/')[-1][6:12]
                        name = '{}_{}_uint16.png'.format(name_frag1, name_frag2)
                        img_info['pseduo_label_path'] = os.path.join(self.pseudo_label_path, name)
                        
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
        depth_gt = Image.open(depth_path)
        
        # Get the size of the image
        target_size = image.size  # This returns a tuple (width, height)

        # Resize depth_gt to match the size of image
        depth_gt = depth_gt.resize(target_size, Image.NEAREST)
                
        if self.mode == 'train':
            if self.with_pseudo_label:
                loaded_depth_map = np.asarray(Image.open(self.data_infos[idx]['pseduo_label_path']), dtype=np.float32) / 256
                loaded_depth_map = F.interpolate(torch.tensor(loaded_depth_map).unsqueeze(0).unsqueeze(0), (1440, 1920), mode='nearest').squeeze().numpy()
                pseudo_depth = Image.fromarray(loaded_depth_map)
            if self.with_pseudo_label:
                image, gt_info = aug_rotate(image, [depth_gt, pseudo_depth], self.transform_cfg.degree, input_format='PIL')
                depth_gt, pseudo_depth = gt_info[0], gt_info[1]
            else:
                image, depth_gt = aug_rotate(image, depth_gt, self.transform_cfg.degree, input_format='PIL')
        
        image = np.asarray(image).astype(np.float32).copy()
        depth_gt = np.asarray(depth_gt).astype(np.float32).copy() / 1000.0 # mm to m
        if self.with_pseudo_label:
            pseudo_depth = np.asarray(pseudo_depth).astype(np.float32).copy()
            
        # div 255
        image = image / 255.0
        
        if self.mode == 'train':
            # process the data
            image = aug_color(image)
            if self.with_pseudo_label:
                image, gt_info = aug_flip(image, [depth_gt, pseudo_depth])
                depth_gt, pseudo_depth = gt_info[0], gt_info[1]
            else:
                image, depth_gt = aug_flip(image, depth_gt)
        
        # process for the coarse input
        image_tensor = to_tensor(image)
        # image_hr_tensor = copy.deepcopy(image_tensor)
        image_lr_tensor = self.resize(image_tensor.unsqueeze(dim=0)).squeeze(dim=0) # feed into deep branch (lr, zoe)
        depth_gt_tensor = to_tensor(depth_gt)
        if self.with_pseudo_label:
            pseudo_depth_tensor = to_tensor(pseudo_depth)
            
        if self.transform_cfg.get('random_crop', False):
            h, w = self.patch_raw_shape[0], self.patch_raw_shape[1]
            if self.with_pseudo_label:
                image_tensor, gt_info, crop_info = random_crop(image_tensor, [depth_gt_tensor, pseudo_depth_tensor], self.patch_raw_shape)
                crop_depths, pseudo_depth_crop_tensor = gt_info[0], gt_info[1]
            else:
                image_tensor, crop_depths, crop_info = random_crop(image_tensor, depth_gt_tensor, self.patch_raw_shape)
            crop_images = self.resize(image_tensor.unsqueeze(dim=0)).squeeze(dim=0)
            # bboxs = torch.tensor([crop_info[1], crop_info[0], crop_info[1]+w, crop_info[0]+h])
            if self.pre_norm_bbox:
                bboxs = torch.tensor([
                    crop_info[1] / self.transform_cfg.image_raw_shape[1] * self.transform_cfg.network_process_size[1], 
                    crop_info[0] / self.transform_cfg.image_raw_shape[0] * self.transform_cfg.network_process_size[0], 
                    (crop_info[1]+w) / self.transform_cfg.image_raw_shape[1] * self.transform_cfg.network_process_size[1], 
                    (crop_info[0]+h) / self.transform_cfg.image_raw_shape[0] * self.transform_cfg.network_process_size[0]])
                
            else:
                bboxs = torch.tensor([crop_info[1], crop_info[0], crop_info[1]+w, crop_info[0]+h])
            
        
        if self.mode == 'train':
            return_dict = \
                {'image_lr': image_lr_tensor, 
                 'image_hr': torch.tensor([1440, 1920]),
                 'crops_image_hr': crop_images, 
                 'depth_gt': depth_gt_tensor, 
                 'crop_depths': crop_depths,
                 'bboxs': bboxs}
            if self.with_pseudo_label:
                return_dict['pseudo_label'] = pseudo_depth_crop_tensor
            
        else:
            boundary = get_boundaries(depth_gt_tensor.squeeze(dim=0), th=1, dilation=0) # for eval maybe
            boundary_tensor = to_tensor(boundary)
            
            img_file_basename, _ = osp.splitext(img_file_path)
            img_file_basename = img_file_basename.replace('/', '_')[1:]
            img_file_basename = img_file_basename[-34:-24] + '_' + img_file_basename[-6:] # bad hack
            
            return_dict = \
                {'image_lr': image_lr_tensor, 
                 'image_hr': image_tensor, 
                 'depth_gt': depth_gt_tensor, 
                 'boundary': boundary_tensor, 
                 'img_file_basename': img_file_basename}
            
        return return_dict
            

    def __len__(self):
        # return len(self.data_infos[:10])
        return len(self.data_infos)
    
    def get_metrics(self, depth_gt, result, disp_gt_edges, image_hr=None, **kwargs):
        
        # image_grad = kornia.filters.spatial_gradient(image_hr)
        # image_grad = (image_grad[:, :, 0, :, :] ** 2 + image_grad[:, :, 1, :, :] ** 2) ** (1/2)
        # image_grad = image_grad.sum(dim=1, keepdim=True)
        # image_grad_max = image_grad.max()
        # edge_area = image_grad.ge(image_grad_max * 0.5)
        # edge_area = edge_area.float()
        # edge_area_extend = kornia.filters.gaussian_blur2d(edge_area, kernel_size=(3, 3), sigma=(3., 3.), border_type='reflect', separable=True)
        # edge_area_extend = F.interpolate(edge_area_extend, size=depth_gt.shape[-2:], mode='bilinear', align_corners=True)
        # edge_area_extend = edge_area_extend > 0
        
        gt_edges = extract_edges(depth_gt.detach().cpu(), use_canny=True, preprocess='log')
        gt_edges_extend = kornia.filters.gaussian_blur2d(torch.tensor(gt_edges).cuda().float().unsqueeze(dim=0).unsqueeze(dim=0), kernel_size=(7, 7), sigma=(5., 5.), border_type='reflect', separable=True)
        gt_edges_extend = gt_edges_extend > 0
        gt_edges_extend = gt_edges_extend.squeeze()
        
        # import matplotlib.pyplot as plt
        # plt.imshow(gt_edges_extend.cpu().numpy())
        # plt.savefig('./work_dir/gt_edges_extend.png')
        # exit(100)
        
        edge_metrics = compute_metrics(
            depth_gt, result, disp_gt_edges=disp_gt_edges, min_depth_eval=self.min_depth, max_depth_eval=self.max_depth, garg_crop=False, eigen_crop=False, dataset='', additional_mask=gt_edges_extend)
        update_metric = {}
        for k, v in edge_metrics.items():
            update_metric['edge_{}'.format(k)] = v.item()
        noedge_metrics = compute_metrics(
            depth_gt, result, disp_gt_edges=disp_gt_edges, min_depth_eval=self.min_depth, max_depth_eval=self.max_depth, garg_crop=False, eigen_crop=False, dataset='', additional_mask=torch.logical_not(gt_edges_extend))
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

    def evaluate_consistency(self, results, **kwargs):
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
        ret_metrics = self.pre_eval_to_metrics_consistency(results)
        
        ret_metric_names = []
        ret_metric_values = []
        for ret_metric, ret_metric_value in ret_metrics.items():
            ret_metric_names.append(ret_metric)
            ret_metric_values.append(ret_metric_value)

        num_table = len(ret_metrics) // 1
        for i in range(num_table):
            names = ret_metric_names[i*1: i*1 + 1]
            values = ret_metric_values[i*1: i*1 + 1]

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
