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

@DATASETS.register_module()
class KittiDataset(Dataset):
    def __init__(
        self,
        mode,
        data_root, 
        split,
        transform_cfg,
        min_depth,
        max_depth,
        patch_raw_shape=(176, 304),
        resize_mode='zoe',
        with_pseudo_label=False,
        pseudo_label_path=None,
        do_kb_crop=True,
        pre_norm_bbox=True):
        
        self.dataset_name = 'kitti'
        
        self.mode = mode
        self.data_root = data_root
        self.split = split
        self.with_pseudo_label = with_pseudo_label
        self.pseudo_label_path = pseudo_label_path
        self.data_infos = self.load_data_list()
        self.min_depth = min_depth
        self.max_depth = max_depth
        
        # load transform info
        # not for zoedepth (also da-zoe): do resize, but no normalization. Consider putting normalization in model forward now
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
        self.do_kb_crop = do_kb_crop
        
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
        data_root = self.data_root
        split = self.split
        
        self.invalid_depth_num = 0
        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_info = dict()
                    
                    img, depth = line.strip().split(" ")[0], line.strip().split(" ")[1]
                    if depth == 'None':
                        continue
                    img_info['depth_map_path'] = osp.join(data_root, 'depth', depth)
                    img_info['img_path'] = osp.join(data_root, 'raw', img)
                    
                    if self.with_pseudo_label:
                        depth_map_pl = depth.replace('/', '_')
                        depth_map_pl = depth_map_pl.replace('_sync_proj_depth_groundtruth_', '_sync_')
                        depth_map_pl = depth_map_pl.replace('_image_02_', '_image_02_data_')
                        depth_map_pl = depth_map_pl.replace('.png', '_uint16.png')
                        depth_map_pl = "{}_{}_{}".format(depth_map_pl[1:10], depth_map_pl[:10], depth_map_pl[11:])
                        img_info['pseduo_label_path'] = osp.join(self.pseudo_label_path, depth_map_pl)

                    img_info['filename'] = img
                    img_infos.append(img_info)
        else:
            raise NotImplementedError 

        img_infos = sorted(img_infos, key=lambda x: x['img_path'])
        return img_infos
    
    def __getitem__(self, idx):
        img_file_path = self.data_infos[idx]['img_path']
        depth_path = self.data_infos[idx]['depth_map_path']

        # image = np.fromfile(open(img_file_path, 'rb'), dtype=np.uint8).reshape(2160, 3840, 3) 
        image = Image.open(img_file_path)
        
        # load depth
        depth_gt = Image.open(depth_path)
        
        if self.with_pseudo_label:
            loaded_depth_map = np.asarray(Image.open(self.data_infos[idx]['pseduo_label_path']), dtype=np.float32) / 256
            pseudo_depth = Image.fromarray(loaded_depth_map)
                
        if self.do_kb_crop:
            height = image.height
            width = image.width
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            depth_gt = depth_gt.crop(
                (left_margin, top_margin, left_margin + 1216, top_margin + 352))
            image = image.crop(
                (left_margin, top_margin, left_margin + 1216, top_margin + 352))
                
        if self.mode == 'train':
            if self.with_pseudo_label:
                image, gt_info = aug_rotate(image, [depth_gt, pseudo_depth], self.transform_cfg.degree, input_format='PIL')
                depth_gt, pseudo_depth = gt_info[0], gt_info[1]
            else:
                image, depth_gt = aug_rotate(image, depth_gt, self.transform_cfg.degree, input_format='PIL')
        
        # convert to np
        image = np.asarray(image, dtype=np.float32) / 255.0
        depth_gt = np.asarray(depth_gt, dtype=np.float32) / 256.0
        if self.with_pseudo_label:
            pseudo_depth = np.asarray(pseudo_depth, dtype=np.float32)

        if self.mode == 'train':
            image = aug_color(image)
            if self.with_pseudo_label:
                image, gt_info = aug_flip(image, [depth_gt, pseudo_depth])
                depth_gt, pseudo_depth = gt_info[0], gt_info[1]
            else:
                image, depth_gt = aug_flip(image, depth_gt)
        
        # process for the coarse input
        image_tensor = to_tensor(image)
        if self.normalize is not None:
            image_tensor = self.normalize(image_tensor) # feed into light branch (hr)
            
        # image_hr_tensor = copy.deepcopy(image_tensor)
        image_lr_tensor = self.resize(image_tensor.unsqueeze(dim=0)).squeeze(dim=0) # feed into deep branch (lr, zoe)
        depth_gt_tensor = to_tensor(depth_gt)
        if self.with_pseudo_label:
            pseudo_depth_tensor = to_tensor(pseudo_depth)
        
        img_file_basename, _ = osp.splitext(self.data_infos[idx]['filename'])
        img_file_basename = img_file_basename.replace('/', '_') # :D
            
        if self.mode == 'train':
            h, w = self.patch_raw_shape[0], self.patch_raw_shape[1]
            if self.with_pseudo_label:
                image_tensor, gt_info, crop_info = random_crop(image_tensor, [depth_gt_tensor, pseudo_depth_tensor], self.patch_raw_shape)
                depth_gt_crop_tensor, pseudo_depth_crop_tensor = gt_info[0], gt_info[1]
            else:
                image_tensor, depth_gt_crop_tensor, crop_info = random_crop(image_tensor, depth_gt_tensor, self.patch_raw_shape)

            crop_images = self.resize(image_tensor.unsqueeze(dim=0)).squeeze(dim=0)
            crop_depths = depth_gt_crop_tensor
            # bboxs = torch.tensor([crop_info[1], crop_info[0], crop_info[1]+w, crop_info[0]+h])
            
            if self.pre_norm_bbox:
                bboxs = torch.tensor([
                    crop_info[1] / self.transform_cfg.image_raw_shape[1] * self.transform_cfg.network_process_size[1], 
                    crop_info[0] / self.transform_cfg.image_raw_shape[0] * self.transform_cfg.network_process_size[0], 
                    (crop_info[1]+w) / self.transform_cfg.image_raw_shape[1] * self.transform_cfg.network_process_size[1], 
                    (crop_info[0]+h) / self.transform_cfg.image_raw_shape[0] * self.transform_cfg.network_process_size[0]])
                
            else:
                bboxs = torch.tensor([crop_info[1], crop_info[0], crop_info[1]+w, crop_info[0]+h])
            
            return_dict = \
                {'image_lr': image_lr_tensor, 
                 'image_hr': torch.tensor([375, 1242]), # save some memory
                 'crops_image_hr': crop_images, 
                 'depth_gt': depth_gt_tensor, 
                 'crop_depths': crop_depths,
                 'bboxs': bboxs,
                 'img_file_basename': img_file_basename}
            
            if self.with_pseudo_label:
                return_dict['pseudo_label'] = pseudo_depth_crop_tensor
        else:
            
            boundary = get_boundaries(depth_gt, th=1, dilation=0) # for eval maybe
            boundary_tensor = to_tensor(boundary)
    
            return_dict = \
                {'image_lr': image_lr_tensor, 
                 'image_hr': image_tensor, 
                 'depth_gt': depth_gt_tensor, 
                 'boundary': boundary_tensor, 
                 'img_file_basename': img_file_basename}
                
        return return_dict
            

    def __len__(self):
        return len(self.data_infos)
    
    def get_metrics(self, depth_gt, result, disp_gt_edges, **kwargs):
        metrics = compute_metrics(depth_gt, result, disp_gt_edges=disp_gt_edges, min_depth_eval=self.min_depth, max_depth_eval=self.max_depth, garg_crop=True, eigen_crop=False, dataset='kitti')
        # print_log('{}: {}\n'.format(kwargs['filename'], metrics), logger='current') # in some case
        return metrics
    
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

        ret_metrics['a1'] = np.nanmean(pre_eval_results[0])
        ret_metrics['a2'] = np.nanmean(pre_eval_results[1])
        ret_metrics['a3'] = np.nanmean(pre_eval_results[2])
        ret_metrics['abs_rel'] = np.nanmean(pre_eval_results[3])
        ret_metrics['rmse'] = np.nanmean(pre_eval_results[4])
        ret_metrics['log_10'] = np.nanmean(pre_eval_results[5])
        ret_metrics['rmse_log'] = np.nanmean(pre_eval_results[6])
        ret_metrics['silog'] = np.nanmean(pre_eval_results[7])
        ret_metrics['sq_rel'] = np.nanmean(pre_eval_results[8])
        ret_metrics['see'] = np.nanmean(pre_eval_results[9])

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

        num_table = len(ret_metrics) // 10
        for i in range(num_table):
            names = ret_metric_names[i*10: i*10 + 10]
            values = ret_metric_values[i*10: i*10 + 10]

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
    
        