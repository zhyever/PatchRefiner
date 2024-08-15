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
import os
from collections import OrderedDict
from prettytable import PrettyTable
from mmengine import print_log
import copy
from estimator.datasets.transformers import aug_color, aug_flip, to_tensor, random_crop, aug_rotate
from estimator.registry import DATASETS
from estimator.utils import get_boundaries, compute_metrics, compute_boundary_metrics, extract_edges, rescale_tensor_train
import cv2
from PIL import Image
import kornia
from skimage import io
import torchmetrics
import json

@DATASETS.register_module()
class CityScapesDataset(Dataset):
    def __init__(
        self,
        mode,
        split,
        transform_cfg,
        min_depth,
        max_depth,
        patch_raw_shape=[256, 512],
        data_root='./data/cityscapes',
        resize_mode='zoe',
        with_pseudo_label=False,
        pseudo_label_path=None,
        with_seg_map=False,
        filter_sky=True,
        pre_norm_bbox=True,
        with_uncert=False,
        base=np.exp(1),
        filter_thr=-0.1,):
        
        self.dataset_name = 'cityscapes'
        
        self.data_root = data_root
        self.mode = mode
        self.split = split
        self.with_pseudo_label = with_pseudo_label
        self.with_uncert = with_uncert
        self.with_seg_map = with_seg_map
        self.pseudo_label_path = pseudo_label_path
        self.filter_sky = filter_sky
        self.data_infos = self.load_data_list()
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.base = base
        self.filter_thr = filter_thr
        
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
            
        self.transform_cfg = transform_cfg
        self.patch_raw_shape = patch_raw_shape
        
        self.precision_metric = torchmetrics.classification.BinaryPrecision()
        self.recall_metric = torchmetrics.classification.BinaryRecall()
        self.f1_metric = torchmetrics.classification.BinaryF1Score()
        self.hammingidstance = torchmetrics.classification.BinaryHammingDistance()
        self.acc = torchmetrics.classification.BinaryAccuracy()
        
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

                    img_info['depth_map_path'] = os.path.join(self.data_root, depth_map)
                    img_info['img_path'] = os.path.join(self.data_root, img)
                    
                    img_info['camera_info'] = os.path.join(self.data_root, img).replace('leftImg8bit', 'camera')
                    img_info['camera_info'] = img_info['camera_info'].replace('.png', '.json')
                    
                    if self.filter_sky:
                        img_info['sky_seg_path'] = img_info['img_path'].replace('leftImg8bit', 'skyArea')
                        
                    if self.with_pseudo_label:
                        depth_map_pl = depth_map.replace('disparity', 'leftImg8bit')
                        depth_map_pl = depth_map_pl.replace('/', '_')
                        depth_map_pl = depth_map_pl.replace('.png', '_uint16.png')
                        img_info['pseduo_label_path'] = os.path.join(self.pseudo_label_path, depth_map_pl)
                    
                    if self.with_uncert:
                        img_info['uncertain_path'] = img_info['pseduo_label_path'].replace('_uint16.png', '_uncert_uint16.png')
                        img_info['count_path'] = img_info['pseduo_label_path'].replace('_uint16.png', '_count_uint16.png')
                        
                        
                    if self.with_seg_map:
                        seg_map = img_info['depth_map_path'].replace('disparity', 'gtFine').replace('.png', '_color.png')
                        img_info['seg_map'] = seg_map
                        
                    img_info['filename'] = img
                    img_info['depth_fields'] = []
                    img_infos.append(img_info)
        else:
            raise NotImplementedError 

        # github issue:: make sure the same order
        img_infos = sorted(img_infos, key=lambda x: x['img_path'])
        return img_infos
    
    def __getitem__(self, idx):
        # fetch base path info
        img_file_path = self.data_infos[idx]['img_path']
        disp_path = self.data_infos[idx]['depth_map_path']
        
        # load image
        image = Image.open(img_file_path).convert("RGB")
        
        # load camera
        json_file = open(self.data_infos[idx]['camera_info'])
        camera_info = json.load(json_file)
        # camera_info = torch.tensor([camera_info['intrinsic']['fx'], camera_info['intrinsic']['fy'], camera_info['intrinsic']['u0'], camera_info['intrinsic']['v0']])
        
        # load depth
        img_d = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        img_d[img_d > 0] = (img_d[img_d > 0] - 1) / 256
        depth_gt = (camera_info['extrinsic']['baseline'] * camera_info['intrinsic']['fx']) / img_d
        depth_gt = np.nan_to_num(depth_gt, posinf=0., neginf=0., nan=0.)
        depth_gt = depth_gt.astype(np.float32)
        h, w = depth_gt.shape
        
        # filter noisy gt
        # depth_gt[:h//4, :] = -1.
        depth_gt[-h//4:, :] = -1.
        depth_gt[:, :w//16] = -1.
        depth_gt[:, -w//16:] = -1.
        
        # load seg map (eval only)
        if self.with_seg_map:
            seg_path = self.data_infos[idx]['seg_map']
            seg_image = Image.open(seg_path).convert("RGB")
            if self.mode == 'infer':
                seg_image_np = np.asarray(seg_image)
                seg_image_np_mask = np.logical_and(seg_image_np[:, :, 0] == 70, seg_image_np[:, :, 1] == 130)
                depth_gt[seg_image_np_mask] = 0
        else:
            seg_image = None
            
        # load sky seg map (train only) (not really adopted)
        if self.mode == 'train':
            if self.filter_sky:
                loaded_sky = np.asarray(Image.open(self.data_infos[idx]['sky_seg_path']), dtype=np.float32)
                if self.transform_cfg.get('input_size_shallow', None) is not None:
                    loaded_sky = F.interpolate(torch.tensor(loaded_sky).unsqueeze(0).unsqueeze(0), self.transform_cfg.input_size_shallow, mode='nearest').squeeze().numpy()
                else:
                    loaded_sky = F.interpolate(torch.tensor(loaded_sky).unsqueeze(0).unsqueeze(0), (1024, 2048), mode='nearest').squeeze().numpy()
                depth_gt[loaded_sky > 0] = -2.
        
        # prepare to be processed
        depth_gt = Image.fromarray(depth_gt)
        pseudo_depth, pseudo_uncert = None, None
        
        # resize
        if self.transform_cfg.get('input_size_shallow', None) is not None:
            image=image.resize(self.transform_cfg.input_size_shallow[::-1], Image.BILINEAR)
            depth_gt=depth_gt.resize(self.transform_cfg.input_size_shallow[::-1], Image.NEAREST)
            if self.with_seg_map:
                seg_image=seg_image.resize(self.transform_cfg.input_size_shallow[::-1], Image.BILINEAR)
        
        # rot. augmentations
        if self.mode == 'train':
            # pl will be only used during training -> resize inside the training cond.
            if self.with_pseudo_label:
                loaded_depth_map = np.asarray(Image.open(self.data_infos[idx]['pseduo_label_path']), dtype=np.float32) / 256
                if self.transform_cfg.get('input_size_shallow', None) is not None:
                    loaded_depth_map = F.interpolate(torch.tensor(loaded_depth_map).unsqueeze(0).unsqueeze(0), self.transform_cfg.input_size_shallow, mode='nearest').squeeze().numpy()
                else:
                    loaded_depth_map = F.interpolate(torch.tensor(loaded_depth_map).unsqueeze(0).unsqueeze(0), (1024, 2048), mode='nearest').squeeze().numpy()
                pseudo_depth = Image.fromarray(loaded_depth_map)
                
            if self.with_uncert:
                loaded_uncert_map = np.asarray(Image.open(self.data_infos[idx]['uncertain_path']), dtype=np.float32) / 256 # will be 0-1
                loaded_count_map = np.asarray(Image.open(self.data_infos[idx]['count_path']), dtype=np.float32) / 256 # will real counts
                loaded_uncert_map[loaded_count_map < (16+9+9+9+128) * self.filter_thr] = 1.0
                
                if self.transform_cfg.get('input_size_shallow', None) is not None:
                    loaded_uncert_map = F.interpolate(torch.tensor(loaded_uncert_map).unsqueeze(0).unsqueeze(0), self.transform_cfg.input_size_shallow, mode='nearest').squeeze().numpy()
                else:
                    loaded_uncert_map = F.interpolate(torch.tensor(loaded_uncert_map).unsqueeze(0).unsqueeze(0), (1024, 2048), mode='nearest').squeeze().numpy()
                pseudo_uncert = Image.fromarray(loaded_uncert_map)
            
            image, gt_info = aug_rotate(image, [depth_gt, pseudo_depth, seg_image, pseudo_uncert], self.transform_cfg.degree, input_format='PIL')
            depth_gt, pseudo_depth, seg_image, pseudo_uncert = gt_info[0], gt_info[1], gt_info[2], gt_info[3]
        
        # convert to np for more processes
        image = np.asarray(image).astype(np.float32).copy()
        image = image / 255.0
        
        depth_gt = np.asarray(depth_gt).astype(np.float32).copy()
        disp_gt = depth_gt.copy()
        
        if self.with_seg_map:
            seg_image = np.asarray(seg_image).astype(np.float32).copy()
            seg_image = seg_image / 1.0 # no need to / 256
        if self.with_pseudo_label:
            pseudo_depth = np.asarray(pseudo_depth).astype(np.float32).copy()
        if self.with_uncert:
            pseudo_uncert = np.asarray(pseudo_uncert).astype(np.float32).copy()
            pseudo_uncert = np.log(1 + pseudo_uncert) / np.log(self.base) # log(x) here
            pseudo_uncert = rescale_tensor_train(torch.tensor(pseudo_uncert), 0, 1).numpy()
            
        # more augs
        if self.mode == 'train':
            # process the data
            image = aug_color(image)
            image, gt_info = aug_flip(image, [depth_gt, pseudo_depth, pseudo_uncert])
            depth_gt, pseudo_depth, pseudo_uncert = gt_info[0], gt_info[1], gt_info[2]
            
        # process for the coarse input
        image_tensor = to_tensor(image)
        if self.normalize is not None:
            image_tensor = self.normalize(image_tensor) # feed into light branch (hr)
            
        # from numpy to tensor
        image_lr_tensor = self.resize(image_tensor.unsqueeze(dim=0)).squeeze(dim=0) # feed into deep branch (lr, zoe)
        depth_gt_tensor = to_tensor(depth_gt)
        pseudo_depth_tensor = to_tensor(pseudo_depth) if self.with_pseudo_label else None
        pseudo_uncert_tensor = to_tensor(pseudo_uncert) if self.with_uncert else None
        seg_image = to_tensor(seg_image) if self.with_seg_map else None
        
        img_file_basename, _ = osp.splitext(self.data_infos[idx]['filename'])
        img_file_basename = img_file_basename.replace('/', '_')
        
        # crop
        if self.transform_cfg.get('random_crop', False):
            h, w = self.patch_raw_shape[0], self.patch_raw_shape[1]
            image_tensor, gt_info, crop_info = random_crop(image_tensor, [depth_gt_tensor, pseudo_depth_tensor, pseudo_uncert_tensor], self.transform_cfg.random_crop_size)
            depth_gt_crop_tensor, pseudo_depth_crop_tensor, pseudo_uncert_crop_tensor = gt_info[0], gt_info[1], gt_info[2]
            
            crop_images = self.resize(image_tensor.unsqueeze(dim=0)).squeeze(dim=0)
            crop_depths = depth_gt_crop_tensor
            # bboxs = torch.tensor([crop_info[1] / (w * 4) * 512, crop_info[0] / (h * 4) * 384, (crop_info[1]+w) / (w * 4) * 512, (crop_info[0]+h) / (h * 4) * 384])
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
                 'image_hr': torch.tensor([2160, 3840]), 
                 'crops_image_hr': crop_images, 
                 'depth_gt': depth_gt_tensor, 
                 'crop_depths': crop_depths,
                 'bboxs': bboxs,
                 'img_file_basename': img_file_basename}
            if self.with_pseudo_label:
                return_dict['pseudo_label'] = pseudo_depth_crop_tensor
            if self.with_uncert:
                
                # pseudo_uncert_crop_tensor = rescale_tensor_train(pseudo_uncert_crop_tensor, 0, 1)
                # pseudo_uncert_crop_tensor = 1 - torch.exp(-pseudo_uncert_crop_tensor) 
                return_dict['pseudo_uncert'] = pseudo_uncert_crop_tensor
            
        else:
            boundary = get_boundaries(disp_gt, th=1, dilation=0) # for eval maybe
            boundary_tensor = to_tensor(boundary)
            
            return_dict = \
                {'image_lr': image_lr_tensor, 
                 'image_hr': image_tensor, 
                 'depth_gt': depth_gt_tensor, 
                 'boundary': boundary_tensor, 
                 'img_file_basename': img_file_basename}
            if self.with_seg_map:
                return_dict['seg_image'] = seg_image
            
        return return_dict

    def __len__(self):
        # return len(self.data_infos[:10])
        return len(self.data_infos)
    
    def get_metrics(self, depth_gt, result, disp_gt_edges, seg_image=None, image_hr=None, **kwargs):
        
        if depth_gt.shape[-2:] != result.shape[-2:]:
            result = nn.functional.interpolate(
                result, depth_gt.shape[-2:], mode='bilinear', align_corners=False).squeeze()
        
        mask = torch.logical_and(depth_gt > self.min_depth, depth_gt < self.max_depth).squeeze()
        h, w = depth_gt.shape[-2:]
        # mask[:h//4, :] = 0
        mask[-h//4:, :] = 0
        mask[:, :w//16] = 0
        mask[:, -w//16:] = 0
        mask = mask.squeeze()
        
        # calculate seg map edges
        _, seg_map_edge = kornia.filters.canny(seg_image)
        seg_map_edge = seg_map_edge > 0
        seg_map_edge_extend = kornia.filters.gaussian_blur2d(seg_map_edge.float(), kernel_size=(7, 7), sigma=(5., 5.), border_type='reflect', separable=True)
        seg_map_edge_extend = seg_map_edge_extend > 0
        seg_map_edge_extend = seg_map_edge_extend.squeeze()
        
        # calculate gt depth map edge (nosiy)
        gt_edges = extract_edges(depth_gt.detach().cpu(), use_canny=True, preprocess='log')
        gt_edges_extend = kornia.filters.gaussian_blur2d(torch.tensor(gt_edges).cuda().float().unsqueeze(dim=0).unsqueeze(dim=0), kernel_size=(7, 7), sigma=(5., 5.), border_type='reflect', separable=True)
        gt_edges_extend = gt_edges_extend > 0
        gt_edges_extend = gt_edges_extend.squeeze()
        
        # calculate hr rgb map edge (nosiy)
        hr_grad = kornia.filters.spatial_gradient(image_hr)
        hr_grad = (hr_grad[:, :, 0, :, :] ** 2 + hr_grad[:, :, 1, :, :] ** 2) ** (1/2)
        hr_grad_sum = hr_grad.sum(dim=1, keepdim=True)
        hr_edge = hr_grad_sum.ge(0.05 * hr_grad_sum.max())
        hr_edge_extend = kornia.filters.gaussian_blur2d(hr_edge.float(), kernel_size=(3, 3), sigma=(3., 3.), border_type='reflect', separable=True)
        hr_edge = hr_edge_extend > 0
        hr_edge = hr_edge.float().squeeze()
        
        # calculate pred edges (MDE challenge)
        pred_edges = extract_edges(result.detach().cpu(), use_canny=True, preprocess='log')
        pred_edges = pred_edges > 0
        pred_edges = pred_edges.squeeze()
        pred_edges = torch.from_numpy(pred_edges)
    
        # edge mask (w/o the valid value mask):
        edge_mask = torch.logical_and(seg_map_edge.squeeze(), gt_edges_extend) # edge = seg edge + (and) gt edge
        
        # flatten_mask (contain the valid value mask):
        flatten_mask = torch.logical_and(mask, torch.logical_not(edge_mask)) # valid and ~(edge)
        flatten_mask = torch.logical_and(flatten_mask, torch.logical_not(hr_edge)) # and ~(hr image edge)
        
        # squeeze
        depth_gt = depth_gt.squeeze().detach().cpu()
        result = result.squeeze().detach().cpu()
        disp_gt_edges = disp_gt_edges.squeeze().detach().cpu()
        edge_mask = edge_mask.squeeze().detach().cpu()
        flatten_mask = flatten_mask.squeeze().detach().cpu()
        pred_edges = pred_edges.squeeze().detach().cpu()
        
        update_metric = {}
        metrics = compute_metrics(
            depth_gt, 
            result, 
            disp_gt_edges=disp_gt_edges, 
            min_depth_eval=self.min_depth, 
            max_depth_eval=self.max_depth, 
            garg_crop=False, eigen_crop=False, 
            dataset='', 
            additional_mask=flatten_mask)
        
        for k, v in metrics.items():
            update_metric['{}'.format(k)] = v.item()
        
        if self.with_seg_map:
            metric_dict = { \
                'precision': self.precision_metric,
                'recall': self.recall_metric,
                'f1_score': self.f1_metric,
                'hamming': self.hammingidstance,
                'acc': self.acc}
            
            metrics = compute_boundary_metrics(
                depth_gt, 
                result,
                gt_edges=edge_mask,
                pred_edges=pred_edges,
                valid_mask=mask.squeeze().detach().cpu(),
                metric_dict=metric_dict)
            
            for k, v in metrics.items():
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
        
        if self.with_seg_map:
            ret_metrics['EdgeAcc'] = np.nanmean(pre_eval_results[10])
            ret_metrics['EdgeComp'] = np.nanmean(pre_eval_results[11])
            ret_metrics['precision'] = np.nanmean(pre_eval_results[12])
            ret_metrics['recall'] = np.nanmean(pre_eval_results[13])
            ret_metrics['f1_score'] = np.nanmean(pre_eval_results[14])
            ret_metrics['hamming'] = np.nanmean(pre_eval_results[15])
            ret_metrics['acc'] = np.nanmean(pre_eval_results[16])
        
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

        if self.with_seg_map:
            # num_table = len(ret_metrics) // 34
            num_table = len(ret_metrics) // 17
        else:
            # num_table = len(ret_metrics) // 30
            num_table = len(ret_metrics) // 10
            
            
        for i in range(num_table):
            if self.with_seg_map:
                names = ret_metric_names[i*17: i*17 + 17]
                values = ret_metric_values[i*17: i*17 + 17]
            else:
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