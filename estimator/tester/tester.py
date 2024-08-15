import os
import cv2
import mmengine.analysis
import wandb
import numpy as np
import torch
import mmengine
from mmengine.optim import build_optim_wrapper
import torch.optim as optim
import matplotlib.pyplot as plt
from mmengine.dist import get_dist_info, collect_results_cpu, collect_results_gpu
from mmengine import print_log
from estimator.utils import colorize, colorize_infer_pfv1, colorize_rescale
import torch.nn.functional as F
from tqdm import tqdm
from mmengine.utils import mkdir_or_exist
import copy
from skimage import io
import kornia
from PIL import Image
from estimator.utils import extract_edges, rescale_tensor
import time
from mmengine.fileio import dump

class Tester:
    """
    Tester class
    """
    def __init__(
        self, 
        config,
        runner_info,
        dataloader,
        model):
       
        self.config = config
        self.runner_info = runner_info
        self.dataloader = dataloader
        self.model = model
        self.collect_input_args = config.collect_input_args
    
    def collect_input(self, batch_data):
        collect_batch_data = dict()
        for k, v in batch_data.items():
            if isinstance(v, torch.Tensor):
                if k in self.collect_input_args:
                    collect_batch_data[k] = v.cuda()
        return collect_batch_data
    
    @torch.no_grad()
    def run(self, cai_mode='p16', process_num=4, image_raw_shape=[2160, 3840], patch_split_num=[4, 4]):
        
        results = []
        dataset = self.dataloader.dataset
        loader_indices = self.dataloader.batch_sampler
        
        rank, world_size = get_dist_info()
        if self.runner_info.rank == 0:
            prog_bar = mmengine.utils.ProgressBar(len(dataset))

        for idx, (batch_indices, batch_data) in enumerate(zip(loader_indices, self.dataloader)):
            
            batch_data_collect = self.collect_input(batch_data)
            
            tile_cfg = dict()
            tile_cfg['image_raw_shape'] = image_raw_shape
            tile_cfg['patch_split_num'] = patch_split_num # use a customized value instead of the default [4, 4] for 4K images
            result, log_dict = self.model(mode='infer', cai_mode=cai_mode, process_num=process_num, tile_cfg=tile_cfg, **batch_data_collect) # might use test/val to split cases
            # uncertainty = log_dict['uncertainty']
            
            if self.runner_info.save:
                
                if self.runner_info.gray_scale:
                    color_pred = colorize(result, cmap='gray_r')[:, :, [2, 1, 0]]
                else:    
                    if self.dataloader.dataset.dataset_name == 'cityscapes':
                        color_pred = colorize(result, cmap='magma_r', vminp=0, vmaxp=100)[:, :, [2, 1, 0]]
                    elif self.dataloader.dataset.dataset_name == 'kitti':
                        color_pred = colorize(result, cmap='magma_r', vminp=0, vmaxp=100)[:, :, [2, 1, 0]]
                    elif self.dataloader.dataset.dataset_name == 'scannet':
                        color_pred = colorize(result, cmap='magma_r', vminp=0, vmaxp=100)[:, :, [2, 1, 0]]
                    else:
                        color_pred = colorize(result, cmap='Spectral', vminp=0, vmaxp=100)[:, :, [2, 1, 0]]
                    color_pred = colorize(result, cmap='Spectral', vminp=0, vmaxp=100)[:, :, [2, 1, 0]]
                        
                cv2.imwrite(os.path.join(self.runner_info.work_dir, '{}.png'.format(batch_data['img_file_basename'][0])), color_pred)
            
                # Save as PNG
                # raw_depth = Image.fromarray((result.clone().squeeze().detach().cpu().numpy()*256).astype('uint16'))
                # raw_depth.save(os.path.join(self.runner_info.work_dir, '{}_uint16.png'.format(batch_data['img_file_basename'][0])))
                
                # # calculate pred edges
                # pred_edges = extract_edges(result.detach().cpu(), use_canny=True, preprocess='log')
                # pred_edges = pred_edges > 0
                # pred_edges = torch.tensor(pred_edges).unsqueeze(0).unsqueeze(0)
                # pred_edges_extend = kornia.filters.gaussian_blur2d(pred_edges.float(), kernel_size=(3, 3), sigma=(3., 3.), border_type='reflect', separable=True)
                # pred_edges_extend = pred_edges_extend > 0
                # pred_edges = pred_edges_extend.squeeze()
                # pred_edges = pred_edges.squeeze().numpy()
                # cv2.imwrite(os.path.join(self.runner_info.work_dir, '{}_edge.png'.format(batch_data['img_file_basename'][0])), pred_edges * 256)
                
                # color_pred = colorize(uncertainty, cmap='jet', vminp=0, vmaxp=100)[:, :, [2, 1, 0]]
                # cv2.imwrite(os.path.join(self.runner_info.work_dir, '{}_uncert.png'.format(batch_data['img_file_basename'][0])), color_pred)
                # uncertainty = uncertainty / torch.max(uncertainty)
                

            if batch_data_collect.get('depth_gt', None) is not None:
                # validate the effectiveness of uncertainty
                # batch_data_collect['depth_gt'][uncertainty > torch.quantile(uncertainty, 0.9)] = 0
                # result[uncertainty > torch.quantile(uncertainty, 0.9)] = 0
                
                metrics = dataset.get_metrics(
                    batch_data_collect['depth_gt'], 
                    result, 
                    seg_image=batch_data_collect.get('seg_image', None),
                    disp_gt_edges=batch_data.get('boundary', None), 
                    image_hr=batch_data.get('image_hr', None).cuda(),
                    filename=batch_data['img_file_basename'][0])
                results.extend([metrics])
            
            if self.runner_info.rank == 0:
                batch_size = len(result) * world_size
                for _ in range(batch_size):
                    prog_bar.update()
        
        if batch_data_collect.get('depth_gt', None) is not None:   
            results = collect_results_gpu(results, len(dataset))
            if self.runner_info.rank == 0:
                ret_dict = dataset.evaluate(results)
    
    
    
    @torch.no_grad()
    def generate_pl(self, cai_mode='p16', process_num=4, image_raw_shape=[2160, 3840], patch_split_num=[4, 4], count_thr=0.05):
        
        results = []
        dataset = self.dataloader.dataset
        loader_indices = self.dataloader.batch_sampler
        
        rank, world_size = get_dist_info()
        if self.runner_info.rank == 0:
            prog_bar = mmengine.utils.ProgressBar(len(dataset))

        for idx, (batch_indices, batch_data) in enumerate(zip(loader_indices, self.dataloader)):
            
            batch_data_collect = self.collect_input(batch_data)
            
            tile_cfg = dict()
            tile_cfg['image_raw_shape'] = image_raw_shape
            tile_cfg['patch_split_num'] = patch_split_num # use a customized value instead of the default [4, 4] for 4K images
            result, log_dict = self.model(mode='infer', cai_mode=cai_mode, process_num=process_num, tile_cfg=tile_cfg, **batch_data_collect) # might use test/val to split cases
            uncertainty = log_dict['uncertainty']
            
            if self.runner_info.save:
                
                if self.runner_info.gray_scale:
                    color_pred = colorize(result, cmap='gray_r', vminp=0, vmaxp=100)[:, :, [2, 1, 0]]
                else:
                    color_pred = colorize(result, cmap='magma_r', vminp=0, vmaxp=100)[:, :, [2, 1, 0]]
                cv2.imwrite(os.path.join(self.runner_info.work_dir, '{}.png'.format(batch_data['img_file_basename'][0])), color_pred)
            
                # Save as PNG
                raw_depth = Image.fromarray((result.clone().squeeze().detach().cpu().numpy()*256).astype('uint16'))
                raw_depth.save(os.path.join(self.runner_info.work_dir, '{}_uint16.png'.format(batch_data['img_file_basename'][0])))
                
                # Save as PNG
                uncertainty_norm = rescale_tensor(uncertainty, 0, 1)
                # uncertainty_norm[log_dict['count_map']<177*count_thr] = 1
                uncertainty_norm_color = colorize(uncertainty_norm, cmap='jet', vminp=0, vmaxp=100)[:, :, [2, 1, 0]]
                cv2.imwrite(os.path.join(self.runner_info.work_dir, '{}_uncert.png'.format(batch_data['img_file_basename'][0])), uncertainty_norm_color)
                uncert = Image.fromarray((uncertainty.clone().squeeze().detach().cpu().numpy()*256).astype('uint16'))
                uncert.save(os.path.join(self.runner_info.work_dir, '{}_uncert_uint16.png'.format(batch_data['img_file_basename'][0])))
                
                count = Image.fromarray((log_dict['count_map'].clone().squeeze().detach().cpu().numpy()*256).astype('uint16'))
                count.save(os.path.join(self.runner_info.work_dir, '{}_count_uint16.png'.format(batch_data['img_file_basename'][0])))
                # count_map_norm_color = colorize(count_map_norm, cmap='jet', vminp=0, vmaxp=100)[:, :, [2, 1, 0]]
                # cv2.imwrite(os.path.join(self.runner_info.work_dir, '{}_count.png'.format(batch_data['img_file_basename'][0])), count_map_norm_color)
                
            if self.runner_info.rank == 0:
                batch_size = len(result) * world_size
                for _ in range(batch_size):
                    prog_bar.update()
    
    @torch.no_grad()
    def show_gts(self, cai_mode='p16', process_num=4, image_raw_shape=[2160, 3840], patch_split_num=[4, 4]):
        
        results = []
        dataset = self.dataloader.dataset
        loader_indices = self.dataloader.batch_sampler
        
        rank, world_size = get_dist_info()
        if self.runner_info.rank == 0:
            prog_bar = mmengine.utils.ProgressBar(len(dataset))

        for idx, (batch_indices, batch_data) in enumerate(zip(loader_indices, self.dataloader)):
            
            batch_data_collect = self.collect_input(batch_data)
            
            depth_gt = batch_data_collect['depth_gt']
            
            invalid_mask = torch.logical_or(depth_gt <= 0, depth_gt >= 80).squeeze().detach().cpu().numpy()
            
            if self.runner_info.save:
                
                color_pred = colorize(depth_gt, cmap='magma_r', invalid_mask=invalid_mask)[:, :, [2, 1, 0]]
                cv2.imwrite(os.path.join(self.runner_info.work_dir, '{}.png'.format(batch_data['img_file_basename'][0])), color_pred)
            
            if self.runner_info.rank == 0:
                batch_size = len(depth_gt) * world_size
                for _ in range(batch_size):
                    prog_bar.update()
    
    @torch.no_grad()
    def benchmark(self, cai_mode='p16', process_num=4, image_raw_shape=[2160, 3840], patch_split_num=[4, 4], repeat_times=2, log_interval=10):
        results = []
        dataset = self.dataloader.dataset
        loader_indices = self.dataloader.batch_sampler
        
        benchmark_dict = dict(unit='img / s')
        overall_fps_list = []
        
        for time_index in range(repeat_times):
            print(f'Run {time_index + 1}:')
            
            num_warmup = 5
            pure_inf_time = 0
            total_iters = 50

            # benchmark with 200 batches and take the average
            # for i, batch_data in enumerate(dataset):
            for i, (batch_indices, batch_data) in enumerate(zip(loader_indices, self.dataloader)):
                
                batch_data_collect = self.collect_input(batch_data)
                tile_cfg = dict()
                tile_cfg['image_raw_shape'] = image_raw_shape
                tile_cfg['patch_split_num'] = patch_split_num # use a customized value instead of the default [4, 4] for 4K images
            
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_time = time.perf_counter()

                with torch.no_grad():
                    self.model(mode='infer', cai_mode=cai_mode, process_num=process_num, tile_cfg=tile_cfg, **batch_data_collect) # might use test/val to split cases

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start_time

                if i >= num_warmup:
                    pure_inf_time += elapsed
                    if (i + 1) % log_interval == 0:
                        fps = (i + 1 - num_warmup) / pure_inf_time
                        print(f'Done image [{i + 1:<3}/ {total_iters}], '
                            f'fps: {fps:.3f} img / s')

                if (i + 1) == total_iters:
                    fps = (i + 1 - num_warmup) / pure_inf_time
                    print(f'Overall fps: {fps:.3f} img / s\n')
                    benchmark_dict[f'overall_fps_{time_index + 1}'] = round(fps, 2)
                    overall_fps_list.append(fps)
                    break
                    
                break
        
        benchmark_dict['average_fps'] = round(np.mean(overall_fps_list), 2)
        benchmark_dict['fps_variance'] = round(np.var(overall_fps_list), 4)
        print(f'Average fps of {repeat_times} evaluations: '
            f'{benchmark_dict["average_fps"]}')
        print(f'The variance of {repeat_times} evaluations: '
            f'{benchmark_dict["fps_variance"]}')

        analysis_results = mmengine.analysis.get_model_complexity_info(
            self.model, 
            inputs=(
                'infer', 
                batch_data_collect['image_lr'], 
                batch_data_collect['image_hr'], 
                None, None, None, None, 
                tile_cfg, cai_mode, process_num))
        
        
        
        for key, val in analysis_results.items():
            print(f'{key}: {val}')
        print("Model Flops:{}".format(analysis_results['flops_str']))
        print("Model Parameters:{}".format(analysis_results['params_str']))
        
        with open(os.path.join(self.runner_info.work_dir, 'benchmark.txt'), 'w') as f:
            f.write(analysis_results['out_table'])

        # old model
        # Model Flops:14.418T
        # Model Parameters:0.714G
        
        # new model
        # Model Flops:12.534T
        # Model Parameters:0.387G
        
        # raw zoe:
        # Done image [10 / 200], fps: 8.112 img / s
        # Done image [20 / 200], fps: 10.427 img / s 0.1
        # Done image [30 / 200], fps: 11.666 img / s
        # Done image [40 / 200], fps: 12.288 img / s
        # Done image [50 / 200], fps: 12.665 img / s
        # Done image [60 / 200], fps: 12.922 img / s
        # Done image [70 / 200], fps: 13.113 img / s
        # xxx 0.1s/image
        
        # raw zoe + some processing
        # 6.78 image/s, 0.147 s/image

        # 1zoe + 16lightweight model vs 1zoe + 16zoe 
        # Done image [10 / 200], fps: 3.059 img / s (0.3) Done image [10 / 200], fps: 0.728 img / s (1.373)
        # Done image [20 / 200], fps: 3.091 img / s Done image [20 / 200], fps: 0.773 img / s
        # Done image [30 / 200], fps: 2.927 img / s Done image [30 / 200], fps: 0.751 img / s
        # Done image [40 / 200], fps: 2.925 img / s
        # Done image [50 / 200], fps: 2.937 img / s
        # Done image [60 / 200], fps: 2.689 img / s
        # Done image [70 / 200], fps: 2.738 img / s
        # 2.73 image/s 0.366 s/image || 0.73 image/s 1.369 s/image
        
        #
        # Run 1: + fusion      
        # Done image [10 / 200], fps: 1.030 img / s (1) Done image [10 / 200], fps: 0.509 img / s (1.964)
        # Done image [20 / 200], fps: 1.317 img / s Done image [20 / 200], fps: 0.550 img / s
        # Done image [30 / 200], fps: 1.339 img / s Done image [30 / 200], fps: 0.553 img / s
        # Done image [40 / 200], fps: 1.368 img / s
        # Done image [50 / 200], fps: 1.328 img / s
        # Done image [60 / 200], fps: 1.363 img / s
        # Done image [70 / 200], fps: 1.366 img / s
        # 1.4 image/s 0.714 s/image || 0.56 image/s 1.786 s/image