import os
import os.path as osp
import argparse
import torch
import time
from torch.utils.data import DataLoader
from mmengine.utils import mkdir_or_exist
from mmengine.config import Config, DictAction
from mmengine.logging import MMLogger

from estimator.utils import RunnerInfo, setup_env, log_env, fix_random_seed
from estimator.models.builder import build_model
from estimator.datasets.builder import build_dataset
from estimator.tester import Tester
from estimator.models.patchfusion import PatchFusion
from mmengine import print_log
from estimator.utils import get_boundaries, compute_metrics, compute_boundary_metrics, extract_edges
from estimator.utils import colorize
import cv2
import kornia
import numpy as np
def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--work-dir', 
        help='the dir to save logs and models', 
        default=None)
    parser.add_argument(
        '--test-type',
        type=str,
        default='normal',
        help='evaluation type')
    parser.add_argument(
        '--ckp-path',
        type=str,
        help='ckp_path')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--save',
        action='store_true',
        default=False,
        help='save colored prediction & depth predictions')
    parser.add_argument(
        '--cai-mode', 
        type=str,
        default='m1',
        help='m1, m2, or rx')
    parser.add_argument(
        '--process-num',
        type=int, default=4,
        help='batchsize number for inference')
    parser.add_argument(
        '--tag',
        type=str, default='',
        help='infer_infos')
    parser.add_argument(
        '--gray-scale',
        action='store_true',
        default=False,
        help='use gray-scale color map')
    parser.add_argument(
        '--image-raw-shape',
        nargs='+', default=[2160, 3840])
    parser.add_argument(
        '--patch-split-num',
        nargs='+', default=[4, 4])
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def main():
    args = parse_args()

    image_raw_shape=[int(num) for num in args.image_raw_shape]
    patch_split_num=[int(num) for num in args.patch_split_num]
        
    # load config
    cfg = Config.fromfile(args.config)
    
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use ckp path as default work_dir if cfg.work_dir is None
        if '.pth' in args.ckp_path:
            args.work_dir = osp.dirname(args.ckp_path)
        else:
            args.work_dir = osp.join('work_dir', args.ckp_path.split('/')[1])
        cfg.work_dir = args.work_dir
        
    mkdir_or_exist(cfg.work_dir)
    cfg.ckp_path = args.ckp_path
    
    # fix seed
    seed = cfg.get('seed', 5621)
    fix_random_seed(seed)
    
    # start dist training
    if cfg.launcher == 'none':
        distributed = False
        timestamp = torch.tensor(time.time(), dtype=torch.float64)
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(timestamp.item()))
        rank = 0
        world_size = 1
        env_cfg = cfg.get('env_cfg')
    else:
        distributed = True
        env_cfg = cfg.get('env_cfg', dict(dist_cfg=dict(backend='nccl')))
        rank, world_size, timestamp = setup_env(env_cfg, distributed, cfg.launcher)
    
    # build dataloader
    cfg.val_dataloader.batch_size = 1
    cfg.val_dataloader.num_workers = 1
    dataloader_config = cfg.val_dataloader
    # cfg.val_dataloader.dataset.mode = 'gen'
    dataset = build_dataset(cfg.val_dataloader.dataset)
    
    dataset.image_resolution = image_raw_shape
    
    # extract experiment name from cmd
    config_path = args.config
    exp_cfg_filename = config_path.split('/')[-1].split('.')[0]
    ckp_name = args.ckp_path.replace('/', '_').replace('.pth', '')
    dataset_name = dataset.dataset_name
    # log_filename = 'eval_{}_{}_{}_{}.log'.format(timestamp, exp_cfg_filename, ckp_name, dataset_name)
    log_filename = 'eval_{}_{}_{}_{}_{}.log'.format(exp_cfg_filename, args.tag, ckp_name, dataset_name, timestamp)
    
    # prepare basic text logger
    log_file = osp.join(args.work_dir, log_filename)
    log_cfg = dict(log_level='INFO', log_file=log_file)
    log_cfg.setdefault('name', timestamp)
    log_cfg.setdefault('logger_name', 'patchstitcher')
    # `torch.compile` in PyTorch 2.0 could close all user defined handlers
    # unexpectedly. Using file mode 'a' can help prevent abnormal
    # termination of the FileHandler and ensure that the log file could
    # be continuously updated during the lifespan of the runner.
    log_cfg.setdefault('file_mode', 'a')
    logger = MMLogger.get_instance(**log_cfg)
    
    # save some information useful during the training
    runner_info = RunnerInfo()
    runner_info.config = cfg # ideally, cfg should not be changed during process. information should be temp saved in runner_info
    runner_info.logger = logger # easier way: use print_log("infos", logger='current')
    runner_info.rank = rank
    runner_info.distributed = distributed
    runner_info.launcher = cfg.launcher
    runner_info.seed = seed
    runner_info.world_size = world_size
    runner_info.work_dir = cfg.work_dir
    runner_info.timestamp = timestamp
    runner_info.save = args.save
    runner_info.log_filename = log_filename
    runner_info.gray_scale = args.gray_scale
    
    if runner_info.save:
        mkdir_or_exist(args.work_dir)
        runner_info.work_dir = args.work_dir
    # log_env(cfg, env_cfg, runner_info, logger)
    
    if runner_info.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    else:
        val_sampler = None
    
    val_dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=dataloader_config.num_workers,
        pin_memory=True,
        persistent_workers=True,
        sampler=val_sampler)

    for batch_data in val_dataloader:
        image_hr = batch_data['image_hr'].cuda()
        depth_gt = batch_data['depth_gt'].cuda()
        seg_image = batch_data['seg_image'].cuda()
        file_name = batch_data['img_file_basename']
        
        max_depth = 250
        mask = torch.logical_and(depth_gt > 1e-3, depth_gt < max_depth).squeeze()
        h, w = depth_gt.shape[-2:]
        # mask[:h//4, :] = 0
        mask[-h//4:, :] = 0
        mask[:, :w//16] = 0
        mask[:, -w//16:] = 0
        mask = mask.squeeze()
        
        # calculate seg map edges
        seg_image_temp = seg_image
        seg_image_temp_grad = kornia.filters.spatial_gradient(seg_image_temp)
        seg_image_temp_grad = (seg_image_temp_grad[:, :, 0, :, :] ** 2 + seg_image_temp_grad[:, :, 1, :, :] ** 2) ** (1/2)
        seg_image_temp_grad = seg_image_temp_grad.sum(dim=1, keepdim=True)
        seg_map_edge = seg_image_temp_grad.ge(1e-6)
        seg_map_edge = seg_map_edge.float()
        
        _, seg_map_edge = kornia.filters.canny(seg_image)
        seg_map_edge = seg_map_edge > 0
        
        seg_map_edge_extend = kornia.filters.gaussian_blur2d(torch.tensor(seg_map_edge).float(), kernel_size=(7, 7), sigma=(5., 5.), border_type='reflect', separable=True)
        seg_map_edge_extend = seg_map_edge_extend > 0
        seg_map_edge_extend = seg_map_edge_extend.squeeze()
        
        # calculate gt map edge (nosiy)
        gt_edges = extract_edges(depth_gt.detach().cpu(), use_canny=True, preprocess='log')
        gt_edges_extend = kornia.filters.gaussian_blur2d(torch.tensor(gt_edges).cuda().float().unsqueeze(dim=0).unsqueeze(dim=0), kernel_size=(7, 7), sigma=(5., 5.), border_type='reflect', separable=True)
        gt_edges_extend = gt_edges_extend > 0
        gt_edges_extend = gt_edges_extend.squeeze()
        
        # calculate hr rgb map edge (nosiy)
        hr_grad = kornia.filters.spatial_gradient(image_hr)
        hr_grad = (hr_grad[:, :, 0, :, :] ** 2 + hr_grad[:, :, 1, :, :] ** 2) ** (1/2)
        hr_grad_sum = hr_grad.sum(dim=1, keepdim=True)
        hr_edge = hr_grad_sum.ge(0.05 * hr_grad_sum.max())
        hr_edge_extend = kornia.filters.gaussian_blur2d(torch.tensor(hr_edge).float(), kernel_size=(3, 3), sigma=(3., 3.), border_type='reflect', separable=True)
        hr_edge = hr_edge_extend > 0
        hr_edge = hr_edge.float().squeeze()
        
        
        # edge mask (w/o the valid value mask):
        edge_mask = torch.logical_and(seg_map_edge.squeeze(), gt_edges_extend) # seg edge and gt edge = edge
        
        # flatten_mask (contain the valid value mask):
        flatten_mask = torch.logical_and(mask, torch.logical_not(edge_mask)) # valid and ~(edge)
        flatten_mask = torch.logical_and(flatten_mask, torch.logical_not(hr_edge)) # and ~(hr image edge)
        gt_range = colorize(depth_gt.detach().cpu().squeeze(), invalid_mask=torch.logical_not(mask).detach().cpu(), vmin=1e-3, vmax=max_depth)
        gt_flatten_1 = colorize(depth_gt.detach().cpu().squeeze(), invalid_mask=torch.logical_not(edge_mask).detach().cpu(), vmin=1e-3, vmax=max_depth)
        gt_flatten_2 = colorize(depth_gt.detach().cpu().squeeze(), invalid_mask=torch.logical_not(flatten_mask.detach().cpu()), vmin=1e-3, vmax=max_depth)
        
        cv2.imwrite(os.path.join(args.work_dir, '{}_gt_range.png'.format(batch_data['img_file_basename'][0])), gt_range)
        cv2.imwrite(os.path.join(args.work_dir, '{}_gt_flatten1.png'.format(batch_data['img_file_basename'][0])), gt_flatten_1)
        cv2.imwrite(os.path.join(args.work_dir, '{}_gt_flatten2.png'.format(batch_data['img_file_basename'][0])), gt_flatten_2)
        
        # calculate pred edges
        # pred_edges = kornia.filters.sobel(result.unsqueeze(dim=0).unsqueeze(dim=0), normalized=True, eps=1e-6).squeeze()
        # pred_edges = pred_edges / result.squeeze()
        # pred_edges = pred_edges > 0.02
        
        # gt_edges = extract_edges(depth_gt.detach().cpu(), use_canny=True, preprocess='log')
        
        # exit(1200)
if __name__ == '__main__':
    main()