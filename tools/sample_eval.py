from PIL import Image
import numpy as np
from estimator.utils import get_boundaries, compute_metrics
import torch

for i in range(200):
    try:
        depth_gt = Image.open('/ibex/ai/home/liz0l/codes/PatchRefiner/data/kitti/depth/2011_09_26_drive_0036_sync/proj_depth/groundtruth/image_02/{:0>10d}.png'.format(i))
        
        height = depth_gt.height
        width = depth_gt.width
        top_margin = int(height - 352)
        left_margin = int((width - 1216) / 2)
        depth_gt = depth_gt.crop(
            (left_margin, top_margin, left_margin + 1216, top_margin + 352))

        baseline = np.asarray(Image.open('/ibex/ai/home/liz0l/codes/PatchRefiner/work_dir/zoedepth/kitti/pr/vis/011_09_26_2011_09_26_drive_0036_sync_image_02_data_{:0>10d}_uint16.png'.format(i)).resize((depth_gt.size[0], depth_gt.size[1]), Image.Resampling.BILINEAR), dtype=np.float32) / 256
        ours =  np.asarray(Image.open('/ibex/ai/home/liz0l/codes/PatchRefiner/work_dir/zoedepth/kitti/pr_sync_ssi_midas_grad_weight1_select/vis/011_09_26_2011_09_26_drive_0036_sync_image_02_data_{:0>10d}_uint16.png'.format(i)).resize((depth_gt.size[0], depth_gt.size[1]), Image.Resampling.BILINEAR), dtype=np.float32) / 256
        
    except FileNotFoundError:
        continue
    
    


    depth_gt = np.asarray(depth_gt, dtype=np.float32) / 256.0
    depth_gt = torch.tensor(depth_gt)
    baseline = torch.tensor(baseline)
    ours = torch.tensor(ours)

    print(depth_gt.shape, baseline.shape, ours.shape)
    rmse_base = compute_metrics(depth_gt, baseline, disp_gt_edges=baseline, min_depth_eval=0, max_depth_eval=80, garg_crop=True, eigen_crop=False, dataset='kitti')['rmse']
    rmse_ours = compute_metrics(depth_gt, ours, disp_gt_edges=baseline, min_depth_eval=0, max_depth_eval=80, garg_crop=True, eigen_crop=False, dataset='kitti')['rmse']
    
    print(rmse_base, rmse_ours, rmse_ours > rmse_ours, i)
