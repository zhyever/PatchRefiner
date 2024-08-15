import os
from PIL import Image

scannet_sample_image = '/ibex/ai/home/liz0l/projects/codes/datasets/scannet_pp_select_val_lr/d755b3d9d8/iphone/rgb/frame_006140.jpg'
scannet_depth_1 = '/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/zoedepth/scannet/coarse_pretrain/vis_sp/d755b3d9d8_006140.png'
scannet_depth_2 = '/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/zoedepth/scannet/pr_sync_ssi_midas_grad_weight10/vis_sp/d755b3d9d8_006140.png'
scannet_sample_image = Image.open(scannet_sample_image)
scannet_depth_1 = Image.open(scannet_depth_1)
scannet_depth_2 = Image.open(scannet_depth_2)
scannet_depth_1 = scannet_depth_1.resize(scannet_sample_image.size)
scannet_depth_2 = scannet_depth_2.resize(scannet_sample_image.size)
scannet_depth_1.save('/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/sample_images/scan_id1_zoe.jpg')
scannet_depth_2.save('/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/sample_images/scan_id1_ours.jpg')
scannet_sample_image.save('/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/sample_images/scan_id1.jpg')

scannet_sample_image = '/ibex/ai/home/liz0l/projects/codes/datasets/scannet_pp_select_val_lr/5eb31827b7/iphone/rgb/frame_001430.jpg'
scannet_depth_1 = '/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/zoedepth/scannet/coarse_pretrain/vis_sp/5eb31827b7_001430.png'
scannet_depth_2 = '/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/zoedepth/scannet/pr_sync_ssi_midas_grad_weight10/vis_sp/5eb31827b7_001430.png'
scannet_sample_image = Image.open(scannet_sample_image)
scannet_depth_1 = Image.open(scannet_depth_1)
scannet_depth_2 = Image.open(scannet_depth_2)
scannet_depth_1 = scannet_depth_1.resize(scannet_sample_image.size)
scannet_depth_2 = scannet_depth_2.resize(scannet_sample_image.size)
scannet_depth_1.save('/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/sample_images/scan_id2_zoe.jpg')
scannet_depth_2.save('/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/sample_images/scan_id2_ours.jpg')
scannet_sample_image.save('/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/sample_images/scan_id2.jpg')


scannet_sample_image = '/ibex/ai/home/liz0l/projects/codes/datasets/scannet_pp_select_val_lr/09c1414f1b/iphone/rgb/frame_001570.jpg'
scannet_depth_1 = '/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/zoedepth/scannet/coarse_pretrain/vis_sp/09c1414f1b_001570.png'
scannet_depth_2 = '/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/zoedepth/scannet/pr_sync_ssi_midas_grad_weight10/vis_sp/09c1414f1b_001570.png'
scannet_sample_image = Image.open(scannet_sample_image)
scannet_depth_1 = Image.open(scannet_depth_1)
scannet_depth_2 = Image.open(scannet_depth_2)
scannet_depth_1 = scannet_depth_1.resize(scannet_sample_image.size)
scannet_depth_2 = scannet_depth_2.resize(scannet_sample_image.size)
scannet_depth_1.save('/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/sample_images/scan_id3_zoe.jpg')
scannet_depth_2.save('/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/sample_images/scan_id3_ours.jpg')
scannet_sample_image.save('/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/sample_images/scan_id3.jpg')


cs_sample_image = '/ibex/ai/home/liz0l/projects/codes/datasets/cityscape/leftImg8bit/val/munster/munster_000165_000019_leftImg8bit.png'
cs_depth_1 = '/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/zoedepth/cityscapes/coarse_pretrain/vis_spectral/leftImg8bit_val_munster_munster_000165_000019_leftImg8bit.png'
cs_depth_2 = '/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/zoedepth/cityscapes/pr_sync_ssi_midas_grad_weight75e-2_best/vis_spectral/leftImg8bit_val_munster_munster_000165_000019_leftImg8bit.png'
# cs_depth_2 = '/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/zoedepth/cityscapes/pr_sync_ssi_midas_grad_weight5/vis_sp/leftImg8bit_val_munster_munster_000165_000019_leftImg8bit.png'
cs_sample_image = Image.open(cs_sample_image)
cs_depth_1 = Image.open(cs_depth_1)
cs_depth_2 = Image.open(cs_depth_2)
cs_depth_1 = cs_depth_1.resize(cs_sample_image.size)
cs_depth_2 = cs_depth_2.resize(cs_sample_image.size)
cs_depth_1.save('/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/sample_images/cs_id1_zoe.jpg')
cs_depth_2.save('/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/sample_images/cs_id1_ours.jpg')
cs_sample_image.save('/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/sample_images/cs_id1.jpg')


cs_sample_image = '/ibex/ai/home/liz0l/projects/codes/datasets/cityscape/leftImg8bit/val/munster/munster_000094_000019_leftImg8bit.png'
cs_depth_1 = '/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/zoedepth/cityscapes/coarse_pretrain/vis_spectral/leftImg8bit_val_munster_munster_000094_000019_leftImg8bit.png'
cs_depth_2 = '/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/zoedepth/cityscapes/pr_sync_ssi_midas_grad_weight75e-2_best/vis_spectral/leftImg8bit_val_munster_munster_000094_000019_leftImg8bit.png'
# cs_depth_2 = '/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/zoedepth/cityscapes/pr_sync_ssi_midas_grad_weight5/vis_sp/leftImg8bit_val_munster_munster_000094_000019_leftImg8bit.png'
cs_sample_image = Image.open(cs_sample_image)
cs_depth_1 = Image.open(cs_depth_1)
cs_depth_2 = Image.open(cs_depth_2)
cs_depth_1 = cs_depth_1.resize(cs_sample_image.size)
cs_depth_2 = cs_depth_2.resize(cs_sample_image.size)
cs_depth_1.save('/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/sample_images/cs_id2_zoe.jpg')
cs_depth_2.save('/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/sample_images/cs_id2_ours.jpg')
cs_sample_image.save('/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/sample_images/cs_id2.jpg')

cs_sample_image = '/ibex/ai/home/liz0l/projects/codes/datasets/cityscape/leftImg8bit/val/lindau/lindau_000025_000019_leftImg8bit.png'
cs_depth_1 = '/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/zoedepth/cityscapes/coarse_pretrain/vis_spectral/leftImg8bit_val_lindau_lindau_000025_000019_leftImg8bit.png'
cs_depth_2 = '/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/zoedepth/cityscapes/pr_sync_ssi_midas_grad_weight75e-2_best/vis_spectral/leftImg8bit_val_lindau_lindau_000025_000019_leftImg8bit.png'
# cs_depth_2 = '/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/zoedepth/cityscapes/pr_sync_ssi_midas_grad_weight5/vis_sp/leftImg8bit_val_munster_munster_000094_000019_leftImg8bit.png'
cs_sample_image = Image.open(cs_sample_image)
cs_depth_1 = Image.open(cs_depth_1)
cs_depth_2 = Image.open(cs_depth_2)
cs_depth_1 = cs_depth_1.resize(cs_sample_image.size)
cs_depth_2 = cs_depth_2.resize(cs_sample_image.size)
cs_depth_1.save('/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/sample_images/cs_id3_zoe.jpg')
cs_depth_2.save('/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/sample_images/cs_id3_ours.jpg')
cs_sample_image.save('/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/sample_images/cs_id3.jpg')

image_path = '/ibex/ai/home/liz0l/projects/codes/PatchRefiner/data/kitti/raw/2011_10_03/2011_10_03_drive_0047_sync/image_02/data/0000000576.png'
image = Image.open(image_path)
height = image.height
width = image.width
top_margin = int(height - 352)
left_margin = int((width - 1216) / 2)
image = image.crop(
    (left_margin, top_margin, left_margin + 1216, top_margin + 352))
kitti_depth_1 = '/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/zoedepth/kitti/coarse_pretrain/vis_sp/2011_10_03_2011_10_03_drive_0047_sync_image_02_data_0000000576.png'
kitti_depth_2 = '/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/zoedepth/kitti/pr_sync_ssi_midas_grad_weight1_select/vis_sp/2011_10_03_2011_10_03_drive_0047_sync_image_02_data_0000000576.png'
kitti_depth_1 = Image.open(kitti_depth_1)
kitti_depth_2 = Image.open(kitti_depth_2)
kitti_depth_1 = kitti_depth_1.resize(image.size)
kitti_depth_2 = kitti_depth_2.resize(image.size)
kitti_depth_1.save('/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/sample_images/kitti_id1_zoe.jpg')
kitti_depth_2.save('/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/sample_images/kitti_id1_ours.jpg')
image.save('/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/sample_images/kitti_id1.jpg')

image_path = '/ibex/ai/home/liz0l/projects/codes/PatchRefiner/data/kitti/raw/2011_09_29/2011_09_29_drive_0071_sync/image_02/data/0000000951.png'
image = Image.open(image_path)
height = image.height
width = image.width
top_margin = int(height - 352)
left_margin = int((width - 1216) / 2)
image = image.crop(
    (left_margin, top_margin, left_margin + 1216, top_margin + 352))
kitti_depth_1 = '/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/zoedepth/kitti/coarse_pretrain/vis_sp/2011_09_29_2011_09_29_drive_0071_sync_image_02_data_0000000951.png'
kitti_depth_2 = '/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/zoedepth/kitti/pr_sync_ssi_midas_grad_weight1_select/vis_sp/2011_09_29_2011_09_29_drive_0071_sync_image_02_data_0000000951.png'
kitti_depth_1 = Image.open(kitti_depth_1)
kitti_depth_2 = Image.open(kitti_depth_2)
kitti_depth_1 = kitti_depth_1.resize(image.size)
kitti_depth_2 = kitti_depth_2.resize(image.size)
kitti_depth_1.save('/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/sample_images/kitti_id2_zoe.jpg')
kitti_depth_2.save('/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/sample_images/kitti_id2_ours.jpg')
image.save('/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/sample_images/kitti_id2.jpg')

image_path = '/ibex/ai/home/liz0l/projects/codes/PatchRefiner/data/kitti/raw/2011_09_26/2011_09_26_drive_0009_sync/image_02/data/0000000032.png'
image = Image.open(image_path)
height = image.height
width = image.width
top_margin = int(height - 352)
left_margin = int((width - 1216) / 2)
image = image.crop(
    (left_margin, top_margin, left_margin + 1216, top_margin + 352))
kitti_depth_1 = '/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/zoedepth/kitti/coarse_pretrain/vis_sp/2011_09_26_2011_09_26_drive_0009_sync_image_02_data_0000000032.png'
kitti_depth_2 = '/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/zoedepth/kitti/pr_sync_ssi_midas_grad_weight1_select/vis_sp/2011_09_26_2011_09_26_drive_0009_sync_image_02_data_0000000032.png'
kitti_depth_1 = Image.open(kitti_depth_1)
kitti_depth_2 = Image.open(kitti_depth_2)
kitti_depth_1 = kitti_depth_1.resize(image.size)
kitti_depth_2 = kitti_depth_2.resize(image.size)
kitti_depth_1.save('/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/sample_images/kitti_id3_zoe.jpg')
kitti_depth_2.save('/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/sample_images/kitti_id3_ours.jpg')
image.save('/ibex/ai/home/liz0l/projects/codes/PatchRefiner/work_dir/sample_images/kitti_id3.jpg')
