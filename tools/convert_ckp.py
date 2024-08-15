# import torch
# ckp_pth = '/home/liz0l/shortcuts/monodepth3_checkpoints/ZoeDepthv1_16-Apr_12-07-9614edef0d28_latest.pt'
# ckp = torch.load(ckp_pth)
# ckp_update = dict()
# ckp_update['model_state_dict'] = ckp['model']
# torch.save(ckp_update, './work_dir/ZoeDepthv1_kitti.pth')


import torch
ckp_pth = './work_dir/zoedepth/u4k/coarse_pretrain/checkpoint_24.pth'
ckp = torch.load(ckp_pth)
ckp_update = ckp['model_state_dict']
torch.save(ckp_update, './work_dir/zoedepth/u4k/coarse_pretrain/checkpoint_24.pt')