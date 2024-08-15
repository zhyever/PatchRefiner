import torch
import numpy as np
import matplotlib.pyplot as plt
from estimator.utils import colorize
from estimator.utils import RandomBBoxQueries
import kornia
from estimator.utils import get_boundaries, compute_metrics, compute_boundary_metrics, extract_edges
from pytorch3d.loss import chamfer_distance
import torch.nn.functional as F

def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1

def generate_pointcloud_ply(xyz, color, pc_file):
    # how to generate a pointcloud .ply file using xyz and color
    # xyz    ndarray  3,N  float
    # color  ndarray  3,N  uint8
    df = np.zeros((6, xyz.shape[1]))
    df[0] = xyz[0]
    df[1] = xyz[1]
    df[2] = xyz[2]
    df[3] = color[0]
    df[4] = color[1]
    df[5] = color[2]
    float_formatter = lambda x: "%.4f" % x
    points =[]
    for i in df.T:
        points.append("{} {} {} {} {} {} 0\n".format
                      (float_formatter(i[0]), float_formatter(i[1]), float_formatter(i[2]),
                       int(i[3]), int(i[4]), int(i[5])))
    file = open(pc_file, "w")
    file.write('''ply
    format ascii 1.0
    element vertex %d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    property uchar alpha
    end_header
    %s
    ''' % (len(points), "".join(points)))
    file.close()
    
# data_dict = torch.load('./work_dir/data_dict.pth')

# fx = data_dict['fx'].cpu()
# fy = data_dict['fy'].cpu()
# u0 = data_dict['u0'].cpu()
# v0 = data_dict['v0'].cpu()

# depth = data_dict['depth'].cpu()
# bs, h, w = depth.shape
        
# intrinsics = torch.zeros((bs, 4, 4))
# intrinsics[:, 0, 0] = fx
# intrinsics[:, 0, 2] = u0
# intrinsics[:, 1, 1] = fy
# intrinsics[:, 1, 2] = v0
# intrinsics[:, 2, 2] = 1.
# intrinsics[:, 3, 3] = 1.

# meshgrid = np.meshgrid(range(w), range(h), indexing='xy')
# id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
# id_coords = depth.new_tensor(id_coords)
# pix_coords = torch.cat([id_coords[0].view(-1).unsqueeze(dim=0), id_coords[1].view(-1).unsqueeze(dim=0)], 0)
# ones = torch.ones(1, w * h)
# pix_coords = torch.cat([pix_coords, ones], dim=0) # 3xHW

# plt.imshow(depth[1])
# plt.savefig('./work_dir/depth.png')

# point_clouds = []
# inv_K_list = []
# gt_edges_list = []
# for idx, intrinsics_bs in enumerate(intrinsics):
#     inv_K = torch.linalg.inv(intrinsics_bs)
#     inv_K_list.append(inv_K)
    
#     gt_edge = torch.from_numpy(extract_edges(depth[idx].detach().cpu(), use_canny=True, preprocess='log'))
#     gt_edges_list.append(gt_edge)
    
#     # cam_points = torch.matmul(inv_K[:3, :3], pix_coords)
#     # cam_points = torch.einsum('cn,n->cn', cam_points, depth[idx].view(-1))
#     # point_clouds.append(cam_points)

# inv_K = torch.stack(inv_K_list, dim=0)
# pix_coords = torch.stack([pix_coords] * len(inv_K_list), dim=0)
# cam_points = torch.bmm(inv_K[:, :3, :3], pix_coords)
# point_clouds = torch.einsum('bcn,bn->bcn', cam_points, depth.flatten(-2))
# gt_edges = torch.stack(gt_edges_list, dim=0).unsqueeze(dim=1)
# # img_tensor = torch.ones((h, w, 3))
# # img_tensor = colorize(depth[1:2, :, :], cmap='magma_r')[:, :, [2, 1, 0]]
# # img_tensor_flatten = torch.tensor(img_tensor).permute(2, 0, 1).flatten(start_dim=1)
# # img_tensor_flatten = img_tensor_flatten.int()
# # generate_pointcloud_ply(point_clouds[1], img_tensor_flatten.numpy(), './work_dir/pc.ply')


# # torch.max(pc[0]), torch.min(pc[0])
# # torch.max(pc[1]), torch.min(pc[1])
# # torch.max(pc[2]), torch.min(pc[2])

# # # x, y, z
# # anchor_generator = Random3DBBoxQueries(1, h, w, [3, 5, 7, 9], N=100)

# point_clouds = point_clouds.reshape(bs, 3, h, w)

# anchor_generator = RandomBBoxQueries(bs, h, w, [3, 7, 15, 31, 251], N=100)

# depth_edge_extend = kornia.filters.gaussian_blur2d(gt_edges.float(), kernel_size=(7, 7), sigma=(5., 5.), border_type='reflect', separable=True)
# depth_edge = depth_edge_extend > 0

# for win_size in [3, 7, 15, 31, 251]:
#     # # target points
#     # # to get target points, create an list of indices that cover the bboxes and index the depth map
#     # # mask out that are invalid
#     # # skip if too many are masked or all are masked
#     abs_coords = anchor_generator.absolute[win_size]  # B, N, 2; babs[b,n] = [x,y]
#     B, N, _two = abs_coords.shape
#     k = win_size // 2
#     # base_xx, baseyy = np.tile(range(width), height), np.repeat(range(height), width)
#     x = torch.arange(-k, k+1)
#     y = torch.arange(-k, k+1)
#     Y, X = torch.meshgrid(y, x)
#     base_coords = torch.stack((X, Y), dim=0)[None, None,...].to(depth.device)  # .shape 1, 1, 2, k, k
    
#     coords = abs_coords[...,None,None] + base_coords  # shape B, N, 2, k, k
    
#     x = coords[:,:,0,:,:]
#     y = coords[:,:,1,:,:]
#     flatten_indices = y * w + x  # .shape B, N, k, k
    
#     flatten_flatten_indices = flatten_indices.flatten(2)  # .shape B, N, kxk
#     depths = depth.unsqueeze(dim=1).expand(-1, N, -1, -1).flatten(2)  # .shape B, N, HxW
#     depth_edges = depth_edge.expand(-1, N, -1, -1).flatten(2)  # .shape B, N, HxW
#     print(point_clouds.shape)
#     point_cloud_process = point_clouds.unsqueeze(dim=1).expand(-1, N, -1, -1, -1).flatten(3)  # .shape B, N, 3, HxW
    
#     # print(depths.shape, flatten_flatten_indices.shape, flatten_flatten_indices.max(), flatten_flatten_indices.min())
    
#     # target_points = depths[flatten_flatten_indices]  # .shape B, N, kxk
#     target_points = torch.gather(depths, dim=-1, index=flatten_flatten_indices.long())
#     depth_edge_points = torch.gather(depth_edges, dim=-1, index=flatten_flatten_indices.long())
#     point_cloud_process = torch.gather(point_cloud_process, dim=-1, index=flatten_flatten_indices.long().unsqueeze(dim=-2).repeat(1, 1, 3, 1))

#     # merge N boxes into batch
#     target_points = target_points.flatten(0, 1)  # .shape BxN, kxk
#     depth_edge_points = depth_edge_points.flatten(0, 1)
#     point_cloud_process = point_cloud_process.flatten(0, 1)
#     print(point_cloud_process.shape)
    

#     mask = torch.logical_and(target_points > 1e-3, target_points < 80)

#     target_points[~mask] = float('inf')
#     target_points = torch.sort(target_points, dim=-1).values
#     target_lengths = torch.sum(mask.long(), dim=-1)
    
#     # find out which of the boxes have mostly invalid regions >50%
#     portion = target_lengths.float() / float(target_points.size(1))
#     valid_value = portion > 0.5

#     edge_portion = torch.sum(depth_edge_points.long(), dim=-1) / float(target_points.size(1))
#     valid_edge = edge_portion > 0.2
    
#     valids = torch.logical_and(valid_value, valid_edge)
#     print(torch.sum(valids))
    
#     # target[win_size] = target_points, target_lengths, valids
    

# NOTE: new start!!!!!
data_dict = torch.load('./work_dir/data_dict.pth')
prediction = data_dict['prediction']
pseudo_label = data_dict['pseudo_label']
gt_depth = data_dict['gt_depth']
camera_info = data_dict['camera_info']
bboxs = data_dict['bboxs']
min_depth = 0
max_depth = 80

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
    print("too few pixels")

assert prediction.shape == pseudo_label.shape, f"Shape mismatch: Expected same shape but got {prediction.shape} and {pseudo_label.shape}."

scale, shift = compute_scale_and_shift(pseudo_label, gt_depth, valid_mask) # compute scale and shift with valid_value mask

scaled_pseudo_label = scale.view(-1, 1, 1) * pseudo_label + shift.view(-1, 1, 1) # scaled preditcion aligned with target

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

h, w = h_t, w_t
pesudo_edge_list = []
for idx in range(bs):
    pesudo_edge = torch.from_numpy(extract_edges(pseudo_label[idx].detach().cpu(), use_canny=True, preprocess='log')).cuda()
    pesudo_edge_list.append(pesudo_edge)
pesudo_edges = torch.stack(pesudo_edge_list, dim=0).unsqueeze(dim=1)
pesudo_edges_extend = kornia.filters.gaussian_blur2d(pesudo_edges.float(), kernel_size=(7, 7), sigma=(5., 5.), border_type='reflect', separable=True)
pesudo_edges = pesudo_edges_extend > 0
region_num = 100

h, w = h_t, w_t
flag = True
# flag = False

if flag is False:
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
    
    pc_prediction = pc_prediction.unsqueeze(dim=1).expand(-1, region_num, -1, -1).flatten(3)  # .shape B, N, 3, HxW
    pc_pseudo_label = pc_pseudo_label.unsqueeze(dim=1).expand(-1, region_num, -1, -1).flatten(3)  # .shape B, N, 3, HxW

else:
    pc_pseudo_label = pseudo_label.unsqueeze(dim=1).expand(-1, region_num, -1, -1).flatten(-2)
    pc_prediction = prediction.unsqueeze(dim=1).expand(-1, region_num, -1, -1).flatten(-2)

# img_tensor = torch.ones((h, w, 3))
# img_tensor = colorize(gt_depth[0:1, :, :].detach().cpu(), cmap='magma_r')[:, :, [2, 1, 0]]
# img_tensor_flatten = torch.tensor(img_tensor).permute(2, 0, 1).flatten(start_dim=1)
# img_tensor_flatten = img_tensor_flatten.int()

# print(gt_depth.shape)
# plt.imshow(gt_depth[0].detach().cpu())
# plt.savefig('./work_dir/depth.png')

# generate_pointcloud_ply(pc_pseudo_label[2].detach().cpu(), img_tensor_flatten.numpy(), './work_dir/pc_scale_pl.ply')
# generate_pointcloud_ply(gt[2].detach().cpu(), img_tensor_flatten.numpy(), './work_dir/pc_gt.ply')
# generate_pointcloud_ply(pc_prediction[2].detach().cpu(), img_tensor_flatten.numpy(), './work_dir/pc_pred.ply')

# process windows
sampling_mask = sampling_mask.unsqueeze(dim=1).expand(-1, region_num, -1, -1).flatten(2)
pesudo_edges = pesudo_edges.expand(-1, region_num, -1, -1).flatten(2)  # .shape B, N, HxW
# pc_prediction = pc_prediction.unsqueeze(dim=1).expand(-1, region_num, -1, -1).flatten(3)  # .shape B, N, 3, HxW
# pc_pseudo_label = pc_pseudo_label.unsqueeze(dim=1).expand(-1, region_num, -1, -1).flatten(3)  # .shape B, N, 3, HxW

loss = torch.DoubleTensor([0.0]).cuda()
w_window = 1.0
w_window_sum = 0.0
# window_size = [3, 7, 15, 31, 63]
window_size = [63]
anchor_generator = RandomBBoxQueries(4, 256, 512, window_size, N=region_num)
for idx, win_size in enumerate(window_size):
    # # target points
    # # to get target points, create an list of indices that cover the bboxes and index the depth map
    # # mask out that are invalid
    # # skip if too many are masked or all are masked
    abs_coords = anchor_generator.absolute[win_size].to(prediction.device)  # B, N, 2; babs[b,n] = [x,y]
    abs_coords[:, :, 0] = 350
    abs_coords[:, :, 1] = 64
    
    B, N, _two = abs_coords.shape
    k = win_size // 2
    # base_xx, baseyy = np.tile(range(width), height), np.repeat(range(height), width)
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
    # merge N boxes into batch
    sampling_mask_sample = sampling_mask_sample.flatten(0, 1) # BxN HxW
    pesudo_edges_sample = pesudo_edges_sample.flatten(0, 1) # BxN HxW
    pc_pseudo_label_len = sampling_mask_sample.float().sum(dim=-1).long()
    
    if flag is False:
        pc_prediction_sample = torch.gather(pc_prediction, dim=-1, index=flatten_flatten_indices.long().unsqueeze(dim=-2).repeat(1, 1, 3, 1))
        pc_pseudo_label_sample = torch.gather(pc_pseudo_label, dim=-1, index=flatten_flatten_indices.long().unsqueeze(dim=-2).repeat(1, 1, 3, 1))
        pc_prediction_sample = pc_prediction_sample.flatten(0, 1) # BxN HxW
        pc_pseudo_label_sample = pc_pseudo_label_sample.flatten(0, 1) # BxN HxW
        pc_pseudo_label_sample[~sampling_mask_sample.unsqueeze(dim=-2).repeat(1, 3, 1)] = float('inf')
        values, indices = torch.sort(pc_pseudo_label_sample[:, 0, :], dim=-1)
        expanded_indices = indices.unsqueeze(1).expand(-1, 3, -1)
        pc_pseudo_label_sample = torch.gather(pc_pseudo_label_sample, 2, expanded_indices)
    else:
        pc_prediction_sample = torch.gather(pc_prediction, dim=-1, index=flatten_flatten_indices.long())
        pc_pseudo_label_sample = torch.gather(pc_pseudo_label, dim=-1, index=flatten_flatten_indices.long())
        pc_prediction_sample = pc_prediction_sample.flatten(0, 1) # BxN HxW
        pc_pseudo_label_sample = pc_pseudo_label_sample.flatten(0, 1) # BxN HxW
        
        pc_pseudo_label_sample[~sampling_mask_sample] = float('inf')
        values, indices = torch.sort(pc_pseudo_label_sample, dim=-1)
        pc_pseudo_label_sample = torch.gather(pc_pseudo_label_sample, -1, indices)
        print(pc_pseudo_label_sample.shape)


    pc_pseudo_label_sample_copy = pc_pseudo_label_sample.clone()
    

    edge_portion = torch.sum(pesudo_edges_sample.long(), dim=-1) / float(pc_pseudo_label_sample.size(-1))
    value_portion = torch.sum(sampling_mask_sample.long(), dim=-1) / float(pc_pseudo_label_sample.size(-1))
    
    valids = torch.logical_and(edge_portion > 0.1, value_portion > 0.1)
    
    pc_prediction_sample_valid, pc_pseudo_label_sample_valid, pc_pseudo_label_len_valid = pc_prediction_sample[valids], pc_pseudo_label_sample[valids], pc_pseudo_label_len[valids]
    pc_pseudo_label_sample_copy_valid = pc_pseudo_label_sample_copy[valids]
    
    # print(pc_prediction_sample_valid.shape)
    # print(pc_pseudo_label_sample_valid.shape)
    # pc_prediction_sample_valid_back = pc_prediction_sample_valid.reshape(-1, 3, 63, 63)
    # pc_pseudo_label_sample_valid_back = pc_pseudo_label_sample_valid.reshape(-1, 3, 63, 63)
    # pc_pseudo_label_sample_copy_valid = pc_pseudo_label_sample_copy_valid.reshape(-1, 3, 63, 63)
    # plt.subplot(3, 1, 1)
    # plt.imshow(pc_prediction_sample_valid_back[0, -1, :, :].detach().cpu())
    # plt.subplot(3, 1, 2)
    # plt.imshow(pc_pseudo_label_sample_valid_back[0, -1, :, :].detach().cpu())
    # plt.subplot(3, 1, 3)
    # plt.imshow(pc_pseudo_label_sample_copy_valid[0, -1, :, :].detach().cpu())
    
    if pc_pseudo_label_len_valid.shape[0] == 0:
        continue
    
    print(pc_prediction_sample_valid.shape)
    # loss_win, _ = chamfer_distance(pc_prediction_sample_valid.permute(0, 2, 1), pc_pseudo_label_sample_valid.permute(0, 2, 1), y_lengths=pc_pseudo_label_len_valid)
    # loss_win, _ = chamfer_distance(pc_prediction_sample_valid.unsqueeze(dim=-1), pc_pseudo_label_sample_valid.unsqueeze(dim=-1), y_lengths=pc_pseudo_label_len_valid)
    loss_win, _ = chamfer_distance(pc_pseudo_label_sample_valid.unsqueeze(dim=-1), pc_pseudo_label_sample_valid.unsqueeze(dim=-1), x_lengths=pc_pseudo_label_len_valid, y_lengths=pc_pseudo_label_len_valid)

    print(loss_win)
    loss += loss_win * w_window
    w_window_sum += w_window

loss = loss / w_window_sum
print(loss)
