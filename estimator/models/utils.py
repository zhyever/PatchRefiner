import torch
import cv2
import numpy as np
import torch.nn.functional as F
import copy

def get_activation(name, bank):
    def hook(model, input, output):
        bank[name] = output
    return hook

class HookTool: 
    def __init__(self):
        self.feat = None 

    def hook_in_fun(self, module, fea_in, fea_out):
        self.feat = fea_in
        
    def hook_out_fun(self, module, fea_in, fea_out):
        self.feat = fea_out

# class RunningAverageMap:
#     """ Saving avg depth estimation results."""
#     def __init__(self, average_map, count_map):
#         self.average_map = average_map
#         self.count_map = count_map
#         self.average_map = self.average_map / self.count_map
    
#     def update(self, pred_map, ct_map):
#         self.average_map = (pred_map + self.count_map * self.average_map) / (self.count_map + ct_map)
#         self.count_map = self.count_map + ct_map
        
#     def resize(self, resolution):
#         temp_avg_map = self.average_map.unsqueeze(0).unsqueeze(0)
#         temp_count_map = self.count_map.unsqueeze(0).unsqueeze(0)
#         self.average_map = F.interpolate(temp_avg_map, size=resolution).squeeze()
#         self.count_map = F.interpolate(temp_count_map, size=resolution, mode='bilinear', align_corners=True).squeeze()


class RunningAverageMap:
    """ Saving avg depth estimation results."""
    def __init__(self, average_map, count_map, update_flag=False):
        self.mask = (count_map > 0)
        self.count_map = count_map
        self.average_map_init = average_map # m1 returned
        self.average_map = average_map
        self.update_flag = False
        
    def update(self, pred_map, ct_map, **kwargs):
        self.update_flag = True
        # mask = torch.logical_and(ct_map > 0, self.count_map > 0)
        mask = ct_map > 0
        self.average_map[mask] = (pred_map[mask] * ct_map[mask] + self.count_map[mask] * self.average_map[mask]) / (self.count_map[mask] + ct_map[mask])
        self.count_map[mask] = self.count_map[mask] + ct_map[mask]
        
    def resize(self, resolution):
        temp_avg_map = self.average_map.unsqueeze(0).unsqueeze(0)
        temp_count_map = self.count_map.unsqueeze(0).unsqueeze(0)
        
        self.average_map = F.interpolate(temp_avg_map, size=resolution).squeeze()
        self.count_map = F.interpolate(temp_count_map, size=resolution, mode='bilinear', align_corners=True).squeeze()
    
    def get_avg_map(self):
        if self.update_flag:
            return self.average_map
        else:
            return self.average_map_init

# def generatemask(size):
#     # Generates a Guassian mask
#     mask = np.zeros(size, dtype=np.float32)
#     sigma = int(size[0]/16)
#     k_size = int(2 * np.ceil(2 * int(size[0]/16)) + 1)
#     mask[int(0.1*size[0]):size[0] - int(0.1*size[0]), int(0.1*size[1]): size[1] - int(0.1*size[1])] = 1
#     mask = cv2.GaussianBlur(mask, (int(k_size), int(k_size)), sigma)
#     mask = (mask - mask.min()) / (mask.max() - mask.min())
#     mask = mask.astype(np.float32)
#     return mask

def generatemask(size, border=0.1):
    # Generates a Guassian mask
    mask = np.zeros(size, dtype=np.float32)
    sigma = int(size[0]/16)
    k_size = int(2 * np.ceil(2 * int(size[0]/16)) + 1)
    mask[int(border*size[0]):size[0] - int(border*size[0]), int(border*size[1]): size[1] - int(border*size[1])] = 1
    mask = cv2.GaussianBlur(mask, (int(k_size), int(k_size)), sigma)
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = mask.astype(np.float32)
    return mask