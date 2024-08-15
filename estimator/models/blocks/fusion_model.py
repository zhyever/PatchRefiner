
import torch
import torch.nn as nn
import torch.nn.functional as F
from estimator.registry import MODELS
from estimator.models.blocks.convs import SingleConvCNNLN, DoubleConv, SingleConv, SingleConvLN

class UpSample(nn.Module):
    """Upscaling then cat and DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels, in_channels)

    def forward_hardcode(self, x1, x2, pred1, pred2, update_depth=None):
        x1 = F.interpolate(x1, x2.shape[-2:], mode='bilinear', align_corners=True)
        pred1 = F.interpolate(pred1, x2.shape[-2:], mode='bilinear', align_corners=True)
        pred2 = F.interpolate(pred2, x2.shape[-2:], mode='bilinear', align_corners=True)
        if update_depth is not None:
            update_depth = F.interpolate(update_depth, x2.shape[-2:], mode='bilinear', align_corners=True)
            x = torch.cat([x1, x2, pred1, pred2, update_depth], dim=1)
        else:
            x = torch.cat([x1, x2, pred1, pred2], dim=1)
        return self.conv(x)
    
    def forward(self, x1, feat_list):
        ''' Args:
            x1: the feature map from the skip connection
            feat_list: the feature map list from the encoder and everything you want to concate with current feat maps
        '''
        upscale_feat_list = [x1]
        for feat in feat_list:
            upscale_feat_list.append(F.interpolate(feat, x1.shape[-2:], mode='bilinear', align_corners=True))
            
        x = torch.cat(upscale_feat_list, dim=1)
        return self.conv(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

@MODELS.register_module()
class FusionUnet(nn.Module):
    def __init__(self, input_chl=[32*2, 256*2, 256*2], temp_chl=[32, 256, 256], dec_chl=[256, 32]):
        super(FusionUnet, self).__init__()
            
        self.input_chl = input_chl
        self.temp_chl = temp_chl
        
        self.encoder_layers_1 = nn.ModuleList()
        self.encoder_layers_2 = nn.ModuleList()
        
        for idx, (inp_c, tmp_c) in enumerate(zip(input_chl, temp_chl)):
            layer = SingleConvCNNLN(inp_c, tmp_c)
            self.encoder_layers_1.append(layer)
            layer = SingleConvCNNLN(tmp_c + 2, tmp_c)
            self.encoder_layers_2.append(layer)

        self.decoder_layers = nn.ModuleList()
        temp_chl = temp_chl[::-1]
        _chl = temp_chl[0]
        temp_chl = temp_chl[1:]
        for tmp_c, dec_c in zip(temp_chl, dec_chl):            
            layer = UpSample(tmp_c + _chl + 2, dec_c)
            _chl = dec_c
            self.decoder_layers.append(layer)

        if len(dec_chl) != 0:
            self.final_conv = nn.Conv2d(dec_chl[-1], 1, 3, 1, 1, bias=False)
        else:
            self.final_conv = nn.Conv2d(_chl, 1, 3, 1, 1, bias=False)
        # nn.init.zeros_(self.final_conv.weight)
        
    def forward(
        self, 
        c_feat,
        f_feat,
        pred1, 
        pred2,
        update_base=None):
        
        temp_feat_list = []
        for idx, (c, f) in enumerate(zip(c_feat, f_feat)):
            feat = torch.cat([c, f], dim=1)
            f = self.encoder_layers_1[idx](feat)
            pred1_lvl = F.interpolate(pred1, f.shape[-2:], mode='bilinear', align_corners=True)
            pred2_lvl = F.interpolate(pred2, f.shape[-2:], mode='bilinear', align_corners=True)
            f = torch.cat([f, pred1_lvl, pred2_lvl], dim=1)
            f = self.encoder_layers_2[idx](f)
            temp_feat_list.append(f)
        
        dec_feat = temp_feat_list[0]
        temp_feat_list = temp_feat_list[::-1]
        _feat = temp_feat_list[0]
        temp_feat_list = temp_feat_list[1:]
        
        for idx, (feat, dec_layer) in enumerate(zip(temp_feat_list, self.decoder_layers)):
            dec_feat = dec_layer.forward_hardcode(_feat, feat, pred1, pred2)
            # dec_feat = self.decoder_layers[idx](feat, [_feat, pred1, pred2])
            _feat = dec_feat
            
        final_feat = dec_feat
        final_offset = self.final_conv(final_feat)
        # print(torch.min(final_offset), torch.max(final_offset))
        
        if update_base is not None:
            depth_prediction = update_base + final_offset
            depth_prediction = torch.clamp(depth_prediction, min=0)
        else:
            depth_prediction = final_offset
        
        return depth_prediction
    

       