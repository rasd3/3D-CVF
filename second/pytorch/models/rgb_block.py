import time

import numpy as np
import spconv
import torch
from torch import nn
from torch.nn import functional as F
import cv2
import os
import string
import random

class PyramidFeatures(nn.Module):
    '''
    FPN pyramid layer
    '''
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)
        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5 = self.P5_1(C5)
        P5_up = self.P5_upsampled(P5)
        P5 = self.P5_2(P5)

        P4 = self.P4_1(C4)
        P4 = P4 + P5_up
        P4_up = self.P4_upsampled(P4)
        P4 = self.P4_2(P4)

        P3 = self.P3_1(C3)
        P3 = P3 + P4_up
        P3 = self.P3_2(P3)

        P6 = self.P6(C5)
        P7 = self.P7_1(P6)
        P7 = self.P7_2(P7)

        return [P3, P4, P5, P6, P7]

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, \
                            kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes))
        self.stride = stride

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    '''
    ResNet's repeated Bottleneck
    '''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):

        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


def vis_feature_act(f_view, layer_list, layer_name, fol_name, seed_name):
    if not os.path.exists('./'+fol_name):
        os.makedirs('./'+fol_name)
    for img_id in range(1):
        input_view = f_view.cpu().detach().numpy()
        input_view = np.array(input_view[img_id,:,:,:])
        input_view = np.transpose(input_view,(1,2,0))
        # import pdb; pdb.set_trace()
        input_view = input_view + np.abs(np.min(input_view))
        input_view = input_view/np.max(input_view)*255
        input_view = input_view[:,:,::-1]
        input_view = np.flip(input_view,1)
        seed_name = id_generator()

        for idx in range(len(layer_list)) :
            ori_run = np.array(layer_list[idx])
            if 'fuse' not in layer_name[idx]:
                ori_run = np.transpose(ori_run,(0,3,2,1))
                ori_run = np.flip(ori_run, axis=1)
            else :
                ori_run = np.transpose(ori_run,(0,2,3,1))
            layer_tot_ori = []
            f1 = ori_run[img_id]
            feat_n = f1*f1
            feat_n = feat_n.sum(0).sum(0)
            max_idx = feat_n.argmax()
            print("imageid", img_id, " max_idx", max_idx)
            ori_img = cv2.resize(ori_run[img_id,:,:,max_idx],dsize=(0, 0),fx=1, fy=1)
            ori_img = ori_img/np.max(ori_img)*255
            ori_img = np.array(ori_img, np.uint8)
            ori_img = cv2.applyColorMap(ori_img, cv2.COLORMAP_JET)
            cv2.imwrite('./'+fol_name+'/'+seed_name+'_'+layer_name[idx].replace('/','_')+'.png', ori_img)
        cv2.imwrite('./'+fol_name+'/'+seed_name+'_input_image.png', input_view)

def vis_feature(f_view, layer_list, layer_name, fol_name):
    if not os.path.exists('./'+fol_name):
        os.makedirs('./'+fol_name)
    for img_id in range(1):

        input_view = f_view.cpu().detach().numpy()
        input_view = np.array(input_view[img_id,:,:,:])
        input_view = np.array(input_view)
        input_view = np.transpose(input_view,(1,2,0))
        # import pdb; pdb.set_trace()
        input_view = input_view + np.abs(np.min(input_view))
        input_view = input_view/np.max(input_view)*255
        input_view = input_view[:,:,::-1]
        seed_name = id_generator()
        for idx in range(len(layer_list)) :
            ori_run = np.array(layer_list[idx])
            if 'fuse' not in layer_name[idx]:
                ori_run = np.transpose(ori_run,(0,3,2,1))
                ori_run = np.flip(ori_run, axis=1)
            else :
                ori_run = np.transpose(ori_run,(0,2,3,1))
            layer_tot_ori = []
            if idx == 0:
                f1 = ori_run[img_id]
                feat_n = f1*f1
                feat_n = feat_n.sum(0).sum(0)
                max_idx = feat_n.argmax()

            print("imageid", img_id, " max_idx", max_idx)
            ori_img = cv2.resize(ori_run[img_id,:,:,max_idx],dsize=(0, 0),fx=1, fy=1)
            ori_img = ori_img/np.max(ori_img)*255
            ori_img = np.array(ori_img, np.uint8)
            ori_img = cv2.applyColorMap(ori_img, cv2.COLORMAP_JET)
            # cv2.imwrite('./'+fol_name+'/img'+str(img_id)+'/'+layer_name[idx].replace('/','_')+'.png', ori_img)
            cv2.imwrite('./'+fol_name+'/'+seed_name+'_'+layer_name[idx].replace('/','_')+'.png', ori_img)
        cv2.imwrite('./'+fol_name+'/'+seed_name+'_input_image.png', input_view)
    return seed_name

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def feature_crop_interp(self, feature, idx_c, w_size=200,h_size=176):
    '''cropping from projection coordinates'''
    grid_num = 2
    num_coord = idx_c.shape[1]
    a1 = torch.tensor([[-0.5,-0.5], [-0.5,0.5], [0.5,-0.5], [0.5, 0.5]]).cuda()

    batch_size = feature.shape[0]
    f_w, f_h = feature.shape[2], feature.shape[3]
    crop_feature_all = []
    
    for i in range(idx_c.shape[0]):
        idx = idx_c[i]
        mask_ori = torch.mul(idx >= 0, idx <= 1).sum(dim=1) != 2
        idx[mask_ori] = 0
        idx_upsamp = idx*torch.tensor([f_w,f_h]).view(1,2).cuda().to(torch.float32) 
        idx_upsamp = idx_upsamp + self.idx_offset
        mask_w = torch.mul(idx_upsamp[:,0] >= 0, idx_upsamp[:,0] < w_size)
        mask_h = torch.mul(idx_upsamp[:,1] >= 0, idx_upsamp[:,1] < h_size)
        mask_wh = torch.stack((mask_w,mask_h), dim=1)
        idx_upsamp = idx_upsamp * mask_wh.cuda().to(torch.float)
        rep_coord = idx_upsamp.repeat_interleave(grid_num**2,dim=0)
        rep_a1 = a1.repeat(num_coord,1).cuda()
        c_coord = torch.floor(rep_coord+rep_a1) ## minus debug!
        cen_coord = c_coord+0.5
        rep_mask = mask_ori.repeat_interleave(grid_num**2, dim=0)
        w_coord = ((cen_coord-rep_coord)**2).sum(1).sqrt()
        w_norm = w_coord[0::4] + w_coord[1::4] + w_coord[2::4] + w_coord[3::4]
        w_norm = w_norm.repeat_interleave(grid_num**2,dim=0)
        w_coord = w_coord/w_norm
        w_coord[rep_mask] = 0
        c_coord[rep_mask,:] = 0
        mask = torch.mul(c_coord[:,0] >= 0, c_coord[:,0] < f_w) + torch.mul(c_coord[:,1] >= 0, c_coord[:,1] < f_h) != 2
        w_coord[mask] = 0
        c_coord[mask, :] = 0
        temp = feature[i,:,c_coord[:,0].to(torch.int64), c_coord[:,1].to(torch.int64)] * w_coord.view(1,num_coord*grid_num**2)
        crop_feature_each = temp[:,0::4] + temp[:,1::4] + temp[:,2::4] + temp[:,3::4]
        crop_feature_all.append(crop_feature_each)
    crop_feature = torch.stack(crop_feature_all,dim=0)
    crop_features_cc = crop_feature.reshape(batch_size, -1, w_size, h_size)
    return crop_features_cc


def feature_crop(feature, idx_c,w_size=200,h_size=176):
    '''cropping from projection coordinates'''
    batch_size = feature.shape[0]
    f_w, f_h = feature.shape[2], feature.shape[3]
    num_coord = idx_c.shape[1] 
    crop_feature = torch.zeros(batch_size, feature.shape[1], num_coord).cuda()
    # crop_feature_all = []
    for i in range(idx_c.shape[0]):
        idx = idx_c[i]
        mask = torch.mul(idx > 0, idx < 1).sum(dim=1) == 2
        mask = mask.view(-1,1)
        idx = idx * mask.to(torch.float32)
        w_crd = (idx[:, 0] * f_w).to(torch.int64)
        h_crd = (idx[:, 1] * f_h).to(torch.int64)
        crop_feature[i,:,:] = feature[i, :, w_crd, h_crd]
    crop_features_cc = crop_feature.reshape(batch_size, -1, w_size, h_size)
    return crop_features_cc


class BasicGate(nn.Module):
    def __init__(self, g_channel):
        super(BasicGate, self).__init__()
        self.g_channel = g_channel
        self.spatial_basic = nn.Conv2d(self.g_channel, 1, kernel_size=3, stride=1, padding=1)
    def forward(self, bev, rgb):
        bev_map = self.spatial_basic(bev)
        scale = torch.sigmoid(bev_map) # broadcasting
        return rgb * scale

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, bev, rgb):
        bev_compress = self.compress(bev)
        bev_out = self.spatial(bev_compress)
        scale = F.sigmoid(bev_out) # broadcasting
        return rgb * scale
