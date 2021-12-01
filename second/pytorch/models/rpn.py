import time
from enum import Enum

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import spconv
import torchplus
from torchplus.nn import Empty, GroupNorm, Sequential
from torchplus.tools import change_default_args
from second.pytorch.models.rgb_block import *
from torch.autograd import Variable

class RPN_SECOND_FUSION(nn.Module):
    """Compare with RPN, RPNV2 support arbitrary number of stage.
    """
    def __init__(self,
                 use_norm=True,
                 num_class=2,
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2],
                 num_filters=[128, 128, 256],
                 upsample_strides=[1, 2, 4],
                 num_upsample_filters=[256, 256, 256],
                 num_input_features=128,
                 num_anchor_per_loc=2,
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 use_groupnorm=False,
                 num_groups=32,
                 use_bev=False,
                 box_code_size=7,
                 use_rc_net=False,
                 name='rpn'):
        super(RPN_SECOND_FUSION, self).__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self._use_direction_classifier = use_direction_classifier
        self._use_bev = use_bev
        self._use_rc_net = use_rc_net
        # assert len(layer_nums) == 3
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(upsample_strides) == len(layer_nums)
        assert len(num_upsample_filters) == len(layer_nums)
        """
        factors = []
        for i in range(len(layer_nums)):
            assert int(np.prod(layer_strides[:i + 1])) % upsample_strides[i] == 0
            factors.append(np.prod(layer_strides[:i + 1]) // upsample_strides[i])
        assert all([x == factors[0] for x in factors])
        """
        if use_norm:
            if use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)

        in_filters = [num_input_features, *num_filters[:-1]]
        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        blocks = []
        deblocks = []
        
        for i, layer_num in enumerate(layer_nums):
            block = Sequential(
                nn.ZeroPad2d(1),
                Conv2d(
                    in_filters[i], num_filters[i], 3, stride=layer_strides[i]),
                BatchNorm2d(num_filters[i]),
                nn.ReLU(),
            )
            for j in range(layer_num):
                block.add(
                    Conv2d(num_filters[i], num_filters[i], 3, padding=1))
                block.add(BatchNorm2d(num_filters[i]))
                block.add(nn.ReLU())
            blocks.append(block)
            deblock = Sequential(
                ConvTranspose2d(
                    num_filters[i],
                    num_upsample_filters[i],
                    upsample_strides[i],
                    stride=upsample_strides[i]),
                BatchNorm2d(num_upsample_filters[i]),
                nn.ReLU(),
            )
            deblocks.append(deblock)
        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)
        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)
        #########################
        det_num = sum(num_upsample_filters)
        #########################
        self.conv_cls = nn.Conv2d(det_num, num_cls, 1)
        self.conv_box = nn.Conv2d(
            det_num, num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                det_num, num_anchor_per_loc * 2, 1)

        if self._use_rc_net:
            self.conv_rc = nn.Conv2d(
                det_num, num_anchor_per_loc * box_code_size, 1)
        ##########################################################
        self.f_in_planes_det = 64
        net_type = 'FPN18'
        if net_type == 'FPN50':
            num_blocks = [3,4,6,3]
            bb_block = Bottleneck
        elif net_type == 'FPN18':
            num_blocks = [2,2,2,2]
            bb_block = BasicBlock

        # For RGB Feature Network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer_det(bb_block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer_det(bb_block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer_det(bb_block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer_det(bb_block, 512, num_blocks[3], stride=2)
        if net_type == 'FPN18':
            fpn_sizes = [
                    self.layer2[1].conv2.out_channels,
                    self.layer3[1].conv2.out_channels,
                    self.layer4[1].conv2.out_channels]
        else:
            fpn_sizes = [self.layer2[num_blocks[1]-1].conv3.out_channels, self.layer3[num_blocks[2]-1].conv3.out_channels, self.layer4[num_blocks[3]-1].conv3.out_channels]

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

    def _make_layer_det(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.f_in_planes_det, planes, stride))
            self.f_in_planes_det = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, f_view, idxs_norm, bev=None):

        ups = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            ups.append(self.deblocks[i](x))
        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        else:
            x = ups[0]
        bev_feature = x
        ###################### FPN-18 ##########################
        with torch.no_grad():
            f1 = self.maxpool(F.relu(self.bn1(self.conv1(f_view))))
            f2 = self.layer1(f1)
            f3 = self.layer2(f2)
            f4 = self.layer3(f3)
            f5 = self.layer4(f4)
            f_view_features = self.fpn([f3, f4, f5])
            fuse_features2 = F.relu(f_view_features[0]).permute(0,1,3,2)
            fuse_features = fuse_features2.permute(0,1,3,2)
            #######################################################

            ##################### Z concat #########################
            crop_feature_all = []
            for i in range(idxs_norm.size()[1]):
                crop_feature_0 = feature_crop(fuse_features, idxs_norm[:,i,:,:])
                crop_feature_all.append(crop_feature_0)
            #######################################################

            crop_feature_all = torch.stack(crop_feature_all, dim=2)
            N, C, D, H, W = crop_feature_all.shape
            crop_feature_all = crop_feature_all.view(N,C*D,H,W)

        ########################################
        box_preds = self.conv_box(bev_feature)
        cls_preds = self.conv_cls(bev_feature)
        # [N, C, y(H), x(W)]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        ret_dict = {
            "box_preds": box_preds,
            "cls_preds": cls_preds,
        }

        return ret_dict
        
class RPN_FUSION(nn.Module):
    """Compare with RPN, RPNV2 support arbitrary number of stage.
    """
    def __init__(self,
                 use_norm=True,
                 num_class=2,
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2],
                 num_filters=[128, 128, 256],
                 upsample_strides=[1, 2, 4],
                 num_upsample_filters=[256, 256, 256],
                 num_input_features=128,
                 num_anchor_per_loc=2,
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 use_groupnorm=False,
                 num_groups=32,
                 use_bev=False,
                 box_code_size=7,
                 use_rc_net=False,
                 name='rpn'):
        super(RPN_FUSION, self).__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self._use_direction_classifier = use_direction_classifier
        self._use_bev = use_bev
        self._use_rc_net = use_rc_net
        # assert len(layer_nums) == 3
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(upsample_strides) == len(layer_nums)
        assert len(num_upsample_filters) == len(layer_nums)
        """
        factors = []
        for i in range(len(layer_nums)):
            assert int(np.prod(layer_strides[:i + 1])) % upsample_strides[i] == 0
            factors.append(np.prod(layer_strides[:i + 1]) // upsample_strides[i])
        assert all([x == factors[0] for x in factors])
        """
        if use_norm:
            if use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)

        in_filters = [num_input_features, *num_filters[:-1]]
        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        blocks = []
        deblocks = []
        
        for i, layer_num in enumerate(layer_nums):
            # in_f = 256 if i == 0 else in_filters[i]
            in_f = in_filters[i]
            block = Sequential(
                nn.ZeroPad2d(1),
                Conv2d(
                    in_f, num_filters[i], 3, stride=layer_strides[i]),
                BatchNorm2d(num_filters[i]),
                nn.ReLU(),
            )
            for j in range(layer_num):
                block.add(
                    Conv2d(num_filters[i], num_filters[i], 3, padding=1))
                block.add(BatchNorm2d(num_filters[i]))
                block.add(nn.ReLU())
            blocks.append(block)
            deblock = Sequential(
                ConvTranspose2d(
                    num_filters[i],
                    num_upsample_filters[i],
                    upsample_strides[i],
                    stride=upsample_strides[i]),
                BatchNorm2d(num_upsample_filters[i]),
                nn.ReLU(),
            )
            deblocks.append(deblock)
        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)
        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)
        #########################
        det_num = sum(num_upsample_filters)
        #########################
        self.conv_cls = nn.Conv2d(det_num, num_cls, 1)
        self.conv_box = nn.Conv2d(
            det_num, num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                det_num, num_anchor_per_loc * 2, 1)

        if self._use_rc_net:
            self.conv_rc = nn.Conv2d(
                det_num, num_anchor_per_loc * box_code_size, 1)
        ##########################################################
        self.f_in_planes_det = 64
        net_type = 'FPN18'
        if net_type == 'FPN50':
            num_blocks = [3,4,6,3]
            bb_block = Bottleneck
        elif net_type == 'FPN18':
            num_blocks = [2,2,2,2]
            bb_block = BasicBlock

        # For RGB Feature Network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer_det(bb_block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer_det(bb_block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer_det(bb_block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer_det(bb_block, 512, num_blocks[3], stride=2)
        if net_type == 'FPN18':
            fpn_sizes = [
                    self.layer2[1].conv2.out_channels,
                    self.layer3[1].conv2.out_channels,
                    self.layer4[1].conv2.out_channels]
        else:
            fpn_sizes = [self.layer2[num_blocks[1]-1].conv3.out_channels, self.layer3[num_blocks[2]-1].conv3.out_channels, self.layer4[num_blocks[3]-1].conv3.out_channels]

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])
        ####################################################################
        # Fusion Layer
        num_z_feat = 3
        n_feats = 128
        self.rgb_refine = Sequential(
            nn.Conv2d(256*num_z_feat, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, n_feats, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(n_feats),
            nn.ReLU(),
        )
        self.fusion_refine = Sequential(
            nn.Conv2d(n_feats*2, n_feats*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_feats*2),
            nn.ReLU(),
            nn.Conv2d(n_feats*2, n_feats, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(n_feats),
            nn.ReLU(),
        )
        self.bev_gate = BasicGate(n_feats)
        self.crop_gate = BasicGate(n_feats)

    def _make_layer_det(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.f_in_planes_det, planes, stride))
            self.f_in_planes_det = planes * block.expansion
        return nn.Sequential(*layers)

    def feature_crop_interp(self, feature, idx_c, w_size=200,h_size=176):
       '''Auto-Calibrated projection'''
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

           #########################################
           feature_for_offset = feature[i,:,c_coord[:,0].to(torch.int64), c_coord[:,1].to(torch.int64)] * w_coord.view(1,num_coord*grid_num**2)
           self.off_input = feature_for_offset.permute(1,0)
           self.off_input = self.off_input.view(35200, 4, 256).contiguous()
           self.off_input = self.off_input.view(35200, 1024).contiguous()
           self.off_input = self.idx_linear(self.off_input)

           idx_upsamp = idx_upsamp + self.off_input
           mask_w = torch.mul(idx_upsamp[:,0] >= 0, idx_upsamp[:,0] < w_size)
           mask_h = torch.mul(idx_upsamp[:,1] >= 0, idx_upsamp[:,1] < h_size)
           mask_wh = torch.stack((mask_w,mask_h), dim=1)
           idx_upsamp = idx_upsamp * mask_wh.cuda().to(torch.float)
           rep_coord = idx_upsamp.repeat_interleave(grid_num**2,dim=0)
           #########################################

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
           # off_input = temp.permute(1,0)
           # off_input = off_input.view(35200, 4, 256)
           # off_input = off_input.view(35200, 1024)


           crop_feature_each = temp[:,0::4] + temp[:,1::4] + temp[:,2::4] + temp[:,3::4]
           crop_feature_all.append(crop_feature_each)
           # crop_feature1[i,:,:] = feature[i,:,c_coord[:,0].to(torch.int64), c_coord[:,1].to(torch.int64)] * w_coord.view(1,num_coord*grid_num**2)
           # crop_feature[i,:,:] = crop_feature1[i,:,:][:,0::4] + crop_feature1[i,:,:][:,1::4] + crop_feature1[i,:,:][:,2::4] + crop_feature1[i,:,:][:,3::4]
       crop_feature = torch.stack(crop_feature_all,dim=0)
       crop_features_cc = crop_feature.reshape(batch_size, -1, w_size, h_size)
       return crop_features_cc

    def forward(self, x, f_view, idxs_norm, bev=None):

        ups = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            ups.append(self.deblocks[i](x))
        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        else:
            x = ups[0]
        ###################### FPN-18 ##########################
        bev_feature = x
        f1 = self.maxpool(F.relu(self.bn1(self.conv1(f_view))))
        f2 = self.layer1(f1)
        f3 = self.layer2(f2)
        f4 = self.layer3(f3)
        f5 = self.layer4(f4)
        f_view_features = self.fpn([f3, f4, f5])
        fuse_features2 = F.relu(f_view_features[0]).permute(0,1,3,2)
        fuse_features = fuse_features2.permute(0,1,3,2)

        ##################### Z concat #########################
        crop_feature_all = []
        for i in range(idxs_norm.size()[1]):
            crop_feature_0 = feature_crop(fuse_features, idxs_norm[:,i,:,:])
            # crop_feature_0 = self.feature_crop_interp(fuse_features, idxs_norm[:,i,:,:])
            crop_feature_all.append(crop_feature_0)
        #######################################################
        crop_feature_all = torch.stack(crop_feature_all, dim=2)
        N, C, D, H, W = crop_feature_all.shape
        crop_feature_all = crop_feature_all.view(N,C*D,H,W)



        crop_feature = self.rgb_refine(crop_feature_all)
        bev_gated_s1 = self.bev_gate(bev_feature, bev_feature)
        rgb_gated_s1 = self.crop_gate(bev_feature, crop_feature)

        fused_feature_add = torch.cat((bev_gated_s1, rgb_gated_s1),dim=1)
        concat_feat = self.fusion_refine(fused_feature_add)
        
        ########################################
        box_preds = self.conv_box(bev_feature)
        cls_preds = self.conv_cls(concat_feat)
        # [N, C, y(H), x(W)]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        ret_dict = {
            "box_preds": box_preds,
            "cls_preds": cls_preds,
        }
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(concat_feat)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict["dir_cls_preds"] = dir_cls_preds
        if self._use_rc_net:
            rc_preds = self.conv_rc(reg_fused)
            rc_preds = rc_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict["rc_preds"] = rc_preds
        ret_dict['gated_concat_feat'] = concat_feat.permute(1,0,2,3)
        ret_dict['gated_bev_feat'] = bev_feature.permute(1,0,2,3)
        return ret_dict




class Squeeze(nn.Module):
    def forward(self, x):
        return x.squeeze(2)
