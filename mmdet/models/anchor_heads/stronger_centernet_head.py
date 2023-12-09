import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init, kaiming_init
import numpy as np

from mmdet.ops import ModulatedDeformConvPack as dcn_v2
from mmdet.core import multi_apply, bbox_areas, force_fp32
from mmdet.core.anchor.guided_anchor_target import calc_region
from mmdet.models.losses import ct_focal_loss, giou_loss
from mmdet.models.utils import (build_norm_layer, bias_init_with_prob, ConvModule)
from mmdet.ops.nms import simple_nms
from .anchor_head import AnchorHead
from ..registry import HEADS

class SPP(nn.Module): # size 2X, channel 1/2X
    def __init__(self,in_channel):
        super(SPP,self).__init__()
        self.weight = nn.Conv2d(in_channel*4, 4, 1, stride=1, padding=0, dilation=1)
    def forward(self,x):
        x1 = F.max_pool2d(x,3,stride=1,padding=1) # raw  is 5,9,13
        x2 = F.max_pool2d(x,5,stride=1,padding=2)
        x3 = F.max_pool2d(x,7,stride=1,padding=3)
        weight = self.weight(torch.cat((x,x1,x2,x3),dim=1))
        weight = F.softmax(weight,dim=1)
        x = x * weight[:,0:1,:,:] + x1 * weight[:,1:2,:,:] + x2 * weight[:,2:3,:,:] + x3 * weight[:,3:4,:,:]
        return x

class Fuse(nn.Module): # size 2X, channel 1/2X
    def __init__(self,in_channel):
        super(Fuse,self).__init__()
        self.weight = nn.Conv2d(in_channel*2, 2, 1, stride=1, padding=0, dilation=1)
    def forward(self,x1,x2):
        weight = self.weight(torch.cat((x1,x2),dim=1))
        weight = F.softmax(weight,dim=1)
        return x1 * weight[:,0:1,:,:] + x2 * weight[:,1:2,:,:]

class Fuse_two(nn.Module): # size 2X, channel 1/2X
    def __init__(self,in_channel):
        super(Fuse_two,self).__init__()
        self.mask = nn.Sequential(
                     nn.Conv2d(in_channel, in_channel, 3, stride=1, padding=1, dilation=1,bias = True),
                     nn.Sigmoid()
                     )
    def forward(self,x1,x2):
        mask = self.mask(x2) # x1 is main flow
        return x1 * (1-mask) + x2 * mask

class Fuse_three(nn.Module): # size 2X, channel 1/2X
    def __init__(self,in_channel):
        super(Fuse_three,self).__init__()
        self.mask_2 = nn.Sequential(
                     nn.Conv2d(in_channel, in_channel, 3, stride=1, padding=1, dilation=1,bias = True),
                     nn.Sigmoid()
                     )
        self.mask_3 = nn.Sequential(
                     nn.Conv2d(in_channel, in_channel, 3, stride=1, padding=1, dilation=1,bias = True),
                     nn.Sigmoid()
                     )
    def forward(self,x1,x2,x3):
        mask_2 = self.mask_2(x2) # x1 is main flow
        mask_3 = self.mask_3(x3)
        return x1 * (2 - mask_2 - mask_3) + x2 * mask_2 + x3 * mask_3

class Head_hm(nn.Module): # size 2X, channel 1/2X
    def __init__(self):
        super(Head_hm,self).__init__()
        self.layer1 = nn.Sequential(
                     nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1,bias = True),
                     nn.ReLU(inplace=True),
                     )
        self.layer2 = nn.Sequential(
                     nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1,bias = True),
                     nn.ReLU(inplace=True),
                     )
        self.layer3 = nn.Sequential(
                     nn.Conv2d(128, 80, 1, stride=1, padding=0, dilation=1,bias = True),
                     )
        self.init_weights()

    def init_weights(self):

        for _, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.layer3[0], std=0.01, bias=bias_cls)


    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class Head_wh(nn.Module): # size 2X, channel 1/2X
    def __init__(self,backbone):
        super(Head_wh,self).__init__()
        self.backbone = backbone
        if self.backbone == "resnet50" or self.backbone == "resnet101" or self.backbone == "resnet152" or self.backbone == "res2net101"\
            or self.backbone == "darknet53":
            self.layer0 = nn.Sequential(
                        nn.Conv2d(128, 128, 3, stride=1, padding=1, dilation=1,bias = False),
                        nn.ReLU(inplace=True),
                        )
        self.layer1 = nn.Sequential(
                     nn.Conv2d(128, 64, 3, stride=1, padding=1, dilation=1,bias = False),
                     nn.ReLU(inplace=True),
                     )
        self.layer2 = nn.Sequential(
                     nn.Conv2d(64, 4, 1, stride=1, padding=0, dilation=1,bias = True),
                     )
        self.init_weights()

    def init_weights(self):
        for _, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)

    def forward(self,x):
        if self.backbone == "resnet50" or self.backbone == "resnet101" or self.backbone == "resnet152" or self.backbone == "res2net101"\
            or self.backbone == "darknet53":
            x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class Up2X(nn.Module): # size 2X, channel 1/2X
    def __init__(self,in_channel):
        super(Up2X,self).__init__()
        self.up = nn.Sequential(
                     nn.Conv2d(in_channel, in_channel//2, 3, stride=1, padding=1, dilation=1,bias = False),
                     nn.BatchNorm2d(in_channel//2),
                     nn.ReLU(inplace=True),
                     nn.UpsamplingBilinear2d(scale_factor=2),
                     )
    def forward(self,x):
        x = self.up(x)
        return x

class Down2X(nn.Module): # size 1/2X, channel 2X
    def __init__(self,in_channel):
        super(Down2X,self).__init__()
        self.down = nn.Sequential(
                     nn.Conv2d(in_channel, in_channel*2, 3, stride=2, padding=1, dilation=1,bias = False),
                     nn.BatchNorm2d(in_channel*2),
                     nn.ReLU(inplace=True),
                     )
    def forward(self,x):
        x = self.down(x)
        return x


class CBR(nn.Module):
    '''
    input:[256,32,32],[128,64,64],[64,128,128]
    '''
    def __init__(self,in_channel,out_channel,dilation):
        super(CBR,self).__init__()
        self.cbr = nn.Sequential(
                    nn.Conv2d(in_channels = in_channel, out_channels = out_channel, kernel_size = 3, stride = 1, padding = 1, dilation=dilation, bias = False),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(inplace=True),
                    )
    def forward(self,x):
        x = self.cbr(x)
        return x

class ResUnit(nn.Module):
    def __init__(self,in_channel,out_channel,number,dilation=1):
        super(ResUnit,self).__init__()
        self.resunit = nn.ModuleList()
        for _ in range(number):
            self.resunit.append(CBR(in_channel,out_channel,dilation=dilation))
    def forward(self,x):
        res = x
        for m in self.resunit:
            x = m(x)
        return x + res


class LBFFM(nn.Module):
    """docstring for TTFnet"""
    def __init__(self,backbone, affm = False):
        super(LBFFM, self).__init__()
        self.backbone = backbone
        self.affm = affm

        self.down21_1 = Down2X(in_channel = 256)
        self.down32_1 = Down2X(in_channel = 128)
        self.up12_1 = Up2X(in_channel = 512)
        self.up23_1 = Up2X(in_channel = 256)

        self.down21_2 = Down2X(in_channel = 256)
        self.down32_2 = Down2X(in_channel = 128)
        self.up12_2 = Up2X(in_channel = 512)
        self.up23_2 = Up2X(in_channel = 256)

        self.down21_3 = Down2X(in_channel = 256)
        self.down32_3 = Down2X(in_channel = 128)
        self.up12_3 = Up2X(in_channel = 512)
        self.up23_3 = Up2X(in_channel = 256)

        if self.affm == True:
            # fuse attention
            self.Fuse1_1 = Fuse_two(in_channel = 512)
            self.Fuse1_2 = Fuse_three(in_channel = 256)
            self.Fuse1_3 = Fuse_two(in_channel = 128)

            self.Fuse2_1 = Fuse_two(in_channel = 512)
            self.Fuse2_2 = Fuse_three(in_channel = 256)
            self.Fuse2_3 = Fuse_two(in_channel = 128)

            self.Fuse3_1 = Fuse_two(in_channel = 512)
            self.Fuse3_2 = Fuse_three(in_channel = 256)
            self.Fuse3_3 = Fuse_two(in_channel = 128)

        if self.backbone == "resnet50" or self.backbone == "resnet101" or self.backbone == "resnet152" or self.backbone == "res2net101":
            self.shortcut1_0 = nn.Sequential(
                                CBR(in_channel = 2048,out_channel = 512,dilation = 1),
                                )
            self.shortcut2_0 = nn.Sequential(
                                CBR(in_channel = 1024,out_channel = 256,dilation = 1),
                                )
            self.shortcut3_0 = nn.Sequential(
                                CBR(in_channel = 512,out_channel = 128,dilation = 1),
                                )

        if self.backbone == "darknet53":
            self.shortcut1_0 = nn.Sequential(
                                CBR(in_channel = 1024,out_channel = 512,dilation = 1),
                                )
            self.shortcut2_0 = nn.Sequential(
                                CBR(in_channel = 512,out_channel = 256,dilation = 1),
                                )
            self.shortcut3_0 = nn.Sequential(
                                CBR(in_channel = 256,out_channel = 128,dilation = 1),  
                                )


        self.shortcut1_1 = nn.Sequential(
                            ResUnit(in_channel = 512,out_channel = 512,number = 2,dilation = 1),
                            CBR(in_channel = 512,out_channel = 512,dilation = 1),
                            )
        self.shortcut2_1 = nn.Sequential(
                            ResUnit(in_channel = 256,out_channel = 256,number = 2,dilation = 1),
                            CBR(in_channel = 256,out_channel = 256,dilation = 1),
                            )


        self.shortcut1_2 = nn.Sequential(
                            ResUnit(in_channel = 512,out_channel = 512,number = 2,dilation = 1),
                            CBR(in_channel = 512,out_channel = 512,dilation = 1),
                            )
        self.shortcut3_2 = nn.Sequential(
                            ResUnit(in_channel = 128,out_channel = 128,number = 2,dilation = 1),
                            CBR(in_channel = 128,out_channel = 128,dilation = 1),
                            )


        self.shortcut2_3 = nn.Sequential(
                            ResUnit(in_channel = 256,out_channel = 256,number = 2,dilation = 1),
                            CBR(in_channel = 256,out_channel = 256,dilation = 1),
                            )
        self.shortcut3_3 = nn.Sequential(
                            ResUnit(in_channel = 128,out_channel = 128,number = 2,dilation = 1),
                            CBR(in_channel = 128,out_channel = 128,dilation = 1),
                            )

    #     self.init_weights()

    # def init_weights(self):
    #     for _, m in self.named_modules():
    #         if isinstance(m, nn.Conv2d):
    #             kaiming_init(m)

    def forward(self, feats):
        """

        Args:
            feats: list(tensor).

        Returns:
            hm: tensor, (batch, 80, h, w).
            wh: tensor, (batch, 4, h, w) or (batch, 80 * 4, h, w).
        """
        x1 = feats[-1] # 512 channel
        x2 = feats[-2] # 256 channel
        x3 = feats[-3] # 128 channel
        #x4 = feats[-4] # 128 channel

        if self.backbone == "resnet50" or self.backbone == "resnet101" or self.backbone == "resnet152" or self.backbone == "res2net101"\
            or self.backbone == "darknet53":
            x1 = self.shortcut1_0(x1)
            x2 = self.shortcut2_0(x2)
            x3 = self.shortcut3_0(x3)

        if self.affm == True:
            x1 = self.shortcut1_1(self.Fuse1_1(x1, self.down21_1(x2))) # attention used on up and down, not main flow
            x2 = self.shortcut2_1(self.Fuse1_2(x2, self.up12_1(x1), self.down32_1(x3)))
            x3 = self.Fuse1_3(x3, self.up23_1(x2))

            x1 = self.shortcut1_2(self.Fuse2_1(x1, self.down21_2(x2)))
            x2 = self.Fuse2_2(x2, self.up12_2(x1), self.down32_2(x3))
            x3 = self.shortcut3_2(self.Fuse2_3(x3, self.up23_2(x2)))

            x1 = self.Fuse3_1(x1, self.down21_3(x2))
            x2 = self.shortcut2_3(self.Fuse3_2(x2, self.up12_3(x1), self.down32_3(x3)))
            x3 = self.shortcut3_3(self.Fuse3_3(x3, self.up23_3(x2)))

        else:
            x1 = self.shortcut1_1(x1 + self.down21_1(x2)) # attention used on up and down, not main flow
            x2 = self.shortcut2_1(x2 + self.up12_1(x1) + self.down32_1(x3))
            x3 = x3 + self.up23_1(x2)

            x1 = self.shortcut1_2(x1 + self.down21_2(x2))
            x2 = x2 + self.up12_2(x1) + self.down32_2(x3)
            x3 = self.shortcut3_2(x3 + self.up23_2(x2))

            x1 = x1 + self.down21_3(x2)
            x2 = self.shortcut2_3(x2 + self.up12_3(x1) + self.down32_3(x3))
            x3 = self.shortcut3_3(x3 + self.up23_3(x2))

        return x1,x2,x3

class DH(nn.Module):
    """docstring for TTFnet"""
    def __init__(self,backbone):
        super(DH, self).__init__()
        self.wh_offset_base = 16
        # heads
        self.hm = Head_hm()
        self.wh = Head_wh(backbone)

        # 1 for 512 channel
        self.mdcn1 = nn.Sequential(
                     dcn_v2(512, 256, 3, stride=1,padding=1, dilation=1, deformable_groups=1),
                     nn.BatchNorm2d(256),
                     nn.ReLU(inplace=True),
                     nn.UpsamplingBilinear2d(scale_factor=2),
                     )
        self.mdcn2 = nn.Sequential(
                     dcn_v2(256, 128, 3, stride=1,padding=1, dilation=1, deformable_groups=1),
                     nn.BatchNorm2d(128),
                     nn.ReLU(inplace=True),
                     nn.UpsamplingBilinear2d(scale_factor=2),
                     )
        self.init_weights()

    def init_weights(self):

        for _, m in self.mdcn1.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for _, m in self.mdcn2.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]

        x2 = x2 + self.mdcn1(x1)
        x3 = x3 + self.mdcn2(x2)

        hm = self.hm(x3)
        wh = F.relu(self.wh(x3)) * self.wh_offset_base
        return hm, wh
    
class DDH(nn.Module):
    """docstring for TTFnet"""
    def __init__(self,backbone,affm=False):
        super(DDH, self).__init__()
        self.wh_offset_base = 16
        self.affm = affm
        # heads
        self.hm = Head_hm()
        self.wh = Head_wh(backbone)

        if self.affm == True:
            # sum attention, heatmap and wh
            self.Fuse_two_hm1 = Fuse_two(in_channel = 256)
            self.Fuse_two_hm2 = Fuse_two(in_channel = 128)
            self.Fuse_two_wh1 = Fuse_two(in_channel = 256)
            self.Fuse_two_wh2 = Fuse_two(in_channel = 128)

        # 1 for 512 channel
        self.mdcn1 = nn.Sequential(
                     dcn_v2(512, 256, 3, stride=1,padding=1, dilation=1, deformable_groups=1),
                     nn.BatchNorm2d(256),
                     nn.ReLU(inplace=True),
                     nn.UpsamplingBilinear2d(scale_factor=2),
                     )
        self.mdcn2 = nn.Sequential(
                     dcn_v2(256, 128, 3, stride=1,padding=1, dilation=1, deformable_groups=1),
                     nn.BatchNorm2d(128),
                     nn.ReLU(inplace=True),
                     nn.UpsamplingBilinear2d(scale_factor=2),
                     )


        self.mdcn1_hm = nn.Sequential(
                     dcn_v2(512, 256, 3, stride=1,padding=1, dilation=1, deformable_groups=1),
                     nn.BatchNorm2d(256),
                     nn.ReLU(inplace=True),
                     nn.UpsamplingBilinear2d(scale_factor=2),
                     )
        self.mdcn2_hm = nn.Sequential(
                     dcn_v2(256, 128, 3, stride=1,padding=1, dilation=1, deformable_groups=1),
                     nn.BatchNorm2d(128),
                     nn.ReLU(inplace=True),
                     nn.UpsamplingBilinear2d(scale_factor=2),
                     )


        self.mdcn1_wh = nn.Sequential(
                     dcn_v2(512, 256, 3, stride=1,padding=1, dilation=1, deformable_groups=1),
                     nn.BatchNorm2d(256),
                     nn.ReLU(inplace=True),
                     nn.UpsamplingBilinear2d(scale_factor=2),
                     )
        self.mdcn2_wh = nn.Sequential(
                     dcn_v2(256, 128, 3, stride=1,padding=1, dilation=1, deformable_groups=1),
                     nn.BatchNorm2d(128),
                     nn.ReLU(inplace=True),
                     nn.UpsamplingBilinear2d(scale_factor=2),
                     )
        
        self.init_weights()

    def init_weights(self):

        for _, m in self.mdcn1.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for _, m in self.mdcn2.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for _, m in self.mdcn1_hm.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for _, m in self.mdcn2_hm.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for _, m in self.mdcn1_wh.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for _, m in self.mdcn2_wh.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]

        x2 = x2 + self.mdcn1(x1)
        x3 = x3 + self.mdcn2(x2)

        if self.affm == True:
            x_hm = self.mdcn1_hm(x1)
            x_hm = self.mdcn2_hm(self.Fuse_two_hm1(x_hm,x2)) # attention is computed on x2
            x_hm = self.Fuse_two_hm2(x_hm,x3)

            x_wh = self.mdcn1_wh(x1)
            x_wh = self.mdcn2_wh(self.Fuse_two_wh1(x_wh,x2))
            x_wh = self.Fuse_two_wh2(x_wh,x3)
        else:
            x_hm = self.mdcn1_hm(x1)
            x_hm = self.mdcn2_hm(x_hm + x2) # attention is computed on x2
            x_hm = x_hm + x3

            x_wh = self.mdcn1_wh(x1)
            x_wh = self.mdcn2_wh(x_wh + x2)
            x_wh = x_wh + x3

        hm = self.hm(x_hm)
        wh = F.relu(self.wh(x_wh)) * self.wh_offset_base
        return hm, wh
    
@HEADS.register_module
class Stronger_Centernet_Head(AnchorHead):

    def __init__(self,
                 backbone = "resnet18",
                 affm = False,
                 ddh = False,
                 num_classes=81,
                 wh_offset_base=16.,
                 wh_area_process='log',
                 wh_agnostic=True,
                 wh_gaussian=True,
                 alpha=0.54,
                 beta=0.54,
                 hm_weight=1.,
                 wh_weight=5.,
                 max_objs=128):
        super(AnchorHead, self).__init__()

        assert wh_area_process in [None, 'norm', 'log', 'sqrt']

        self.num_classes = num_classes
        self.wh_offset_base = wh_offset_base
        self.wh_area_process = wh_area_process
        self.wh_agnostic = wh_agnostic
        self.wh_gaussian = wh_gaussian
        self.alpha = alpha
        self.beta = beta
        self.hm_weight = hm_weight
        self.wh_weight = wh_weight
        self.max_objs = max_objs
        self.fp16_enabled = False
        self.backbone = backbone
        self.affm = affm
        self.ddh = ddh

        #self.down_ratio = base_down_ratio // 2 ** len(planes)
        self.down_ratio = 8
        self.num_fg = num_classes - 1
        self.wh_planes = 4 if wh_agnostic else 4 * self.num_fg
        self.base_loc = None

        self.neck = LBFFM(backbone = self.backbone, affm = self.affm)
        if self.ddh == True:
            self.head = DDH(backbone = self.backbone,affm = self.affm)
        else:
            self.head = DH(backbone = self.backbone)

    def init_weights(self):
        pass

    def forward(self, feats):
        return self.head(self.neck(feats))

    @force_fp32(apply_to=('pred_heatmap', 'pred_wh'))
    def get_bboxes(self,
                   pred_heatmap,
                   pred_wh,
                   img_metas,
                   cfg,
                   rescale=False):
        batch, cat, height, width = pred_heatmap.size()
        pred_heatmap = pred_heatmap.detach().sigmoid_()
        wh = pred_wh.detach()

        # perform nms on heatmaps
        heat = simple_nms(pred_heatmap)  # used maxpool to filter the max score

        topk = getattr(cfg, 'max_per_img', 100)
        # (batch, topk)
        scores, inds, clses, ys, xs = self._topk(heat, topk=topk)
        xs = xs.view(batch, topk, 1) * self.down_ratio
        ys = ys.view(batch, topk, 1) * self.down_ratio

        wh = wh.permute(0, 2, 3, 1).contiguous()
        wh = wh.view(wh.size(0), -1, wh.size(3))
        inds = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), wh.size(2))
        wh = wh.gather(1, inds)

        if not self.wh_agnostic:
            wh = wh.view(-1, topk, self.num_fg, 4)
            wh = torch.gather(wh, 2, clses[..., None, None].expand(
                clses.size(0), clses.size(1), 1, 4).long())

        wh = wh.view(batch, topk, 4)
        clses = clses.view(batch, topk, 1).float()
        scores = scores.view(batch, topk, 1)

        bboxes = torch.cat([xs - wh[..., [0]], ys - wh[..., [1]],
                            xs + wh[..., [2]], ys + wh[..., [3]]], dim=2)

        result_list = []
        score_thr = getattr(cfg, 'score_thr', 0.01)
        for batch_i in range(bboxes.shape[0]):
            scores_per_img = scores[batch_i]
            scores_keep = (scores_per_img > score_thr).squeeze(-1)

            scores_per_img = scores_per_img[scores_keep]
            bboxes_per_img = bboxes[batch_i][scores_keep]
            labels_per_img = clses[batch_i][scores_keep]
            img_shape = img_metas[batch_i]['pad_shape']
            bboxes_per_img[:, 0::2] = bboxes_per_img[:, 0::2].clamp(min=0, max=img_shape[1] - 1)
            bboxes_per_img[:, 1::2] = bboxes_per_img[:, 1::2].clamp(min=0, max=img_shape[0] - 1)

            if rescale:
                scale_factor = img_metas[batch_i]['scale_factor']
                bboxes_per_img /= bboxes_per_img.new_tensor(scale_factor)

            bboxes_per_img = torch.cat([bboxes_per_img, scores_per_img], dim=1)
            labels_per_img = labels_per_img.squeeze(-1)
            result_list.append((bboxes_per_img, labels_per_img))
        return result_list

    @force_fp32(apply_to=('pred_heatmap', 'pred_wh'))
    def loss(self,
             pred_heatmap,
             pred_wh,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        all_targets = self.target_generator(gt_bboxes, gt_labels, img_metas)
        hm_loss, wh_loss = self.loss_calc(pred_heatmap, pred_wh, *all_targets)
        return {'losses/loss_heatmap': hm_loss, 'losses/loss_wh': wh_loss}

    def _topk(self, scores, topk):
        batch, cat, height, width = scores.size()

        # both are (batch, 80, topk)
        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), topk)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()

        # both are (batch, topk). select topk from 80*topk
        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), topk)
        topk_clses = (topk_ind / topk).int()
        topk_ind = topk_ind.unsqueeze(2)
        topk_inds = topk_inds.view(batch, -1, 1).gather(1, topk_ind).view(batch, topk)
        topk_ys = topk_ys.view(batch, -1, 1).gather(1, topk_ind).view(batch, topk)
        topk_xs = topk_xs.view(batch, -1, 1).gather(1, topk_ind).view(batch, topk)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def gaussian_2d(self, shape, sigma_x=1, sigma_y=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        h = np.exp(-(x * x / (2 * sigma_x * sigma_x) + y * y / (2 * sigma_y * sigma_y)))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def draw_truncate_gaussian(self, heatmap, center, h_radius, w_radius, k=1):
        h, w = 2 * h_radius + 1, 2 * w_radius + 1
        sigma_x = w / 6
        sigma_y = h / 6
        gaussian = self.gaussian_2d((h, w), sigma_x=sigma_x, sigma_y=sigma_y)
        gaussian = heatmap.new_tensor(gaussian)

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]

        left, right = min(x, w_radius), min(width - x, w_radius + 1)
        top, bottom = min(y, h_radius), min(height - y, h_radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[h_radius - top:h_radius + bottom,
                          w_radius - left:w_radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap

    def target_single_image(self, gt_boxes, gt_labels, feat_shape):
        """

        Args:
            gt_boxes: tensor, tensor <=> img, (num_gt, 4).
            gt_labels: tensor, tensor <=> img, (num_gt,).
            feat_shape: tuple.

        Returns:
            heatmap: tensor, tensor <=> img, (80, h, w).
            box_target: tensor, tensor <=> img, (4, h, w) or (80 * 4, h, w).
            reg_weight: tensor, same as box_target
        """
        output_h, output_w = feat_shape
        heatmap_channel = self.num_fg

        heatmap = gt_boxes.new_zeros((heatmap_channel, output_h, output_w))
        fake_heatmap = gt_boxes.new_zeros((output_h, output_w))
        box_target = gt_boxes.new_ones((self.wh_planes, output_h, output_w)) * -1
        reg_weight = gt_boxes.new_zeros((self.wh_planes // 4, output_h, output_w))

        if self.wh_area_process == 'log':
            boxes_areas_log = bbox_areas(gt_boxes).log()
        elif self.wh_area_process == 'sqrt':
            boxes_areas_log = bbox_areas(gt_boxes).sqrt()
        else:
            boxes_areas_log = bbox_areas(gt_boxes)
        boxes_area_topk_log, boxes_ind = torch.topk(boxes_areas_log, boxes_areas_log.size(0))

        if self.wh_area_process == 'norm':
            boxes_area_topk_log[:] = 1.

        gt_boxes = gt_boxes[boxes_ind]
        gt_labels = gt_labels[boxes_ind]

        feat_gt_boxes = gt_boxes / self.down_ratio
        feat_gt_boxes[:, [0, 2]] = torch.clamp(feat_gt_boxes[:, [0, 2]], min=0,
                                               max=output_w - 1)
        feat_gt_boxes[:, [1, 3]] = torch.clamp(feat_gt_boxes[:, [1, 3]], min=0,
                                               max=output_h - 1)
        feat_hs, feat_ws = (feat_gt_boxes[:, 3] - feat_gt_boxes[:, 1],
                            feat_gt_boxes[:, 2] - feat_gt_boxes[:, 0])

        # we calc the center and ignore area based on the gt-boxes of the origin scale
        # no peak will fall between pixels
        ct_ints = (torch.stack([(gt_boxes[:, 0] + gt_boxes[:, 2]) / 2,
                                (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2],
                               dim=1) / self.down_ratio).to(torch.int)

        h_radiuses_alpha = (feat_hs / 2. * self.alpha).int()
        w_radiuses_alpha = (feat_ws / 2. * self.alpha).int()
        if self.wh_gaussian and self.alpha != self.beta:
            h_radiuses_beta = (feat_hs / 2. * self.beta).int()
            w_radiuses_beta = (feat_ws / 2. * self.beta).int()

        if not self.wh_gaussian:
            # calculate positive (center) regions
            r1 = (1 - self.beta) / 2
            ctr_x1s, ctr_y1s, ctr_x2s, ctr_y2s = calc_region(gt_boxes.transpose(0, 1), r1)
            ctr_x1s, ctr_y1s, ctr_x2s, ctr_y2s = [torch.round(x.float() / self.down_ratio).int()
                                                  for x in [ctr_x1s, ctr_y1s, ctr_x2s, ctr_y2s]]
            ctr_x1s, ctr_x2s = [torch.clamp(x, max=output_w - 1) for x in [ctr_x1s, ctr_x2s]]
            ctr_y1s, ctr_y2s = [torch.clamp(y, max=output_h - 1) for y in [ctr_y1s, ctr_y2s]]

        # larger boxes have lower priority than small boxes.
        for k in range(boxes_ind.shape[0]):
            cls_id = gt_labels[k] - 1

            fake_heatmap = fake_heatmap.zero_()
            self.draw_truncate_gaussian(fake_heatmap, ct_ints[k],
                                        h_radiuses_alpha[k].item(), w_radiuses_alpha[k].item())
            heatmap[cls_id] = torch.max(heatmap[cls_id], fake_heatmap)

            if self.wh_gaussian:
                if self.alpha != self.beta:
                    fake_heatmap = fake_heatmap.zero_()
                    self.draw_truncate_gaussian(fake_heatmap, ct_ints[k],
                                                h_radiuses_beta[k].item(),
                                                w_radiuses_beta[k].item())
                box_target_inds = fake_heatmap > 0
            else:
                ctr_x1, ctr_y1, ctr_x2, ctr_y2 = ctr_x1s[k], ctr_y1s[k], ctr_x2s[k], ctr_y2s[k]
                box_target_inds = torch.zeros_like(fake_heatmap, dtype=torch.uint8)
                box_target_inds[ctr_y1:ctr_y2 + 1, ctr_x1:ctr_x2 + 1] = 1

            if self.wh_agnostic:
                box_target[:, box_target_inds] = gt_boxes[k][:, None]
                cls_id = 0
            else:
                box_target[(cls_id * 4):((cls_id + 1) * 4), box_target_inds] = gt_boxes[k][:, None]

            if self.wh_gaussian:
                local_heatmap = fake_heatmap[box_target_inds]
                ct_div = local_heatmap.sum()
                local_heatmap *= boxes_area_topk_log[k]
                reg_weight[cls_id, box_target_inds] = local_heatmap / ct_div
            else:
                reg_weight[cls_id, box_target_inds] = \
                    boxes_area_topk_log[k] / box_target_inds.sum().float()

        return heatmap, box_target, reg_weight

    def target_generator(self, gt_boxes, gt_labels, img_metas):
        """

        Args:
            gt_boxes: list(tensor). tensor <=> image, (gt_num, 4).
            gt_labels: list(tensor). tensor <=> image, (gt_num,).
            img_metas: list(dict).

        Returns:
            heatmap: tensor, (batch, 80, h, w).
            box_target: tensor, (batch, 4, h, w) or (batch, 80 * 4, h, w).
            reg_weight: tensor, same as box_target.
        """
        with torch.no_grad():
            feat_shape = (img_metas[0]['pad_shape'][0] // self.down_ratio,
                          img_metas[0]['pad_shape'][1] // self.down_ratio)
            heatmap, box_target, reg_weight = multi_apply(
                self.target_single_image,
                gt_boxes,
                gt_labels,
                feat_shape=feat_shape
            )

            heatmap, box_target = [torch.stack(t, dim=0).detach() for t in [heatmap, box_target]]
            reg_weight = torch.stack(reg_weight, dim=0).detach()

            return heatmap, box_target, reg_weight

    def loss_calc(self,
                  pred_hm,
                  pred_wh,
                  heatmap,
                  box_target,
                  wh_weight):
        """

        Args:
            pred_hm: tensor, (batch, 80, h, w).
            pred_wh: tensor, (batch, 4, h, w) or (batch, 80 * 4, h, w).
            heatmap: tensor, same as pred_hm.
            box_target: tensor, same as pred_wh.
            wh_weight: tensor, same as pred_wh.

        Returns:
            hm_loss
            wh_loss
        """
        H, W = pred_hm.shape[2:]
        pred_hm = torch.clamp(pred_hm.sigmoid_(), min=1e-4, max=1 - 1e-4)
        hm_loss = ct_focal_loss(pred_hm, heatmap) * self.hm_weight

        mask = wh_weight.view(-1, H, W)
        avg_factor = mask.sum() + 1e-4

        if self.base_loc is None or H != self.base_loc.shape[1] or W != self.base_loc.shape[2]:
            base_step = self.down_ratio
            shifts_x = torch.arange(0, (W - 1) * base_step + 1, base_step,
                                    dtype=torch.float32, device=heatmap.device)
            shifts_y = torch.arange(0, (H - 1) * base_step + 1, base_step,
                                    dtype=torch.float32, device=heatmap.device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            self.base_loc = torch.stack((shift_x, shift_y), dim=0)  # (2, h, w)

        # (batch, h, w, 4)
        pred_boxes = torch.cat((self.base_loc - pred_wh[:, [0, 1]],
                                self.base_loc + pred_wh[:, [2, 3]]), dim=1).permute(0, 2, 3, 1)
        # (batch, h, w, 4)
        boxes = box_target.permute(0, 2, 3, 1)
        wh_loss = giou_loss(pred_boxes, boxes, mask, avg_factor=avg_factor) * self.wh_weight

        return hm_loss, wh_loss

