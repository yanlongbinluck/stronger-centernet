import torch
torch.set_printoptions(threshold=100000)
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init, kaiming_init
import numpy as np
from mmdet.ops import ModulatedDeformConvPack
from mmdet.core import multi_apply, bbox_areas, force_fp32
from mmdet.core.anchor.guided_anchor_target import calc_region
from mmdet.models.losses import ct_focal_loss, giou_loss
from mmdet.models.utils import (build_norm_layer, bias_init_with_prob, ConvModule)
from mmdet.ops.nms import simple_nms
from .anchor_head import AnchorHead
from ..registry import HEADS
import cv2
import time

@HEADS.register_module 
class TTFHead(AnchorHead): # 这里面的参数会被config中的参数覆盖，找config中type为TTFHead的dict

    def __init__(self,
                 inplanes=(64, 128, 256, 512),
                 planes=(256, 128, 64),
                 use_dla=False,
                 base_down_ratio=32,
                 head_conv=256,
                 wh_conv=64,
                 hm_head_conv_num=2,
                 wh_head_conv_num=2,
                 num_classes=81,
                 shortcut_kernel=3,
                 norm_cfg=dict(type='BN'),
                 shortcut_cfg=(1, 2, 3),
                 wh_offset_base=16.,
                 wh_area_process='log',
                 wh_agnostic=True,
                 wh_gaussian=True, # 需要计算wh的高斯椭圆
                 alpha=0.54,   # hm计算高斯椭圆的参数
                 beta=0.54,    # wh计算高斯椭圆的参数
                 hm_weight=1., # 根据论文，hm损失的权重为1
                 wh_weight=5., # wh损失的权重
                 max_objs=128):
        super(AnchorHead, self).__init__() 
        # 这里初始化的是AnchorHead的父类，不是AnchorHead。self是TTFHead类的实例
        # 在ttfnet中，AnchorHead的__init__()没有执行，故也没有调用self._init_layers()等等
        # 在TTFHead类中，重写了init_weights，故AnchorHead中的init_weights也不会执行；
        
        assert len(planes) in [2, 3, 4]
        shortcut_num = min(len(inplanes) - 1, len(planes))
        assert shortcut_num == len(shortcut_cfg)
        assert wh_area_process in [None, 'norm', 'log', 'sqrt']

        self.planes = planes
        self.use_dla = use_dla
        self.head_conv = head_conv
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
        self.down_ratio = base_down_ratio // 2 ** len(planes) # 32/8=4,原始网络resnet18的缩放倍数是32，又上采样3次，即8倍，所以最后出来的特征是4倍放缩
        self.num_fg = num_classes - 1
        self.wh_planes = 4 if wh_agnostic else 4 * self.num_fg # 设置wh特征的厚度是否是每个类都要设4层，默认是不需要
        self.base_loc = None

        # repeat upsampling n times. 32x to 4x by default.
        self.deconv_layers = nn.ModuleList([
            self.build_upsample(inplanes[-1], planes[0], norm_cfg=norm_cfg),
            self.build_upsample(planes[0], planes[1], norm_cfg=norm_cfg)
        ])
        for i in range(2, len(planes)):
            self.deconv_layers.append(
                self.build_upsample(planes[i - 1], planes[i], norm_cfg=norm_cfg))

        padding = (shortcut_kernel - 1) // 2
        self.shortcut_layers = self.build_shortcut(
            inplanes[:-1][::-1][:shortcut_num], planes[:shortcut_num], shortcut_cfg,
            kernel_size=shortcut_kernel, padding=padding)

        # heads
        self.wh = self.build_head(self.wh_planes, wh_head_conv_num, wh_conv)
        self.hm = self.build_head(self.num_fg, hm_head_conv_num)

    def build_shortcut(self,
                       inplanes,
                       planes,
                       shortcut_cfg,
                       kernel_size=3,
                       padding=1): # 创建Unet的结构
        assert len(inplanes) == len(planes) == len(shortcut_cfg)

        shortcut_layers = nn.ModuleList()
        for (inp, outp, layer_num) in zip(
                inplanes, planes, shortcut_cfg):
            assert layer_num > 0
            layer = ShortcutConv2d(
                inp, outp, [kernel_size] * layer_num, [padding] * layer_num)
            shortcut_layers.append(layer)
        return shortcut_layers

    def build_upsample(self, inplanes, planes, norm_cfg=None): # 创建Unet的长采样层，用DCNv2实现，厚度逐渐减半，尺度变为2倍
        mdcn = ModulatedDeformConvPack(inplanes, planes, 3, stride=1,
                                       padding=1, dilation=1, deformable_groups=1)
        up = nn.UpsamplingBilinear2d(scale_factor=2)

        layers = []
        layers.append(mdcn)
        if norm_cfg:
            layers.append(build_norm_layer(norm_cfg, planes)[1])
        layers.append(nn.ReLU(inplace=True))
        layers.append(up)

        return nn.Sequential(*layers)

    def build_head(self, out_channel, conv_num=1, head_conv_plane=None):# 检测HEAD是两个分支，相比于centernet，不要offset
        head_convs = []
        head_conv_plane = self.head_conv if not head_conv_plane else head_conv_plane
        for i in range(conv_num):
            inp = self.planes[-1] if i == 0 else head_conv_plane
            head_convs.append(ConvModule(inp, head_conv_plane, 3, padding=1))

        inp = self.planes[-1] if conv_num <= 0 else head_conv_plane
        head_convs.append(nn.Conv2d(inp, out_channel, 1))
        return nn.Sequential(*head_convs)

    def init_weights(self):
        for _, m in self.shortcut_layers.named_modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)

        for _, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for _, m in self.hm.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.hm[-1], std=0.01, bias=bias_cls)

        for _, m in self.wh.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)

    def forward(self, feats):
        """

        Args:
            feats: list(tensor).

        Returns:
            hm: tensor, (batch, 80, h, w).
            wh: tensor, (batch, 4, h, w) or (batch, 80 * 4, h, w).
        """
        x = feats[-1]
        if not self.use_dla:
            for i, upsample_layer in enumerate(self.deconv_layers):
                x = upsample_layer(x)
                if i < len(self.shortcut_layers):
                    shortcut = self.shortcut_layers[i](feats[-i - 2])
                    x = x + shortcut

        hm = self.hm(x)
        wh = F.relu(self.wh(x)) * self.wh_offset_base

        return hm, wh # 前向传播，返回两个输出

    @force_fp32(apply_to=('pred_heatmap', 'pred_wh')) # 这个是测试时才会用的，通过网络输出得到检测结果
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
        heat = simple_nms(pred_heatmap)  # used maxpool to filter the max score，将热点图执行最大池化，代替nms，这是centernet中的代码

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
        for batch_i in range(bboxes.shape[0]): # 测试时，batch=1。 bboxes.shape[0]是batch
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
            '''
            result_list：[(tensor(100,5),tensor(100,))]
            当batch=1时，result_list是个list，里面是100*5的tensor(bboxes_per_img)，100*1的tensor(labels_per_img)，
            1*5对应4个原图上的xyxy坐标（原图的size不固定），以及score；100*1是检测框的类别id；这里是取的top100个检测框

            如果batch>1,则是[(a,b),(a,b),(a,b)...]
            '''
        return result_list

    @force_fp32(apply_to=('pred_heatmap', 'pred_wh')) # 在混合精度训练时有用，强行将tensor转为fp32
    def loss(self,
             pred_heatmap,
             pred_wh,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        all_targets = self.target_generator(gt_bboxes, gt_labels, img_metas) # 调用下面的函数生成标签
        hm_loss, wh_loss = self.loss_calc(pred_heatmap, pred_wh, *all_targets) # 调用下面的函数计算loss，all_targets中包含target的3项
        return {'losses/ttfnet_loss_heatmap': hm_loss, 'losses/ttfnet_loss_wh': wh_loss} 
        # loss必须要是dict，且带'loss'字符串，方便后边寻找，系数为1和5

    def _topk(self, scores, topk): # 测试时调用，centernet中的代码
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
        # masked_heatmap是heatmap中的一块，它俩是共享内存的，修改masked_heatmap也就是修改heatmap。

        masked_gaussian = gaussian[h_radius - top:h_radius + bottom,
                          w_radius - left:w_radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap) 
            # 比较两个tensor，返回的masked_heatmap是在每个位置上取最大值。
        return heatmap

    def target_single_image(self, gt_boxes, gt_labels, feat_shape): # 输入单张图的GT，类别，应该要输出的大小(128,128)，返回单张图的标签
        """

        Args:
            gt_boxes: tensor, tensor <=> img, (num_gt, 4).这个是原图已经调整到512*512之后，原图上的GT框的xyxy坐标
            gt_labels: tensor, tensor <=> img, (num_gt,).
            feat_shape: tuple.

        Returns:
            heatmap: tensor, tensor <=> img, (80, h, w).
            box_target: tensor, tensor <=> img, (4, h, w) or (80 * 4, h, w).
            reg_weight: tensor, same as box_target
        """
        # 输入：gt_boxes, gt_labels, feat_shape
        # 形状为n*4,n的两个tensor,(128,128)的特征大小
        output_h, output_w = feat_shape
        heatmap_channel = self.num_fg

        heatmap = gt_boxes.new_zeros((heatmap_channel, output_h, output_w))
        fake_heatmap = gt_boxes.new_zeros((output_h, output_w)) # (128,128)
        box_target = gt_boxes.new_ones((self.wh_planes, output_h, output_w)) * -1 # 这里初始化成-1了
        # x.new_one(3,2)，意思是得到一个形状(3,2)，全是1的tensor，但各种属性跟x一样，比如dtype，device等
        # 得到一个(4,128,128)的tensor，centernet中wh的特征厚度是2，这里是4

        reg_weight = gt_boxes.new_zeros((self.wh_planes // 4, output_h, output_w))
        # (1,128,128)的全0

        if self.wh_area_process == 'log':
            boxes_areas_log = bbox_areas(gt_boxes).log()
        elif self.wh_area_process == 'sqrt':
            boxes_areas_log = bbox_areas(gt_boxes).sqrt()
        else:
            boxes_areas_log = bbox_areas(gt_boxes) # bbox_areas计算bbox的面积

        boxes_area_topk_log, boxes_ind = torch.topk(boxes_areas_log, boxes_areas_log.size(0)) 
        # 按某个dim，返回top k个最大值
        # 这里是只有一个dim，按这个dim返回len()个
        # 将bbox的面积从大到小排列返回，并返回当前的数是之前数中的哪个的index 


        if self.wh_area_process == 'norm':
            boxes_area_topk_log[:] = 1.

        gt_boxes = gt_boxes[boxes_ind] # 按面积从大到小，将GT框的顺序重排
        gt_labels = gt_labels[boxes_ind] # 类别也一样

        feat_gt_boxes = gt_boxes / self.down_ratio # GT框除以4，得到特征上的坐标，xyxy
        feat_gt_boxes[:, [0, 2]] = torch.clamp(feat_gt_boxes[:, [0, 2]], min=0,
                                               max=output_w - 1)
        feat_gt_boxes[:, [1, 3]] = torch.clamp(feat_gt_boxes[:, [1, 3]], min=0,
                                               max=output_h - 1)
        feat_hs, feat_ws = (feat_gt_boxes[:, 3] - feat_gt_boxes[:, 1],
                            feat_gt_boxes[:, 2] - feat_gt_boxes[:, 0]) # 特征图上的GT框坐标，xyxy转为hw

        # we calc the center and ignore area based on the gt-boxes of the origin scale
        # no peak will fall between pixels
        ct_ints = (torch.stack([(gt_boxes[:, 0] + gt_boxes[:, 2]) / 2,
                                (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2],
                               dim=1) / self.down_ratio).to(torch.int)

        h_radiuses_alpha = (feat_hs / 2. * self.alpha).int()  # alpha参数
        w_radiuses_alpha = (feat_ws / 2. * self.alpha).int()
        if self.wh_gaussian and self.alpha != self.beta:
            h_radiuses_beta = (feat_hs / 2. * self.beta).int() # beta参数
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
        for k in range(boxes_ind.shape[0]): # 对于单张图中的GT遍历，将对应类别的椭圆放入对应层
            #print('No.{} GT'.format(k+1))
            cls_id = gt_labels[k] - 1

            fake_heatmap = fake_heatmap.zero_() # 128*128，全是0
            
            self.draw_truncate_gaussian(fake_heatmap, ct_ints[k],
                                        h_radiuses_alpha[k].item(), w_radiuses_alpha[k].item()) 
                                        # 画热图，修改函数外边的fake_heatmap，热图在fake_heatmap上
                                        # 热图的范围[0,1]
                                        # 这里返回的热图虽然看起来是只有一个小椭圆，但实际大于0的范围是个矩形，只是值太小，看不出来
                                        # 很显然矩形的范围比椭圆更大
                                        # 一个GT，计算一次热图

            heatmap[cls_id] = torch.max(heatmap[cls_id], fake_heatmap) # heatmap是全0,两个取大的之后，赋值给heatmap
            # 这里生成最终的hm热图

            if self.wh_gaussian:
                if self.alpha != self.beta: # alpha和beta对应论文，表示构建高斯热图时，w，h与生成高斯椭圆半径是否同比例
                    fake_heatmap = fake_heatmap.zero_() # 将fake_heatmap重新原地置0，为wh再计算一次热图。输入参数不一样，这个是beta
                    self.draw_truncate_gaussian(fake_heatmap, ct_ints[k],
                                                h_radiuses_beta[k].item(),
                                                w_radiuses_beta[k].item())

                # data = fake_heatmap.cpu().numpy()
                # data = data*255
                # data = data.astype('uint8')
                # cv2.imwrite('./plot_label/wh_hm/{}.jpg'.format(k),data) # wh的热图，每个GT对应一张热图，所以共有k张图 
                
                box_target_inds = fake_heatmap > 0 # 128*128,取出热图中大于0的位置，是一个矩形。返回tensor仍然跟原来tensor一致，很方便


            else:
                ctr_x1, ctr_y1, ctr_x2, ctr_y2 = ctr_x1s[k], ctr_y1s[k], ctr_x2s[k], ctr_y2s[k]
                box_target_inds = torch.zeros_like(fake_heatmap, dtype=torch.uint8)
                box_target_inds[ctr_y1:ctr_y2 + 1, ctr_x1:ctr_x2 + 1] = 1

            if self.wh_agnostic: # 如果wh只需要4层的厚度
                box_target[:, box_target_inds] = gt_boxes[k][:, None] 
                cls_id = 0
                # gt_boxes[k]是第k个GT的xyxy坐标，原图512*512上面的，是(4,)的，然后转为(4,1)的
                # [:, None]是维度扩增
                # 然后将4个数分别赋值给wh特征(4*128*128)对应的4层上，赋值的位置由box_target_inds确定，上边选出的wh热图中大于0的地方，是个矩形
                # box_target默认是4*128*128、全-1的tensor

                # 这里因为每一层上赋的值都一样，所以效果是对于4张图上同一位置处的矩形是不同颜色，但同一张图上的某个矩形颜色是一样的
                # 先放大矩形，再放小矩形，如果有重叠，是小矩形替换大矩形
            

            else:
                box_target[(cls_id * 4):((cls_id + 1) * 4), box_target_inds] = gt_boxes[k][:, None]

            if self.wh_gaussian: # 如果wh也是高斯椭圆的，而不是像centernet中的一个点

                local_heatmap = fake_heatmap[box_target_inds] # 用矩形的index索引原热图，返回的是一串数。即返回的是热图中大于0的数
                # fake_heatmap (128,128)
                # box_target_inds (128,128)

                ct_div = local_heatmap.sum()
                local_heatmap *= boxes_area_topk_log[k] 
                # boxes_area_topk_log是所有GT的面积从大到小的序列，boxes_area_topk_log[k]是第k大的，是一个数

                reg_weight[cls_id, box_target_inds] = local_heatmap / ct_div 
                # 一个GT对应一张热图，将选出来的热图中大于0的那一串数
                # 乘以bbox的面积，再除以那一串数的和
                # 然后赋值给reg_weight，reg_weight默认是128*128的全0tensor
            else:
                reg_weight[cls_id, box_target_inds] = boxes_area_topk_log[k] / box_target_inds.sum().float()
                # 如果不用高斯的wh热图，那对于不同GT，权重就是不同颜色的矩形块


        # # 画热图
        # for i in range(20):
        #     data = heatmap[i].cpu().numpy()
        #     data = data*255
        #     data = data.astype('uint8')
        #     cv2.imwrite('./plot_label/hm/{}.jpg'.format(i),data)

        # for i in range(4):
        #     data = box_target[i].cpu().numpy()
        #     data = data/np.max(data)
        #     data[data < 0] = 0
        #     data = data*255
        #     data = data.astype('uint8')
        #     cv2.imwrite('./plot_label/wh_target/{}.jpg'.format(i),data)

        # data = reg_weight[0].cpu().numpy()
        # data = data/np.max(data)
        # data = data*255
        # data = data.astype('uint8')
        # cv2.imwrite('./plot_label/reg_weight/reg_weight.jpg',data)

        # print('sleep............')
        # time.sleep(100)



        

        # print(heatmap.dtype,box_target.dtype)
        # print(torch.max(heatmap),torch.min(heatmap))
        # print(torch.max(box_target),torch.min(box_target))
        return heatmap, box_target, reg_weight
        # 返回单张图的标签
        # voc数据集：
        # torch.Size([20, 128, 128]) torch.Size([4, 128, 128]) torch.Size([1, 128, 128])
        # 范围[0,1]的热图；[0,]直接是512*512图上的坐标绝对值；[0,]的权重

    def target_generator(self, gt_boxes, gt_labels, img_metas):
        #print(img_metas)
        """

        Args:
            gt_boxes: list(tensor). tensor <=> image, (gt_num, 4).[[[]],[[]],[[]]],list中有3个n*4的tensor，batch=3
            gt_labels: list(tensor). tensor <=> image, (gt_num,).  [[],[],[]]      list中有3个n*1的tensor，batch=3
            img_metas: list(dict).                                  [{},{},{}]     list中有{}，每张图像的原始信息

        Returns:
            heatmap: tensor, (batch, 80, h, w).
            box_target: tensor, (batch, 4, h, w) or (batch, 80 * 4, h, w).
            reg_weight: tensor, same as box_target.
        """

        '''
        如下：
        [tensor([[141.3120,   0.0000, 510.9760, 499.7120]], device='cuda:0'), # GT框
         tensor([[ 34.8400, 103.9059, 233.4960, 435.2000],[189.4640,  63.2471, 485.4000, 414.1176]], device='cuda:0'), 
         tensor([[ 25.6240,  72.2643, 429.0800, 384.3844]], device='cuda:0')]

        [tensor([12], device='cuda:0'), 
         tensor([12, 12], device='cuda:0'), 20类，类别index
         tensor([7], device='cuda:0')]

        [{'filename': 'data/VOCdevkit/VOC2007/JPEGImages/009405.jpg', 'ori_shape': (375, 500, 3), 'img_shape': (512, 512, 3), 'pad_shape': (512, 512, 3), 'scale_factor': array([1.024    , 1.3653333, 1.024    , 1.3653333], dtype=float32), 'flip': False, 'img_norm_cfg': {'mean': array([123.675, 116.28 , 103.53 ], dtype=float32), 'std': array([58.395, 57.12 , 57.375], dtype=float32), 'to_rgb': True}}, 
        {'filename': 'data/VOCdevkit/VOC2007/JPEGImages/009870.jpg', 'ori_shape': (340, 500, 3), 'img_shape': (512, 512, 3), 'pad_shape': (512, 512, 3), 'scale_factor': array([1.024    , 1.5058824, 1.024    , 1.5058824], dtype=float32), 'flip': True, 'img_norm_cfg': {'mean': array([123.675, 116.28 , 103.53 ], dtype=float32), 'std': array([58.395, 57.12 , 57.375], dtype=float32), 'to_rgb': True}}, 
        {'filename': 'data/VOCdevkit/VOC2007/JPEGImages/002420.jpg', 'ori_shape': (333, 500, 3), 'img_shape': (512, 512, 3), 'pad_shape': (512, 512, 3), 'scale_factor': array([1.024    , 1.5375376, 1.024    , 1.5375376], dtype=float32), 'flip': True, 'img_norm_cfg': {'mean': array([123.675, 116.28 , 103.53 ], dtype=float32), 'std': array([58.395, 57.12 , 57.375], dtype=float32), 'to_rgb': True}}
        ]
        '''

        with torch.no_grad():
            feat_shape = (img_metas[0]['pad_shape'][0] // self.down_ratio,
                          img_metas[0]['pad_shape'][1] // self.down_ratio) # 得出输出特征是512/4=128
            heatmap, box_target, reg_weight = multi_apply(
                self.target_single_image,
                gt_boxes,
                gt_labels,
                feat_shape=feat_shape
            ) 
            # 调用很牛叉的multi_apply(),将处理单张图的函数送进去，将[,,]装的多张图的信息送进去，直接出来多张图输出结果
            # 这个函数在core/utils/misc.py中

            heatmap, box_target = [torch.stack(t, dim=0).detach() for t in [heatmap, box_target]]
            reg_weight = torch.stack(reg_weight, dim=0).detach()
            return heatmap, box_target, reg_weight
            # 返回一个batch的label
            # torch.Size([3, 20, 128, 128]) torch.Size([3, 4, 128, 128]) torch.Size([3, 1, 128, 128])
            # batch=3

    def loss_calc(self,
                  pred_hm,
                  pred_wh,
                  heatmap,
                  box_target,
                  wh_weight): # 计算loss的主函数
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
        hm_loss = ct_focal_loss(pred_hm, heatmap) * self.hm_weight # 这个函数在mmdet/models/losses/focal_loss.py中，是新加的函数

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


class ShortcutConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes,
                 paddings,
                 activation_last=False):
        super(ShortcutConv2d, self).__init__()
        assert len(kernel_sizes) == len(paddings)

        layers = []
        for i, (kernel_size, padding) in enumerate(zip(kernel_sizes, paddings)):
            inc = in_channels if i == 0 else out_channels
            layers.append(nn.Conv2d(inc, out_channels, kernel_size, padding=padding))
            if i < len(kernel_sizes) - 1 or activation_last:
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        y = self.layers(x)
        return y
