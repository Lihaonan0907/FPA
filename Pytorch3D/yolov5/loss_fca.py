# Loss functions
import fnmatch
from PIL import Image

import torch
import torch.nn as nn
import math
import numpy as np
from typing import Tuple
# from models.utils.general import bbox_iou
from utils.torch_utils import de_parallel
import os
import torch.nn.functional as F
from torchvision import transforms
import torchvision
import time
from torch.utils.data import Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# The training entry point can override this module-level device after loading config.
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def set_device(device_like):
    """Set the module-level device used by legacy helper code."""
    global device
    device = torch.device(device_like)
    return device


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps

'''
对一批图像补丁（adv_patch）进行多种随机变换，包括调整对比度、亮度、添加随机噪声、旋转、缩放、平移等操作。
初始化参数：
对比度（contrast）、亮度（brightness）、噪声因子（noise_factor）、
缩放范围（min_scale 和 max_scale）、平移参数（translation_x 和 translation_y）等。

数据形状和尺寸处理：
首先，代码获取输入图像批次的形状信息，并获取补丁的形状信息。
然后，通过 mask 变量提取补丁的遮罩信息（通常表示补丁的存在与否），
并从 adv_patch 中移除遮罩通道。这是因为后续的操作会应用于图像通道，而不包括遮罩。

调整对比度和亮度：
通过随机生成对比度和亮度参数，对 adv_patch 中的图像进行对比度和亮度的调整。
添加随机噪声：
通过生成随机噪声并添加到 adv_patch 中，引入图像的随机性

缩放和平移：
随机生成缩放比例和平移参数，然后根据这些参数对补丁进行缩放和平移操作。
仿射变换：
根据前面的缩放和平移参数，构造仿射矩阵 theta，然后使用 PyTorch 的 F.affine_grid 和 F.grid_sample 函数对图像进行仿射变换。
这些操作可以实现缩放和平移，同时考虑了遮罩信息，以确保图像和遮罩的一致性。
最终输出：
输出经过数据增强操作后的图像批次 adv_batch，同时返回了用于评估的坐标信息 gt。
gt 包含了增强后的图像中补丁的位置信息，用于后续的评估或监督任务。
'''
class PatchTransformer(nn.Module):
    """Apply spatial transforms to a 4-channel adversarial patch and return the transformed patch plus bbox labels."""
    """PatchTransformer: transforms batch of patches

    Module providing the functionality necessary to transform a batch of patches, randomly adjusting brightness and
    contrast, adding random amount of noise, and rotating randomly. Resizes patches according to as size based on the
    batch of labels, and pads them to the dimension of an image.

    """

    def __init__(self):
        super(PatchTransformer, self).__init__()
        # self.min_contrast = 0.9
        # self.max_contrast = 1.1
        # self.min_brightness = -0.1
        # self.max_brightness = 0.1
        # self.noise_factor = 0.02
        # self.min_contrast = 1.0
        # self.max_contrast = 1.0
        # self.min_brightness = 0.0
        # self.max_brightness = 0.0
        # self.noise_factor = 0.0
        # self.min_scale = -0.28  # log 0.75
        # self.max_scale = 0.47  # log 1.60
        #没有必要在这里缩放
        self.min_scale = 0.0  # log 0.75
        self.max_scale = 0.0  # log 1.60

        # self.translation_x = 0.8
        # self.translation_y = 1.0
        #不需要在这里位移
        # self.translation_x = 0.6
        # self.translation_y = 0.5
        self.translation_x = 0.4
        self.translation_y = 0.6
    def forward(self, img_batch, adv_patch):
        # import matplotlib.pyplot as plt
        B, _, Ht, Wt = img_batch.shape
        _, _, Ho, Wo = adv_patch.shape
        adv_patch = adv_patch[:B]

        mask = (adv_patch[:, -1:, ...] > 0).to(adv_patch) #[10,1,2048,2048]
        adv_patch = adv_patch[:, :-1, ...] #[10,3,2048,2048]

        # contrast = adv_patch.new(size=[B]).uniform_(self.min_contrast, self.max_contrast)
        # contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) #对比度

        # brightness = adv_patch.new(size=[B]).uniform_(self.min_brightness, self.max_brightness)
        # brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  #亮度

        # noise = adv_patch.new(adv_patch.shape).uniform_(-1, 1) * self.noise_factor

        # adv_patch = adv_patch * contrast + brightness + noise #增加随机噪声,全部都是作用于补丁上
        # # adv = adv_patch
        # # with open('output.txt3', 'a') as file:
        # #     file.write(f'Iteration {adv_patch}:\n')
        # adv_patch = adv_patch.clamp(0, 1) #[4,3,800,800]


        adv_patch = torch.cat([adv_patch, mask], dim=1) #[4,4,800,800]
        # adv = adv_patch

        
        #随机数的范围由 self.min_scale 和 self.max_scale 控制，并且通过 exp() 函数将其转换为正数，以表示缩放因子。
        scale = adv_patch.new(size=[B]).uniform_(self.min_scale, self.max_scale).exp() #[4]
        '''
        mask 是补丁的遮罩，通过将其每个元素的非零索引提取出来，得到了一个列表，
        列表中的每个元素是一个包含最小和最大非零值索引的张量。
        这个列表的长度与批次大小相同，最终通过 torch.stack 得到一个张量 mesh_bord，其形状为 [B, 4]。
        '''
        mesh_bord = torch.stack([torch.cat([m[0].nonzero().min(0).values, m[0].nonzero().max(0).values]) for m in mask])
        '''
        归一化坐标：mesh_bord 中的坐标被归一化到 [-1, 1] 的范围内。
        首先，通过 mesh_bord.new([Ho, Wo, Ho, Wo]) 创建一个张量，然后将 mesh_bord 除以该张量，
        乘以 2 并减去 1，以将坐标映射到 [-1, 1] 的范围内。这是因为 mesh_bord 表示的是坐标相对于图像尺寸的比例
        '''
        mesh_bord = mesh_bord / mesh_bord.new([Ho, Wo, Ho, Wo]) * 2 - 1 #[4,4]
        '''
        计算平移参数：首先，从 pos_param 中解绑（unbind） txmin、tymin、txmax 和 tymax 四个坐标参数。
        然后，计算 xdiff 和 ydiff，它们分别表示 txmax 和 txmin 之间的差异，
        并且在差异小于零时进行了修剪（clamp），以确保它们都是非负数。接下来，计算 xmiddle 和 ymiddle，
        它们分别表示 txmax 和 txmin 之间的中间值。
        最后，使用随机均匀分布的随机数对 tx 和 ty 进行随机扰动。
        这些扰动通过 xdiff、ydiff、xmiddle 和 ymiddle 来调整，
        并且受到 self.translation_x 和 self.translation_y 的控制。
        '''
        #mesh_bord = mesh_bord / scale
        pos_param = mesh_bord + mesh_bord.new([1, 1, -1, -1]) * scale.unsqueeze(-1) #再scale后添加一个维度：unsqueeze(-1)
        tymin, txmin, tymax, txmax = pos_param.unbind(-1)
        xdiff = (-txmax + txmin).clamp(min=0)
        xmiddle = (txmax + txmin) / 2
        ydiff = (-tymax + tymin).clamp(min=0)
        ymiddle = (tymax + tymin) / 2
        tx = txmin.new(txmin.shape).uniform_(-0.5, 0.5) * xdiff * self.translation_x + xmiddle #平移参数:translation_x
        ty = tymin.new(tymin.shape).uniform_(-0.5, 0.5) * ydiff * self.translation_y + ymiddle
        #创建一个大小为 (B, 2, 3) 的张量 theta，其中 B 表示批次大小。theta 用于定义仿射变换矩阵，其中包括缩放、旋转和平移操作。
        #theta[:, 0, 0] 和 theta[:, 1, 1] 分别设置为 scale，表示在 x 和 y 方向上的缩放。
        #将 tx 和 ty 分别设置为 theta 张量的平移因子。这里 theta[:, 0, 2] 表示 x 方向上的平移，theta[:, 1, 2] 表示 y 方向上的平移。
        theta = adv_patch.new_zeros(B, 2, 3)
        theta[:, 0, 0] = scale
        theta[:, 0, 1] = 0
        theta[:, 1, 0] = 0
        theta[:, 1, 1] = scale
        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty
        #F.affine_grid 函数根据 theta 张量生成仿射变换的网格。
        #这个网格用于指定如何对输入图像进行仿射变换，从而实现缩放、旋转和平移
        grid = F.affine_grid(theta, img_batch.shape)
        #F.grid_sample 函数将 adv_patch 应用到生成的仿射网格上，进行仿射变换。
        #这会将 adv_patch 中的内容根据 theta 中定义的变换应用到输入图像上。
        adv_batch = F.grid_sample(adv_patch, grid, padding_mode='zeros')


        #通过 mask 提取出仿射变换后的图像中的有效区域（非零区域），然后使用 img_batch 中的相应像素值替换掉无效区域。
        #这是为了确保变换后的图像与输入图像在有效区域上是一致的
        mask = adv_batch[:, -1:]
        adv_batch = adv_batch[:, :-1] * mask + img_batch * (1 - mask)

        #计算 gt，它是仿射变换后图像中有效区域的边界框坐标。
        gt = torch.stack([torch.cat([m[0].nonzero().min(0).values, m[0].nonzero().max(0).values]) for m in mask])
        gt = gt[:, [1, 0, 3, 2]].unbind(0)
        return adv_batch, gt

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y
def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

def non_max_suppression(prediction, conf_thres=0.01, iou_thres=0.01, classes=None, agnostic=False,
                            multi_label=False, labels=(), max_det=300):
        """Runs Non-Maximum Suppression (NMS) on inference and logits results

        Returns:
             list of detections, on (n,6) tensor per image [xyxy, conf, cls] and pruned input logits (n, number-classes)
        """
        #prediction为Model输出，形状为[1，17640，85]
        nc = prediction.shape[2] - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Checks
        assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

        # Settings
        min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 10.0  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        #准备输出prediction.shape[0] = logits.shape[0] = batchsize
        t = time.time()
        output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
        
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            # xi 为第几个batch
            # xc 尺寸为[1,17640]，为布尔量，为true表示该anchor有目标,为false表示没目标
            # x[xc[xi]] 表示第i个batch存在目标的anchor的信息，此处为[9,85]
            x = x[xc[xi]]  # confidence
            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                l = labels[xi]
                v = torch.zeros((len(l), nc + 5), device=x.device)
                v[:, :4] = l[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
            # log_ *= x[:, 4:5]
            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = xywh2xyxy(x[:, :4])

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            else:  # best class only
                conf, j = x[:, 5:].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]
            assert log_[i].shape[0] == x[i].shape[0]
            if (time.time() - t) > time_limit:
                print(f'WARNING: NMS time limit {time_limit}s exceeded')
                break  # time limit exceeded

def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU
    
def get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors, only_objectness=1, validation=False,
                     name=None):
    anchor_step = len(anchors) // num_anchors
    device = output.device
    if output.dim() == 3:
        output = output.unsqueeze(0)
    batch = output.size(0)
    assert (output.size(1) == (5 + num_classes) * num_anchors)
    h = output.size(2)
    w = output.size(3)

    output = output.view(batch * num_anchors, 5 + num_classes, h * w)
    output = output.transpose(0, 1).contiguous()
    output = output.view(5 + num_classes, batch * num_anchors * h * w)
    # grid_x = torch.linspace(0, w-1, w).repeat(h,1).repeat(batch*num_anchors, 1, 1).view(batch*num_anchors*h*w).to(output)
    # grid_y = torch.linspace(0, h-1, h).repeat(w,1).t().repeat(batch*num_anchors, 1, 1).view(batch*num_anchors*h*w).to(output)
    grid_y, grid_x = torch.meshgrid([torch.arange(w, device=device), torch.arange(h, device=device)])
    grid_x = grid_x.repeat(batch * num_anchors, 1, 1).flatten()
    grid_y = grid_y.repeat(batch * num_anchors, 1, 1).flatten()
    xs = torch.sigmoid(output[0]) + grid_x
    ys = torch.sigmoid(output[1]) + grid_y

    anchor_tensor = torch.tensor(anchors, device=device).view(num_anchors, anchor_step)
    # anchor_w = anchor_tensor.index_select(1, torch.LongTensor([0]))
    # anchor_h = anchor_tensor.index_select(1, torch.LongTensor([1]))
    anchor_w = anchor_tensor[:, 0:1]
    anchor_h = anchor_tensor[:, 1:2]
    anchor_w = anchor_w.repeat(batch, 1).repeat(1, 1, h * w).view(batch * num_anchors * h * w)
    anchor_h = anchor_h.repeat(batch, 1).repeat(1, 1, h * w).view(batch * num_anchors * h * w)
    ws = torch.exp(output[2]) * anchor_w
    hs = torch.exp(output[3]) * anchor_h

    det_confs = torch.sigmoid(output[4])
    # cls_confs = torch.nn.Softmax()(Variable(output[5:5+num_classes].transpose(0,1))).data

    if name == 'yolov2':
        cls_confs = output[5:5 + num_classes].transpose(0, 1).softmax(-1)
    elif name == 'yolov3':
        cls_confs = output[5:5 + num_classes].transpose(0, 1).sigmoid()
    else:
        raise ValueError

    cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
    cls_max_confs = cls_max_confs.view(-1)
    cls_max_ids = cls_max_ids.view(-1)

    raw_boxes = torch.stack([xs/w, ys/h, ws/w, hs/h, det_confs, cls_max_confs, cls_max_ids], 1).view(batch, -1, 7)
    if only_objectness:
        conf = det_confs
    else:
        conf = det_confs * cls_max_confs
    inds = (conf > conf_thresh).view(batch, -1)

    all_boxes = [b[i] for b, i in zip(raw_boxes, inds)]

    if (not only_objectness) and validation:
        raise NotImplementedError

    return all_boxes

def get_region_boxes_general(output, model, conf_thresh, name=None, img_size=640, lab_filter=None):
    if name is None:
        if type(output) is list:
            name = 'yolov3'
        else:
            name = 'yolov2'
    if name == 'yolov2':
        num_classes = model.num_classes
        anchors = model.anchors
        num_anchors = model.num_anchors
        if isinstance(output, list):
            assert len(output) == 1
            output = output[0]
        all_boxes = get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors, name=name)
    elif name == 'yolov3':
        boxes = []
        y1 =  [[10,13, 16,30, 33,23] , 
             [30,61, 62,45, 59,119],  # P4/16
             [116,90, 156,198, 373,326]]
        y1 = np.array(y1)
        y22 = [8,16,32]
        i=0
        for o,  in zip(output):
            B, A, W, H, C = o.shape
            y = y1[i]/y22[i]
            y=y.tolist()
            i=i+1
            b = get_region_boxes(o.permute(0, 1, 4, 2, 3).contiguous().view(B, A * C, W, H), conf_thresh,
                                 82, y, 3,
                                 name=name)
            boxes.append(b)
        i=0
        all_boxes = [torch.cat([boxes[i][j] for i in range(len(output))], 0) for j in range(output[0].shape[0])]
        # all_boxes = boxes[0]
        # for b in boxes[1:]:
        #     for i, bc in enumerate(b):
        #         all_boxes[i].extend(bc)
    elif name in ['rcnn', 'faster_rcnn', 'mask_rcnn']:
        all_boxes = []
        for d in output:
            boxes = [(d['boxes'][:, 0] + d['boxes'][:, 2]) / (2 * img_size),
                     (d['boxes'][:, 1] + d['boxes'][:, 3]) / (2 * img_size),
                     (d['boxes'][:, 2] - d['boxes'][:, 0]) / img_size, (d['boxes'][:, 3] - d['boxes'][:, 1]) / img_size]
            boxes = boxes + [d['scores'], d['scores'], d['labels'] - 1]
            boxes = torch.stack(boxes, 1)
            boxes = boxes[boxes[:, 4] > conf_thresh]
            # boxes = boxes.tolist()
            all_boxes.append(boxes)

    elif name == 'detr':

        bboxes = output['pred_boxes']
        # print(output['pred_logits'].shape)
        scores, labels = output['pred_logits'].softmax(dim=-1)[..., :-1].max(-1)
        bboxes = torch.cat([bboxes, scores.unsqueeze(-1), scores.unsqueeze(-1), labels.unsqueeze(-1)-1], -1)
        all_boxes = []
        for boxes in bboxes:
            boxes = boxes[boxes[:, 4] > conf_thresh]
            all_boxes.append(boxes)

    elif name == 'deformable-detr':
        bboxes = output['pred_boxes']
        scores, labels = torch.max(output['logits'].softmax(dim=-1), dim=-1)
        bboxes = torch.cat([bboxes, scores.unsqueeze(-1), scores.unsqueeze(-1), labels.unsqueeze(-1)-1], -1)
        all_boxes = []
        for boxes in bboxes:
            boxes = boxes[boxes[:, 4] > conf_thresh]
            all_boxes.append(boxes)

    elif 'mmdet' in name:
        if 'mask' in name:
            output = [results[0] for results in output]
        # output = [results[0] for results in output]
        all_boxes = []
        for results in output:
            boxes = []
            for i, preds in enumerate(results):
                x1 = preds[:, 0]
                y1 = preds[:, 1]
                x2 = preds[:, 2]
                y2 = preds[:, 3]
                score = preds[:, 4]
                box = [(x1 + x2) / (2 * img_size),
                       (y1 + y2) / (2 * img_size),
                       (x2 - x1) / img_size,
                       (y2 - y1) / img_size,
                       score,
                       score,
                       np.zeros(score.shape) + i]
                box = np.stack(box, 1)
                box = box[box[:, 4] > conf_thresh]
                # box = box.tolist()
                if len(box) > 0:
                    boxes.append(box)
            boxes = torch.from_numpy(np.concatenate(boxes, 0))
            all_boxes.append(boxes)
    else:
        raise ValueError
    if lab_filter is not None:
        for i in range(len(all_boxes)):
            all_boxes[i] = all_boxes[i][all_boxes[i][:, 6] == lab_filter]
    return all_boxes

class YOLOv3MaxProbExtractor(nn.Module):
    """MaxProbExtractor: extracts max class probability for class from YOLO output.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """

    def __init__(self, cls_id, num_cls, model,figsize):
        super(YOLOv3MaxProbExtractor, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.figsize = figsize
        self.model = model

    # for v3 training output
    
    def forward(self, YOLOoutputs, gt, loss_type, iou_thresh):
        max_probs = []
        det_loss = []
        num = 0
        box_all = get_region_boxes_general(YOLOoutputs, self.model, conf_thresh=0.2, name="yolov3")
        box_all = box_all
        for i in range(len(box_all)):
            boxes = box_all[i]
            assert boxes.shape[1] == 7
            # boxes = boxes.view(-1,7) # [x,y,w,h,obj_conf,cls_score,class_idx]
            w1 = boxes[...,2] - boxes[..., 0]/2
            h1 = boxes[...,3] - boxes[..., 1]/2
            w2 = boxes[...,2] + boxes[..., 0]/2
            h2 = boxes[...,3] + boxes[..., 1]/2
            bbox = torch.stack([w1,h1,w2,h2],dim=-1).to(device) #在这一步要得到结果中每一幅图的最佳预测值
            ious = torchvision.ops.box_iou(bbox.view(-1,4).detach()*self.figsize,gt[i].unsqueeze(0)).squeeze(-1).to(device)
            mask = ious.ge(iou_thresh)
            if True:
                mask = mask.logical_and(boxes[...,6]==0)
            ious = ious[mask]
            scores = boxes[...,4][mask]
            if len(ious) > 0:
                if loss_type == 'max_iou':
                    _, ids = torch.max(ious, dim=0) # get the bbox w/ biggest iou compared to gt
                    det_loss.append(scores[ids])
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'max_conf':
                    det_loss.append(scores.max())
                    max_probs.append(scores.max())
                    num += 1
                elif loss_type == 'softplus_max':
                    max_conf = - torch.log(1.0 / scores.max() - 1.0)
                    max_conf = F.softplus(max_conf)
                    det_loss.append(max_conf)
                    max_probs.append(scores.max())
                    num += 1
                elif loss_type == 'softplus_sum':
                    max_conf = (F.softplus(- torch.log(1.0 / scores - 1.0)) * ious.detach()).sum()
                    det_loss.append(max_conf)
                    max_probs.append(scores.mean())
                    num += len(scores)

                elif loss_type == 'max_iou_mtiou':
                    _, ids = torch.max(ious, dim=0) # get the bbox w/ biggest iou compared to gt
                    det_loss.append(scores[ids] * ious[ids])
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'max_conf_mtiou':
                    _, ids = torch.max(scores, dim=0)
                    det_loss.append(scores[ids] * ious[ids])
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'softplus_max_mtiou':
                    _, ids = torch.max(scores, dim=0)
                    max_conf = - torch.log(1.0 / scores[ids] - 1.0)
                    max_conf = F.softplus(max_conf) * ious[ids]
                    det_loss.append(max_conf)
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'softplus_sum_mtiou':
                    max_conf = (F.softplus(- torch.log(1.0 / scores - 1.0)) * ious).sum()
                    det_loss.append(max_conf)
                    max_probs.append(scores.mean())
                    num += len(scores)


                elif loss_type == 'max_iou_adiou':
                    _, ids = torch.max(ious, dim=0) # get the bbox w/ biggest iou compared to gt
                    det_loss.append(scores[ids] + ious[ids])
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'max_conf_adiou':
                    _, ids = torch.max(scores, dim=0)
                    det_loss.append(scores[ids] + ious[ids])
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'softplus_max_adiou':
                    _, ids = torch.max(scores, dim=0)
                    max_conf = - torch.log(1.0 / scores[ids] - 1.0)
                    max_conf = F.softplus(max_conf) + ious[ids]
                    det_loss.append(max_conf)
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'softplus_sum_adiou':
                    max_conf = (F.softplus(- torch.log(1.0 / scores - 1.0)) + ious).sum()
                    det_loss.append(max_conf)
                    max_probs.append(scores.mean())
                    num += len(scores)

                elif loss_type == 'softplus_max_adspiou':
                    _, ids = torch.max(scores, dim=0)
                    max_conf = - torch.log(1.0 / scores[ids] - 1.0)
                    max_conf = F.softplus(max_conf) + F.softplus(- torch.log(1.0 / ious[ids] - 1.0))
                    det_loss.append(max_conf)
                    max_probs.append(scores[ids])
                    num += 1

                elif loss_type == 'softplus_sum_adspiou':
                    max_conf = (F.softplus(- torch.log(1.0 / scores - 1.0)) + F.softplus(- torch.log(1.0 / ious - 1.0))).sum()
                    det_loss.append(max_conf)
                    max_probs.append(scores.mean())
                    num += len(scores)

                else:
                    raise ValueError
            else:
                det_loss.append(ious.new([0.0])[0])
                max_probs.append(ious.new([0.0])[0])
        det_loss = torch.stack(det_loss).mean()
        max_probs = torch.stack(max_probs)
        if num < 1:
            raise RuntimeError()
        return det_loss, max_probs
    
class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
class ComputeLoss_WTO:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLoss_WTO, self).__init__()
        print("computing wto....")
        self.device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters #model.module
        # print(dir(model))
        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=self.device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=self.device))
        #二元交叉熵损失（Binary Cross-Entropy Loss）的实例

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets
        #计算平滑的 BCE 损失目标。eps 参数从超参数 h 中获取。

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma 
        #如果 g 大于 0，则会对 BCEcls 和 BCEobj 进行修改，应用焦点损失（Focal Loss）。
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.model[-1]#model.module.model[-1] #if is_parallel(model) else model.model[-1]  # Detect() mode.module
        #获取了目标检测模型中的最后一层，通常是检测层。

        # print("det",det)
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        # print("self.balance..",self.balance)
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        # self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        self.BCEcls, self.BCEobj, self.hyp, self.autobalance = BCEcls, BCEobj, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            # print(k, getattr(det, k))
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
       
        lcls, lbox, lobj = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        # print("p_call", p, len(p))
        # tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
        # print("tcls",tcls,"tbox",tbox,"indices", indices, 'anc',anchors)
        # Losses
        # print("p_call",p.shape)
        for i, pi in enumerate(p):  # layer index, layer predictions,分别计算3个维度下的值
            # b [0,0,0]
            # a [0, 1, 2]
            # gj [10, 10, 10]
            # gi [10, 10, 10]
            

            # b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            # tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            ##############################################
            #真实的标签选择有问题
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  
            # target obj，显然有问题吧！！
            ###################################################

            # n = b.shape[0]  # number of targets
            # print("tobj",i,tobj.size(),i,n,b,a,gj, gi)
            # if n:
                # ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets  目标对应的预测子集

                # Regression
                # pxy = ps[:, :2].sigmoid() * 2. - 0.5
                # pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                # pbox = torch.cat((pxy, pwh), 1)  # predicted box
                # iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                # lbox += (1.0 - iou).mean()  # original iou loss
                # print(iou.mean(), tobj.size(),len(b), len(a), len(gj), len(gi))
                # lbox += iou.mean()  # adversarial iou loss
                # print(b,a,gj, gi)
        
                # tobj[b, a, gj, gi] = 1.0 # (1.0 - self.gr) + self.gr * (1-iou).detach().clamp(0).type(tobj.dtype)  # iou ratio
         

                # Classification
                # if self.nc > 1:  # cls loss (only if multiple classes)
                #     t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                #     print(t.size(), n, tcls[i], self.cp)
                #     t[range(n), tcls[i]] = self.cp
                #     lcls += torch.max(torch.mean(ps[:, 5:] * t, dim=0))  #

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        # lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        # lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size
        # print(lbox, lobj, lcls, self.hyp['box'],self.hyp['obj'],self.hyp['cls'])

        loss =   lobj  
        
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        # print("self.na...",self.na)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        # print("nt",nt,len(targets[0]))
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        # print(na, targets.size())
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)#.resize(-1, 1, 1)  # same as .repeat_interleave(nt)
        # print(ai[:, :, None].size(),targets.repeat(na, 1, 1).size())
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices
        # targets = torch.cat((targets.repeat(na, 1, 1).resize(na, 16, 5), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            # [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets
        # print("self.nl", self.nl, len(p[1]), len(p[0]), len(p[2]), nt)
        for i in range(self.nl):
        
            anchors = self.anchors[i]
            # print("pi", p[i].shape,len(p[i]), targets.size())
            # tensor([ 1.,  1., 80., 80., 80., 80.,  1.], device='cuda:0')
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
            gain = gain.long()

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j),))
                t = t.repeat((off.shape[0], 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
             
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
class ComputeLoss:    #攻击损失函数
    """YOLO-style adversarial loss wrapper used to compute the attack objective during training."""
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLoss, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters，得到模型参数

        # Define criteria定义标准
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        '''
        nn.BCEloss()衡量输出与目标之间的二分类交叉熵
        功能：二分类交叉熵，输入值取值在[0,1]
        weight: 默认None，用于计算损失的手动尺度化的权重、张量
        nn.BCEWithLogitsLoss()
        此损失函数将 Sigmoid 层和 BCELoss 整合在一起
        比简单地将 Sigmoid 层加上 BCELoss 损失更稳定，因为使用了 log-sun-exp 技巧获得数值稳定性
        '''
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets
        #调用了一个名为 smooth_BCE 的函数，用于实现类别标签的平滑化。平滑化是为了减少模型对训练数据的过度拟合。

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
        #如果超参数中设置了 Focal Loss 的参数 (fl_gamma 大于零)，则应用 Focal Loss，这有助于处理难以训练的样本。

        det = model.model[-1]  #获取模型最后一层
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.hyp, self.autobalance = BCEcls, BCEobj,h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        #p:0[16,3,80,80,85];1[16,3,40,40,85];2[16,3,20,20,85]
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device) #初始化
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            # b [0,0,0]
            # a [0, 1, 2]
            # gj [10, 10, 10]
            # gi [10, 10, 10]
            b, a, gj, gi = indices[i]  # 从 indices 中获取当前预测层的图像索引 b、锚框索引 a，以及特征图上的网格索引 gj 和 gi。
            tobj = torch.zeros_like(pi[..., 0], device=device)  # [16,3,80,80]

            n = b.shape[0]  # number of targets，计算目标的数量 n，并检查是否有目标存在。
            if n:  #pi[16,3,80,80,85]
                ps = pi[b, a, gj, gi]  # [12,85]，取那12个框对应的输出

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                # lbox += (1.0 - iou).mean()  # original iou loss，这里最小化就相当于最大化iou了，攻击的目的是最小化iou
                lbox += iou.mean()  #预测的坐标信息 ps[:, :2] 和 ps[:, 2:4] 进行逆变换，计算预测的包围框 pbox。然后，通过 bbox_iou 计算预测框与目标框的 IoU（Intersection over Union）。
                #最后，将计算的 IoU 值添加到 lbox 中，这是用于计算损失的变量。

                tobj[b, a, gj, gi] = 1.0#将 tobj 中对应目标的位置设置为 1，表示目标存在
                # (1.0 - self.gr) + self.gr * (1-iou).detach().clamp(0).type(tobj.dtype)
                # iou ratio，iou比率

                # Classification
                if self.nc > 1:  # 如果存在多个类别 (self.nc > 1)，则创建一个与 ps 的类别部分相同形状的张量 t，其中所有元素初始化为 self.cn。然后，将目标的类别对应的位置设置为 self.cp。
                    #最后，计算交叉熵损失（classification loss），并将其添加到 lcls 中。
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  #[12,80]取的是类别信息
                    t[range(n), tcls[i]] = self.cp
                    lcls += torch.max(torch.mean(ps[:, 5:] * t, dim=0))  #

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
 
        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach(),lbox, lobj, lcls

    def build_targets(self, p, targets): #target[16,6]
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets;3,16(batch_size)
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device).long()  # normalized to gridspace gain,[7]，用于将坐标值归一化到网格空间
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # [3,16]]，锚框张量[na，nt]，每一列包含了锚框的索引
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  #[3,16,7]
        #targets.repeat(na, 1, 1):[16,6]--[16*na,6]
        #ai[3,16,1]  沿着第三个维度1凭借，所以最后一个维度为1+6， [16*3,6+1]
        g = 0.5  #偏置值，用于后续处理目标在特征图上的位置
        off = torch.tensor([[0, 0],
                            # [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets  [1,2]，偏移张量

        for i in range(self.nl):  #nl为3,3个锚框

            anchors, shape = self.anchors[i], p[i].shape #获取当层的锚框和输出的形状
            # tensor([ 1.,  1., 80., 80., 80., 80.,  1.], device='cuda:0')
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain；
            #更新了 gain 张量的部分值，用于将坐标转换为相对于特征图的比例。具体来说，这里用了当前预测层的宽和高（p[i].shape[3] 和 p[i].shape[2]）。
            #指的是候选框的大小
            # Match targets to anchors
            t = targets * gain #这一步是为了将目标的坐标转换为相对于当前预测层的特征图的比例。转换成当前的框的大小；[3,16,7]
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio[3,16,2]
                #计算目标的宽高比与锚框的宽高比之比 r，然后判断是否满足匹配条件 self.hyp['anchor_t']。
                # 匹配条件的判断使用了 torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']。
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # 返回的张量是一个一维张量，其中的每个元素表示相应行中最大值。[3,16]
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter，只要满足条件的候选框[12,7]因为只有12个ture

                # Offsets
                gxy = t[:, 2:4]  # grid xy [12,2]，计算目标在特征图上的网格坐标
                gxi = gain[[2, 3]] - gxy  # 计算相应的逆变换
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T #[12]
                j = torch.stack((torch.ones_like(j),)) #[1,12]
                t = t.repeat((off.shape[0], 1, 1))[j]#将目标张量 t 复制 off.shape[0] 次，得到新的目标张量 t。这一步的目的是为了处理 off 中的每一个偏移量。
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j] #[12,2]，计算偏移量 offsets，这是一个张量，用于将目标的坐标转换为相对于特征图的偏移量。
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class  图标号[12]，标签类别[12]
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # 锚框索引
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box,0:[12,4]；1:[40,4]；2:[30,4]
            anch.append(anchors[a])  #0:[12,2]；
            tcls.append(c)  # 0：[12]

        return tcls, tbox, indices, anch


class ComputeLoss12:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = is_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

    def __call__(self, p, targets):  # predictions, targets
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()
    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
    

class MaxProbExtractor(nn.Module):
    """MaxProbExtractor: extracts max class probability for class from YOLO output.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """

    def __init__(self):
        super(MaxProbExtractor, self).__init__()
       
        self.n_classes = 80
        self.objective_class_id = None
        self.loss_target = "obj * cls"

    def forward(self, output: torch.Tensor):
        """
        output must be of the shape [batch, -1, 5 + num_cls]
        """
        # get values neccesary for transformation
        assert (output.size(-1) == (5 + self.n_classes))

        class_confs = output[:, :, 5:5 + self.n_classes]  # [6, 14175, 87]
        objectness_score = output[:, :, 4]
        # objectness_score = torch.max(objectness_score, dim=1)[0]  
        # objectness_score = torch.sigmoid(objectness_score)  # [batch, -1, 5 + num_cls] -> [batch, -1], no need to run sigmoid here
        # objectness_score = output[:, :, 4]  # [batch, -1, 5 + num_cls] -> [batch, -1], no need to run sigmoid here


        if self.objective_class_id is not None:
            # norm probs for object classes to [0, 1]
            class_confs = torch.nn.Softmax(dim=2)(class_confs)
            # only select the conf score for the objective class
            class_confs = class_confs[:, :, self.objective_class_id]
        else:
            # get class with highest conf for each box if objective_class_id is None
            class_confs = torch.max(class_confs, dim=2)[0]  # [batch, -1, 4] -> [batch, -1]
        #不指定class_id，让其不识别为任何类别，取的是每个框里的最大类别的概率

        #两种，第一种就是计算两者相乘的值，但objectness不加sigmoid几乎为0
        #第二种就是分开计算cls和obj：obj：计算最大的候选框包含的置信度分数+cls：目标类别在候选框中的概率值
        
        confs_if_object =  objectness_score * class_confs
        max_conf, _ = torch.max(confs_if_object, dim=1)
        return max_conf


# class MaxProbExtractor(nn.Module):
#     """MaxProbExtractor: extracts max class probability for class from YOLO output.

#     Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

#     """

#     def __init__(self):
#         super(MaxProbExtractor, self).__init__()
       
#         self.n_classes = 80
#         self.objective_class_id = 2
#         self.loss_target = "obj * cls"

#     def forward(self, output: torch.Tensor):
#         """
#         output must be of the shape [batch, -1, 5 + num_cls]
#         """
#         # get values neccesary for transformation
#         assert (output.size(-1) == (5 + self.n_classes))

#         class_confs = output[:, :, 5:5 + self.n_classes]  # [6, 14175, 87]
#         objectness_score = output[:, :, 4]
#         # objectness_score = torch.max(objectness_score, dim=1)[0]  
#         # objectness_score = torch.sigmoid(objectness_score)  # [batch, -1, 5 + num_cls] -> [batch, -1], no need to run sigmoid here
#         # objectness_score = output[:, :, 4]  # [batch, -1, 5 + num_cls] -> [batch, -1], no need to run sigmoid here


#         maxoo,_ = torch.max(objectness_score,dim=1)
#         # max2,max2_index = torch.max(max1,dim=1)

#         if self.objective_class_id is not None:
#             # norm probs for object classes to [0, 1]
#             class_confs = torch.nn.Softmax(dim=2)(class_confs)
#             max1,_ = torch.max(class_confs,dim=1)
#             max2,max2_index = torch.max(max1,dim=1)
#             # only select the conf score for the objective class
#             class_confs = class_confs[:, :, self.objective_class_id]
#         else:
#             # get class with highest conf for each box if objective_class_id is None
#             class_confs = torch.max(class_confs, dim=2)[0]  # [batch, -1, 4] -> [batch, -1]
#         #不指定class_id，让其不识别为任何类别，取的是每个框里的最大类别的概率

#         #两种，第一种就是计算两者相乘的值，但objectness不加sigmoid几乎为0
#         #第二种就是分开计算cls和obj：obj：计算最大的候选框包含的置信度分数+cls：目标类别在候选框中的概率值
        
#         confs_if_object =  objectness_score * class_confs
#         max_conf, _ = torch.max(confs_if_object, dim=1)
#         return max_conf

class SaliencyLoss(nn.Module):
    """Implementation of the colorfulness metric as the saliency loss.
    The smaller the value, the less colorful the image.
    Reference: https://infoscience.epfl.ch/record/33994/files/HaslerS03.pdf
    """

    def __init__(self):
        super(SaliencyLoss, self).__init__()

    def forward(self, adv_patch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            adv_patch: Float Tensor of shape [C, H, W] where C=3 (R, G, B channels)
        """
        # print(adv_patch.size())
        adv_patch = adv_patch.squeeze(0).permute(2,0,1)
        assert adv_patch.shape[0] == 3
        r, g, b = adv_patch
        rg = r - g
        yb = 0.5 * (r + g) - b

        mu_rg, sigma_rg = torch.mean(rg) + 1e-8, torch.std(rg) + 1e-8
        mu_yb, sigma_yb = torch.mean(yb) + 1e-8, torch.std(yb) + 1e-8
        sl = torch.sqrt(sigma_rg**2 + sigma_yb**2) + \
            (0.3 * torch.sqrt(mu_rg**2 + mu_yb**2))
        return sl / torch.numel(adv_patch)


class TotalVariationLoss(nn.Module):
    """Total variation regularizer for encouraging spatial smoothness in the optimized texture."""
    """TotalVariationLoss: calculates the total variation of a patch.
    Module providing the functionality necessary to calculate the total vatiation (TV) of an adversarial patch.
    Reference: https://en.wikipedia.org/wiki/Total_variation
    """

    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def forward(self, adv_patch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            adv_patch: Tensor of shape [C, H, W] 
        """
        # calc diff in patch rows
        tvcomp_r = torch.sum(
            torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1]+0.000001), dim=0)
        tvcomp_r = torch.sum(torch.sum(tvcomp_r, dim=0), dim=0)
        # calc diff in patch columns
        tvcomp_c = torch.sum(
            torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :]+0.000001), dim=0)
        tvcomp_c = torch.sum(torch.sum(tvcomp_c, dim=0), dim=0)
        tv = tvcomp_r + tvcomp_c
        return tv / torch.numel(adv_patch)

class NPSCalculator(nn.Module):
    """NMSCalculator: calculates the non-printability score of a patch.

    Module providing the functionality necessary to calculate the non-printability score (NMS) of an adversarial patch.

    """
    
    def __init__(self, printability_file, img_size):
        super(NPSCalculator, self).__init__()
        self.printability_array = nn.Parameter(self.get_printability_array(printability_file, img_size),
                                               requires_grad=False)
    
    def forward(self, adv_patch):
        # calculate euclidian distance between colors in patch and colors in printability_array
        # square root of sum of squared difference
        color_dist = (adv_patch - self.printability_array.to(adv_patch.device) + 0.000001)
        color_dist = color_dist ** 2
        color_dist = torch.sum(color_dist, 1) + 0.000001
        color_dist = torch.sqrt(color_dist)
        # only work with the min distance
        color_dist_prod = torch.min(color_dist, 0)[0]  # test: change prod for min (find distance to closest color)
        # calculate the nps by summing over all pixels
        nps_score = torch.sum(color_dist_prod, 0)
        nps_score = torch.sum(nps_score, 0)
        return nps_score / torch.numel(adv_patch)
    
    def get_printability_array(self, printability_file, side):
        printability_list = []
        
        # read in printability triplets and put them in a list
        with open(printability_file) as f:
            for line in f:
                printability_list.append(line.split(","))
        
        printability_array = []
        for printability_triplet in printability_list:
            printability_imgs = []
            red, green, blue = printability_triplet
            printability_imgs.append(np.full((side, side), red))
            printability_imgs.append(np.full((side, side), green))
            printability_imgs.append(np.full((side, side), blue))
            printability_array.append(printability_imgs)
        
        printability_array = np.asarray(printability_array)
        printability_array = np.float32(printability_array)
        pa = torch.from_numpy(printability_array)
        return pa
    
class NPSLoss(nn.Module):
    """Non-printability-score loss built from a list of printable RGB triplets."""
    """NMSLoss: calculates the non-printability-score loss of a patch.
    Module providing the functionality necessary to calculate the non-printability score (NMS) of an adversarial patch.
    However, a summation of the differences is used instead of the total product to calc the NPSLoss
    Reference: https://users.ece.cmu.edu/~lbauer/papers/2016/ccs2016-face-recognition.pdf
        Args: 
            triplet_scores_fpath: str, path to csv file with RGB triplets sep by commas in newlines
            size: Tuple[int, int], Tuple with height, width of the patch
    """

    def __init__(self, triplet_scores_fpath: str, size: Tuple[int, int]):
        super(NPSLoss, self).__init__()
        self.printability_array = nn.Parameter(self.get_printability_array(
            triplet_scores_fpath, size), requires_grad=False)

    def forward(self, adv_patch):
        # calculate euclidian distance between colors in patch and colors in printability_array
        # square root of sum of squared difference
        # print(adv_patch.size())
        device = self.printability_array.device
        adv_patch = adv_patch.permute(0,3,1,2).to(device)
        # self.printability_array.to(device)
        # print(self.printability_array, adv_patch)
        color_dist = (adv_patch - self.printability_array + 0.000001)
        color_dist = color_dist ** 2
        color_dist = torch.sum(color_dist, 1) + 0.000001
        color_dist = torch.sqrt(color_dist)
        # use the min distance
        color_dist_prod = torch.min(color_dist, 0)[0]
        # calculate the nps by summing over all pixels
        nps_score = torch.sum(color_dist_prod, 0)
        nps_score = torch.sum(nps_score, 0)
        return nps_score / torch.numel(adv_patch)

    def get_printability_array(self, triplet_scores_fpath: str, size: Tuple[int, int]) -> torch.Tensor:
        """
        Get printability tensor array holding the rgb triplets (range [0,1]) loaded from triplet_scores_fpath
        Args: 
            triplet_scores_fpath: str, path to csv file with RGB triplets sep by commas in newlines
            size: Tuple[int, int], Tuple with height, width of the patch
        """
        ref_triplet_list = []
        # read in reference printability triplets into a list
        with open(triplet_scores_fpath, 'r', encoding="utf-8") as f:
            for line in f:
                ref_triplet_list.append(line.strip().split(","))

        p_h, p_w,_ = size
        printability_array = []
        for ref_triplet in ref_triplet_list:
            r, g, b = map(float, ref_triplet)
            ref_tensor_img = torch.stack([torch.full((p_h, p_w), r),
                                          torch.full((p_h, p_w), g),
                                          torch.full((p_h, p_w), b)])
            printability_array.append(ref_tensor_img.float())
        return torch.stack(printability_array)
    
class InriaDataset(Dataset):
    def __init__(self, img_dir, lab_dir, mask_dir, imgsize, shuffle=True): 
        n_images = len(fnmatch.filter(os.listdir(img_dir), '*.npz'))
        n_labels = len(fnmatch.filter(os.listdir(lab_dir), '*.txt'))
        n_mask_images = len(fnmatch.filter(os.listdir(mask_dir), '*.png'))

        assert n_images == n_labels, "Number of images and number of labels don't match"
        self.len = n_images
        self.img_dir = img_dir
        self.lab_dir = lab_dir
        self.mask_dir = mask_dir
        self.imgsize = imgsize

        self.img_names = fnmatch.filter(os.listdir(img_dir), '*.npz')
        self.shuffle = shuffle
        self.img_paths = []
        for img_name in self.img_names:
            self.img_paths.append(os.path.join(self.img_dir, img_name))
        self.mask_paths = []
        for img_name in self.img_names:
            mask_path = os.path.join(self.mask_dir, img_name).replace('.npz', '.png')
            self.mask_paths.append(mask_path)
        self.lab_paths = []
        for img_name in self.img_names:
            lab_path = os.path.join(self.lab_dir, img_name).replace('.npz', '.txt')
            self.lab_paths.append(lab_path)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        assert idx <= len(self), 'index range error'
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        mask_path = os.path.join(self.mask_dir, self.img_names[idx]).replace('.npz', '.png')
        lab_path = os.path.join(self.lab_dir, self.img_names[idx]).replace('.npz', '.txt')
        
        #npz文件转图像：
        cat_data = np.load(img_path)
        image = cat_data['img']
        cam_trans = cat_data['cam_trans']
        veh_trans = cat_data['veh_trans']

        mask_image = Image.open(mask_path).convert('RGB')

        if os.path.getsize(lab_path):       #check to see if label file contains data. 
            label = np.loadtxt(lab_path)
        else:
            label = np.ones([5])
        label = torch.from_numpy(label).float()
        # if label.dim() == 1:
        #     label = label.unsqueeze(0)
        #image[800,800,3]
        image = Image.fromarray(image)
        image,mask_image,label = self.pad_and_scale(image,mask_image,label)
        transform = transforms.ToTensor()
        
        image = transform(image)
        mask_image = transform(mask_image)
        # label = self.pad_lab(label)
        return image,mask_image, label, cam_trans, veh_trans

    def pad_and_scale(self, img, mask, lab):
        """
        Args:
            img:
        Returns:
        """
        w,h = img.size
        w_m,h_m = mask.size
        if w==h:
            padded_img = img
        else:
            dim_to_pad = 1 if w<h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h,h), color=(127,127,127))
                padded_img.paste(img, (int(padding), 0))
                lab[:, [1]] = (lab[:, [1]] * w + padding) / h
                lab[:, [3]] = (lab[:, [3]] * w / h)
            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(127,127,127))
                padded_img.paste(img, (0, int(padding)))
                lab[:, [2]] = (lab[:, [2]] * h + padding) / w
                lab[:, [4]] = (lab[:, [4]] * h  / w)

        if w_m==h_m:
            padded_img_m = mask
        else:
            dim_to_pad = 1 if w_m<h_m else 2
            if dim_to_pad == 1:
                padding_m = (h_m - w_m) / 2
                padded_img_m = Image.new('RGB', (h_m,h_m), color=(127,127,127))
                padded_img_m.paste(mask, (int(padding_m), 0))
            else:
                padding_m = (w_m - h_m) / 2
                padded_img_m = Image.new('RGB', (w_m, w_m), color=(127,127,127))
                padded_img_m.paste(mask, (0, int(padding_m)))

        resize = transforms.Resize((self.imgsize,self.imgsize))
        padded_img = resize(padded_img)     #choose here
        padded_img_m = resize(padded_img_m)     #choose here
        return padded_img, padded_img_m, lab

    # def pad_lab(self, lab):
    #     pad_size = self.max_n_labels - lab.shape[0]
    #     if(pad_size>0):
    #         padded_lab = F.pad(lab, (0, 0, 0, pad_size), value=1)
    #     else:
    #         padded_lab = lab
    #     return padded_lab

# class InriaDataset(Dataset):
#     def __init__(self, img_dir, imgsize, shuffle=True, if_square=True):
#         n_png_images = len(fnmatch.filter(os.listdir(img_dir), '*.png'))
#         n_jpg_images = len(fnmatch.filter(os.listdir(img_dir), '*.jpg'))
#         n_images = n_png_images + n_jpg_images
#         # n_labels = len(fnmatch.filter(os.listdir(lab_dir), '*.txt'))
#         # assert n_images == n_labels, "Number of images and number of labels don't match"
#         self.len = n_images
#         self.img_dir = img_dir
#         # self.lab_dir = lab_dir
#         self.imgsize = imgsize
#         self.img_names = fnmatch.filter(os.listdir(img_dir), '*.png') + fnmatch.filter(os.listdir(img_dir), '*.jpg')
#         self.shuffle = shuffle
#         self.img_paths = []
#         self.if_square = if_square
#         for img_name in self.img_names:
#             self.img_paths.append(os.path.join(self.img_dir, img_name))


#     def __len__(self):
#         return self.len

#     def __getitem__(self, idx):
#         assert idx <= len(self), 'index range error'
#         img_path = os.path.join(self.img_dir, self.img_names[idx])
#         # lab_path = os.path.join(self.lab_dir, self.img_names[idx]).replace('.jpg', '.txt').replace('.png', '.txt')
#         image = Image.open(img_path).convert('RGB')


#         image = self.pad_and_scale(image)
#         transform = transforms.ToTensor()
#         image = transform(image)
#         # label = self.pad_lab(label)
#         return image

#     def pad_and_scale(self, img):
#         """

#         Args:
#             img:

#         Returns:

#         """
#         w, h = img.size
#         if w==h:
#             padded_img = img
#         elif self.if_square:
#             a = min(w, h)
#             ww = (w - a) // 2
#             hh = (h - a) // 2
#             padded_img = img.crop([ww, hh, ww+a, hh+a])
#         else:
#             dim_to_pad = 1 if w<h else 2
#             if dim_to_pad == 1:
#                 padding = (h - w) / 2
#                 padded_img = Image.new('RGB', (h,h), color=(127,127,127))
#                 padded_img.paste(img, (int(padding), 0))
#                 # lab[:, [1]] = (lab[:, [1]] * w + padding) / h
#                 # lab[:, [3]] = (lab[:, [3]] * w / h)
#             else:
#                 padding = (w - h) / 2
#                 padded_img = Image.new('RGB', (w, w), color=(127,127,127))
#                 padded_img.paste(img, (0, int(padding)))
#                 # lab[:, [2]] = (lab[:, [2]] * h + padding) / w
#                 # lab[:, [4]] = (lab[:, [4]] * h  / w)
#         resize = transforms.Resize((self.imgsize, self.imgsize))
#         padded_img = resize(padded_img)     #choose here
#         return padded_img

#     def pad_lab(self, lab):
#         pad_size = self.max_n_labels - lab.shape[0]
#         if(pad_size>0):
#             padded_lab = F.pad(lab, (0, 0, 0, pad_size), value=1)
#         else:
#             padded_lab = lab
#         return padded_lab
