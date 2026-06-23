
#encoding=utf-8
import torch

from PIL import Image 
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize


from models.experimental import attempt_load 
from Object3D import image_render
from utils.plots import save_one_box,  colors, Annotator 
from utils.dataloaders import create_dataloader,LoadImages
from skimage.color import rgb2hsv
from utils.general import TQDM_BAR_FORMAT, check_img_size, check_dataset, colorstr, non_max_suppression, scale_boxes, check_amp, labels_to_class_weights
import numpy as np 
import cv2 as cv
from utils.autobatch import check_train_batch_size
from tqdm import tqdm
from pathlib import Path
from loss_fca import ComputeLoss,ComputeLoss_WTO,SaliencyLoss,TotalVariationLoss,MaxProbExtractor,NPSLoss
import loss_fca as loss_module
import argparse
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import yaml 
import os
import time

import torch.nn as nn
from torch.utils.data import Dataset 
from utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, smart_optimizer,
                               smart_resume, torch_distributed_zero_first)
from torch.utils.data import DataLoader
from utils.loggers import Loggers
import random
import os
import torch
import numpy as np
# import trimesh
from torch.autograd import Variable 
import torchvision.transforms.functional as TF
from voronoi import voro 
from style_VAE import VAE,DiffusionModel, loss_fn
from fusion import load_background, fusion
from loss_fca import *
import yaml
from detect import *
from models.common import DetectMultiBackend

torch.manual_seed(42)

device = torch.device("cuda:0")
device1 = torch.device("cuda:0")

WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parents[1]


def _resolve_config_path(config_path):
    """Resolve config paths from cwd, repo root, or the YOLOv5 project root."""
    config_file = Path(config_path)
    if config_file.is_absolute():
        return config_file
    for base in (Path.cwd(), REPO_ROOT, PROJECT_ROOT):
        candidate = base / config_file
        if candidate.exists():
            return candidate
    return PROJECT_ROOT / config_file


def _resolve_path(path_value, fallback=None):
    """Resolve runtime asset paths relative to Pytorch3D/yolov5."""
    candidate = path_value if path_value not in (None, "") else fallback
    if candidate in (None, ""):
        return None
    candidate_path = Path(candidate)
    if not candidate_path.is_absolute():
        candidate_path = PROJECT_ROOT / candidate_path
    return str(candidate_path)


def _load_runtime_config(config_path):
    config_file = _resolve_config_path(config_path)
    if not config_file.exists():
        print(f"warning: config file not found: {config_file}, using built-in defaults")
        return {}
    with open(config_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config file must contain a mapping: {config_file}")
    return data


def _loss_weights(cfg):
    weights = cfg.get('loss_weights', {}) if isinstance(cfg, dict) else {}
    return {
        'adversarial': weights.get('adversarial', 1.0),
        'detector': weights.get('detector', 1.0),
        'tv': weights.get('tv', 10.0),
        'nps': weights.get('nps', 1.0),
        'color': weights.get('color', 0.0),
        'color_ratio': weights.get('color_ratio', 0.0),
        'df': weights.get('df', 0.0),
    }

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

 
   
def resize(img):
    
    resized_images = torch.nn.functional.interpolate(img.permute(0, 3, 1, 2), 
                                                     size=(640, 640), 
                                                     mode='bilinear', 
                                                     align_corners=False)
    # resized_images = resized_images.permute(0, 2, 3, 1)   
    return resized_images
def resize_480(img, rgb=True):
    if img.size()[1]!=3:
        img = img.permute(0,3,1,2)
    resized_images = torch.nn.functional.interpolate(img, 
                                                     size=(640, 640), 
                                                     mode='bilinear', 
                                                     align_corners=False)
    if rgb:
        resized_images = resized_images.permute(0, 2, 3, 1)   
    return resized_images
def rgb_to_hsv(img):
        eps=1e-8
        img = img.permute(0,3,1,2)
        hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(img.device)

        hue[ img[:,2]==img.max(1)[0] ] = 4.0 + ( (img[:,0]-img[:,1]) / ( img.max(1)[0] - img.min(1)[0] + eps) ) [ img[:,2]==img.max(1)[0] ]
        hue[ img[:,1]==img.max(1)[0] ] = 2.0 + ( (img[:,2]-img[:,0]) / ( img.max(1)[0] - img.min(1)[0] + eps) ) [ img[:,1]==img.max(1)[0] ]
        hue[ img[:,0]==img.max(1)[0] ] = (0.0 + ( (img[:,1]-img[:,2]) / ( img.max(1)[0] - img.min(1)[0] + eps) ) [ img[:,0]==img.max(1)[0] ]) % 6

        hue[img.min(1)[0]==img.max(1)[0]] = 0.0
        hue = hue/6

        saturation = ( img.max(1)[0] - img.min(1)[0] ) / ( img.max(1)[0] + eps )
        saturation[ img.max(1)[0]==0 ] = 0

        value = img.max(1)[0]
        
        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)
        hsv = torch.cat([hue, saturation, value],dim=1)
        return hsv
class RGB_HSV(nn.Module):
#     RGB or HSV's shape: (B * C * H * W)
# RGB or HSV's range: [0, 1)
    def __init__(self, eps=1e-8):
        super(RGB_HSV, self).__init__()
        self.eps = eps

    def rgb_to_hsv(self, img):
        img = img.permute(0,3,1,2)
        hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(img.device)

        hue[ img[:,2]==img.max(1)[0] ] = 4.0 + ( (img[:,0]-img[:,1]) / ( img.max(1)[0] - img.min(1)[0] + self.eps) ) [ img[:,2]==img.max(1)[0] ]
        hue[ img[:,1]==img.max(1)[0] ] = 2.0 + ( (img[:,2]-img[:,0]) / ( img.max(1)[0] - img.min(1)[0] + self.eps) ) [ img[:,1]==img.max(1)[0] ]
        hue[ img[:,0]==img.max(1)[0] ] = (0.0 + ( (img[:,1]-img[:,2]) / ( img.max(1)[0] - img.min(1)[0] + self.eps) ) [ img[:,0]==img.max(1)[0] ]) % 6

        hue[img.min(1)[0]==img.max(1)[0]] = 0.0
        hue = hue/6

        saturation = ( img.max(1)[0] - img.min(1)[0] ) / ( img.max(1)[0] + self.eps )
        saturation[ img.max(1)[0]==0 ] = 0

        value = img.max(1)[0]
        
        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)
        hsv = torch.cat([hue, saturation, value],dim=1)
        return hsv

    def hsv_to_rgb(self, hsv):
        h,s,v = hsv[:,0,:,:],hsv[:,1,:,:],hsv[:,2,:,:]
        #对出界值的处理
        h = h%1
        s = torch.clamp(s,0,1)
        v = torch.clamp(v,0,1)
  
        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)
        
        hi = torch.floor(h * 6)
        f = h * 6 - hi
        p = v * (1 - s)
        q = v * (1 - (f * s))
        t = v * (1 - ((1 - f) * s))
        
        hi0 = hi==0
        hi1 = hi==1
        hi2 = hi==2
        hi3 = hi==3
        hi4 = hi==4
        hi5 = hi==5
        
        r[hi0] = v[hi0]
        g[hi0] = t[hi0]
        b[hi0] = p[hi0]
        
        r[hi1] = q[hi1]
        g[hi1] = v[hi1]
        b[hi1] = p[hi1]
        
        r[hi2] = p[hi2]
        g[hi2] = v[hi2]
        b[hi2] = t[hi2]
        
        r[hi3] = p[hi3]
        g[hi3] = q[hi3]
        b[hi3] = v[hi3]
        
        r[hi4] = t[hi4]
        g[hi4] = p[hi4]
        b[hi4] = v[hi4]
        
        r[hi5] = v[hi5]
        g[hi5] = p[hi5]
        b[hi5] = q[hi5]
        
        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)
        rgb = torch.cat([r, g, b], dim=1)
        return rgb
    
class ReparameterizedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = torch.randn(2048, 2048,3)  # 预先采样噪声
        
    def forward(self, x):
        eps = self.eps.view(x.size()).to(device)  # 重塑eps和x相同shape
        return x + 0.1 * eps   # 用eps替换高斯噪声
def gaussian_regularization_loss(texture_para, texture_mask):

    loss = torch.mean(((texture_para-0.5)*texture_mask) ** 2)
    return  loss


def color_loss(image, texture_mask):
    # print(image.size())
    image = image.squeeze()

    # 计算图像中不在黄色区间的像素的损失
    # 可以使用任何定义黄色区间的方法，例如基于颜色空间或阈值等
    yellow_min = torch.tensor([238/255, 213/255, 183/255]).to(device)  # 黄色区间的下界
    yellow_max = torch.tensor([1, 248/255, 220/255]).to(device)    # 黄色区间的上界
    # 将图像的像素值限制在黄色区间内
    # image = torch.clamp(image, yellow_min, yellow_max)
    # 计算不在黄色区间的像素的损失
    loss = torch.mean(((image - yellow_min)*texture_mask)**2) + torch.mean(((image - yellow_max)*texture_mask)**2)
    # loss = torch.norm((image)*texture_mask, 2) + torch.norm((image - yellow_max)*texture_mask,2)
    return loss


def mse_loss(img, img_bg, texture_mask):
    loss = torch.mean(((img - img_bg)*texture_mask)**2) 
    return loss
class NonZeroLoss(nn.Module):
    def __init__(self):
        super(NonZeroLoss, self).__init__()

    def forward(self, output):
        non_zero_count = (output != 1.0).sum()
        return non_zero_count


class ResizeNet(nn.Module):
    def __init__(self):
        super(ResizeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        return out
def load_bg():
    back_img_cv  = load_background()
    transform = transforms.ToTensor()
        # Convert the image to PyTorch tensor
    back_img = transform(back_img_cv).to(device).unsqueeze(0)
        # back_img = torch.nn.functional.interpolate(back_img,size=(480, 480),  mode='bilinear',  align_corners=False)
    back_img = resize_480(back_img, rgb=True)
    return back_img, back_img_cv

# 定义一个损失函数，包括内容损失和颜色比例损失
def color_ratio_loss(fake_image, texture_mask, target_green_ratio=0.6, target_yellow_ratio=0.4):
    # 计算内容损失，可以使用像素差异、感知损失等方法
    # print(fake_image.size())
    # 计算颜色比例损失
    green_pixels = (fake_image[:, :, :, 1] > 0.5).float().mean()  # 绿色通道
    yellow_pixels = ((fake_image[:, :, :, 0] > 0.5) & (fake_image[:, :, :, 1] > 0.5)).float().mean()  # 红色和绿色通道
    # 计算与目标比例的差异
    # green_loss = torch.abs(green_pixels - target_green_ratio)
    # yellow_loss = torch.abs(yellow_pixels - target_yellow_ratio)
    loss = torch.mean(((green_pixels - target_green_ratio)*texture_mask)**2) + torch.mean(((yellow_pixels - target_yellow_ratio)*texture_mask)**2)
    # 定义权重以平衡内容损失和颜色比例损失
    # content_weight = 1.0
    color_ratio_weight = 1.0  # 调整这个权重以控制颜色比例的重要性
    # 计算总损失
    # total_loss =  color_ratio_weight * (green_loss + yellow_loss)

    return loss

def get_loader(train_path,lab_path,mask_dir,batch_size,shuffle):
    loader = torch.utils.data.DataLoader(InriaDataset(train_path,lab_path,mask_dir, 640, shuffle=shuffle),
                                             batch_size= batch_size,
                                             shuffle=True,
                                             num_workers=0)
    return loader

'''
转换gt
'''
def gt_trans(gt,size,batch_size): #gt[10,]
    gt_list = []
    for i in range(batch_size):
        min_x = gt[i][0]
        max_x = gt[i][2]
        min_y = gt[i][1]
        max_y = gt[i][3]
        c_x = ((min_x+max_x)*1.0/2)/size
        c_y = ((min_y+max_y)*1.0/2)/size
        w = (max_x-min_x)*1.0/size
        h = (max_y-min_y)*1.0/size
        gt_z = torch.cat((c_x.unsqueeze(0), c_y.unsqueeze(0), w.unsqueeze(0), h.unsqueeze(0)), dim=0)
        a = torch.tensor([i,2]).to(device)
        a= torch.cat((a,gt_z), dim=0)
        # print(gt_z)
        gt_list.append(a)
    gt_tensor = torch.stack(gt_list, dim=0)
    return gt_tensor

def cv_save(img_batch,mask_batch,p_img_batch,flag, output_dir=None):
    batch_size = img_batch.shape[0]
    base_dir = Path(output_dir) if output_dir else PROJECT_ROOT / 'outputs' / 'gen'
    bk_dir = base_dir / 'bk'
    mask_dir = base_dir / 'mask'
    render_dir = base_dir / 'render'
    bk_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    render_dir.mkdir(parents=True, exist_ok=True)
    for i in range(batch_size):
        img1 = img_batch.detach()[i]#[:,:,:3].cpu().numpy()*255.0
        img1 = img1.permute(1,2,0)        
        img1 = (img1[:,:,:3].cpu().numpy()*255).astype(np.uint8) #从tensor转换为numpy
        # img1 = img1[..., ::-1]
        cv.imwrite(str(bk_dir / f'{i}.png'), img1)

        img2 = mask_batch.detach()[i]#[:,:,:3].cpu().numpy()*255.0
        img2 = img2.permute(1,2,0)        
        img2 = (img2[:,:,:3].cpu().numpy()*255).astype(np.uint8) #从tensor转换为numpy
        img2 = img2[..., ::-1]
        cv.imwrite(str(mask_dir / f'{i}.png'), img2)

        img3 = p_img_batch.detach()[i]#[:,:,:3].cpu().numpy()*255.0
        img3 = img3.permute(1,2,0)        
        img3 = (img3[:,:,:3].cpu().numpy()*255).astype(np.uint8) #从tensor转换为numpy
        img3 = img3[..., ::-1]
        cv.imwrite(str(render_dir / f'{i}.png'), img3)

def attack(opt):

    global device, device1

    runtime_cfg = getattr(opt, 'config_dict', {}) or {}
    device_name = runtime_cfg.get('device', opt.device or 'cuda:0')
    device = torch.device(device_name)
    device1 = device
    if hasattr(loss_module, 'set_device'):
        loss_module.set_device(device)
    else:
        loss_module.device = device

    loss_weights = _loss_weights(runtime_cfg)
    renderer_cfg = runtime_cfg.get('renderer', {}) if isinstance(runtime_cfg, dict) else {}
    renderer_cfg = dict(renderer_cfg)
    renderer_cfg.setdefault('device', str(device))

    #初始化部分参数
    epochs, batch_size, weights = opt.epochs, opt.batch_size, _resolve_path(opt.weights)
    dataset_root = _resolve_path(runtime_cfg.get('dataset_path', opt.datapath))
    mask_dir = os.path.join(dataset_root, 'train_new/')
    train_path = os.path.join(dataset_root, 'train/')
    lab_path = os.path.join(dataset_root, 'train_label_new/')
    output_dir = _resolve_path(runtime_cfg.get('output_dir', 'outputs/gen'))
    texture_checkpoint = _resolve_path(runtime_cfg.get('texture_checkpoint_path', 'checkpoints/texture_para.pt'))
    texture_init_path = _resolve_path(runtime_cfg.get('texture_init_path', 'data/3Dmodels/car/ao_di/texture.jpg'))
    texture_bg_path = _resolve_path(runtime_cfg.get('texture_bg_path', 'datasets/car/single/highway/train/1.jpg'))
    attack_data_yaml = _resolve_path(runtime_cfg.get('data_yaml', opt.data))
    nps_values_path = _resolve_path(runtime_cfg.get('nps_values_path', '30values.txt'))
    object3d_data_dir = _resolve_path(runtime_cfg.get('object3d_data_dir', 'data/car_model'))
    diffusion_step_t = int(runtime_cfg.get('diffusion_step_t', 0))
    Path(texture_checkpoint).parent.mkdir(parents=True, exist_ok=True)

    # with open(opt.data) as f:
    #     data_dict = yaml.safe_load(f)  # data dict
    train_loader = get_loader(train_path,lab_path,mask_dir,batch_size, True)
    print(f'One training epoch has {len(train_loader.dataset)} images')

   
    #初始化纹理和背景图
    texture_img =  Image.open(texture_init_path).convert('RGB') #texture_carla  texture
    texture_img = torch.FloatTensor(np.array(texture_img) / 255.)
    print("the size of the texture image is ", texture_img.size())
    texture_bg =  Image.open(texture_bg_path).convert('RGB')  
    resize_new=Compose([
            Resize((480,480)),
            ToTensor()
        ])
    texture_bg = resize_new(texture_bg)
    texture_bg = texture_bg.permute(1,2,0).unsqueeze(0).to(device)
    print(texture_bg.size())

    #检测模型处理-yolov5
    imgsz = 640
    model = attempt_load(weights).to(device)
    data_dict = check_dataset(attack_data_yaml)  # check if None
    nc =  int(data_dict['nc'])  # number of classes
    print("the size of the nc ", nc)
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple
    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
    model.hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers 
    compute_loss = ComputeLoss(model)
     
    #扩散模型初始化
    diffusion = DiffusionModel().to(device)
    size_batch = texture_bg.size()
    noise = torch.randn(size_batch).to(device)

    #初始伪装加载
    model.eval()
    texture_para = torch.tensor(texture_img).unsqueeze(0).to(device)
    texture_para_ori = texture_para.clone().cpu()
    texture_mask = (texture_para.detach().sum(dim=3) > 0.04).detach().float().unsqueeze(-1)

    #检测器初始化
    model_det = DetectMultiBackend(weights, device=device, dnn=False, data=attack_data_yaml, fp16=False)

    texture_para_pt = texture_checkpoint
    texture_para_cal = texture_para.clone().to(device)
    if os.path.exists(texture_para_pt):
        print("loading existing parameter")
        ckpt = torch.load(texture_para_pt, map_location=device)
        if isinstance(ckpt, dict):
            diffusion.load_state_dict(ckpt)
            print("loaded diffusion state_dict")
        elif torch.is_tensor(ckpt):
            # 兼容旧版：该文件以前保存的是直接优化得到的纹理Tensor
            print("warning: checkpoint is a texture Tensor (old format), skip diffusion load")
            texture_para_cal = ckpt.to(device)
        else:
            print("warning: unknown checkpoint format, train diffusion from scratch")
    else:
        print("no existing diffusion parameter, train from scratch")

    #训练参数初始化
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    learning_rate = 0.01
    loss_min = 1e7
    loss_max = 2.5
    loss = 0
    last_loss = 0
    

    #部分函数初始化
    patch_transformer = PatchTransformer().to(device)
    loss_saliency = SaliencyLoss().to(device)
    loss_vt = TotalVariationLoss().to(device)
    loss_det = MaxProbExtractor().to(device)
    loss_nz = NonZeroLoss().to(device)
    loss_nps = NPSLoss(nps_values_path,[480,480,3]).to(device)
    optimizer = torch.optim.Adam(diffusion.parameters(), lr = opt.lr) 



    for epoch in range(epochs): 
        epoch_start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        num = 0
        for i_batch, (img_batch,mask_batch,lab_batch,camera_batch,veh_batch) in tqdm(enumerate(train_loader)):
            #img_batch[10,3,640,640]
            #mask_batch[10,3,640,640]
            #lab_batch[10,1,5]
            #camera_batch[10,2,3]
            num = num + 1
            img_batch = img_batch.to(device) #加载背景图片
            mask_batch = mask_batch.to(device) 
            camera_batch = camera_batch.to(device) 
            veh_batch = veh_batch.to(device)

            #需要的标签格式为：tensor[10,6]
            num_samples = lab_batch.size(0)
            indices = torch.arange(num_samples).unsqueeze(1)
            lab_batch = torch.cat([indices,lab_batch], dim=1).to(device)

            loss_df = 0
            x_t = diffusion(noise, diffusion_step_t)
            x_t = x_t.permute(0, 2, 3, 1)
            x_t = x_t * texture_mask
            texture_para_cal = torch.clamp(x_t, min=0.0, max=1.0)

            imgs,_ = image_render(camera_batch,veh_batch,img_batch,texture_para_cal,'color', data_dir=object3d_data_dir, renderer_cfg=renderer_cfg) #[10,640,640,4]
            imgs = imgs[:,:,:,:3]
            images = imgs.permute(0, 3, 1, 2) #得到渲染图像，[10,3,640,640]
            images = images[:, [2, 1, 0], :, :] #BGR转RGB
            # images = images.permute(0, 2, 3, 1) #[10,640,640,3]
            
            #渲染图像和mask图像融合，得到最终的图像
            #image_batch:tensor[10,3,640,640]
            #mask_batch:tensor[10,3,640,640]
            #images:tensor[10,3,640,640]
            # 根据 mask_batch 生成对应的图像
            #由于角度对应不是特别好，要保证车能够完全显示出来

            images_mask = torch.where(images == 1, 0, 1)
            generated_images = img_batch *(1-mask_batch)
            p_img_batch = generated_images * (1-images_mask) + images * images_mask 
            p_img_batch = p_img_batch[:, [2, 1, 0], :, :]
            # p_img_batch = p_img_batch.permute(0, 2, 3, 1) #[10,640,640,3]




            # for i in range(batch_size):
            #     tank_img = p_img_batch.detach()[i]#[:,:,:3].cpu().numpy()*255.0
            #     tank_img = (tank_img[:,:,:3].cpu().numpy()*255).astype(np.uint8) #从tensor转换为numpy
            #     tank_img = tank_img[..., ::-1]
            #     tank_img = cv.resize(tank_img, (640, 640))
            #     cv.imwrite(str(Path(output_dir) / 'render' / f'tank_{i}.png'), tank_img)
           
            # p_img_batch, gtt = patch_transformer(img_batch, images)
            # gt = gt_trans(gtt,imgsz,batch_size)
            cv_save(img_batch,mask_batch,p_img_batch,'color', output_dir=output_dir)

            loss_df += loss_fn(p_img_batch, img_batch)  

            # #计算背景和真实的区别，应该是用沙漠或者丛林的背景来和坦克的纹理图来计算，而不是和back+img计算吧
            # #back_img为当前的随机背景，imgs_a为放入坦克并渲染后的图像，但是环境不一样之后，很难去拟合这一部分 
            #        
            texture_para_np = texture_para_cal.squeeze().detach().cpu().numpy()
            texture_para_np_uint8 = (texture_para_np * 255).astype(np.uint8)
            texture_mask_np = (np.int32(texture_mask.squeeze().detach().cpu().numpy() * 255)).astype(np.uint8)

                    
            img = Image.fromarray(texture_mask_np)
            img.save('texture_mask.jpg')
            im = Image.fromarray(texture_para_np_uint8)
            im.save('texture.jpg')
            # texture_para_mean = texture_para_cal.mean(-1)

            if num%20 == 0:
                run(source = str(Path(output_dir) / 'render'), model = model_det)           



            # imgs = resize(imgs) #[10,3,640,640]
                    
            outputs = model(p_img_batch)  
            loss_vt_ = loss_vt(texture_para_cal).mean()  

            loss_color_ratio = color_ratio_loss(texture_para_cal, texture_mask)
            loss0, loss_item, lbox, lobj, lcls = compute_loss(outputs[1], lab_batch)  
            loss_nps_ = loss_nps(texture_para_cal) 
            loss0 = loss0[0]
            lbox = lbox[0]
            lobj = lobj[0]
            lcls = lcls[0]
            loss_norm0 = loss_nz(texture_para_cal*texture_mask) 
            loss_color = color_loss(texture_para_cal, texture_mask) 
            loss_det_ = torch.mean(loss_det(outputs[0])) ## detection loss
            # print((texture_para_cal.size(), texture_bg.size(), texture_mask.size()))
            loss_rescon = mse_loss(texture_para_cal, texture_bg, texture_mask)
            # loss = 0.0*loss_df + 10*loss_det_ + 10*loss0 + 3*loss_nps_ + 0.2*loss_color_ratio + 3*loss_vt_ + 0*loss_color #+loss0_bin + loss_vt_bin +  loss_color #loss0 + loss_det_ + loss_norm0 + loss_sali + loss_tv#+ 1e-5* loss_norm0
            
            # if loss0 > loss_max:
            #     loss0 = last_loss.to(device)

            loss = (loss_weights['detector'] * loss_det_
                    + loss_weights['adversarial'] * loss0
                    + loss_weights['nps'] * loss_nps_
                    + loss_weights['color'] * loss_color
                    + loss_weights['tv'] * loss_vt_
                    + loss_weights['color_ratio'] * loss_color_ratio
                    + loss_weights['df'] * loss_df)       #color_ratio_loss,基于RGB混合权重来约束颜色

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
  
            if loss.item() <  loss_min:
                print("saving the parameter")
                loss_min = loss.item()
                torch.save(diffusion.state_dict(), texture_para_pt)
                    
            

            print('Epoch %d Loss %.5f adv %.5f det %.5f tv %.5f nps %.5f lbox %.5f lobj %.5f lcls %.5f' % (epoch, loss.data.cpu().numpy(), loss0.data.cpu().numpy(), loss_det_.data.cpu().numpy(), loss_vt_.data.cpu().numpy(), 
            loss_nps_.data.cpu().numpy(),lbox.data.cpu().numpy(), lobj.data.cpu().numpy(), lcls.data.cpu().numpy()))
            # del outputs, loss, loss0, loss_det_, loss_nps_, loss_vt_
            torch.cuda.empty_cache()

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        if torch.cuda.is_available():
            peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        else:
            peak_mem_mb = 0.0
        print(f"\n{'='*55}")
        print(f"✅ Epoch {epoch} 性能监控 (Performance Monitor):")
        print(f" ⏳ 运行耗时 (Time): {epoch_duration:.2f} 秒")
        print(f" 💾 峰值显存 (Peak GPU Memory): {peak_mem_mb:.2f} MB")
        print(f"{'='*55}\n")
    

def parse_opt(known=False):
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', '--cfg', dest='config', type=str, default='configs/default.yaml', help='runtime config yaml')
    config_args, remaining = config_parser.parse_known_args()
    runtime_cfg = _load_runtime_config(config_args.config)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '--cfg', dest='config', type=str, default=config_args.config, help='runtime config yaml')

    parser.add_argument('--weights', type=str, default=runtime_cfg.get('yolo_weights', 'best.pt'), help='initial weights path')

    parser.add_argument('--epochs', type=int, default=runtime_cfg.get('epochs', 5000), help='total training epochs')
    parser.add_argument('--data', type=str, default=runtime_cfg.get('data_yaml', 'data/attack.yaml'), help='dataset.yaml path')
    parser.add_argument('--lr', type=float, default=runtime_cfg.get('learning_rate', 0.1), help='learning rate')
    parser.add_argument('--batch-size', type=int, default=runtime_cfg.get('batch_size', 8), help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=runtime_cfg.get('image_size', 640), help='train, val image size (pixels)')
    parser.add_argument('--datapath', type=str, default=runtime_cfg.get('dataset_path', 'datasets/carla_dataset'), help='data path')


    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='image --cache ram/disk')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default=runtime_cfg.get('device', ''), help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')

    # Logger arguments
    parser.add_argument('--entity', default=None, help='Entity')
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='Upload data, "val" option')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='Version of dataset artifact to use')

    parser.add_argument("--loss_type", default='max_iou', help='max_iou, max_conf, softplus_max, softplus_sum')
    parser.add_argument("--train_iou", type=float, default=0.01, help='')

    opt = parser.parse_known_args(remaining)[0] if known else parser.parse_args(remaining)
    opt.config_dict = runtime_cfg
    return opt


# import random

 



class UnityDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, label_dir, img_size=640):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.img_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')]
        self.label_files = [os.path.join(label_dir, f) for f in os.listdir(label_dir)]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        label_path = self.label_files[idx]

        # 读取图片并进行预处理
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.img_size, self.img_size))
        img = torch.FloatTensor(np.array(img) / 255.).permute(2,0,1)
        # print(img.size())

        # 解析标签文件
        label = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                class_id, x, y, w, h = line.strip().split(',')
                x, y, w, h = float(x), float(y), float(w), float(h)
                label.append([int(class_id), x, y, w, h])

        return img, torch.FloatTensor(label)

if __name__ == "__main__":
    opt = parse_opt()
    # test()
    attack(opt)
    # render(15)
