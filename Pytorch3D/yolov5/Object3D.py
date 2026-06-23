#encoding=utf-8
import torch
import matplotlib.pyplot as plt
from skimage.io import imread

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj 
from pytorch3d.ops import sample_points_from_meshes
import pytorch3d
# Data structures and functions for rendering
from pytorch3d.structures import Meshes, join_meshes_as_batch, join_meshes_as_scene, Pointclouds
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)
from PIL import Image
# add path for demo utils functions 
import sys
import os
import tabulate
import numpy as np
from pathlib import Path
from fusion import fusion
import cv2 as cv

from plot_image_grid import image_grid
import os
from collections import Counter
from loss_fca import *
import math
from math import *
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def resize(img):
    
    resized_images = torch.nn.functional.interpolate(img.permute(0, 3, 1, 2), 
                                                     size=(640, 640), 
                                                     mode='bilinear', 
                                                     align_corners=False)
    # resized_images = resized_images.permute(0, 2, 3, 1)   
    return resized_images

def show_mesh_info(mesh):
  print(f"mesh.verts_padded().shape = {mesh.verts_padded().shape}")
  print(f"mesh.faces_padded().shape = {mesh.faces_padded().shape}")
  print(f"mesh.textures.maps_padded().shape = {mesh.textures.maps_padded().shape}")
  print("--------")
  print("VERTS UVS:")
  print(f"mesh.textures.verts_uvs_padded().shape = {mesh.textures.verts_uvs_padded().shape}")
  print(mesh.textures.verts_uvs_padded()[0,:5])

  print("FACES UVS:")
  print(f"mesh.textures.faces_uvs_padded().shape = {mesh.textures.faces_uvs_padded().shape}")
  print(mesh.textures.faces_uvs_padded()[0,:5])

# at this point we have a predicted class per point and can apply those classifications to the origin mesh
def majority_vote(x: np.ndarray) -> int:
    """A helper function to count and return the most occuring value in an array."""
    if x.shape[0] == 1:
        return x[0]
    else:
        if not np.any(x):
            return np.bincount(x).argmax()
        else:
            # for data structs containing negative values
            return Counter(x).most_common(1)[0][0]


def image_render(camera_trans,veh,img, texture_img,flag, data_dir=None, renderer_cfg=None):
    """Render a textured 3D vehicle with Pytorch3D.

    Args:
        camera_trans: camera parameters from dataset loader.
        veh: vehicle parameters from dataset loader.
        img: background image batch.
        texture_img: texture tensor [H,W,3] or [1,H,W,3].
        flag: unused flag preserved for compatibility.
        data_dir: optional override for 3D model directory.
        renderer_cfg: optional dict for image_size/fov/yaw_offset/pitch_offset.
    """
    sys.path.append(os.path.abspath(''))
    renderer_cfg = renderer_cfg or {}
    device_name = renderer_cfg.get('device')
    if device_name:
        device = torch.device(device_name)
    elif torch.is_tensor(texture_img):
        device = texture_img.device
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    # batch_size = camera_trans.shape[0]
    # Set paths through config; fallback stays local to the released YOLOv5 folder.
    DATA_DIR = Path(data_dir) if data_dir else Path(__file__).resolve().parent / "data" / "car_model"
    obj_filename = str(DATA_DIR / "raw.obj") #all_camou_carla


    # Load obj file 
    mesh = load_objs_as_meshes([obj_filename], load_textures=True, device=device,texture_wrap = 'clamp')
    texture_image=mesh.textures.maps_padded()
    # 兼容 texture_img 可能是 [H,W,3] 或 [1,H,W,3]
    if texture_img.dim() == 3:
        texture_img = texture_img.unsqueeze(0)

    tex_h, tex_w = texture_image.shape[1], texture_image.shape[2]
    patch_h, patch_w = texture_img.shape[1], texture_img.shape[2]

    # 原逻辑是在大UV图的右侧区域粘贴 patch；若当前UV图本身仅为 480x480，则直接全图覆盖
    if tex_h == patch_h and tex_w == patch_w:
        texture_image[:, :patch_h, :patch_w, :] = texture_img
    elif tex_w >= 3360 + patch_w and tex_h >= patch_h:
        texture_image[:, :patch_h, 3360:3360+patch_w, :] = texture_img
    else:
        # 兜底：贴到左上角，避免切片越界
        h = min(tex_h, patch_h)
        w = min(tex_w, patch_w)
        texture_image[:, :h, :w, :] = texture_img[:, :h, :w, :]
    verts = torch.tensor(mesh.textures.verts_uvs_padded()).to(device)
    faces = torch.tensor(mesh.textures.faces_uvs_padded()).to(device) # faces_list()返回一个包含所有网格的面列表的元组，因此我们获取第一个网格的面列表
    texture_image_new = pytorch3d.renderer.mesh.textures.TexturesUV(maps=texture_image, 
                                                                    faces_uvs=faces,
                                                                    verts_uvs= verts,
                                                                    padding_mode = 'zeros', 
                                                                    align_corners = True,
                                                                    sampling_mode = 'nearest')
    mesh.textures = texture_image_new
    # Define the settings for rasterization and shading. Defaults preserve the original output.
    image_size = int(renderer_cfg.get('image_size', 640))
    pitch_offset = float(renderer_cfg.get('pitch_offset', 0.0))
    yaw_offset = float(renderer_cfg.get('yaw_offset', 0.0))
    fov = float(renderer_cfg.get('fov', 90.0))
    raster_settings = RasterizationSettings(
        image_size=image_size, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
        max_faces_per_bin=15000,
    )
    # Lighting/material defaults are kept intact; only the data path and optional camera offsets are configurable.
    lights = DirectionalLights(ambient_color=((0.8, 0.8, 0.8), ), 
                               diffuse_color=((0.3, 0.3, 0.3), ), 
                               specular_color=((0.2, 0.2, 0.2), ), 
                               direction=((100, 100, 100),),
                               device=device)
    materials = Materials(
        device=device,
        ambient_color=((1, 1, 1),),
        diffuse_color=((1.0, 1.0, 1.0),),
        specular_color=((1, 1, 1),),
        shininess=30.0
    )
    #image_batch[10,3,640,640]
    #mask_batch[10,3,640,640]
    #lab_batch[10,1,5]
    #camera_batch[10,2,3]
    # dis, ele, azi =[], [], []
    batch = camera_trans.shape[0]
    meshes = mesh.extend(batch)
    dis = [0.0] * batch
    ele = [0.0] * batch
    azi = [0.0] * batch   
    result_images = torch.empty((0, image_size, image_size, 4), dtype=torch.float32).to(device)
    for i in range(batch): #要保证总数/batch，能够除尽
        x_1 = camera_trans[i][0][0]
        y_1 = camera_trans[i][0][1]
        z_1 = camera_trans[i][0][2]
        rotal = veh[i][1][1]
        x = y_1
        y = z_1
        z = -x_1
        dis[i] = math.sqrt(x*x+y*y+z*z)
        ele[i] = asin(y*1.0/(dis[i]+1e-10))*180*1.0/3.14 + pitch_offset

        if -100 <= rotal <= -81:
            azi[i] = atan(x*1.0/(z+1e-10))*180*1.0/3.14 + yaw_offset + 0
        elif 0 <= rotal <= 10:
            azi[i] = atan(x*1.0/(z+1e-10))*180*1.0/3.14 + yaw_offset - 90
        elif -10 <= rotal < 0:
            azi[i] = atan(x*1.0/(z+1e-10))*180*1.0/3.14 + yaw_offset + 90
        elif 81 <= rotal <= 100:
            azi[i] = atan(x*1.0/(z+1e-10))*180*1.0/3.14 + yaw_offset + 0
        elif -169 <= rotal <= -160:
            azi[i] = atan(x*1.0/(z+1e-10))*180*1.0/3.14 + yaw_offset - 77         
        elif 170 <= rotal <= 190:
            azi[i] = atan(x*1.0/(z+1e-10))*180*1.0/3.14 + yaw_offset - 90
        elif -190 <= rotal <= -170:
            azi[i] = atan(x*1.0/(z+1e-10))*180*1.0/3.14 + yaw_offset + 90

    R, T = look_at_view_transform(dist=dis, elev=ele, azim=azi)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=fov)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device, 
            cameras=cameras,
            lights=lights
        )
    )
    images = renderer(meshes, cameras=cameras, lights=lights, materials=materials) #tensor[1,640,640,4]
    # for i in range(batch_size):
    #     tank_img = images.detach()[i]#[:,:,:3].cpu().numpy()*255.0
    #     tank_img = (tank_img[:,:,:3].cpu().numpy()*255).astype(np.uint8) #从tensor转换为numpy
    #     tank_img = tank_img[..., ::-1]
    #     tank_img = cv.resize(tank_img, (640, 640))
    #     cv.imwrite(str(Path('outputs/gen/render') / f'tank_{i}.png'), tank_img)
   
    if device.type == "cuda":
        torch.cuda.empty_cache() 
    return(images, texture_img)

 
