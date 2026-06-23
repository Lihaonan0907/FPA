import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from PIL import Image
import torch
import cv2

# 加载图像
def voro(image_tensor, device):
    # image = Image.open('texture.jpg')
    # image_array = np.array(image)
    # image_tensor = torch.from_numpy(image_array)
    # 获取图像的形状和像素值
    image_tensor = image_tensor.squeeze(0)
    image_tensor_out = image_tensor.clone()
    h, w, _ = image_tensor.size()
    image_array = image_tensor.detach().cpu().numpy()
    pixels = image_array.reshape(-1, 3)

    # 选择最大的20个像素
    max_pixels_indices = np.argpartition(np.linalg.norm(pixels, axis=1), -1000)[-1000:]
    max_pixels = pixels[max_pixels_indices]

    # 确定控制点的位置
    control_points = torch.tensor([(index // w, index % w) for index in max_pixels_indices])
    control_points = control_points.float()

    # 构建Voronoi图
    vor = Voronoi(control_points)
 
    regions = vor.point_region  # 获取每个点对应的区域索引
    values = torch.arange(1, len(vor.regions))  # 为每个区域指定一个唯一的值
    # output = torch.zeros_like(image_tensor, dtype=torch.long)  # 创建与输入图像相同大小的输出张量

    for i, region_index in enumerate(regions):
        indices = vor.regions[region_index]  # 获取区域的顶点索引

        if -1 not in indices:  # 排除无限区域
            region_points = vor.vertices[vor.regions[region_index]]
            region_points = torch.tensor(region_points)
            region_points = torch.clamp(region_points, 0, h-1).long()
            region_mask = torch.zeros((h, w), dtype=torch.long).to(device)
            region_mask[region_points[:, 0], region_points[:, 1]] = 1

            region_mean = torch.zeros(3)
            for channel in range(3):
                image_chan = image_tensor[:, :, channel]
                num = torch.sum(region_mask).item()
                # print(image_chan.size(),torch.sum(region_mask),region_mask.size())
                region_mean[channel] = torch.sum(image_chan * region_mask) / num

            polygon = vor.vertices[indices]  # 获取区域的顶点坐标
            polygon = torch.from_numpy(polygon)
            polygon = torch.round(polygon).long()  # 四舍五入为整数坐标

            mask = torch.zeros((h, w), dtype=torch.uint8)  # 创建当前区域的掩码
            cv2.fillPoly(mask.numpy(), [polygon.numpy()], 1)  # 使用多边形填充掩码
            # print(region_mean.long())
            image_tensor_out[mask == 1] = region_mean.float().to(device)
    image_tensor_out = image_tensor_out.unsqueeze(0).float()
    return image_tensor_out
    # output = output.numpy()

    # # 显示结果
    # fig, ax = plt.subplots()
    # ax.imshow(output)

    # # 隐藏坐标轴
    # ax.axis('off')

    # # 显示结果
    # plt.show()
def voro_with_mask(image_tensor, texture_mask, device): 
    # 获取图像的形状和像素值
    image_tensor = image_tensor.squeeze(0)
    image_tensor_out = image_tensor.clone()
    h, w, _ = image_tensor.size()
    image_array = image_tensor.detach().cpu().numpy()
    pixels = image_array.reshape(-1, 3)

    # 选择最大的20个像素
    num = 100
    max_pixels_indices = np.argpartition(np.linalg.norm(pixels, axis=1), -num)[-num:]
    max_pixels = pixels[max_pixels_indices]

    # 确定控制点的位置
    control_points = torch.tensor([(index // w, index % w) for index in max_pixels_indices])
    control_points = control_points.float()

    # 构建Voronoi图
    vor = Voronoi(control_points)
 
    regions = vor.point_region  # 获取每个点对应的区域索引
    values = torch.arange(1, len(vor.regions))  # 为每个区域指定一个唯一的值
    # output = torch.zeros_like(image_tensor, dtype=torch.long)  # 创建与输入图像相同大小的输出张量
    # print(texture_mask.size())
    texture_mask = texture_mask.squeeze(0)[:, :, 0].squeeze(-1)
    for i, region_index in enumerate(regions):
        indices = vor.regions[region_index]  # 获取区域的顶点索引

        if -1 not in indices:  # 排除无限区域
            region_points = vor.vertices[vor.regions[region_index]]
            region_points = torch.tensor(region_points)
            region_points = torch.clamp(region_points, 0, h-1).long()
            region_mask = torch.zeros((h, w), dtype=torch.long).to(device)
            region_mask = region_mask*texture_mask
            region_mask[region_points[:, 0], region_points[:, 1]] = 1
            region_mean = torch.sum(image_tensor * region_mask[..., None]) / torch.sum(region_mask)

            # region_mean = torch.zeros(3)
            # for channel in range(3):
            #     image_chan = image_tensor[:, :, channel]
            #     num = torch.sum(region_mask).item()
            #     # print(image_chan.size(),torch.sum(region_mask),region_mask.size())
            #     region_mean[channel] = torch.sum(image_chan * region_mask) / num

            polygon = vor.vertices[indices]  # 获取区域的顶点坐标
            polygon = torch.from_numpy(polygon)
            polygon = torch.round(polygon).long()  # 四舍五入为整数坐标

            mask = torch.zeros((h, w), dtype=torch.uint8)  # 创建当前区域的掩码
            cv2.fillPoly(mask.numpy(), [polygon.numpy()], 1)  # 使用多边形填充掩码
            # print(region_mean.long())
            image_tensor_out[mask == 1] = region_mean.float().to(device)
    image_tensor_out = image_tensor_out.unsqueeze(0).float()
    return image_tensor_out
 

