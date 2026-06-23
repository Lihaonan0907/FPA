import cv2 as cv
import numpy as np
import os, random
import copy
from pathlib import Path


def load_background(background_dir=None):
    """Load one random background image from a configurable directory."""
    background_dir = Path(background_dir) if background_dir else Path(__file__).resolve().parent / "data" / "bk_train"
    if not background_dir.exists():
        raise FileNotFoundError(f"background directory not found: {background_dir}")
    land_img = random.choice(os.listdir(background_dir))
    land_img = background_dir / land_img
    img1 = cv.imread(str(land_img))
    img1 = cv.resize(img1, (640, 640))
    return img1
# 加载两张图片
def fusion(img_ori,img2):
    # img1 = load_background()
    #img2 2048,2048,4
    img1 = copy.deepcopy(img_ori)
    img2 = (img2[:,:,:3].cpu().numpy()*255).astype(np.uint8) #从tensor转换为numpy
    img2 = img2[..., ::-1]
    img2 = cv.resize(img2, (640, 640))
    # if t==0:
    #     img2 = cv.resize(img2, (500, 500))
    # elif t==1:
    #     img2 = cv.resize(img2, (350, 350))
    # elif t==2:
    #     img2 = cv.resize(img2, (250, 250))

    img2 = (1-(img2==255).astype('uint8'))*img2
    
    # 我希望把 LOGO 防止到左上角，所以创建了一个区域（roi)
    rows,cols,channels = img2.shape
    
    x = 0
    y = 0
    roi = img1[x:x+rows, y:y+cols]
 

    # 将 LOGO 图转换为灰度图，阈值为 10， 最大值为 255 
    img2gray = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
    # mask = 1-(img2gray==255).astype('uint8')
    ret, mask = cv.threshold(img2gray, 1, 255, cv.THRESH_BINARY)
    # print(dir(mask), mask.dtype)

    mask_inv = cv.bitwise_not(mask).astype(np.uint8)
    # print(mask_inv.shape)
    black_pixels = np.where(mask_inv == 0)
    if len(black_pixels[0]) > 0:
    # 寻找黑色图案的上下左右的极值点
        min_x = np.min(black_pixels[1])
        max_x = np.max(black_pixels[1])
        min_y = np.min(black_pixels[0])
        max_y = np.max(black_pixels[0])
        # print("Min X:", min_x)
        # print("Max X:", max_x)
        # print("Min Y:", min_y)
        # print("Max Y:", max_y)
    else:
        print("No black pixels found in mask_inv.")
    # gt = [min_x+x,max_x+x,min_y+y,max_y+y]

    c_x = ((min_x+x+max_x+x)*1.0/2)/640
    c_y = ((min_y+y+max_y+y)*1.0/2)/640
    w = (max_x-min_x)*1.0/640
    h = (max_y-min_y)*1.0/640
    gt = [c_x,c_y,w,h]
    # print(gt)

    # 现在将 ROI 抠出黑色区域
    img1_bg = cv.bitwise_and(roi,roi,mask = mask_inv)

    # 抠出 LOGO 图像
    img2_fg = cv.bitwise_and(img2,img2,mask = mask)

    # 将抠出的 LOGO 防入 ROI 区域中
    dst = cv.add(img1_bg,img2_fg)
    img1[x:x+rows, y:y+cols ] = dst

    return img1,gt
