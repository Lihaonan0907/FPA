本目录保留原 YOLOv5 训练入口和相关模块：

- `main_gen.py`: 伪装纹理训练主入口。
- `loss_fca.py`: 对抗损失、TV 平滑损失、NPS 不可打印损失等。
- `style_VAE.py`: 扩散/生成模块。
- `Object3D.py`: PyTorch3D 渲染入口。

发布版不再写死个人绝对路径。请优先修改仓库根目录的 `configs/default.yaml`。
其它重要代码：loss_fca（损失函数代码）、style_VAE（扩散模型代码）、Object3D（渲染器代码）；
loss_fca：①修改device；②PatchTransformer（用于设置图像合成的位置和提取标签）；③对抗损失（ComputeLoss函数）、对抗损失（MaxProbExtractor函数）、平滑度损失（TotalVariationLoss函数）、不可打印分数（NPSLoss函数）；
style_VAE：主要函数DiffusionModel（），主要参数设置迭代次数t；
Object3D：①修改device；②修改3D模型路径（DATA_DIR）；③设置纹理部署位置；④设置俯仰角等参数；
main_gen：①train_loader（数据集位置）；②加载检测模型（路径修改，模型更改）；③texture_para_pt（伪装初始化）；④损失权重初始化
