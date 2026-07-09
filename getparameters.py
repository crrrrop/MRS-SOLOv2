# 计算SOLOv2的参数量 修改argparser.py来选择backbone的的模型;修改对应的结构来选择是否用MSCA注意力;
import paddle
from config import *  # 从config模块导入所有内容

from model.solo import *  # 从model.solo模块导入所有内容
from tools.argparser import ArgParser  # 从tools.argparser模块导入ArgParser类

parser = ArgParser()  # 创建参数解析器对象
cfg = parser.get_cfg()  # 获取配置文件对象
# 创建模型
Backbone = select_backbone(cfg.backbone_type)  # 选择骨干网络类型
backbone = Backbone(**cfg.backbone)  # 初始化骨干网络
FPN = select_fpn(cfg.fpn_type)  # 选择特征金字塔网络类型
fpn = FPN(**cfg.fpn)  # 初始化特征金字塔网络
print("||已初始化特征金字塔网络")
MaskFeatHead = select_head(cfg.mask_feat_head_type)  # 选择掩码特征头类型
mask_feat_head = MaskFeatHead(**cfg.mask_feat_head)  # 初始化掩码特征头
Loss = select_loss(cfg.solo_loss_type)  # 选择损失函数类型
solo_loss = Loss(**cfg.solo_loss)  # 初始化损失函数
Head = select_head(cfg.head_type)  # 选择头部类型
head = Head(solo_loss=solo_loss, nms_cfg=cfg.nms_cfg, **cfg.head)  # 初始化头部
model = SOLOv2(backbone, fpn, mask_feat_head, head)  # 初始化整体模型


# 打印模型的参数信息以及计算总参数量
total_params = 0
print("Model parameters:")
for name, param in model.named_parameters():
    param_count = param.numel()  # 计算参数总数
    total_params += param_count
    print(f"Name: {name}, Shape: {param.shape}, Parameters: {param_count}")

print(f"\nTotal parameters: {total_params}")

total_params_backbone = 0
print("Backbone parameters:")
for name, param in backbone.named_parameters():
    param_count = param.numel()  # 计算参数总数
    total_params_backbone += param_count
    # print(f"Name: {name}, Shape: {param.shape}, Parameters: {param_count}")

print(f"\nTotal parameters in backbone: {total_params_backbone}")