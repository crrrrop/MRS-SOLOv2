#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-11-21 09:13:23
#   Description : paddle2.0_solov2
#
# ================================================================



class SOLOv2_r50_fpn_8gpu_3x_Config(object):
    def __init__(self):
        # 自定义数据集
        self.train_path = 'dataset/line/annotations/instances_line_train.json'
        self.val_path = 'dataset/line/annotations/instance_line_val.json'
        self.classes_path = 'data/pod_classes.txt'
        self.train_pre_path = 'dataset/line/train/'   # 训练集图片相对路径
        self.val_pre_path = 'dataset/line/val/'     # 验证集图片相对路径
        self.num_classes = 6                              # 数据集类别数

        # # AIStudio下的COCO数据集
        # self.train_path = '../data/data7122/annotations/instances_train2017.json'
        # self.val_path = '../data/data7122/annotations/instances_val2017.json'
        # self.classes_path = 'data/coco_classes.txt'
        # self.train_pre_path = '../data/data7122/train2017/'  # 训练集图片相对路径
        # self.val_pre_path = '../data/data7122/val2017/'      # 验证集图片相对路径
        # self.test_path = '../data/data7122/annotations/image_info_test-dev2017.json'      # test集
        # self.test_pre_path = '../data/data7122/test2017/'    # test集图片相对路径
        # self.num_classes = 80                                # 数据集类别数

        # Windows下的COCO数据集
        # self.train_path = '../COCO/annotations/instances_train2017.json'
        # self.val_path = '../COCO/annotations/instances_val2017.json'
        # self.classes_path = 'data/coco_classes.txt'
        # self.train_pre_path = '../COCO/train2017/'  # 训练集图片相对路径
        # self.val_pre_path = '../COCO/val2017/'      # 验证集图片相对路径
        # self.test_path = '../COCO/annotations/image_info_test-dev2017.json'      # test集
        # self.test_pre_path = '../COCO/test2017/'    # test集图片相对路径
        # self.num_classes = 80                       # 数据集类别数


        # ========= 一些设置 =========
        self.train_cfg = dict(
            batch_size=2,
            num_threads=5,   # 读数据的线程数
            max_batch=2,     # 最大读多少个批
            model_path= None,
            # model_path='./weights/1000.pdparams',
            save_iter=1750,   # 每隔几步保存一次模型 10个epoch
            eval_iter=5250,   # 每隔几步计算一次eval集的mAP 30个epoch
            max_iters=23800,   # 训练多少步  改了 136个epoch
            mixup_epoch=36,     # 前几轮进行mixup
            cutmix_epoch=-1,    # 前几轮进行cutmix
        )
        self.learningRate = dict(
            base_lr=0.01,  # 基础学习率
            PiecewiseDecay=dict(  # 分段衰减配置
                gamma=0.01,  # 学习率衰减系数，每次衰减时将学习率乘以这个系数
                milestones=[60000, 240000],  # 学习率衰减的步数，达到这些步数时进行衰减
            ),
            LinearWarmup=dict(  # 线性预热配置
                start_factor=0.0001,  # 预热的起始因子，即预热开始时的学习率为 base_lr * start_factor
                steps=1500,  # 预热的步数，在这段步数内线性增加学习率  改了
            ),
        )

        self.optimizerBuilder = dict(
            optimizer=dict(  # 优化器配置
                momentum=0.9,  # 动量因子，用于动量优化器
                type='Momentum',  # 优化器类型，这里使用的是动量优化器
            ),
            regularizer=dict(  # 正则化配置
                factor=0.0001,  # 正则化因子
                type='L2',  # 正则化类型，这里使用的是L2正则化
            ),
        )


        # 验证。用于train.py、eval.py、test_dev.py
        self.eval_cfg = dict(
            model_path='weights_vd+MSCA×4/best_model.pdparams',
            # model_path='./weights/1000.pdparams',
            target_size=800,
            draw_image=False,    # 是否画出验证集图片
            draw_thresh=0.15,    # 如果draw_image==True，那么只画出分数超过draw_thresh的物体的预测框。
            eval_batch_size=1,   # 验证时的批大小。
        )

        # 测试。用于demo.py
        self.test_cfg = dict(
            model_path='weights_vd/best_model.pdparams',   # dygraph_solov2_r50_fpn_8gpu_3x.pdparams
            # model_path='./weights/1000.pdparams',
            target_size=500,
            # target_size=320,  # 320的话很多豆荚都识别不到，再研究下这个参数的作用！！！
            draw_image=True,
            draw_thresh=0.15,   # 如果draw_image==True，那么只画出分数超过draw_thresh的物体的预测框。
        )


        # ============= 模型相关 =============
        # self.use_ema = True  # 使用EMA (指数移动平均)，暂时被注释掉了
        self.use_ema = False  # 不使用EMA
        self.ema_decay = 0.9998  # EMA衰减率
        self.backbone_type = 'Resnet50Vd'  # 使用Resnet50Vb作为骨干网络
        self.backbone = dict(  # 骨干网络的配置参数
            norm_type='bn',  # 归一化类型：批归一化
            feature_maps=[2, 3, 4, 5],  # 提取的特征图层级
            dcn_v2_stages=[],  # 使用可变形卷积的阶段
            downsample_in3x3=True,  # 是否在3x3卷积层进行下采样
            freeze_at=2,  # 冻结层数
            freeze_norm=False,  # 是否冻结归一化层
            norm_decay=0.,  # 归一化层的衰减率
        )
        self.fpn_type = 'FPN'  # 使用FPN（特征金字塔网络）
        self.fpn = dict(  # FPN的配置参数
            in_channels=[2048, 1024, 512, 256],  # 输入通道数
            num_chan=256,  # 输出通道数
            min_level=2,  # 最小层级
            max_level=6,  # 最大层级
            spatial_scale=[0.03125, 0.0625, 0.125, 0.25],  # 空间尺度
            has_extra_convs=False,  # 是否有额外的卷积层
            use_c5=False,  # 是否使用C5层
            reverse_out=True,  # 是否反转输出
        )
        self.mask_feat_head_type = 'MaskFeatHead'  # 使用Mask特征头
        self.mask_feat_head = dict(  # Mask特征头的配置参数
            in_channels=256,  # 输入通道数
            out_channels=128,  # 输出通道数
            norm_type='gn',  # 归一化类型：群归一化
            start_level=0,  # 开始层级
            end_level=3,  # 结束层级
            num_classes=256,  # 类别数量
        )
        self.head_type = 'SOLOv2Head'  # 使用SOLOv2头
        self.head = dict(  # SOLOv2头的配置参数
            num_classes=self.num_classes + 1,  # 类别数量+1（包括背景）
            in_channels=256,  # 输入通道数
            norm_type='gn',  # 归一化类型：群归一化
            num_convs=4,  # 卷积层数量
            seg_feat_channels=512,  # 分割特征通道数
            strides=[8, 8, 16, 32, 32],  # 步幅
            sigma=0.2,  # Sigma值
            kernel_out_channels=256,  # 卷积核输出通道数
            num_grids=[40, 36, 24, 16, 12],  # 网格数量
        )
        self.solo_loss_type = 'SOLOv2Loss'  # 使用SOLOv2损失函数
        self.solo_loss = dict(  # SOLOv2损失函数的配置参数
            ins_loss_weight=3.0,  # 实例损失权重
            focal_loss_gamma=2.0,  # Focal损失的Gamma值
            focal_loss_alpha=0.25,  # Focal损失的Alpha值
        )
        self.nms_cfg = dict(  # 非极大值抑制的配置参数
            score_thr=0.1,  # 得分阈值
            update_thr=0.05,  # 更新阈值
            mask_thr=0.5,  # Mask阈值
            nms_pre=500,  # NMS前处理的数量
            max_per_img=100,  # 每张图像的最大数量
            kernel="gaussian",  # 使用高斯核
            sigma=2.,  # 高斯核的Sigma值
        )


        # ============= 预处理相关 =============
        self.context = {'fields': ['image', 'im_id', 'gt_segm']}  # 上下文信息，包括图像、图像ID和Ground Truth分割
        # DecodeImage
        self.decodeImage = dict(  # 图像解码的配置参数
            to_rgb=True,  # 转换为RGB格式
            with_mixup=False,  # 是否使用Mixup增强
            with_cutmix=False,  # 是否使用CutMix增强
        )
        # Poly2Mask
        self.poly2Mask = dict(  # 多边形转掩码的配置参数
        )
        # ResizeImage
        self.resizeImage = dict(  # 图像缩放的配置参数
            target_size=[640, 672, 704, 736, 768, 800],  # 目标尺寸
            max_size=1333,  # 最大尺寸
            interp=1,  # 插值方法
            use_cv2=True,  # 是否使用OpenCV
            resize_box=True,  # 是否缩放边框
        )
        # RandomFlipImage
        self.randomFlipImage = dict(  # 随机翻转的配置参数
            prob=0.5,  # 翻转概率
        )
        # NormalizeImage
        self.normalizeImage = dict(  # 图像归一化的配置参数
            mean=[123.675, 116.28, 103.53],  # 均值
            std=[58.395, 57.12, 57.375],  # 标准差
            is_scale=False,  # 是否缩放
            is_channel_first=False,  # 是否将通道放在第一维
        )
        # Permute
        self.permute = dict(  # 通道变换的配置参数
            to_bgr=False,  # 是否转换为BGR格式
            channel_first=True,  # 是否将通道放在第一维
        )
        # PadBatch
        self.padBatch = dict(  # 批次填充的配置参数
            pad_to_stride=32,  # 填充到的步幅
        )
        # Gt2Solov2Target
        self.gt2Solov2Target = dict(  # 目标生成的配置参数
            num_grids=[40, 36, 24, 16, 12],  # 网格数量
            scale_ranges=[[1, 96], [48, 192], [96, 384], [192, 768], [384, 2048]],  # 尺度范围
            coord_sigma=0.2,  # 坐标Sigma值
        )

        # 预处理顺序。增加一些数据增强时这里也要加上，否则train.py中相当于没加！
        self.sample_transforms_seq = []  # 初始化样本变换序列
        self.sample_transforms_seq.append('decodeImage')  # 添加图像解码
        self.sample_transforms_seq.append('poly2Mask')  # 添加多边形转掩码
        self.sample_transforms_seq.append('resizeImage')  # 添加图像缩放
        self.sample_transforms_seq.append('randomFlipImage')  # 添加随机翻转
        self.sample_transforms_seq.append('normalizeImage')  # 添加图像归一化
        self.sample_transforms_seq.append('permute')  # 添加通道变换
        self.batch_transforms_seq = []  # 初始化批次变换序列
        self.batch_transforms_seq.append('padBatch')  # 添加批次填充
        self.batch_transforms_seq.append('gt2Solov2Target')  # 添加目标生成


