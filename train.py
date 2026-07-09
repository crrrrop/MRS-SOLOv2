#! /usr/bin/env python  # 指定脚本的解释器路径
# coding=utf-8  # 指定文件的编码格式为UTF-8
# ================================================================
#
#   Author      : miemie2013  # 作者信息
#   Created date: 2020-11-21 09:13:23  # 创建日期
#   Description : paddle2.0_solov2  # 描述信息
#
# ================================================================

from collections import deque  # 导入双端队列数据结构
import time  # 导入时间处理模块
import threading  # 导入线程模块
import datetime  # 导入日期时间处理模块
from collections import OrderedDict  # 导入有序字典数据结构
import os  # 导入操作系统接口模块
import json  # 导入JSON处理模块
import argparse  # 导入命令行参数解析模块
import textwrap  # 导入文本换行模块

from config import *  # 从config模块导入所有内容
from model.EMA import ExponentialMovingAverage  # 从model.EMA模块导入ExponentialMovingAverage类

from model.solo import *  # 从model.solo模块导入所有内容
from tools.argparser import ArgParser  # 从tools.argparser模块导入ArgParser类
from tools.cocotools import get_classes, catid2clsid, clsid2catid  # 从tools.cocotools模块导入get_classes, catid2clsid, clsid2catid函数
from model.decode_np import Decode  # 从model.decode_np模块导入Decode类
from tools.cocotools import eval  # 从tools.cocotools模块导入eval函数
from tools.data_process import data_clean, get_samples  # 从tools.data_process模块导入data_clean和get_samples函数
from tools.transform import *  # 从tools.transform模块导入所有内容
from pycocotools.coco import COCO  # 从pycocotools.coco模块导入COCO类

import logging  # 导入日志记录模块

FORMAT = '%(asctime)s-%(levelname)s: %(message)s'  # 设置日志格式
logging.basicConfig(level=logging.INFO, format=FORMAT)  # 配置基本的日志记录设置
logger = logging.getLogger(__name__)  # 获取日志记录器实例

def multi_thread_op(i, num_threads, batch_size, samples, context, with_mixup, sample_transforms):  # 定义一个多线程操作函数
    for k in range(i, batch_size, num_threads):  # 遍历当前线程负责的样本索引
        for sample_transform in sample_transforms:  # 遍历样本转换步骤
            if isinstance(sample_transform, MixupImage):  # 判断当前转换步骤是否为混合增强
                if with_mixup:  # 如果启用了混合增强
                    samples[k] = sample_transform(samples[k], context)  # 应用混合增强转换
            else:
                samples[k] = sample_transform(samples[k], context)  # 应用其他转换步骤



def read_train_data(cfg,
                    train_indexes,
                    train_steps,
                    train_records,
                    batch_size,
                    _iter_id,
                    train_dic,
                    use_gpu,
                    context, with_mixup, sample_transforms, batch_transforms):  # 定义读取训练数据的函数
    iter_id = _iter_id  # 初始化迭代器ID
    num_threads = cfg.train_cfg['num_threads']  # 获取配置中的线程数
    while True:  # 无限个epoch
        # 每个epoch之前洗乱
        np.random.shuffle(train_indexes)  # 打乱训练数据索引
        for step in range(train_steps):  # 遍历训练步骤
            iter_id += 1  # 增加迭代器ID

            key_list = list(train_dic.keys())  # 获取训练字典的键列表
            key_len = len(key_list)  # 获取键的数量
            while key_len >= cfg.train_cfg['max_batch']:  # 如果键的数量大于等于最大批次
                time.sleep(0.01)  # 休眠0.01秒
                key_list = list(train_dic.keys())  # 更新键列表
                key_len = len(key_list)  # 更新键的数量

            # ==================== train ====================
            n_layers = len(cfg.gt2Solov2Target['num_grids'])  # 获取配置中的网格层数
            images = [None] * batch_size  # 初始化图像列表
            fg_nums = [None] * batch_size  # 初始化前景数量列表
            ins_labels = [None] * n_layers  # 初始化实例标签列表
            cate_labels = [None] * n_layers  # 初始化类别标签列表
            grid_orders = [None] * n_layers  # 初始化网格顺序列表
            for idx in range(n_layers):
                ins_labels[idx] = [None] * batch_size  # 初始化每层实例标签
                cate_labels[idx] = [None] * batch_size  # 初始化每层类别标签
                grid_orders[idx] = [None] * batch_size  # 初始化每层网格顺序

            samples = get_samples(train_records, train_indexes, step, batch_size, with_mixup)  # 获取样本数据
            # sample_transforms用多线程
            threads = []
            for i in range(num_threads):
                t = threading.Thread(target=multi_thread_op, args=(i, num_threads, batch_size, samples, context, with_mixup, sample_transforms))  # 创建多线程处理样本转换
                threads.append(t)  # 添加线程到线程列表
                t.start()  # 启动线程
            # 等待所有线程任务结束。
            for t in threads:
                t.join()  # 等待线程结束

            # batch_transforms
            for batch_transform in batch_transforms:
                samples = batch_transform(samples, context)  # 应用批次转换

            # 整理成ndarray
            for k in range(batch_size):
                images[k] = np.expand_dims(samples[k]['image'].astype(np.float32), 0)  # 获取图像并扩展维度
                fg_nums[k] = np.expand_dims(samples[k]['fg_num'].astype(np.int32), 0)  # 获取前景数量并扩展维度
                for idx in range(n_layers):
                    ins_labels[idx][k] = samples[k]['ins_label%d' % idx].astype(np.int32)  # 获取实例标签
                    cate_labels[idx][k] = samples[k]['cate_label%d' % idx].astype(np.int32)  # 获取类别标签
                    grid_orders[idx][k] = np.reshape(samples[k]['grid_order%d' % idx].astype(np.int32), (-1, ))  # 获取网格顺序并重塑形状

            # lod信息
            # lods = [None] * n_layers
            # for idx in range(n_layers):
            #     lod = [0]
            #     for k in range(batch_size):
            #         l = len(grid_orders[idx][k])
            #         lod.append(l)
            #     lods[idx] = lod
            # lods = np.array(lods).astype(np.int32)
            # lods = np.cumsum(lods, axis=1)

            images = np.concatenate(images, 0)  # 拼接图像数组
            fg_nums = np.concatenate(fg_nums, 0)  # 拼接前景数量数组
            for idx in range(n_layers):
                ins_labels[idx] = np.concatenate(ins_labels[idx], 0)  # 拼接实例标签数组
                cate_labels[idx] = np.concatenate(cate_labels[idx], 0)  # 拼接类别标签数组
                grid_orders[idx] = np.concatenate(grid_orders[idx], 0)  # 拼接网格顺序数组

            images = paddle.to_tensor(images, place=place)  # 转换图像为Paddle张量
            fg_nums = paddle.to_tensor(fg_nums, place=place)  # 转换前景数量为Paddle张量
            for idx in range(n_layers):
                ins_labels[idx] = paddle.to_tensor(ins_labels[idx], place=place)  # 转换实例标签为Paddle张量
                cate_labels[idx] = paddle.to_tensor(cate_labels[idx], place=place)  # 转换类别标签为Paddle张量
                grid_orders[idx] = paddle.to_tensor(grid_orders[idx], place=place)  # 转换网格顺序为Paddle张量

            dic = {}
            dic['images'] = images  # 存储图像张量到字典
            dic['fg_nums'] = fg_nums  # 存储前景数量张量到字典
            for idx in range(n_layers):
                dic['ins_label%d' % idx] = ins_labels[idx]  # 存储实例标签张量到字典
                dic['cate_label%d' % idx] = cate_labels[idx]  # 存储类别标签张量到字典
                dic['grid_order%d' % idx] = grid_orders[idx]  # 存储网格顺序张量到字典
            train_dic['%.8d' % iter_id] = dic  # 将字典存储到训练字典中

            # ==================== exit ====================
            if iter_id == cfg.train_cfg['max_iters']:  # 如果达到最大迭代次数
                return 0  # 退出函数


def load_weights(model, model_path):  # 定义加载模型权重的函数
    _state_dict = model.state_dict()  # 获取模型的当前状态字典
    pretrained_dict = paddle.load(model_path)  # 加载预训练模型权重
    new_state_dict = OrderedDict()  # 初始化新的状态字典
    for k, v in pretrained_dict.items():  # 遍历预训练模型的权重项
        if k in _state_dict:  # 如果当前项在模型状态字典中
            shape_1 = _state_dict[k].shape  # 获取当前模型的权重形状
            shape_2 = pretrained_dict[k].shape  # 获取预训练模型的权重形状
            shape_2 = list(shape_2)  # 将预训练模型的权重形状转为列表
            if shape_1 == shape_2:  # 如果两个形状匹配
                new_state_dict[k] = v  # 将预训练权重赋值给新状态字典
            else:
                print('shape mismatch in %s. shape_1=%s, while shape_2=%s.' % (k, shape_1, shape_2))  # 打印形状不匹配的项
    _state_dict.update(new_state_dict)  # 更新模型状态字典
    model.set_state_dict(_state_dict)  # 设置模型状态字典

def clear_model(save_dir):  # 定义清理模型的函数
    path_dir = os.listdir(save_dir)  # 获取保存目录下的文件列表
    it_ids = []  # 初始化迭代器ID列表
    for name in path_dir:  # 遍历文件列表
        sss = name.split('.')  # 分割文件名
        if sss[0] == '':  # 如果文件名为空
            continue  # 跳过
        if sss[0] == 'best_model':  # 如果文件名是最优模型
            it_id = 9999999999  # 设置迭代器ID为9999999999
        else:
            it_id = int(sss[0])  # 将文件名转换为迭代器ID
        it_ids.append(it_id)  # 添加迭代器ID到列表
    if len(it_ids) >= 11 * 1:  # 如果迭代器ID数量大于等于11
        it_id = min(it_ids)  # 获取最小的迭代器ID
        pdparams_path = '%s/%d.pdparams' % (save_dir, it_id)  # 构建参数文件路径
        if os.path.exists(pdparams_path):  # 如果参数文件存在
            os.remove(pdparams_path)  # 删除参数文件

def calc_lr(iter_id, cfg):  # 定义计算学习率的函数
    base_lr = cfg.learningRate['base_lr']  # 获取基础学习率
    piecewiseDecay = cfg.learningRate['PiecewiseDecay']  # 获取分段衰减配置
    linearWarmup = cfg.learningRate['LinearWarmup']  # 获取线性预热配置
    gamma = piecewiseDecay['gamma']  # 获取分段衰减的gamma值
    milestones = piecewiseDecay['milestones']  # 获取里程碑列表
    start_factor = linearWarmup['start_factor']  # 获取线性预热的起始因子
    steps = linearWarmup['steps']  # 获取线性预热的步数
    n = len(milestones)  # 获取里程碑数量
    for i in range(n, 0, -1):  # 倒序遍历里程碑
        if iter_id >= milestones[i-1]:  # 如果当前迭代次数大于等于当前里程碑
            return base_lr * gamma ** i  # 返回衰减后的学习率
    if iter_id <= steps:  # 如果当前迭代次数小于等于线性预热步数
        k = (1.0 - start_factor) / steps  # 计算预热因子
        factor = start_factor + k * iter_id  # 计算当前预热因子
        return base_lr * factor  # 返回预热后的学习率
    return base_lr  # 返回基础学习率

def configure_logging(iter_id):
    # 配置日志文件路径
    if iter_id % 3000 == 0:
        log_filename = f'train_{iter_id}.log'
        logging.basicConfig(level=logging.INFO, format=FORMAT, filename=log_filename, filemode='w')
        logger.info('Logging configured to save to %s', log_filename)

if __name__ == '__main__':
    print("|__________________train_______________________________")
    parser = ArgParser()  # 创建参数解析器对象
    use_gpu = parser.get_use_gpu()  # 获取是否使用GPU的配置
    cfg = parser.get_cfg()  # 获取配置文件对象
    print(paddle.__version__)  # 打印PaddlePaddle版本号
    paddle.disable_static()   # 开启动态图模式
    gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))  # 获取GPU设备ID
    place = paddle.CUDAPlace(gpu_id) if use_gpu else paddle.CPUPlace()  # 根据是否使用GPU选择运行位置

    # 打印，确认一下使用的配置
    print('\n=============== config message ===============')
    print('config file: %s' % str(type(cfg)))  # 打印配置文件的类型
    if cfg.train_cfg['model_path'] is not None:
        print('pretrained_model: %s' % cfg.train_cfg['model_path'])  # 如果有预训练模型路径，打印之
    else:
        print('pretrained_model: None')  # 否则打印无预训练模型
    print('use_gpu: %s' % str(use_gpu))  # 打印是否使用GPU
    print()

    # 种类id初始化
    _catid2clsid = {}
    _clsid2catid = {}
    _clsid2cname = {}
    with open(cfg.val_path, 'r', encoding='utf-8') as f2:  # 打开验证集路径
        dataset_text = ''
        for line in f2:
            line = line.strip()
            dataset_text += line
        eval_dataset = json.loads(dataset_text)  # 解析JSON格式的验证集数据
        categories = eval_dataset['categories']  # 获取分类信息
        for clsid, cate_dic in enumerate(categories):  # 遍历分类信息
            catid = cate_dic['id']
            cname = cate_dic['name']
            _catid2clsid[catid] = clsid  # 建立分类id到类别id的映射
            _clsid2catid[clsid] = catid  # 建立类别id到分类id的映射
            _clsid2cname[clsid] = cname  # 建立类别id到类别名称的映射
    class_names = []
    num_classes = len(_clsid2cname.keys())
    for clsid in range(num_classes):
        class_names.append(_clsid2cname[clsid])  # 根据类别id获取类别名称

    # 步id初始化，无需设置，会自动读取
    iter_id = 0

    # 创建模型
    n_layers = len(cfg.gt2Solov2Target['num_grids'])  # 获取网格数量
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
    print("|| 已初始化整体模型")

    _decode = Decode(model, class_names, place, cfg, for_test=False)  # 初始化解码器

    # optimizer初始化
    regularization = None
    if cfg.optimizerBuilder['regularizer'] is not None:
        reg_args = cfg.optimizerBuilder['regularizer'].copy()  # 复制正则化参数
        reg_type = reg_args['type'] + 'Decay'   # 正则化类型。L1、L2
        reg_factor = reg_args['factor']
        Regularization = select_regularization(reg_type)  # 选择正则化类型
        # 在 优化器 中设置正则化。
        # 不可以加正则化的参数：norm层(比如bn层、affine_channel层、gn层)的scale、offset；卷积层的偏移参数。
        # 如果同时在 可训练参数的ParamAttr 和 优化器optimizer 中设置正则化， 那么在 可训练参数的ParamAttr 中设置的优先级会高于在 optimizer 中的设置。
        # 也就是说，等价于没给    norm层(比如bn层、affine_channel层、gn层)的scale、offset；卷积层的偏移参数    加正则化。
        regularization = Regularization(reg_factor)  # 初始化正则化器
    optim_args = cfg.optimizerBuilder['optimizer'].copy()  # 复制优化器参数
    optim_type = optim_args['type']   # 使用哪种优化器。Momentum、Adam、SGD、...之类的。
    Optimizer = select_optimizer(optim_type)  # 选择优化器类型
    del optim_args['type']
    optimizer = Optimizer(learning_rate=cfg.learningRate['base_lr'],
                          parameters=model.parameters(),
                          weight_decay=regularization,   # 正则化
                          grad_clip=None,   # 梯度裁剪
                          **optim_args)  # 初始化优化器

    # 加载权重
    if cfg.train_cfg['model_path'] is not None:
        # 加载参数, 跳过形状不匹配的。
        load_weights(model, cfg.train_cfg['model_path'])  # 加载模型权重

        strs = cfg.train_cfg['model_path'].split('weights/')
        if len(strs) == 2:
            iter_id = int(strs[1].split('.')[0])  # 获取当前迭代ID

    # 冻结骨干网络，减少显存需求
    backbone.freeze()

    ema = None
    if cfg.use_ema:
        ema = ExponentialMovingAverage(model, cfg.ema_decay)  # 初始化指数移动平均
        ema.register()

    # 训练集
    train_dataset = COCO(cfg.train_path)  # 加载训练集
    print("||已加载训练集")
    train_img_ids = train_dataset.getImgIds()  # 获取训练集图片ID
    train_records = data_clean(train_dataset, train_img_ids, _catid2clsid, cfg.train_pre_path)  # 数据清洗
    num_train = len(train_records)
    train_indexes = [i for i in range(num_train)]
    print("||已加载训练集索引")
    # 验证集
    val_dataset = COCO(cfg.val_path)  # 加载验证集
    print("||已加载验证集")
    val_img_ids = val_dataset.getImgIds()  # 获取验证集图片ID
    val_images = []   # 只跑有gt的图片，跟随PaddleDetection
    for img_id in val_img_ids:
        ins_anno_ids = val_dataset.getAnnIds(imgIds=img_id, iscrowd=False)   # 读取这张图片所有标注anno的id
        if len(ins_anno_ids) == 0:
            continue
        img_anno = val_dataset.loadImgs(img_id)[0]
        val_images.append(img_anno)

    batch_size = cfg.train_cfg['batch_size']  # 获取批量大小
    with_mixup = cfg.decodeImage['with_mixup']  # 是否使用mixup
    with_cutmix = cfg.decodeImage['with_cutmix']  # 是否使用cutmix
    mixup_epoch = cfg.train_cfg['mixup_epoch']  # mixup的epoch数
    cutmix_epoch = cfg.train_cfg['cutmix_epoch']  # cutmix的epoch数
    context = cfg.context  # 上下文环境
    # 预处理
    # sample_transforms
    sample_transforms = []
    for preprocess_name in cfg.sample_transforms_seq:
        if preprocess_name == 'decodeImage':
            preprocess = DecodeImage(**cfg.decodeImage)   # 对图片解码。最开始的一步。
        elif preprocess_name == 'poly2Mask':
            preprocess = Poly2Mask(**cfg.poly2Mask)         # 多边形变掩码
        elif preprocess_name == 'colorDistort':
            preprocess = ColorDistort(**cfg.colorDistort)  # 颜色扰动
        elif preprocess_name == 'randomCrop':
            preprocess = RandomCrop(**cfg.randomCrop)        # 随机裁剪
        elif preprocess_name == 'resizeImage':
            preprocess = ResizeImage(**cfg.resizeImage)        # 多尺度训练
        elif preprocess_name == 'randomFlipImage':
            preprocess = RandomFlipImage(**cfg.randomFlipImage)  # 随机翻转
        elif preprocess_name == 'normalizeImage':
            preprocess = NormalizeImage(**cfg.normalizeImage)     # 图片归一化。
        elif preprocess_name == 'permute':
            preprocess = Permute(**cfg.permute)    # 图片从HWC格式变成CHW格式
        sample_transforms.append(preprocess)
    print("||已完成预处理")
    # batch_transforms
    batch_transforms = []
    for preprocess_name in cfg.batch_transforms_seq:
        if preprocess_name == 'padBatch':
            preprocess = PadBatch(**cfg.padBatch)   # 填充黑边。使这一批图片有相同的大小。
        elif preprocess_name == 'gt2Solov2Target':
            preprocess = Gt2Solov2Target(**cfg.gt2Solov2Target)   # 填写target张量。
        batch_transforms.append(preprocess) # 添加变换到列表

    # 打印预处理变换
    print('\n=============== sample_transforms ===============')
    for trf in sample_transforms:
        print('%s' % str(type(trf)))    # 打印样本变换类型
    print('\n=============== batch_transforms ===============')
    for trf in batch_transforms:
        print('%s' % str(type(trf)))    # 打印样本变换类型

    # 保存模型的目录
    if not os.path.exists('./weights'): os.mkdir('./weights')
    print("||已保存模型的目录")

    time_stat = deque(maxlen=20)    # 初始化时间统计队列，最多保存10个时间点
    start_time = time.time()    # 获取起始时间
    end_time = time.time()  # 获取结束时间

    # 一轮的步数。丢弃最后几个样本。
    train_steps = num_train // batch_size   # 计算每轮的训练步数
    mixup_steps = mixup_epoch * train_steps # 计算混合增强的步数
    cutmix_steps = cutmix_epoch * train_steps   # 计算裁剪增强的步数

    # 打印混合增强和裁剪增强信息
    print('\n=============== mixup and cutmix ===============')
    print('steps_per_epoch: %d' % train_steps)  # 打印每轮步数
    if with_mixup:
        print('mixup_steps: %d' % mixup_steps)  # 打印混合增强步数
    else:
        print('don\'t use mixup.')  # 不使用混合增强
    if with_cutmix:
        print('cutmix_steps: %d' % cutmix_steps)    # 打印裁剪增强步数
    else:
        print('don\'t use cutmix.') # 不使用裁剪增强

    # 读数据的线程
    train_dic ={}   # 初始化训练数据字典
    thr = threading.Thread(target=read_train_data,  # 创建数据读取线程
                           args=(cfg,
                                 train_indexes,
                                 train_steps,
                                 train_records,
                                 batch_size,
                                 iter_id,
                                 train_dic,
                                 use_gpu,
                                 context, with_mixup, sample_transforms, batch_transforms))
    thr.start() # 启动数据读取线程


    best_ap_list = [0.0, 0]  #[map, iter]   # 初始化最佳AP列表 [map, iter]
    train_speed_count = 0   # 初始化训练速度计数器
    train_speed_start = 0.0 # 初始化训练速度起始时间
    while True:   # 无限个epoch
        for step in range(train_steps): # 遍历训练步骤
            iter_id += 1    # 增加迭代ID

            configure_logging(iter_id)  # 配置日志记录

            key_list = list(train_dic.keys())   # 获取训练字典的键列表
            key_len = len(key_list) # 获取键的数量
            while key_len == 0: # 如果键的数量为0
                time.sleep(0.01)    # 休眠0.01秒
                key_list = list(train_dic.keys())   # 更新键列表
                key_len = len(key_list) # 更新键的数量
            dic = train_dic.pop('%.8d'%iter_id) # 弹出当前迭代ID的数据

            # 估计剩余时间
            start_time = end_time   # 更新起始时间
            end_time = time.time()  # 获取当前时间
            time_stat.append(end_time - start_time) # 将时间差添加到时间统计队列
            time_cost = np.mean(time_stat)  # 计算平均时间差
            eta_sec = (cfg.train_cfg['max_iters'] - iter_id) * time_cost    # 估计剩余时间
            eta = str(datetime.timedelta(seconds=int(eta_sec))) # 将剩余时间格式化为字符串

            # ==================== train ====================
            images = dic['images']  # 获取图像数据
            fg_nums = dic['fg_nums']    # 获取前景数量
            ins_labels = [None] * n_layers  # 初始化实例标签
            cate_labels = [None] * n_layers # 初始化类别标签
            grid_orders = [None] * n_layers # 初始化网格顺序

            for idx in range(n_layers):     # 遍历网格层数
                ins_labels[idx] = dic['ins_label%d'%idx]    # 获取实例标签
                cate_labels[idx] = dic['cate_label%d'%idx]  # 获取类别标签
                grid_orders[idx] = dic['grid_order%d'%idx]  # 获取网格顺序

            losses = model.train_model(images, ins_labels, cate_labels, grid_orders, fg_nums)   # 训练模型并获取损失
            all_loss = 0.0  # 初始化总损失
            loss_names = {} # 初始化损失名称字典
            for loss_name in losses.keys(): # 遍历损失名称
                sub_loss = losses[loss_name]    # 获取子损失
                all_loss += sub_loss    # 累加总损失
                loss_names[loss_name] = sub_loss.numpy()[0] # 存储子损失值
            _all_loss = all_loss.numpy()[0] # 获取总损失值

            # 更新权重
            lr = calc_lr(iter_id, cfg)  # 计算当前学习率
            optimizer.set_lr(lr)  # 设置学习率
            all_loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新模型参数
            optimizer.clear_grad()  # 清除梯度
            if cfg.use_ema:  # 如果使用EMA
                ema.update()  # 更新EMA字典

            # ==================== log ====================
            if iter_id % 50 == 0:  # 每20次迭代记录一次日志
                lr = optimizer.get_lr()  # 获取当前学习率
                each_loss = ''  # 初始化损失字符串
                for loss_name in loss_names.keys():  # 遍历损失名称
                    loss_value = loss_names[loss_name]  # 获取损失值
                    each_loss += ' %s: %.3f,' % (loss_name, loss_value)  # 添加到损失字符串
                strs = 'Train iter: {}, lr: {:.9f}, all_loss: {:.3f},{} eta: {}'.format(iter_id, lr, _all_loss, each_loss, eta)  # 格式化日志字符串
                logger.info(strs)  # 记录日志

            # ==================== train_speed ====================
            mod_iter_id = iter_id % 1000  # 获取迭代ID的模
            step_iter = 200  # 每隔200步计算一次训练速度
            if mod_iter_id >= 20:  # 前20步热身
                if mod_iter_id == 20:
                    train_speed_count = 0  # 重置训练速度计数器
                    train_speed_start = time.time()  # 重置训练速度起始时间
                elif mod_iter_id > 825:
                    pass  # 跳过
                else:
                    train_speed_count += 1  # 增加训练速度计数器
                    if train_speed_count % step_iter == 0:  # 每隔200步计算一次训练速度
                        sts = train_speed_count // step_iter  # 计算训练步数
                        sts *= step_iter
                        cost = time.time() - train_speed_start  # 计算耗时
                        logger.info('Train Speed: %.3f steps per second.' % ((sts / cost), ))  # 记录训练速度

            # ==================== save ====================
            if iter_id % cfg.train_cfg['save_iter'] == 0:  # 每隔指定步数保存一次模型
                if cfg.use_ema:  # 如果使用EMA
                    ema.apply()  # 应用EMA
                save_path = './weights/%d.pdparams' % iter_id  # 构建保存路径
                paddle.save(model.state_dict(), save_path)  # 保存模型状态字典
                if cfg.use_ema:  # 如果使用EMA
                    ema.restore()  # 恢复EMA
                logger.info('Save model to {}'.format(save_path))  # 记录保存日志
                clear_model('weights')  # 清理模型

            # ==================== eval ====================
            if iter_id % cfg.train_cfg['eval_iter'] == 0:  # 每隔指定步数进行一次评估
                if cfg.use_ema:  # 如果使用EMA
                    ema.apply()  # 应用EMA
                model.eval()  # 切换到评估模式
                head.set_dropblock(is_test=True)  # 设置DropBlock为测试模式
                box_ap, mask_ap = eval(_decode, val_images, cfg.val_pre_path, cfg.val_path, cfg.eval_cfg['eval_batch_size'], _clsid2catid, cfg.eval_cfg['draw_image'], cfg.eval_cfg['draw_thresh'])  # 进行评估
                logger.info("box ap: %.3f" % (box_ap[0], ))  # 记录box AP
                model.train()  # 切换到训练模式
                head.set_dropblock(is_test=False)  # 设置DropBlock为训练模式

                # 以mask_ap作为标准
                ap = mask_ap  # 使用mask AP
                if ap[0] > best_ap_list[0]:  # 如果当前AP大于最佳AP
                    best_ap_list[0] = ap[0]  # 更新最佳AP
                    best_ap_list[1] = iter_id  # 更新最佳迭代ID
                    save_path = './weights/best_model.pdparams'  # 构建保存路径
                    paddle.save(model.state_dict(), save_path)  # 保存模型状态字典
                    logger.info('Save model to {}'.format(save_path))  # 记录保存日志
                    clear_model('weights')  # 清理模型
                if cfg.use_ema:  # 如果使用EMA
                    ema.restore()  # 恢复EMA
                logger.info("Best test ap: {}, in iter: {}".format(best_ap_list[0], best_ap_list[1]))  # 记录最佳AP和迭代ID

            # ==================== exit ====================
            if iter_id == cfg.train_cfg['max_iters']:  # 如果达到最大迭代次数
                logger.info('Done.')  # 记录完成日志
                exit(0)  # 退出程序

