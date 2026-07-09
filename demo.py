#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-11-21 09:13:23
#   Description : paddle2.0_solov2
#
# ================================================================
from collections import deque  # 导入deque用于队列操作
import datetime  # 导入datetime模块处理日期和时间
import cv2  # 导入OpenCV模块进行图像处理
import os  # 导入os模块处理文件和目录
import time  # 导入time模块处理时间
import threading  # 导入threading模块处理多线程
import argparse  # 导入argparse模块处理命令行参数
import textwrap  # 导入textwrap模块处理文本换行
import paddle  # 导入PaddlePaddle深度学习框架

from config import *  # 从config模块导入所有配置
from model.decode_np import Decode  # 从model.decode_np模块导入Decode类
from model.solo import *  # 从model.solo模块导入所有内容
from tools.argparser import ArgParser  # 从tools.argparser模块导入ArgParser类
from tools.cocotools import get_classes  # 从tools.cocotools模块导入get_classes函数

import logging  # 导入logging模块处理日志
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'  # 日志格式
logging.basicConfig(level=logging.INFO, format=FORMAT)  # 配置日志基本设置
logger = logging.getLogger(__name__)  # 获取日志记录器


def read_test_data(path_dir,
                   _decode,
                   test_dic):  # 定义读取测试数据的函数
    for k, filename in enumerate(path_dir):  # 遍历目录中的文件
        key_list = list(test_dic.keys())  # 获取测试字典的键列表
        key_len = len(key_list)  # 获取键的数量
        while key_len >= 3:  # 如果键的数量大于等于3
            time.sleep(0.01)  # 休眠0.01秒
            key_list = list(test_dic.keys())  # 更新键列表
            key_len = len(key_list)  # 更新键的数量

        image = cv2.imread('images/test/' + filename)  # 读取测试图像   images/test/
        pimage, ori_shape, resize_shape = _decode.process_image(np.copy(image))  # 处理图像
        dic = {}  # 初始化字典
        dic['image'] = image  # 存储原始图像到字典
        dic['pimage'] = pimage  # 存储处理后的图像到字典
        dic['ori_shape'] = ori_shape  # 存储原始图像形状到字典
        dic['resize_shape'] = resize_shape  # 存储调整后的图像形状到字典
        test_dic['%.8d' % k] = dic  # 将字典存储到测试字典中

def save_img(filename, image):  # 定义保存图像的函数
    cv2.imwrite('images/res/' + filename, image)  # 将图像写入文件

if __name__ == '__main__':
    parser = ArgParser()  # 创建命令行参数解析器
    use_gpu = parser.get_use_gpu()  # 获取是否使用GPU
    cfg = parser.get_cfg()  # 获取配置
    print(paddle.__version__)  # 打印PaddlePaddle版本
    paddle.disable_static()   # 开启动态图
    gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))  # 获取GPU ID
    place = paddle.CUDAPlace(gpu_id) if use_gpu else paddle.CPUPlace()  # 设置计算设备

    # 读取的模型
    model_path = cfg.test_cfg['model_path']  # 获取模型路径

    # 是否给图片画框
    draw_image = cfg.test_cfg['draw_image']  # 获取是否绘制图像
    draw_thresh = cfg.test_cfg['draw_thresh']  # 获取绘制阈值

    # 打印，确认一下使用的配置
    print('\n=============== config message ===============')
    print('config file: %s' % str(type(cfg)))  # 打印配置文件类型
    print('model_path: %s' % model_path)  # 打印模型路径
    print('target_size: %d' % cfg.test_cfg['target_size'])  # 打印目标尺寸
    print('use_gpu: %s' % str(use_gpu))  # 打印是否使用GPU
    print()

    class_names = get_classes(cfg.classes_path)  # 获取类别名称
    num_classes = len(class_names)  # 获取类别数量

    # 创建模型
    Backbone = select_backbone(cfg.backbone_type)  # 选择骨干网络类型
    backbone = Backbone(**cfg.backbone)  # 初始化骨干网络
    FPN = select_fpn(cfg.fpn_type)  # 选择FPN类型
    fpn = FPN(**cfg.fpn)  # 初始化FPN
    MaskFeatHead = select_head(cfg.mask_feat_head_type)  # 选择Mask特征头类型
    mask_feat_head = MaskFeatHead(**cfg.mask_feat_head)  # 初始化Mask特征头
    Head = select_head(cfg.head_type)  # 选择头部类型
    head = Head(solo_loss=None, nms_cfg=cfg.nms_cfg, **cfg.head)  # 初始化头部
    model = SOLOv2(backbone, fpn, mask_feat_head, head)  # 创建SOLOv2模型

    param_state_dict = paddle.load(model_path)  # 加载模型参数
    model.set_state_dict(param_state_dict)  # 设置模型参数
    model.eval()  # 必须调用model.eval()来设置dropout和batch normalization layers在运行推理前，切换到评估模式
    head.set_dropblock(is_test=True)  # 设置dropblock为测试模式

    _decode = Decode(model, class_names, place, cfg, for_test=True)  # 初始化解码器

    if not os.path.exists('images/res_pod/'): os.mkdir('images/res/')  # 如果不存在结果目录则创建
    path_dir = os.listdir('images/test_pod_demo/')  # 获取测试图像目录中的文件列表

    # 读数据的线程
    test_dic = {}  # 初始化测试字典
    thr = threading.Thread(target=read_test_data,
                           args=(path_dir,
                                 _decode,
                                 test_dic))  # 创建读取测试数据的线程
    thr.start()  # 启动线程

    key_list = list(test_dic.keys())  # 获取测试字典的键列表
    key_len = len(key_list)  # 获取键的数量
    while key_len == 0:  # 如果键的数量为0
        time.sleep(0.01)  # 休眠0.01秒
        key_list = list(test_dic.keys())  # 更新键列表
        key_len = len(key_list)  # 更新键的数量
    dic = test_dic['%.8d' % 0]  # 获取第一个测试数据
    image = dic['image']  # 获取原始图像
    pimage = dic['pimage']  # 获取处理后的图像
    ori_shape = dic['ori_shape']  # 获取原始图像形状
    resize_shape = dic['resize_shape']  # 获取调整后的图像形状

    # warm up
    if use_gpu:
        for k in range(10):
            image, boxes, scores, classes = _decode.detect_image(image, pimage, ori_shape, resize_shape, draw_image=False)  # 进行10次预热推理

    time_stat = deque(maxlen=20)  # 初始化时间队列
    start_time = time.time()  # 记录开始时间
    end_time = time.time()  # 记录结束时间
    num_imgs = len(path_dir)  # 获取测试图像数量
    start = time.time()  # 记录开始时间
    for k, filename in enumerate(path_dir):  # 遍历测试图像
        key_list = list(test_dic.keys())  # 获取测试字典的键列表
        key_len = len(key_list)  # 获取键的数量
        while key_len == 0:  # 如果键的数量为0
            time.sleep(0.01)  # 休眠0.01秒
            key_list = list(test_dic.keys())  # 更新键列表
            key_len = len(key_list)  # 更新键的数量
        dic = test_dic.pop('%.8d' % k)  # 获取当前测试数据
        image = dic['image']  # 获取原始图像
        pimage = dic['pimage']  # 获取处理后的图像
        ori_shape = dic['ori_shape']  # 获取原始图像形状
        resize_shape = dic['resize_shape']  # 获取调整后的图像形状

        image, boxes, scores, classes = _decode.detect_image(image, pimage, ori_shape, resize_shape, draw_image, draw_thresh)  # 进行图像检测

        # 估计剩余时间
        start_time = end_time  # 更新开始时间
        end_time = time.time()  # 更新结束时间
        time_stat.append(end_time - start_time)  # 更新时间队列
        time_cost = np.mean(time_stat)  # 计算平均时间
        eta_sec = (num_imgs - k) * time_cost  # 计算剩余时间
        eta = str(datetime.timedelta(seconds=int(eta_sec)))  # 格式化剩余时间

        logger.info('Infer iter {}, num_imgs={}, eta={}.'.format(k, num_imgs, eta))  # 记录推理进度
        if draw_image:  # 如果需要绘制图像
            t2 = threading.Thread(target=save_img, args=(filename, image))  # 创建保存图像的线程
            t2.start()  # 启动线程
            logger.info("Detection bbox results save in images/res_pod/{}".format(filename))  # 记录保存路径
    cost = time.time() - start  # 计算总时间
    logger.info('total time: {0:.6f}s'.format(cost))  # 记录总时间
    logger.info('Speed: %.6fs per image,  %.1f FPS.' % ((cost / num_imgs), (num_imgs / cost)))  # 记录速度和帧率
