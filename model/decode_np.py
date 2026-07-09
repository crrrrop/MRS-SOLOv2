#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-11-21 09:13:23
#   Description : paddle2.0_solov2
#
# ================================================================

import random
import colorsys
import cv2
import threading
import os
import paddle
import paddle.nn.functional as F
import paddle.fluid.layers as L
import numpy as np

from tools.transform import *

class Decode(object):
    def __init__(self, model, all_classes, place, cfg, for_test=True):
        self.all_classes = all_classes
        self.num_classes = len(self.all_classes)
        self.model = model
        self.place = place

        self.context = cfg.context
        self.to_rgb = cfg.decodeImage['to_rgb']
        target_size = cfg.eval_cfg['target_size']
        if for_test:
            target_size = cfg.test_cfg['target_size']
        self.resizeImage = ResizeImage(target_size=target_size, resize_box=False, interp=cfg.resizeImage['interp'],
                                       max_size=cfg.resizeImage['max_size'], use_cv2=cfg.resizeImage['use_cv2'])
        self.normalizeImage = NormalizeImage(**cfg.normalizeImage)
        self.permute = Permute(**cfg.permute)
        self.padBatch = PadBatch(use_padded_im_info=False, pad_to_stride=cfg.padBatch['pad_to_stride'])

    def detect_image(self, image, pimage, ori_shape, resize_shape, draw_image, draw_thresh=0.0, save_mask=False, mask_save_path="", colored_mask_save_path=""):
        pred = self.predict(pimage, ori_shape, resize_shape)
        if pred['scores'][0] < 0:
            boxes = np.array([])
            masks = np.array([])
            scores = np.array([])
            classes = np.array([])
        else:
            masks = pred['masks']
            scores = pred['scores']
            classes = pred['classes'].astype(np.int32)
            boxes = []
            for ms in masks:
                sum_1 = np.sum(ms, axis=0)
                x = np.where(sum_1 > 0.5)[0]
                sum_2 = np.sum(ms, axis=1)
                y = np.where(sum_2 > 0.5)[0]
                if len(x) == 0:
                    x0, x1, y0, y1 = 0, 1, 0, 1
                else:
                    x0, x1, y0, y1 = x[0], x[-1], y[0], y[-1]
                boxes.append([x0, y0, x1, y1])
            boxes = np.array(boxes).astype(np.float32)
        if len(scores) > 0 and draw_image:
            pos = np.where(scores >= draw_thresh)
            boxes2 = boxes[pos]
            scores2 = scores[pos]
            classes2 = classes[pos]
            masks2 = masks[pos]
            self.draw(image, boxes2, scores2, classes2, masks2, save_mask, mask_save_path, colored_mask_save_path)
        return image, boxes, scores, classes

    def detect_batch(self, batch_img, batch_pimage, batch_ori_shape, batch_resize_shape, draw_image, draw_thresh=0.0):
        batch_size = len(batch_img)
        result_image = [None] * batch_size
        result_boxes = [None] * batch_size
        result_scores = [None] * batch_size
        result_classes = [None] * batch_size
        result_masks = [None] * batch_size

        pred = self.predict(batch_pimage, batch_ori_shape, batch_resize_shape)
        if pred['scores'][0] < 0:
            boxes = np.array([])
            masks = np.array([])
            scores = np.array([])
            classes = np.array([])
        else:
            masks = pred['masks']
            scores = pred['scores']
            classes = pred['classes'].astype(np.int32)
            boxes = []
            for ms in masks:
                sum_1 = np.sum(ms, axis=0)
                x = np.where(sum_1 > 0.5)[0]
                sum_2 = np.sum(ms, axis=1)
                y = np.where(sum_2 > 0.5)[0]
                if len(x) == 0:
                    x0, x1, y0, y1 = 0, 1, 0, 1
                else:
                    x0, x1, y0, y1 = x[0], x[-1], y[0], y[-1]
                boxes.append([x0, y0, x1, y1])
            boxes = np.array(boxes).astype(np.float32)
        
        for i in range(batch_size):
            if len(scores) > 0 and draw_image:
                pos = np.where(scores >= draw_thresh)
                boxes2 = boxes[pos]
                scores2 = scores[pos]
                classes2 = classes[pos]
                masks2 = masks[pos]
                self.draw(batch_img[i], boxes2, scores2, classes2, masks2)
            
            result_image[i] = batch_img[i]
            result_boxes[i] = boxes
            result_scores[i] = scores
            result_classes[i] = classes
            result_masks[i] = masks
        
        return result_image, result_boxes, result_scores, result_classes, result_masks






    def draw(self, image, boxes, scores, classes, masks, save_mask=False, mask_save_path="", colored_mask_save_path="", mask_alpha=0.45):
        """绘制图像上的检测结果"""
        image_h, image_w, _ = image.shape  # 获取图像的高和宽

        # 生成至少50种不同的颜色
        hsv_tuples = [(x / 50.0, 1., 1.) for x in range(50)]  # 生成50种不同的HSV颜色值
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))  # 将HSV颜色转换为RGB颜色
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))  # 将颜色值转换为0-255的范围

        random.seed(0)  # 固定随机种子
        random.shuffle(colors)  # 打乱颜色列表
        random.seed(None)  # 重置随机种子

        mask_image = np.zeros((image_h, image_w, 3), dtype=np.uint8) if save_mask else None  # 初始化黑色背景的掩膜图像
        colored_mask_image = np.zeros((image_h, image_w, 3), dtype=np.uint8) if save_mask else None  # 初始化带有颜色的掩膜图像

        # 先绘制掩膜
        for box, score, cl, ms in zip(boxes, scores, classes, masks):  # 遍历检测结果
            x0, y0, x1, y1 = box  # 获取边界框坐标
            left = max(0, np.floor(x0 + 0.5).astype(int))  # 计算左边界
            top = max(0, np.floor(y0 + 0.5).astype(int))  # 计算上边界
            right = min(image.shape[1], np.floor(x1 + 0.5).astype(int))  # 计算右边界
            bottom = min(image.shape[0], np.floor(y1 + 0.5).astype(int))  # 计算下边界

            bbox_color = random.choice(colors)  # 随机选择一种颜色作为掩膜的颜色
            color = np.array(bbox_color)  # 将颜色转换为NumPy数组
            color = np.reshape(color, (1, 1, 3))  # 将颜色数组调整为适合广播的形状
            target_ms = ms[top:bottom, left:right]  # 提取掩膜中目标的区域
            target_ms = np.expand_dims(target_ms, axis=2)  # 为掩膜添加一个通道维度
            target_ms = np.tile(target_ms, (1, 1, 3))  # 将掩膜的单通道复制到三个通道
            target_region = image[top:bottom, left:right, :]  # 提取图像中目标的区域
            target_region = target_ms * (target_region * (1 - mask_alpha) + color * mask_alpha) + (1 - target_ms) * target_region  # 应用掩膜和颜色
            image[top:bottom, left:right, :] = target_region  # 更新图像中的目标区域

            if save_mask:
                mask_image[top:bottom, left:right, :] = target_ms * 255  # 在黑色背景的掩膜图像中添加白色前景
                colored_mask_image[top:bottom, left:right, :] = target_ms * color  # 在带有颜色的掩膜图像中添加颜色

        # 再绘制边界框和类别标签
        for box, score, cl in zip(boxes, scores, classes):
            x0, y0, x1, y1 = box  # 获取边界框坐标
            left = max(0, np.floor(x0 + 0.5).astype(int))  # 计算左边界
            top = max(0, np.floor(y0 + 0.5).astype(int))  # 计算上边界
            right = min(image.shape[1], np.floor(x1 + 0.5).astype(int))  # 计算右边界
            bottom = min(image.shape[0], np.floor(y1 + 0.5).astype(int))  # 计算下边界

            bbox_color = random.choice(colors)  # 随机选择一种颜色作为边界框的颜色

            bbox_thick = 1  # 设置边界框的厚度
            cv2.rectangle(image, (left, top), (right, bottom), bbox_color, bbox_thick)  # 在图像上绘制边界框
            bbox_mess = '%s' % (self.all_classes[cl])  # 生成类别标签
            t_size = cv2.getTextSize(bbox_mess, 0, 0.5, thickness=1)[0]  # 获取文本标签的尺寸
            cv2.rectangle(image, (left, top), (left + t_size[0], top - t_size[1] - 3), bbox_color, -1)  # 绘制文本背景框
            cv2.putText(image, bbox_mess, (left, top - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)  # 在图像上绘制文本标签

        if save_mask:
            cv2.imwrite(mask_save_path, mask_image)  # 保存黑色背景的掩膜图像
            cv2.imwrite(colored_mask_save_path, colored_mask_image)  # 保存带有颜色的掩膜图像

    def process_image(self, img):
        if self.to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        context = self.context
        sample = {}
        sample['image'] = img
        sample['h'] = img.shape[0]
        sample['w'] = img.shape[1]

        sample = self.resizeImage(sample, context)
        sample = self.normalizeImage(sample, context)
        sample = self.permute(sample, context)

        samples = self.padBatch([sample], context)
        sample = samples[0]

        pimage = np.expand_dims(sample['image'], axis=0)
        ori_shape = np.array([[sample['h'], sample['w']]]).astype(np.int32)
        resize_shape = np.array([[sample['im_info'][0], sample['im_info'][1]]]).astype(np.int32)
        return pimage, ori_shape, resize_shape

    def predict(self, image, ori_shape, resize_shape):
        image = paddle.to_tensor(image, place=self.place)
        ori_shape = paddle.to_tensor(ori_shape, place=self.place)
        resize_shape = paddle.to_tensor(resize_shape, place=self.place)
        preds = self.model(image, ori_shape, resize_shape)
        numpy_preds = {}
        for key in preds.keys():
            value = preds[key]
            value = value.numpy()
            numpy_preds[key] = value
        return numpy_preds
