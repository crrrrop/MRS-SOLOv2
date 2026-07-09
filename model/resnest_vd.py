"""Split-Attention"""
import paddle
import paddle.fluid as fluid
import paddle.fluid.dygraph as dygraph
from paddle.fluid.dygraph import Conv2D, BatchNorm, Linear, Pool2D, Sequential

import paddle.nn as nn
import paddle.nn.functional as F

import numpy as np
import os

from model.custom_layers import *

## MSCAAttention注意力机制
class MSCAAttention(dygraph.layers.Layer):
    # SegNext NeurIPS 2022
    # https://github.com/Visual-Attention-Network/SegNeXt/tree/main
    def __init__(self, dim):
        super().__init__()
        self.conv0 = Conv2D(dim, dim, 5, padding=2, groups=dim)
        # self.conv0_1 = Conv2D(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        # self.conv0_2 = Conv2D(dim, dim, (7, 1), padding=(3, 0), groups=dim)
 
        # self.conv1_1 = Conv2D(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        # self.conv1_2 = Conv2D(dim, dim, (11, 1), padding=(5, 0), groups=dim)
 
        # self.conv2_1 = Conv2D(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        # self.conv2_2 = Conv2D(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        # self.conv3 = Conv2D(dim, dim, 1)
        self.conv0 = Conv2D(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = Conv2D(dim, dim, (1, 5), padding=(0, 2), groups=dim)  # 改为5x1
        self.conv0_2 = Conv2D(dim, dim, (5, 1), padding=(2, 0), groups=dim)  # 改为1x5

        self.conv1_1 = Conv2D(dim, dim, (1, 9), padding=(0, 4), groups=dim)  # 改为9x1
        self.conv1_2 = Conv2D(dim, dim, (9, 1), padding=(4, 0), groups=dim)  # 改为1x9

        self.conv2_1 = Conv2D(dim, dim, (1, 15), padding=(0, 7), groups=dim)  # 改为15x1
        self.conv2_2 = Conv2D(dim, dim, (15, 1), padding=(7, 0), groups=dim)  # 改为1x15

        self.conv3 = Conv2D(dim, dim, 1)
 
    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
 
        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)
 
        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)
 
        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2
 
        attn = self.conv3(attn)
 
        return attn * u




## ResNeSt网络模块

class ReLU(dygraph.layers.Layer):
    """ 封装一下relu模块，方便动态图调用 """

    def forward(self, x):
        return fluid.layers.relu(x)

# dropblock 是一种结构化的 dropout 方法，它可以同时丢弃特征图中的一整块激活区域，而不是随机丢弃一个激活单元
class DropBlock2D(dygraph.layers.Layer):
    """ 
    DropBlock2D模块 
    借鉴PaddleDetection： https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.2/ppdet/modeling/ops.py#L116
    """

    def __init__(self, keep_prob, block_size):
        super(DropBlock2D, self).__init__()
        # 需要block的块大小，默认是3x3
        self.block_size = block_size
        # 不做block的概率
        self.keep_prob = keep_prob
        # 自动判断是训练阶段还是测试阶段
        self.is_training = fluid.framework._dygraph_tracer()._train_mode

    def calculate_gamma(self, x_shape):
        feat_shape_tmp = fluid.layers.slice(x_shape, [0], [3], [4])
        feat_shape_tmp = fluid.layers.cast(feat_shape_tmp, dtype="float32")
        feat_shape_t = fluid.layers.reshape(feat_shape_tmp, [1, 1, 1, 1])
        feat_area = fluid.layers.pow(feat_shape_t, factor=2)

        block_shape_t = fluid.layers.fill_constant(
            shape=[1, 1, 1, 1], value=self.block_size, dtype='float32')
        block_area = fluid.layers.pow(block_shape_t, factor=2)

        useful_shape_t = feat_shape_t - block_shape_t + 1
        useful_area = fluid.layers.pow(useful_shape_t, factor=2)

        upper_t = feat_area * (1 - self.keep_prob)
        bottom_t = block_area * useful_area
        output = upper_t / bottom_t
        return output

    def forward(self, x):
        if not self.is_training:
            return x

        input_shape = fluid.layers.shape(x)
        gamma = self.calculate_gamma(input_shape)
        p = fluid.layers.expand_as(gamma, x)

        input_shape_tmp = fluid.layers.cast(input_shape, dtype="int64")
        random_matrix = fluid.layers.uniform_random(
            input_shape_tmp, dtype='float32', min=0.0, max=1.0)
        one_zero_m = fluid.layers.less_than(random_matrix, p)
        one_zero_m.stop_gradient = True
        one_zero_m = fluid.layers.cast(one_zero_m, dtype="float32")

        mask_flag = fluid.layers.pool2d(
            one_zero_m,
            pool_size=self.block_size,
            pool_type='max',
            pool_stride=1,
            pool_padding=self.block_size // 2)
        mask = 1.0 - mask_flag

        elem_numel = fluid.layers.reduce_prod(input_shape)
        elem_numel_m = fluid.layers.cast(elem_numel, dtype="float32")
        elem_numel_m.stop_gradient = True

        elem_sum = fluid.layers.reduce_sum(mask)
        elem_sum_m = fluid.layers.cast(elem_sum, dtype="float32")
        elem_sum_m.stop_gradient = True

        output = x * mask * elem_numel_m / elem_sum_m
        return output


class rSoftMax(dygraph.layers.Layer):
    """
    rSoftMax的实现。
    主要是为了处理最后的通道排列重组，即上图打星星的Split+sum处的处理
    """

    def __init__(self, radix, cardinality):
        super(rSoftMax, self).__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.shape[0]
        if self.radix > 1:
            # 每个cardinality有2个以上split的情况
            x = fluid.layers.reshape(x, (batch, self.cardinality, self.radix, -1))
            x = fluid.layers.transpose(x, perm=[0, 2, 1, 3])
            x = fluid.layers.softmax(x, axis=1)
            x = fluid.layers.reshape(x, (batch, -1))
        else:
            # 每个cardinality有只有1个split的情况
            x = fluid.layers.sigmoid(x)
        return x


class SplAtConv2d(dygraph.layers.Layer):
    """
    Split-Attention核心模块

    Args:
        in_channels: 输入通道数
        channels: 输出通道数
        kernel_size: filter的大小
        stride: 步长
        padding: 补齐
        dilation: 空洞卷积的空洞大小
        groups: 输入首先的分组数，即Cardinal的数量
        bias: 是否使用偏置
        radix: 每个Cardinal继续拆分的个数，所以组卷积个数就是groups * radix
        reduction_factor: 128-64-128这样的缩放因子，resnet中的常规操作
        dropblock_prob: dropblock的概率
    """

    def __init__(self, in_channels, channels, kernel_size,
                 stride=(1, 1), padding=(0, 0), dilation=(1, 1),
                 groups=1, bias=True, freeze_norm=False, radix=2, reduction_factor=4,
                 dropblock_prob=0.0, **kwargs):
        super(SplAtConv2d, self).__init__()
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob
        self.conv = Conv2D(in_channels, channels * radix, kernel_size, stride,
                           padding, dilation, groups=groups * radix, bias_attr=bias)
        self.relu = ReLU()
        self.bn0 = BatchNorm(channels * radix)
        self.fc1 = Conv2D(channels, inter_channels, 1, groups=self.cardinality)
        self.bn1 = BatchNorm(inter_channels)
        self.fc2 = Conv2D(inter_channels, channels * radix, 1, groups=self.cardinality)
        if dropblock_prob > 0.0:
            self.dropblock = DropBlock2D(dropblock_prob, 3)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        # 先对输入进行分组卷积，总共分为groups * radix组
        x = self.conv(x)
        x = self.bn0(x)
        if self.dropblock_prob > 0.0:
            x = self.dropblock(x)
        x = self.relu(x)

        # 组拆分成radix块，并加总 —— split+sum
        batch = x.shape[0]
        if self.radix > 1:
            splited = fluid.layers.split(x, self.radix, dim=1)
            gap = sum(splited)
        else:
            gap = x

        # 每组并行执行attention，并用rsoftmax完成纵向权重归一化
        gap = fluid.layers.adaptive_pool2d(gap, 1, pool_type="avg")
        gap = self.fc1(gap)
        gap = self.bn1(gap)
        gap = self.relu(gap)
        atten = self.fc2(gap)
        atten = self.rsoftmax(atten)
        atten = fluid.layers.reshape(atten, (batch, -1, 1, 1))

        # 对每个cardinal中的radix分别进行加权相加，得到输出结果
        if self.radix > 1:
            attens = fluid.layers.split(atten, self.radix, dim=1)
            # 对应元素相乘，版本1中的实现在paddle1.8中反向传播时会报错！这里已OK！
            out = sum([fluid.layers.elementwise_mul(split, att, axis=0)
                       for (att, split) in zip(attens, splited)])
        else:
            out = atten * x
        return out


class ConvBlock(paddle.nn.Layer):
    def __init__(self, in_c, filters, bn, gn, af, freeze_norm, norm_decay, lr, use_dcn=False, stride=2, downsample_in3x3=True, is_first=False, block_name='', bottleneck_width=64, cardinality=1):
        # 调用父类初始化方法
        super(ConvBlock, self).__init__()
        # 将过滤器数量解包到三个变量中
        filters1, filters2, filters3 = filters
        group_width = int(filters1*(bottleneck_width/64.))*cardinality
        # print(f"ConvBlock: group_width:{group_width}")
        # 根据是否在3x3卷积层进行下采样，确定不同的步幅
        if downsample_in3x3 == True:
            stride1, stride2 = 1, stride
        else:
            stride1, stride2 = stride, 1
        self.is_first = is_first

        # 定义三个卷积层
        self.conv1 = Conv2dUnit(in_c, group_width, 1, stride=stride1, bn=bn, gn=gn, af=af, freeze_norm=freeze_norm, norm_decay=norm_decay, lr=lr, act='relu', name=block_name+'_branch2a')
        # stride=(1, 1), padding=(0, 0),dilation=(1, 1)
        self.conv2 = SplAtConv2d(group_width, group_width, 3, stride=stride2, padding= 1,dilation=1, groups=1, bias=False, freeze_norm=freeze_norm, radix=1, reduction_factor=4, dropblock_prob=0.0)   
        # self.conv2 = Conv2dUnit(filters1, filters2, 3, stride=stride2, bn=bn, gn=gn, af=af, freeze_norm=freeze_norm, norm_decay=norm_decay, lr=lr, act='relu', use_dcn=use_dcn, name=block_name+'_branch2b')
        self.conv3 = Conv2dUnit(group_width, filters3, 1, stride=1, bn=bn, gn=gn, af=af, freeze_norm=freeze_norm, norm_decay=norm_decay, lr=lr, act=None, name=block_name+'_branch2c')

        # print(f"ConvBlock: conv1`s input is:{in_c}, output is:{group_width}")
        # print(f"ConvBlock: conv2`s input is:{group_width}, output is:{group_width}")
        # print(f"ConvBlock: conv3`s input is:{group_width}, output is:{filters3}")

        # 根据是否是第一个块，定义不同的shortcut连接
        if not self.is_first:
            self.avg_pool = paddle.nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
            self.conv4 = Conv2dUnit(in_c, filters3, 1, stride=1, bn=bn, gn=gn, af=af, freeze_norm=freeze_norm, norm_decay=norm_decay, lr=lr, act=None, name=block_name+'_branch1')
        else:
            self.conv4 = Conv2dUnit(in_c, filters3, 1, stride=stride, bn=bn, gn=gn, af=af, freeze_norm=freeze_norm, norm_decay=norm_decay, lr=lr, act=None, name=block_name+'_branch1')
        # print(f"ConvBlock: conv4`s input is:{in_c}, output is:{filters3}. is_first is {self.is_first}")
        # print("----------------------------------------------------------------------------------------")
        self.act = paddle.nn.ReLU()

    # 冻结卷积层的参数
    def freeze(self):
        self.conv1.freeze()
        # self.conv2.freeze()   # conv2为splitconv，没有实现freeze()
        self.conv3.freeze()
        self.conv4.freeze()

    # 定义前向传播过程
    def __call__(self, input_tensor):
        # 通过三个卷积层
        # print("________________ConvBlock____________________")
        # print(f"ConvBlock: input_tensor:{input_tensor.shape}")
        x = self.conv1(input_tensor)
        # print(f"ConvBlock: conv1:{x.shape}")
        x = self.conv2(x)
        # print(f"ConvBlock: conv2:{x.shape}")
        x = self.conv3(x)
        # print(f"ConvBlock: conv3:{x.shape}")
        # 如果不是第一个块，通过平均池化层
        if not self.is_first:
            input_tensor = self.avg_pool(input_tensor)
        # 通过shortcut连接
        shortcut = self.conv4(input_tensor)
        # print(f"ConvBlock: shortcut:{shortcut.shape}")
        # 进行元素级别的加和
        x = x + shortcut
        # print(f"ConvBlock: conv3+conv4:{x.shape}")
        # print("=================================================")
        # 应用激活函数
        x = self.act(x)
        return x


class IdentityBlock(paddle.nn.Layer):
    def __init__(self, in_c, filters, bn, gn, af, freeze_norm, norm_decay, lr, use_dcn=False, block_name='', bottleneck_width=64, cardinality=1):
        super(IdentityBlock, self).__init__()
        filters1, filters2, filters3 = filters
        group_width = int(filters1*(bottleneck_width/64.))*cardinality
        # print(f"IdentityBlock: group_width:{group_width}")

        self.conv1 = Conv2dUnit(in_c,     group_width, 1, stride=1, bn=bn, gn=gn, af=af, freeze_norm=freeze_norm, norm_decay=norm_decay, lr=lr, act='relu', name=block_name+'_branch2a',)
        self.conv2 = SplAtConv2d(group_width, group_width, 3, stride=1, padding= 1,dilation=1, groups=1, bias=False, freeze_norm=freeze_norm, radix=1, reduction_factor=4, dropblock_prob=0.0)
        # self.conv2 = Conv2dUnit(filters1, filters2, 3, stride=1, bn=bn, gn=gn, af=af, freeze_norm=freeze_norm, norm_decay=norm_decay, lr=lr, act='relu', use_dcn=use_dcn, name=block_name+'_branch2b')
        self.conv3 = Conv2dUnit(group_width, filters3, 1, stride=1, bn=bn, gn=gn, af=af, freeze_norm=freeze_norm, norm_decay=norm_decay, lr=lr, act=None, name=block_name+'_branch2c')
        self.act = paddle.nn.ReLU()

        # print(f"IdentityBlock: conv1`s input is:{in_c}, output is:{group_width}")
        # print(f"IdentityBlock: conv2`s input is:{group_width}, output is:{group_width}")
        # print(f"IdentityBlock: conv3`s input is:{group_width}, output is:{filters3}")
        # print("----------------------------------------------------------------------------------------")

    def freeze(self):
        self.conv1.freeze()
        # self.conv2.freeze()   # conv2为splitconv，没有实现freeze()
        self.conv3.freeze()

    def __call__(self, input_tensor):
        # print("________________IdentityBlock____________________")
        # print(f"IdentityBlock: input_tensor:{input_tensor.shape}")
        x = self.conv1(input_tensor)
        # print(f"IdentityBlock: conv1:{x.shape}")
        x = self.conv2(x)
        # print(f"IdentityBlock: conv2:{x.shape}")
        x = self.conv3(x)
        # print(f"IdentityBlock: conv3:{x.shape}")
        x = x + input_tensor
        x = self.act(x)
        # print(f"IdentityBlock: conv3+x:{x.shape}")
        # print("=================================================")
        return x

class ResNeSt(paddle.nn.Layer):
    def __init__(self, norm_type='bn', feature_maps=[3, 4, 5], dcn_v2_stages=[5], downsample_in3x3=False, freeze_at=0, freeze_norm=False, norm_decay=0., lr_mult_list=[1., 1., 1., 1.]):
        super(ResNeSt, self).__init__()
        self.norm_type = norm_type
        self.feature_maps = feature_maps
        assert freeze_at in [0, 1, 2, 3, 4, 5]
        assert len(lr_mult_list) == 4, "lr_mult_list length must be 4 but got {}".format(len(lr_mult_list))
        self.lr_mult_list = lr_mult_list
        self.freeze_at = freeze_at
        assert norm_type in ['bn', 'sync_bn', 'gn', 'affine_channel']
        bn, gn, af = get_norm(norm_type)
        # conv 7×7 --> conv 3×3 -> conv 3×3 -> conv 3×3
        self.stage1_conv1_1 = Conv2dUnit(3,  32, 3, stride=2, bn=bn, gn=gn, af=af, freeze_norm=freeze_norm, norm_decay=norm_decay, act='relu', name='conv1_1')
        self.stage1_conv1_2 = Conv2dUnit(32, 32, 3, stride=1, bn=bn, gn=gn, af=af, freeze_norm=freeze_norm, norm_decay=norm_decay, act='relu', name='conv1_2')
        self.stage1_conv1_3 = Conv2dUnit(32, 64, 3, stride=1, bn=bn, gn=gn, af=af, freeze_norm=freeze_norm, norm_decay=norm_decay, act='relu', name='conv1_3')
        self.pool = paddle.nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

        # stage2
        self.stage2_0 = ConvBlock(64, [64, 64, 256], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[0], stride=1, downsample_in3x3=downsample_in3x3, is_first=True, block_name='res2a', bottleneck_width=64, cardinality=1)
        self.stage2_1 = IdentityBlock(256, [64, 64, 256], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[0], block_name='res2b', bottleneck_width=64, cardinality=1)
        self.stage2_2 = IdentityBlock(256, [64, 64, 256], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[0], block_name='res2c', bottleneck_width=64, cardinality=1)

        # stage3
        use_dcn = 3 in dcn_v2_stages
        self.stage3_0 = ConvBlock(256, [128, 128, 512], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[1], use_dcn=use_dcn, downsample_in3x3=downsample_in3x3, block_name='res3a', bottleneck_width=64, cardinality=1)
        self.stage3_1 = IdentityBlock(512, [128, 128, 512], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[1], use_dcn=use_dcn, block_name='res3b', bottleneck_width=64, cardinality=1)
        self.stage3_2 = IdentityBlock(512, [128, 128, 512], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[1], use_dcn=use_dcn, block_name='res3c', bottleneck_width=64, cardinality=1)
        self.stage3_3 = IdentityBlock(512, [128, 128, 512], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[1], use_dcn=use_dcn, block_name='res3d', bottleneck_width=64, cardinality=1)

        # stage4
        use_dcn = 4 in dcn_v2_stages
        self.stage4_0 = ConvBlock(512, [256, 256, 1024], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[2], use_dcn=use_dcn, downsample_in3x3=downsample_in3x3, block_name='res4a', bottleneck_width=64, cardinality=1)
        self.stage4_1 = IdentityBlock(1024, [256, 256, 1024], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[2], use_dcn=use_dcn, block_name='res4b', bottleneck_width=64, cardinality=1)
        self.stage4_2 = IdentityBlock(1024, [256, 256, 1024], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[2], use_dcn=use_dcn, block_name='res4c', bottleneck_width=64, cardinality=1)
        self.stage4_3 = IdentityBlock(1024, [256, 256, 1024], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[2], use_dcn=use_dcn, block_name='res4d', bottleneck_width=64, cardinality=1)
        self.stage4_4 = IdentityBlock(1024, [256, 256, 1024], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[2], use_dcn=use_dcn, block_name='res4e', bottleneck_width=64, cardinality=1)
        self.stage4_5 = IdentityBlock(1024, [256, 256, 1024], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[2], use_dcn=use_dcn, block_name='res4f', bottleneck_width=64, cardinality=1)

        # stage5
        use_dcn = 5 in dcn_v2_stages
        self.stage5_0 = ConvBlock(1024, [512, 512, 2048], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[3], use_dcn=use_dcn, downsample_in3x3=downsample_in3x3, block_name='res5a', bottleneck_width=64, cardinality=1)
        self.stage5_1 = IdentityBlock(2048, [512, 512, 2048], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[3], use_dcn=use_dcn, block_name='res5b', bottleneck_width=64, cardinality=1)
        self.stage5_2 = IdentityBlock(2048, [512, 512, 2048], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[3], use_dcn=use_dcn, block_name='res5c', bottleneck_width=64, cardinality=1)

        # MSCAAtention
        # self.stage2_A = MSCAAttention(256)
        # self.stage3_A = MSCAAttention(512)
        self.stage4_A = MSCAAttention(1024)
        self.stage5_A = MSCAAttention(2048)

    def forward(self, input_tensor):
        # print(f"0. input_tensor:{input_tensor.shape}")
        x = self.stage1_conv1_1(input_tensor)
        # print(f"1_1. stage1_conv1_1:{x.shape}")
        x = self.stage1_conv1_2(x)
        # print(f"1_2. stage1_conv1_2:{x.shape}")
        x = self.stage1_conv1_3(x)
        # print(f"1_3. stage1_conv1_3:{x.shape}")
        x = self.pool(x)


        # stage2
        x = self.stage2_0(x)
        # print(f"2_1. stage2_0:{x.shape}")
        x = self.stage2_1(x)
        # print(f"2_2. stage2_1:{x.shape}")
        s4 = self.stage2_2(x)
        # print(f"2_3. stage2_2(s4):{s4.shape}")
        # stage3
        x = self.stage3_0(s4)
        # print(f"3_1. stage3_0:{x.shape}")
        x = self.stage3_1(x)
        # print(f"3_2. stage3_1:{x.shape}")
        x = self.stage3_2(x)
        # print(f"3_3. stage3_2:{x.shape}")
        s8 = self.stage3_3(x)
        # print(f"3_4. stage3_3(s8):{s8.shape}")
        # stage4
        x = self.stage4_0(s8)
        # print(f"4_1. stage4_0:{x.shape}")
        x = self.stage4_1(x)
        # print(f"4_2. stage4_1:{x.shape}")
        x = self.stage4_2(x)
        # print(f"4_3. stage4_2:{x.shape}")
        x = self.stage4_3(x)
        # print(f"4_4. stage4_3:{x.shape}")
        x = self.stage4_4(x)
        # print(f"4_5. stage4_4:{x.shape}")
        s16 = self.stage4_5(x)
        # print(f"4_6. stage4_5(s16):{s16.shape}")
        # stage5
        x = self.stage5_0(s16)
        # print(f"5_1. stage5_0:{x.shape}")
        x = self.stage5_1(x)
        # print(f"5_2. stage5_1:{x.shape}")
        s32 = self.stage5_2(x)
        # print(f"5_3. stage5_2(s32):{s32.shape}")

        outs = []
        # s4 = self.stage2_A(s4)
        # s8 = self.stage3_A(s8)
        s16 = self.stage4_A(s16)
        s32 = self.stage5_A(s32)
        if 2 in self.feature_maps:
            outs.append(s4)
        if 3 in self.feature_maps:
            outs.append(s8)
        if 4 in self.feature_maps:
            outs.append(s16)
        if 5 in self.feature_maps:
            outs.append(s32)
        # print("____________outs_____________")
        # for i in range(0,4):
            # print(f"outs[{i}]:{outs[i].shape}")
        # print("__________outs__end__________")
        return outs

    def freeze(self):
        freeze_at = self.freeze_at
        if freeze_at >= 1:
            self.stage1_conv1_1.freeze()
            self.stage1_conv1_2.freeze()
            self.stage1_conv1_3.freeze()
        if freeze_at >= 2:
            self.stage2_0.freeze()
            self.stage2_1.freeze()
            self.stage2_2.freeze()
        if freeze_at >= 3:
            self.stage3_0.freeze()
            self.stage3_1.freeze()
            self.stage3_2.freeze()
            self.stage3_3.freeze()
        if freeze_at >= 4:
            self.stage4_0.freeze()
            self.stage4_1.freeze()
            self.stage4_2.freeze()
            self.stage4_3.freeze()
            self.stage4_4.freeze()
            self.stage4_5.freeze()
        if freeze_at >= 5:
            self.stage5_0.freeze()
            self.stage5_1.freeze()
            self.stage5_2.freeze()


# class Dropout(dygraph.layers.Layer):

#     def __init__(self, prob):
#         super(Dropout, self).__init__()
#         self.keep_prob = prob
#         self.is_test = (not fluid.framework._dygraph_tracer()._train_mode)
#         self.implement_method = "upscale_in_train"

#     def forward(self, x):
#         return fluid.layers.dropout(
#             x, dropout_prob=self.keep_prob, is_test=self.is_test,
#             dropout_implementation=self.implement_method)


# class GlobalAvgPool2d(dygraph.layers.Layer):

#     def forward(self, x):
#         bs = x.shape[0]
#         x = fluid.layers.adaptive_pool2d(x, 1, pool_type="avg")
#         return fluid.layers.reshape(x, (bs, -1))
