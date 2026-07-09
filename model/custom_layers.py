#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date:
#   Description :
#
# ================================================================
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as L
import paddle.nn.functional as F
from paddle.fluid import Variable
from paddle.fluid.data_feeder import check_variable_and_dtype, check_type
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.layers import utils
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Constant, Normal
from paddle.fluid.regularizer import L2Decay

import paddle.fluid.dygraph as dygraph
from paddle.fluid.dygraph import Conv2D, BatchNorm



def deformable_conv(input,
                    offset,
                    mask,
                    num_filters,
                    filter_size,
                    stride=1,
                    padding=0,
                    dilation=1,
                    groups=None,
                    deformable_groups=None,
                    im2col_step=None,
                    filter_param=None,
                    bias_attr=None,
                    modulated=True,
                    name=None):

    check_variable_and_dtype(input, "input", ['float32', 'float64'],
                             'deformable_conv')
    check_variable_and_dtype(offset, "offset", ['float32', 'float64'],
                             'deformable_conv')
    check_type(mask, 'mask', (Variable, type(None)), 'deformable_conv')

    num_channels = input.shape[1]
    assert filter_param is not None, "filter_param should not be None here."

    helper = LayerHelper('deformable_conv', **locals())
    dtype = helper.input_dtype()

    if not isinstance(input, Variable):
        raise TypeError("Input of deformable_conv must be Variable")
    if not isinstance(offset, Variable):
        raise TypeError("Input Offset of deformable_conv must be Variable")

    if groups is None:
        num_filter_channels = num_channels
    else:
        if num_channels % groups != 0:
            raise ValueError("num_channels must be divisible by groups.")
        num_filter_channels = num_channels // groups

    filter_size = utils.convert_to_list(filter_size, 2, 'filter_size')
    stride = utils.convert_to_list(stride, 2, 'stride')
    padding = utils.convert_to_list(padding, 2, 'padding')
    dilation = utils.convert_to_list(dilation, 2, 'dilation')

    input_shape = input.shape
    filter_shape = [num_filters, int(num_filter_channels)] + filter_size

    def _get_default_param_initializer():
        filter_elem_num = filter_size[0] * filter_size[1] * num_channels
        std = (2.0 / filter_elem_num)**0.5
        return Normal(0.0, std, 0)

    pre_bias = helper.create_variable_for_type_inference(dtype)

    if modulated:
        helper.append_op(
            type='deformable_conv',
            inputs={
                'Input': input,
                'Filter': filter_param,
                'Offset': offset,
                'Mask': mask,
            },
            outputs={"Output": pre_bias},
            attrs={
                'strides': stride,
                'paddings': padding,
                'dilations': dilation,
                'groups': groups,
                'deformable_groups': deformable_groups,
                'im2col_step': im2col_step,
            })

    else:
        helper.append_op(
            type='deformable_conv_v1',
            inputs={
                'Input': input,
                'Filter': filter_param,
                'Offset': offset,
            },
            outputs={"Output": pre_bias},
            attrs={
                'strides': stride,
                'paddings': padding,
                'dilations': dilation,
                'groups': groups,
                'deformable_groups': deformable_groups,
                'im2col_step': im2col_step,
            })

    output = helper.append_bias_op(pre_bias, dim_start=1, dim_end=2)
    return output



def get_norm(norm_type):
    bn = 0
    gn = 0
    af = 0
    if norm_type == 'bn':
        bn = 1
    elif norm_type == 'sync_bn':
        bn = 1
    elif norm_type == 'gn':
        gn = 1
    elif norm_type == 'affine_channel':
        af = 1
    return bn, gn, af




class Mish(paddle.nn.Layer):
    def __init__(self):
        super(Mish, self).__init__()

    def _softplus(self, x):
        expf = fluid.layers.exp(fluid.layers.clip(x, -200, 50))
        return fluid.layers.log(1 + expf)

    def __call__(self, x):
        return x * fluid.layers.tanh(self._softplus(x))

class ReLU(dygraph.layers.Layer):
    """ 封装一下relu模块，方便动态图调用 """

    def forward(self, x):
        return fluid.layers.relu(x)


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
                 groups=1, bias=True, radix=2, reduction_factor=4,
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



class Conv2dUnit(paddle.nn.Layer):
    def __init__(self,
                 input_dim,  # 输入通道数
                 filters,  # 卷积核数量（输出通道数）
                 filter_size,  # 卷积核尺寸
                 stride=1,  # 卷积步幅，默认值为1
                 bias_attr=False,  # 是否使用偏置，默认值为False
                 bn=0,  # 是否使用Batch Normalization，默认值为0（不使用）
                 gn=0,  # 是否使用Group Normalization，默认值为0（不使用）
                 af=0,  # 是否使用Affine Channel，默认值为0（不使用）
                 groups=32,  # Group Normalization的组数，默认值为32
                 act=None,  # 激活函数类型，默认值为None（不使用）
                 freeze_norm=False,  # 是否冻结归一化层的参数，默认值为False
                 is_test=False,  # 是否为测试模式，默认值为False
                 norm_decay=0.,  # 归一化层参数的正则化系数，默认值为0
                 lr=1.,  # 学习率，默认值为1
                 bias_lr=None,  # 偏置项的学习率，默认值为None
                 weight_init=None,  # 权重初始化方式，默认值为None
                 bias_init=None,  # 偏置初始化方式，默认值为None
                 use_dcn=False,  # 是否使用可变形卷积，默认值为False
                 name=''):  # 层的名称，默认值为空字符串
        super(Conv2dUnit, self).__init__()
        self.groups = groups  # 保存Group Normalization的组数
        self.filters = filters  # 保存卷积核数量（输出通道数）
        self.filter_size = filter_size  # 保存卷积核尺寸
        self.stride = stride  # 保存卷积步幅
        self.padding = (filter_size - 1) // 2  # 计算填充尺寸，使得输入输出尺寸相同
        self.act = act  # 保存激活函数类型
        self.freeze_norm = freeze_norm  # 保存是否冻结归一化层参数的标志
        self.is_test = is_test  # 保存是否为测试模式的标志
        self.norm_decay = norm_decay  # 保存归一化层参数的正则化系数
        self.use_dcn = use_dcn  # 保存是否使用可变形卷积的标志
        self.name = name  # 保存层的名称

        # 创建卷积层
        conv_name = name  # 卷积层名称
        self.dcn_param = None  # 初始化可变形卷积参数为None
        if use_dcn:  # 如果使用可变形卷积
            self.conv = paddle.nn.Conv2D(input_dim,
                                         filter_size * filter_size * 3,  # 输出通道数
                                         kernel_size=filter_size,  # 卷积核尺寸
                                         stride=stride,  # 卷积步幅
                                         padding=self.padding,  # 填充
                                         weight_attr=ParamAttr(initializer=Constant(0.0), name=conv_name + "_conv_offset.w_0"),  # 偏移权重初始化
                                         bias_attr=ParamAttr(initializer=Constant(0.0), name=conv_name + "_conv_offset.b_0"))  # 偏移偏置初始化
            self.dcn_param = fluid.layers.create_parameter(
                shape=[filters, input_dim, filter_size, filter_size],  # 可变形卷积参数形状
                dtype='float32',  # 数据类型
                attr=ParamAttr(name=conv_name + "_dcn_weights", learning_rate=lr, initializer=weight_init),  # 参数属性
                default_initializer=fluid.initializer.Xavier())  # Xavier初始化
        else:  # 如果不使用可变形卷积
            conv_battr = False  # 初始化偏置属性为False
            if bias_attr:  # 如果使用偏置
                blr = lr  # 偏置学习率
                if bias_lr:  # 如果偏置学习率不为None
                    blr = bias_lr  # 使用指定的偏置学习率
                conv_battr = ParamAttr(name=conv_name + "_bias",  # 偏置参数属性
                                       learning_rate=blr,
                                       initializer=bias_init,
                                       regularizer=L2Decay(0.))  # 不正则化的参数
            self.conv = paddle.nn.Conv2D(input_dim,
                                         filters,  # 输出通道数
                                         kernel_size=filter_size,  # 卷积核尺寸
                                         stride=stride,  # 卷积步幅
                                         padding=self.padding,  # 填充
                                         weight_attr=ParamAttr(name=conv_name + "_weights", learning_rate=lr, initializer=weight_init),  # 权重参数属性
                                         bias_attr=conv_battr)  # 偏置参数属性

        # 创建归一化层
        if conv_name == "conv1":
            norm_name = "bn_" + conv_name  # Batch Normalization层名称
            if gn:
                norm_name = "gn_" + conv_name  # Group Normalization层名称
            if af:
                norm_name = "af_" + conv_name  # Affine Channel层名称
        else:
            norm_name = "bn" + conv_name[3:]  # Batch Normalization层名称
            if gn:
                norm_name = "gn" + conv_name[3:]  # Group Normalization层名称
            if af:
                norm_name = "af" + conv_name[3:]  # Affine Channel层名称
        norm_lr = 0. if freeze_norm else lr  # 归一化层学习率
        pattr = ParamAttr(
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay),  # 归一化层的正则化系数
            name=norm_name + "_scale",
            trainable=False if freeze_norm else True)  # 归一化层参数是否可训练
        battr = ParamAttr(
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay),  # 归一化层的正则化系数
            name=norm_name + "_offset",
            trainable=False if freeze_norm else True)  # 归一化层参数是否可训练
        self.bn = None  # 初始化Batch Normalization层为None
        self.gn = None  # 初始化Group Normalization层为None
        self.af = None  # 初始化Affine Channel层为None
        if bn:
            self.bn = paddle.nn.BatchNorm2D(filters, weight_attr=pattr, bias_attr=battr)  # 创建Batch Normalization层
        if gn:
            self.gn = paddle.nn.GroupNorm(num_groups=groups, num_channels=filters, weight_attr=pattr, bias_attr=battr)  # 创建Group Normalization层
        if af:
            self.af = True  # 标记使用Affine Channel层
            self.scale = fluid.layers.create_parameter(
                shape=[filters],  # Affine Channel层的scale参数形状
                dtype='float32',  # 数据类型
                attr=pattr,
                default_initializer=Constant(1.))  # 默认初始化值为1
            self.offset = fluid.layers.create_parameter(
                shape=[filters],  # Affine Channel层的offset参数形状
                dtype='float32',  # 数据类型
                attr=battr,
                default_initializer=Constant(0.))  # 默认初始化值为0

        # 创建激活函数
        self.act = None  # 初始化激活函数为None
        if act == 'relu':
            self.act = paddle.nn.ReLU()  # 使用ReLU激活函数
        elif act == 'leaky':
            self.act = paddle.nn.LeakyReLU(0.1)  # 使用LeakyReLU激活函数
        elif act == 'mish':
            self.act = Mish()  # 使用Mish激活函数
        elif act is None:
            pass  # 不使用激活函数
        else:
            raise NotImplementedError("Activation \'{}\' is not implemented.".format(act))  # 抛出未实现的激活函数异常

    def freeze(self):
        if self.conv is not None:
            if self.conv.weight is not None:
                self.conv.weight.stop_gradient = True  # 冻结卷积权重
            if self.conv.bias is not None:
                self.conv.bias.stop_gradient = True  # 冻结卷积偏置
        if self.dcn_param is not None:
            self.dcn_param.stop_gradient = True  # 冻结可变形卷积参数
        if self.bn is not None:
            self.bn.weight.stop_gradient = True  # 冻结Batch Normalization权重
            self.bn.bias.stop_gradient = True  # 冻结Batch Normalization偏置
        if self.gn is not None:
            self.gn.weight.stop_gradient = True  # 冻结Group Normalization权重
            self.gn.bias.stop_gradient = True  # 冻结Group Normalization偏置
        if self.af is not None:
            self.scale.stop_gradient = True  # 冻结Affine Channel的scale参数
            self.offset.stop_gradient = True  # 冻结Affine Channel的offset参数

    def forward(self, x):
        if self.use_dcn:
            offset_mask = self.conv(x)  # 计算偏移和掩码
            offset = offset_mask[:, :self.filter_size**2 * 2, :, :]  # 提取偏移
            mask = offset_mask[:, self.filter_size**2 * 2:, :, :]  # 提取掩码
            mask = fluid.layers.sigmoid(mask)  # 对掩码使用Sigmoid激活
            x = deformable_conv(input=x, offset=offset, mask=mask,
                                num_filters=self.filters,  # 可变形卷积的输出通道数
                                filter_size=self.filter_size,  # 卷积核尺寸
                                stride=self.stride,  # 卷积步幅
                                padding=self.padding,  # 填充
                                groups=1,  # 组数
                                deformable_groups=1,  # 可变形组数
                                im2col_step=1,  # im2col步长
                                filter_param=self.dcn_param,  # 可变形卷积的过滤器参数
                                bias_attr=False)  # 不使用偏置
        else:
            x = self.conv(x)  # 卷积操作
        if self.bn:
            x = self.bn(x)  # Batch Normalization
        if self.gn:
            x = self.gn(x)  # Group Normalization
        if self.af:
            x = fluid.layers.affine_channel(x, scale=self.scale, bias=self.offset, act=None)  # Affine Channel
        if self.act:
            x = self.act(x)  # 激活函数
        return x  # 返回结果


class CoordConv(paddle.nn.Layer):
    def __init__(self, coord_conv=True):
        super(CoordConv, self).__init__()
        self.coord_conv = coord_conv

    def __call__(self, input):
        if not self.coord_conv:
            return input
        b = input.shape[0]
        h = input.shape[2]
        w = input.shape[3]
        x_range = L.range(0, w, 1., dtype='float32') / (w - 1) * 2.0 - 1
        y_range = L.range(0, h, 1., dtype='float32') / (h - 1) * 2.0 - 1
        # x_range = paddle.to_tensor(x_range, place=input.place)
        # y_range = paddle.to_tensor(y_range, place=input.place)
        x_range = L.reshape(x_range, (1, 1, 1, -1))  # [1, 1, 1, w]
        y_range = L.reshape(y_range, (1, 1, -1, 1))  # [1, 1, h, 1]
        x_range = L.expand(x_range, [b, 1, h, 1])  # [b, 1, h, w]
        y_range = L.expand(y_range, [b, 1, 1, w])  # [b, 1, h, w]
        offset = L.concat([input, x_range, y_range], axis=1)
        return offset


class SPP(paddle.nn.Layer):
    def __init__(self, seq='asc'):
        super(SPP, self).__init__()
        assert seq in ['desc', 'asc']
        self.seq = seq
        self.max_pool1 = paddle.nn.MaxPool2D(kernel_size=5, stride=1, padding=2)
        self.max_pool2 = paddle.nn.MaxPool2D(kernel_size=9, stride=1, padding=4)
        self.max_pool3 = paddle.nn.MaxPool2D(kernel_size=13, stride=1, padding=6)

    def __call__(self, x):
        x_1 = x
        x_2 = self.max_pool1(x)
        x_3 = self.max_pool2(x)
        x_4 = self.max_pool3(x)
        if self.seq == 'desc':
            out = L.concat([x_4, x_3, x_2, x_1], axis=1)
        else:
            out = L.concat([x_1, x_2, x_3, x_4], axis=1)
        return out


class DropBlock(paddle.nn.Layer):
    def __init__(self,
                 block_size=3,
                 keep_prob=0.9,
                 is_test=False):
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.is_test = is_test

    def __call__(self, input):
        if self.is_test:
            return input

        def CalculateGamma(input, block_size, keep_prob):
            input_shape = fluid.layers.shape(input)
            feat_shape_tmp = fluid.layers.slice(input_shape, [0], [3], [4])
            feat_shape_tmp = fluid.layers.cast(feat_shape_tmp, dtype="float32")
            feat_shape_t = fluid.layers.reshape(feat_shape_tmp, [1, 1, 1, 1])
            feat_area = fluid.layers.pow(feat_shape_t, factor=2)

            block_shape_t = fluid.layers.fill_constant(
                shape=[1, 1, 1, 1], value=block_size, dtype='float32')
            block_area = fluid.layers.pow(block_shape_t, factor=2)

            useful_shape_t = feat_shape_t - block_shape_t + 1
            useful_area = fluid.layers.pow(useful_shape_t, factor=2)

            upper_t = feat_area * (1 - keep_prob)
            bottom_t = block_area * useful_area
            output = upper_t / bottom_t
            return output

        gamma = CalculateGamma(input, block_size=self.block_size, keep_prob=self.keep_prob)
        input_shape = fluid.layers.shape(input)
        p = fluid.layers.expand_as(gamma, input)

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

        output = input * mask * elem_numel_m / elem_sum_m
        return output



