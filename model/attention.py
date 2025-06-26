# From https://github.com/xiaomingnio/pp_liteseg_pytorch/blob/main/UAFM.py#L6

import torch
import torch.nn as nn
import torch.nn.functional as F


def avg_reduce_channel(x):
    # Reduce channel by avg
    # Return cat([avg_ch_0, avg_ch_1, ...])
    if not isinstance(x, (list, tuple)):
        return torch.mean(x, dim=1, keepdim=True)
    elif len(x) == 1:
        return torch.mean(x[0], dim=1, keepdim=True)
    else:
        res = []
        for xi in x:
            res.append(torch.mean(xi, dim=1, keepdim=True))
        return torch.cat(res, dim=1)


def avg_reduce_hw(x):
    # Reduce hw by avg
    # Return cat([avg_pool_0, avg_pool_1, ...])
    if not isinstance(x, (list, tuple)):
        return F.adaptive_avg_pool2d(x, 1)
    elif len(x) == 1:
        return F.adaptive_avg_pool2d(x[0], 1)
    else:
        res = []
        for xi in x:
            res.append(F.adaptive_avg_pool2d(xi, 1))
        return torch.cat(res, dim=1)


def avg_max_reduce_channel_helper(x, use_concat=True):
    # Reduce hw by avg and max, only support single input
    assert not isinstance(x, (list, tuple))
    mean_value = torch.mean(x, dim=1, keepdim=True)
    max_value = torch.max(x, dim=1, keepdim=True)[0]
    # print("mean_value: ", mean_value)
    # print("max_value: ", max_value)

    if use_concat:
        res = torch.cat([mean_value, max_value], dim=1)
    else:
        res = [mean_value, max_value]
    return res


def avg_max_reduce_channel(x):
    # Reduce hw by avg and max
    # Return cat([avg_ch_0, max_ch_0, avg_ch_1, max_ch_1, ...])
    if not isinstance(x, (list, tuple)):
        return avg_max_reduce_channel_helper(x)
    elif len(x) == 1:
        return avg_max_reduce_channel_helper(x[0])
    else:
        res = []
        for xi in x:
            res.extend(avg_max_reduce_channel_helper(xi, False))
        return torch.cat(res, dim=1)


def avg_max_reduce_hw_helper(x, is_training, use_concat=True):
    assert not isinstance(x, (list, tuple))
    avg_pool = F.adaptive_avg_pool2d(x, 1)
    # TODO(pjc): when dim=[2, 3], the paddle.max api has bug for training.
    if is_training:
        max_pool = F.adaptive_max_pool2d(x, 1)
    else:
        max_pool = F.adaptive_max_pool2d(x, 1)

    if use_concat:
        res = torch.cat([avg_pool, max_pool], dim=1)
    else:
        res = [avg_pool, max_pool]
    return res


def avg_max_reduce_hw(x, is_training):
    # Reduce hw by avg and max
    # Return cat([avg_pool_0, avg_pool_1, ..., max_pool_0, max_pool_1, ...])
    if not isinstance(x, (list, tuple)):
        return avg_max_reduce_hw_helper(x, is_training)
    elif len(x) == 1:
        return avg_max_reduce_hw_helper(x[0], is_training)
    else:
        res_avg = []
        res_max = []
        for xi in x:
            avg, max = avg_max_reduce_hw_helper(xi, is_training, False)
            res_avg.append(avg)
            res_max.append(max)
        res = res_avg + res_max
        return torch.cat(res, dim=1)


class ConvINReLU(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1,norm_type="BN"):
        super(ConvINReLU, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=kernel // 2, bias=False)
        # self.bn = nn.BatchNorm2d(out_planes)

        if norm_type == "IN":
            self.bn = nn.InstanceNorm2d(out_planes)
        elif norm_type == "BN":
            self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


class ConvIN(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1, norm_type="BN"):
        super(ConvIN, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=kernel // 2, bias=False)
        # self.bn = nn.BatchNorm2d(out_planes)
        if norm_type == "IN":
            self.bn = nn.InstanceNorm2d(out_planes)
        elif norm_type == "BN":
            self.bn = nn.BatchNorm2d(out_planes)
        elif norm_type == "GN":
            self.bn = nn.GroupNorm(out_planes // 4, out_planes)
        else:
            raise Exception("not right norm_type")

    def forward(self, x):
        out = self.bn(self.conv(x))
        return out


class ConvINAct(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1, norm_type="BN", act_type="leakyrelu"):
        super(ConvINAct, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=kernel // 2, bias=False)
        # self.bn = nn.BatchNorm2d(out_planes)
        if norm_type == "IN":
            self.bn = nn.InstanceNorm2d(out_planes)
        elif norm_type == "BN":
            self.bn = nn.BatchNorm2d(out_planes)
        elif norm_type == "GN":
            self.bn = nn.GroupNorm(out_planes // 4, out_planes)
        else:
            raise Exception("not right norm_type")

        if act_type == "leakyrelu":
            self.act = nn.LeakyReLU(inplace=True)
        else:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # out = self.act(self.bn(self.conv(x)))
        x = self.conv(x)
        x = self.bn(x)
        out = self.act(x)
        return out


class UAFM(nn.Module):
    """
    The base of Unified Attention Fusion Module.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__()

        self.conv_x = ConvINReLU(x_ch, y_ch, kernel=ksize)
        # self.conv_y = ConvINReLU(y_ch, x_ch, kernel=ksize)
        self.conv_out = ConvINReLU(y_ch, out_ch, kernel=3)
        self.resize_mode = resize_mode
        # self.show = (x_ch,y_ch)
    def prepare(self, x, y):
        # print("****************")
        # print(x.size(),self.show)
        # print("****************")
        x = self.conv_x(x)
        # y = self.conv_y(y)
        y = F.interpolate(y, x.shape[2:], mode=self.resize_mode)
        return x, y

    def fuse(self, x, y):
        out = x + y
        out = self.conv_out(out)
        return out

    def forward(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        x, y = self.prepare(x, y)
        out = self.fuse(x, y)
        return out


class UAFM_ChAtten(UAFM):
    """
    The UAFM with channel attention, which uses mean and max values.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = nn.Sequential(
            ConvINAct(4 * y_ch, y_ch // 2, kernel=1, act_type="leakyrelu"),
            ConvIN(y_ch // 2, y_ch, kernel=1))

    def fuse(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        atten = avg_max_reduce_hw([x, y], self.training)
        atten = F.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out


class UAFM_ChAtten_S(UAFM):
    """
    The UAFM with channel attention, which uses mean values.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = nn.Sequential(
            ConvINAct(
                2 * y_ch,
                y_ch // 2,
                kernel=1,
                act_type="leakyrelu"),
            ConvIN(
                y_ch // 2, y_ch, kernel=1))

    def fuse(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        atten = avg_reduce_hw([x, y])
        atten = F.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out


class UAFM_SpAtten(UAFM):
    """
    The UAFM with spatial attention, which uses mean and max values.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = nn.Sequential(
            ConvINReLU(4, 2, kernel=3),
            ConvIN(2, 1, kernel=3))

    def fuse(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        atten = avg_max_reduce_channel([x, y])
        atten = F.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out


class UAFM_SpAtten_S(UAFM):
    """
    The UAFM with spatial attention, which uses mean values.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = nn.Sequential(
            ConvINReLU(2, 2, kernel=3),
            ConvIN(2, 1, kernel=3))

    def fuse(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        atten = avg_reduce_channel([x, y])
        atten = F.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out


class UAFM_CS(UAFM):
    """
    The UAFM with channel-spatial attention, which uses mean and max values.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten_s = nn.Sequential(
            ConvINReLU(4, 2, kernel=3),
            ConvIN(2, 1, kernel=3))

        self.conv_xy_atten_c = nn.Sequential(
            # ConvINAct(2 * (x_ch + y_ch), y_ch // 2, kernel=1, norm_type="GN",act_type="leakyrelu"),
            ConvINAct(4 * y_ch, y_ch // 2, kernel=1, norm_type="GN",act_type="leakyrelu"),
            # ConvINAct(4 * x_ch, 2 * x_ch, kernel=1, norm_type="GN", act_type="leakyrelu"),
            ConvIN(y_ch // 2, y_ch, kernel=1,norm_type="GN"))
            # ConvIN(2 * x_ch, x_ch, kernel=1, norm_type="GN"))

    def fuse(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        atten_s = avg_max_reduce_channel([x, y])
        atten_s = self.conv_xy_atten_s(atten_s)
        atten_c = avg_max_reduce_hw([x, y], self.training)
        atten_c = self.conv_xy_atten_c(atten_c)

        atten = F.sigmoid(atten_c * atten_s)
        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out
