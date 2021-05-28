import math
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils import model_zoo


class CBR(nn.Module):
    """
    This class defines the convolution layer with batch normalization and PReLU activation
    """

    def __init__(self, n_in: int, n_out: int, k_size: int, stride: int = 1) -> None:
        """
        Args:
            n_in (int): number of input channels
            n_out (int): number of output channels
            k_size (int): kernel size
            stride (int): stride rate for down-sampling. Default is 1
        """

        super().__init__()
        padding = int((k_size - 1) / 2)
        self.conv = nn.Conv2d(n_in, n_out, (k_size, k_size), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(n_out, eps=1e-03)
        self.act = nn.PReLU(n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input feature map
        Returns:
            output (torch.Tensor): transformed feature map
        """

        output = self.conv(x)
        output = self.bn(output)
        output = self.act(output)

        return output


class BR(nn.Module):
    """
    This class groups the batch normalization and PReLU activation
    """

    def __init__(self, n_out: int) -> None:
        """
        Args:
            n_out (int): output feature maps
        """

        super().__init__()
        self.bn = nn.BatchNorm2d(n_out, eps=1e-03)
        self.act = nn.PReLU(n_out)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): input feature map
        Returns:
            output (torch.Tensor): normalized and thresholded feature map
        """

        output = self.bn(x)
        output = self.act(output)

        return output


class CB(nn.Module):
    """
       This class groups the convolution and batch normalization
    """

    def __init__(self, n_in: int, n_out: int, k_size: int, stride: int = 1) -> None:
        """
        Args:
            n_in (int): number of input channels
            n_out (int): number of output channels
            k_size (int): kernel size
            stride (int): optional stride for down-sampling. Default 1
        """

        super().__init__()
        padding = int((k_size - 1) / 2)
        self.conv = nn.Conv2d(n_in, n_out, (k_size, k_size), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(n_out, eps=1e-03)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input feature map
        Returns:
            output (torch.Tensor): transformed feature map
        """

        output = self.conv(x)
        output = self.bn(output)

        return output


class C(nn.Module):
    """
    This class is for a convolutional layer.
    """

    def __init__(self, n_in: int, n_out: int, k_size: int, stride: int = 1) -> None:
        """
        Args:
            n_in (int): number of input channels
            n_out (int): number of output channels
            k_size (int): kernel size
            stride (int): optional stride for down-sampling. Default 1
        """

        super().__init__()
        padding = int((k_size - 1) / 2)
        self.conv = nn.Conv2d(n_in, n_out, (k_size, k_size), stride=stride, padding=(padding, padding), bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input feature map
        Returns:
            output (torch.Tensor): transformed feature map
        """

        output = self.conv(x)

        return output


class CDilated(nn.Module):
    """
    This class defines the dilated convolution.
    """

    def __init__(self, n_in: int, n_out: int, k_size: int, stride: int = 1, d: int = 1) -> None:
        """
        Args:
            n_in (int): number of input channels
            n_out (int): number of output channels
            k_size (int): kernel size
            stride (int): optional stride for down-sampling. Default 1
            d (int): optional dilation rate. Default 1
        """

        super().__init__()
        padding = int((k_size - 1) / 2) * d
        self.conv = nn.Conv2d(
            n_in, n_out, (k_size, k_size), stride=stride, padding=(padding, padding), bias=False, dilation=d
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): input feature map
        Returns:
            output (torch.Tensor): transformed feature map
        """

        output = self.conv(x)

        return output


class DownSamplerB(nn.Module):
    def __init__(self, n_in: int, n_out: int) -> None:
        """
        Args:
            n_in (int): number of input channels
            n_out (int): number of output channels
        """

        super().__init__()
        n = int(n_out / 5)
        n1 = n_out - 4 * n
        self.c1 = C(n_in, n, 3, 2)
        self.d1 = CDilated(n, n1, 3, 1, 1)
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 4)
        self.d8 = CDilated(n, n, 3, 1, 8)
        self.d16 = CDilated(n, n, 3, 1, 16)
        self.bn = nn.BatchNorm2d(n_out, eps=1e-3)
        self.act = nn.PReLU(n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input feature map
        Returns:
            output (torch.Tensor): transformed feature map
        """

        output = self.c1(x)
        d1 = self.d1(output)
        d2 = self.d2(output)
        d4 = self.d4(output)
        d8 = self.d8(output)
        d16 = self.d16(output)

        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        combine = torch.cat([d1, add1, add2, add3, add4], 1)

        output = self.bn(combine)
        output = self.act(output)

        return output


class DilatedParllelResidualBlockB(nn.Module):
    """
    This class defines the ESP block, which is based on the following principle
    Reduce ---> Split ---> Transform --> Merge
    """

    def __init__(self, n_in: int, n_out: int, add: bool = True) -> None:
        """
        Args:
            n_in (int): number of input channels
            n_out (int): number of output channels
            add (bool): if true, add a residual connection through identity operation. You can use
                        projection too as in ResNet paper, but we avoid to use it if the dimensions
                        are not the same because we do not want to increase the module complexity
        """

        super().__init__()
        n = int(n_out / 5)
        n1 = n_out - 4 * n
        self.c1 = C(n_in, n, 1, 1)
        self.d1 = CDilated(n, n1, 3, 1, 1)  # dilation rate of 2^0
        self.d2 = CDilated(n, n, 3, 1, 2)  # dilation rate of 2^1
        self.d4 = CDilated(n, n, 3, 1, 4)  # dilation rate of 2^2
        self.d8 = CDilated(n, n, 3, 1, 8)  # dilation rate of 2^3
        self.d16 = CDilated(n, n, 3, 1, 16)  # dilation rate of 2^4
        self.bn = BR(n_out)
        self.add = add

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input feature map
        Returns:
            output (torch.Tensor): transformed feature map
        """

        output = self.c1(x)

        d1 = self.d1(output)
        d2 = self.d2(output)
        d4 = self.d4(output)
        d8 = self.d8(output)
        d16 = self.d16(output)

        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        combine = torch.cat([d1, add1, add2, add3, add4], 1)

        if self.add:
            combine = x + combine

        output = self.bn(combine)

        return output


class _AtrousSpatialPyramidPoolingModule(nn.Module):
    """
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    """

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=(6, 12, 18)):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise "output stride of {} not supported".format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                Norm2d(reduction_dim),
                nn.ReLU(inplace=True),
            )
        )
        # other rates
        for r in rates:
            self.features.append(
                nn.Sequential(
                    nn.Conv2d(in_dim, reduction_dim, kernel_size=3, dilation=r, padding=r, bias=False),
                    Norm2d(reduction_dim),
                    nn.ReLU(inplace=True),
                )
            )
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False), Norm2d(reduction_dim), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = Upsample(img_features, x_size[2:])
        out = img_features

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out


class InputProjectionA(nn.Module):
    """
    This class projects the input image to the same spatial dimensions as the feature map.
    For example, if the input image is 512 x512 x3 and spatial dimensions of feature map size are 56x56xF, then
    this class will generate an output of 56x56x3
    """

    def __init__(self, sampling_times: int) -> None:
        """
        Args:
            sampling_times (int): the rate at which one wants to down-sample the image
        """

        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, sampling_times):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input RGB Image
        Returns:
            output (torch.Tensor): down-sampled image (pyramid-based approach)
        """

        for pool in self.pool:
            x = pool(x)

        return x


def Norm2d(in_channels):
    """
    Custom Norm Function to allow flexible switching
    """

    normalization_layer = nn.BatchNorm2d(in_channels)
    return normalization_layer


def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """

    return nn.functional.interpolate(x, size=size, mode="bilinear", align_corners=True)


def initialize_weights(*models):
    """
    Initialize Model Weights
    """

    for model in models:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


def initialize_pretrained_model(model, num_classes, settings):
    """
    Initialize Pretrain Model Information,
    Download weights, load weights, set variables
    """

    assert num_classes == settings["num_classes"], "num_classes should be {}, but is {}".format(
        settings["num_classes"], num_classes
    )
    weights = model_zoo.load_url(settings["url"])
    model.load_state_dict(weights)
    model.input_space = settings["input_space"]
    model.input_size = settings["input_size"]
    model.input_range = settings["input_range"]
    model.mean = settings["mean"]
    model.std = settings["std"]


class SEModule(nn.Module):
    """
    Squeeze Excitation Module.

    Code adapted from:
    https://github.com/Cadene/pretrained-models.pytorch

    BSD 3-Clause License

    Copyright (c) 2017, Remi Cadene
    All rights reserved.
    """

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class SEResNetBottleneckBase(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.

    Code adapted from:
    https://github.com/Cadene/pretrained-models.pytorch

    BSD 3-Clause License

    Copyright (c) 2017, Remi Cadene
    All rights reserved.
    """

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class SEBottleneck(SEResNetBottleneckBase):
    """
    Bottleneck for SENet154.

    Code adapted from:
    https://github.com/Cadene/pretrained-models.pytorch

    BSD 3-Clause License

    Copyright (c) 2017, Remi Cadene
    All rights reserved.
    """

    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = Norm2d(planes * 2)
        self.conv2 = nn.Conv2d(
            planes * 2, planes * 4, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False
        )
        self.bn2 = Norm2d(planes * 4)
        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1, bias=False)
        self.bn3 = Norm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNetBottleneck(SEResNetBottleneckBase):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).

    Code adapted from:
    https://github.com/Cadene/pretrained-models.pytorch

    BSD 3-Clause License

    Copyright (c) 2017, Remi Cadene
    All rights reserved.
    """

    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False, stride=stride)
        self.bn1 = Norm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, groups=groups, bias=False)
        self.bn2 = Norm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = Norm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNeXtBottleneck(SEResNetBottleneckBase):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.

    Code adapted from:
    https://github.com/Cadene/pretrained-models.pytorch

    BSD 3-Clause License

    Copyright (c) 2017, Remi Cadene
    All rights reserved.
    """

    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False, stride=1)
        self.bn1 = Norm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = Norm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = Norm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


def bnrelu(channels: int) -> nn.Sequential:
    """
    Single Layer BN and Relu
    """
    return nn.Sequential(Norm2d(channels), nn.ReLU(inplace=True))


class GlobalAvgPool2d(nn.Module):
    """
    Global average pooling over the input's spatial dimensions.

    Code adapted from:
    https://github.com/mapillary/inplace_abn/

    BSD 3-Clause License

    Copyright (c) 2017, mapillary
    All rights reserved.
    """

    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    @staticmethod
    def forward(inputs: torch.Tensor) -> torch.Tensor:
        in_size = inputs.size()
        return inputs.view((in_size[0], in_size[1], -1)).mean(dim=2)


class IdentityResidualBlock(nn.Module):
    """
    Identity Residual Block for WideResnet.

    Code adapted from:
    https://github.com/mapillary/inplace_abn/

    BSD 3-Clause License

    Copyright (c) 2017, mapillary
    All rights reserved.
    """

    def __init__(
        self,
        in_channels: int,
        channels: list,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        norm_act: callable = bnrelu,
        dropout: callable = None,
        dist_bn: bool = False,
    ) -> None:
        """Configurable identity-mapping residual block

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        channels : list of int
            Number of channels in the internal feature maps.
            Can either have two or three elements: if three construct
            a residual block with two `3 x 3` convolutions,
            otherwise construct a bottleneck block with `1 x 1`, then
            `3 x 3` then `1 x 1` convolutions.
        stride : int
            Stride of the first `3 x 3` convolution
        dilation : int
            Dilation to apply to the `3 x 3` convolutions.
        groups : int
            Number of convolution groups.
            This is used to create ResNeXt-style blocks and is only compatible with
            bottleneck blocks.
        norm_act : callable
            Function to create normalization / activation Module.
        dropout: callable
            Function to create Dropout Module.
        dist_bn: Boolean
            A variable to enable or disable use of distributed BN
        """

        super(IdentityResidualBlock, self).__init__()
        self.dist_bn = dist_bn

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn

        # Check parameters for inconsistencies
        if len(channels) != 2 and len(channels) != 3:
            raise ValueError("channels must contain either two or three values")
        if len(channels) == 2 and groups != 1:
            raise ValueError("groups > 1 are only valid if len(channels) == 3")

        is_bottleneck = len(channels) == 3
        need_proj_conv = stride != 1 or in_channels != channels[-1]

        self.bn1 = norm_act(in_channels)
        if not is_bottleneck:
            layers = [
                (
                    "conv1",
                    nn.Conv2d(
                        in_channels, channels[0], 3, stride=stride, padding=dilation, bias=False, dilation=dilation
                    ),
                ),
                ("bn2", norm_act(channels[0])),
                (
                    "conv2",
                    nn.Conv2d(channels[0], channels[1], 3, stride=1, padding=dilation, bias=False, dilation=dilation),
                ),
            ]
            if dropout is not None:
                layers = layers[0:2] + [("dropout", dropout())] + layers[2:]
        else:
            layers = [
                ("conv1", nn.Conv2d(in_channels, channels[0], 1, stride=stride, padding=0, bias=False)),
                ("bn2", norm_act(channels[0])),
                (
                    "conv2",
                    nn.Conv2d(
                        channels[0],
                        channels[1],
                        3,
                        stride=1,
                        padding=dilation,
                        bias=False,
                        groups=groups,
                        dilation=dilation,
                    ),
                ),
                ("bn3", norm_act(channels[1])),
                ("conv3", nn.Conv2d(channels[1], channels[2], 1, stride=1, padding=0, bias=False)),
            ]
            if dropout is not None:
                layers = layers[0:4] + [("dropout", dropout())] + layers[4:]
        self.convs = nn.Sequential(OrderedDict(layers))

        if need_proj_conv:
            self.proj_conv = nn.Conv2d(in_channels, channels[-1], 1, stride=stride, padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This is the standard forward function for non-distributed batch norm
        """

        if hasattr(self, "proj_conv"):
            bn1 = self.bn1(x)
            shortcut = self.proj_conv(bn1)
        else:
            shortcut = x.clone()
            bn1 = self.bn1(x)

        out = self.convs(bn1)
        out.add_(shortcut)

        return out


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class ResNetBasicBlock(nn.Module):
    """
    Basic Block for Resnet
    """

    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: callable = None):
        super(ResNetBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = Norm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = Norm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetBottleneck(nn.Module):
    """
    Bottleneck Layer for Resnet
    """

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = Norm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = Norm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = Norm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
