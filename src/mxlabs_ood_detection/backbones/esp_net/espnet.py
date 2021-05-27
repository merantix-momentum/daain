"""
Source code for ESPNet. Heavily based on https://github.com/sacmehta/ESPNet/blob/master/test/Model.py.
Only the CustomESPNet was added as wrapper.
"""
import os

import torch
import torch.nn as nn

from mxlabs_ood_detection.backbones.esp_net.layers import BR, CBR, C, DilatedParllelResidualBlockB, DownSamplerB
from mxlabs_ood_detection.backbones.esp_net.layers import InputProjectionA


class ESPNetEncoder(nn.Module):
    """
    This class defines the ESPNet-C network in the paper
    """

    def __init__(self, classes: int = 20, p: int = 5, q: int = 3) -> None:
        """
        Args:
            classes (int): number of classes in the dataset. Default is 20 for the cityscape dataset
            p (int): depth multiplier
            q (int): depth multiplier
        """

        super().__init__()
        self.level1 = CBR(3, 16, 3, 2)
        self.sample1 = InputProjectionA(1)
        self.sample2 = InputProjectionA(2)

        self.b1 = BR(16 + 3)
        self.level2_0 = DownSamplerB(16 + 3, 64)

        self.level2 = nn.ModuleList()
        for i in range(p):
            self.level2.append(DilatedParllelResidualBlockB(64, 64))
        self.b2 = BR(128 + 3)

        self.level3_0 = DownSamplerB(128 + 3, 128)
        self.level3 = nn.ModuleList()
        for i in range(q):
            self.level3.append(DilatedParllelResidualBlockB(128, 128))
        self.b3 = BR(256)

        self.classifier = C(256, classes, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input RGB image
        Returns:
             output (torch.Tensor): transformed feature map with spatial dimensions 1/8th of the input image
        """

        output0 = self.level1(x)
        inp1 = self.sample1(x)
        inp2 = self.sample2(x)

        output0_cat = self.b1(torch.cat([output0, inp1], 1))
        output1_0 = self.level2_0(output0_cat)  # down-sampled

        for i, layer in enumerate(self.level2):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.b2(torch.cat([output1, output1_0, inp2], 1))

        output2_0 = self.level3_0(output1_cat)  # down-sampled
        for i, layer in enumerate(self.level3):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        output2_cat = self.b3(torch.cat([output2_0, output2], 1))

        output = self.classifier(output2_cat)

        return output


class ESPNet(nn.Module):
    """
    This class defines the ESPNet network
    """

    def __init__(self, classes: int = 20, p: int = 2, q: int = 3, encoder_file: str = None) -> None:
        """
        Args:
            classes (int): number of classes in the dataset. Default is 20 for the cityscape dataset
            p (int): depth multiplier
            q (int): depth multiplier
            encoder_file (str): pretrained encoder weights. Recall that we first trained the ESPNet-C and
                                then attached the RUM-based light weight decoder. See paper for more details.
        """

        super().__init__()
        self.encoder = ESPNetEncoder(classes, p, q)
        if encoder_file is not None:
            self.encoder.load_state_dict(torch.load(encoder_file))
            print("Encoder loaded!")

        # load the encoder modules
        self.modules = []
        for m in iter(self.encoder.children()):
            self.modules.append(m)

        # light-weight decoder
        self.level3_C = C(128 + 3, classes, 1, 1)
        self.br = nn.BatchNorm2d(classes, eps=1e-03)
        self.conv = CBR(16 + classes, classes, 3, 1)

        self.up_l3 = nn.Sequential(
            nn.ConvTranspose2d(classes, classes, 2, stride=2, padding=0, output_padding=0, bias=False)
        )
        self.combine_l2_l3 = nn.Sequential(
            BR(2 * classes), DilatedParllelResidualBlockB(2 * classes, classes, add=False)
        )

        self.up_l2 = nn.Sequential(
            nn.ConvTranspose2d(classes, classes, 2, stride=2, padding=0, output_padding=0, bias=False), BR(classes)
        )

        self.classifier = nn.ConvTranspose2d(classes, classes, 2, stride=2, padding=0, output_padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input RGB image
        Returns:
             output (torch.Tensor): transformed feature map
        """

        output0 = self.modules[0](x)
        inp1 = self.modules[1](x)
        inp2 = self.modules[2](x)

        output0_cat = self.modules[3](torch.cat([output0, inp1], 1))
        output1_0 = self.modules[4](output0_cat)  # down-sampled

        for i, layer in enumerate(self.modules[5]):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.modules[6](torch.cat([output1, output1_0, inp2], 1))

        output2_0 = self.modules[7](output1_cat)  # down-sampled
        for i, layer in enumerate(self.modules[8]):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        # concatenate for feature map width expansion
        output2_cat = self.modules[9](torch.cat([output2_0, output2], 1))

        output2_c = self.up_l3(self.br(self.modules[10](output2_cat)))  # RUM

        # project to C-dimensional space
        output1_c = self.level3_C(output1_cat)
        comb_l2_l3 = self.up_l2(self.combine_l2_l3(torch.cat([output1_c, output2_c], 1)))  # RUM

        concat_features = self.conv(torch.cat([comb_l2_l3, output0], 1))

        output = self.classifier(concat_features)

        return output


class CustomESPNet(nn.Module):
    """A thin wrapper around ESPNet
    a) to load our required configuration of p & q.
    b) that extracts only the first `num_classes` predictions as original implementation of ESPNet treats
       no-class as the 20th class.
    c) that normalizes the input image according to the ESPNet-specific constants.

    NOTE: ESPNet uses it own set of normalizing constants, which are applied to the unnormalized input image of range
    [0, 255], following which it is scaled to [0, 1]. Since our input image is already in [0, 1], we simply re-scale
    the mean and leave the std untouched, leading to  same effect.
    """

    MEAN = torch.FloatTensor([72.3923111, 82.90893555, 73.15840149]) / 255.0
    STD = torch.FloatTensor([45.3192215, 46.15289307, 44.91483307])

    def __init__(self, device: torch.device, num_classes: int, p: int = 2, q: int = 8) -> None:
        """
        Args:
            device (torch.device): torch device to init the model on, default is cpu
            num_classes (int): amount of classes of the output vector
            p (int): depth multiplier
            q (int): depth multiplier
        """

        assert q in [3, 5, 8], ValueError("Supported `q` values for ESPNet are {3, 5, 8}")
        super().__init__()
        # the num_classes handling is most disappointing. leads to many confusions. not changed for backward
        # compatibility reasons.
        self.esp_net = ESPNet(classes=num_classes + 1, p=p, q=q)
        self.esp_net.to(device)
        self.num_classes = num_classes
        self.p = p
        self.q = q

    @staticmethod
    def normalize(x: torch.Tensor, mean: list, std: list) -> torch.Tensor:
        """
        Normalize tensor with 3 channels (rgb image)
        Args:
             x (torch.Tensor): rgb image
             mean (list): mean of dataset in each image channels
             std (list): standard deviation of dataset in each image channels
        Returns:
            y (torch.Tensor): normalized input image
        """

        y = torch.zeros_like(x)
        for c in range(3):
            y[:, c] = (x[:, c] - mean[c]) / std[c]

        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input RGB image
        Returns:
             output (torch.Tensor): transformed feature map
        """

        x = self.normalize(x, self.MEAN, self.STD)
        output = self.esp_net(x)

        return output[:, : self.num_classes, :, :]

    @property
    def out_channels(self):
        return self.esp_net.classifier.out_channels - 1

    @out_channels.setter
    def out_channels(self, value):
        self.esp_net.classifier.out_channels = value

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.esp_net.load_state_dict(state_dict, strict)

    @staticmethod
    def get_weight_path(root: str, p: int, q: int, model_part: str = "decoder") -> str:
        """Returns the weight path. Note that `model_part` refers to the part fo the model.
        Options are `encoder` and `decoder`. `encoder` refers to only the encoder whereas `decoder` refers to the
        full model.
        """
        return os.path.join(root, "espnet", model_part, f"espnet_p_{p}_q_{q}.pth")

    def load_pretrained_weights(self, root):
        self.load_state_dict(
            torch.load(CustomESPNet.get_weight_path(root=root, p=self.p, q=self.q, model_part="decoder"))
        )

        return self
