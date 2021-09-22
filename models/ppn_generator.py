from copy import deepcopy
from effects.xdog import XDoGEffect

import torch
from segmentation_models_pytorch import Unet
from segmentation_models_pytorch.base import SegmentationModel, SegmentationHead, ClassificationHead
from segmentation_models_pytorch.encoders import resnet_encoders
from segmentation_models_pytorch.encoders._base import EncoderMixin
from segmentation_models_pytorch.unet.decoder import UnetDecoder
from torch import nn
from torch.utils import model_zoo
from torchvision.models import ResNet

from effects.identity import IdentityEffect
from effects.structure_tensor import StructureTensorEffect
from effects.watercolor import WatercolorEffect
from helpers.apply_visual_effect import ApplyVisualEffect, xdog_params
from helpers.visual_parameter_def import portrait_preset, coffee_preset


class ResNetWithoutBatchNormEncoder(ResNet, EncoderMixin):
    def __init__(self, weights, in_channels, depth=5):
        parameters = deepcopy(resnet_encoders["resnet50"]["params"])
        self._out_channels = parameters["out_channels"]
        del parameters["out_channels"]

        super().__init__(norm_layer=nn.Identity, **parameters)
        self._depth = depth
        self._in_channels = 3

        del self.fc
        del self.avgpool

        self.relu = nn.LeakyReLU()
        self.layer1.relu = nn.LeakyReLU()
        self.layer2.relu = nn.LeakyReLU()
        self.layer3.relu = nn.LeakyReLU()
        self.layer4.relu = nn.LeakyReLU()

        if weights is not None:
            settings = resnet_encoders["resnet50"]["pretrained_settings"][weights]
            self.load_state_dict(model_zoo.load_url(settings["url"]))

        self.set_in_channels(in_channels)

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("fc.bias")
        state_dict.pop("fc.weight")

        for n in list(state_dict.keys()):
            if "bn" in n or "running" in n or "0.downsample.1" in n:  # remove all batch norm
                del state_dict[n]

        super().load_state_dict(state_dict, **kwargs)


class UnetWithoutBatchNorm(SegmentationModel):
    def __init__(
            self,
            encoder_depth: int = 5,
            encoder_weights="imagenet",
            decoder_use_batchnorm: bool = False,
            decoder_channels=(256, 128, 64, 32, 16),
            decoder_attention_type=None,
            in_channels: int = 3,
            classes: int = 1,
            activation="identity",
            aux_params=None,
    ):
        super().__init__()

        self.encoder = ResNetWithoutBatchNormEncoder(
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "u-{}".format("resnet_without_batchnorm")
        self.initialize()


class OurPPNGenerator(nn.Module):
    def __init__(self, unet_architecture, conv_net, effect, **kwargs):
        super().__init__()
        self.convnet_g = conv_net
        param_names = xdog_params

        if unet_architecture == "classic":
            self.unet = Unet(classes=len(param_names),
                             encoder_weights="imagenet", activation="identity",
                             encoder_name="resnet50")  # decoder_attention_type="scse"
        elif unet_architecture == "random":
            self.unet = Unet(classes=len(param_names),
                             encoder_weights=None, activation="identity",
                             encoder_name="resnet50")
        elif unet_architecture == "no_bn":
            self.unet = UnetWithoutBatchNorm(classes=len(param_names),
                                             decoder_attention_type="scse")
        elif unet_architecture == "none":
            self.unet = None
            param_names = []  # use default preset
        else:
            raise ValueError("architecture found")

        if effect == "xdog":
            self.apply_visual_effect = ApplyVisualEffect(param_names=param_names)
        else:
            raise ValueError("effect not found")

    def forward(self, x):
        x = self.ppn_part_forward(x)
        return self.conv_part_forward(x)

    def conv_part_forward(self, x):
        return self.convnet_g(x)

    def ppn_part_forward(self, x):
        predicted_param = self.predict_parameters(x)

        x = (x * 0.5) + 0.5
        x = self.apply_visual_effect(x, predicted_param)
        return (x - 0.5) / 0.5

    def predict_parameters(self, x):
        if self.unet is None:
            return None

        parameter_prediction = self.unet(x)
        return torch.tanh(parameter_prediction) * 0.5
