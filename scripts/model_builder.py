#!/usr/bin/env python3
"""
Functionality for building models as nn.Module class.

TODO:
    = Implemented intermediate fusion support for mix transformers
    - Implement late fusion
"""
# Import statements
import torch
import torch.nn as nn
import torchvision
import segmentation_models_pytorch as smp
import re
from copy import deepcopy


# Create weight generator for Kim gated fusion module
class kim_wg(nn.Module):
    def __init__(self, channels):
        super(kim_wg, self).__init__()
        self.weight_generator = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(3, 3),
                padding=1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.weight_generator(x)


# Change number of channels using a 1x1x1 conv layer
class conv_channel_changer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_channel_changer, self).__init__()
        self.changer = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.changer(x)


# Squeeze-and-excite then half the number of channels
class se_halver(nn.Module):
    def __init__(self, channels):
        super(se_halver, self).__init__()
        self.halver = nn.Sequential(
            torchvision.ops.SqueezeExcitation(
                input_channels=channels,
                squeeze_channels=channels // 4,
            ),
            nn.Conv2d(
                channels,
                int(channels / (4 / 3)),
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(int(channels / (4 / 3))),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                int(channels / (4 / 3)),
                channels // 2,
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.halver(x)


# Define nn.Module class for model
class Segmenter(nn.Module):
    def __init__(
        self,
        model_name,
        encoder_name,
        encoder_weights,
        n_channels,
        n_classes,
        decoder_attention,
        encoder_freeze,
        fusion=None,
        n_channels_med1=3,
        n_channels_med2=2,
    ):
        """Creates a semantic segmentation model as PyTorch nn.Module class.

        Creates a segmentation model with an encoder. Encoder can be pretrained and frozen.
        Names of networks and weights should follow the Segmentation Models Pytorch (smp) API:
            https://github.com/qubvel/segmentation_models.pytorch
        Or the torchvision API:
            https://pytorch.org/vision/stable/models.html#listing-and-retrieving-available-models
        Only supports the following torchvision models:
            -

        Intermediate fusion is performed by copying the encoder, feeding the input of different
        modalities in the different encoders and concatenating the resulting feature maps before
        the decoder.
        n_channels_med1 and n_channels_med2 are only used with intermediate fusion. They are the input
        channels of the first and second encoder, respectively.
        n_channels should always be equal to or larger than both n_channels_med1 and n_channels_med2.

        Args:
            model_name (str): Name of segmentation model as in smp or torchvision.models.
            encoder_name (str): Name of encoder for segmentation mdoel as in smp.
            encoder_weights (str): Weights for encoder pretraining as in smp. Usually "imagenet".
            n_channels (int): Number of input channels.
            n_classes (int): Number of output classes for segmentation.
            decoder_attention (str): Attention type for decoder. Available for Unet: None or "scse".
            encoder_freeze (bool): If true, freezes parameters of the encoder.
            fusion (str, optional): Fusion of RGB with fluor data: early, intermediate, or late. Defaults to None.
            n_channels_med1(int, optional:) Number of input channels for first encoder in intermediate fusion. Defaults to 3.
            n_channels_med2(int, optional:) Number of input channels for second encoder in intermediate fusion. Defaults to 2.
        """
        super(Segmenter, self).__init__()
        self.fusion = fusion
        self.n_channels_med1 = n_channels_med1
        self.n_channels_med2 = n_channels_med2

        # Set backbone, allow for different models and pretraining
        # Use backbone from segmentation models pytorch (smp)
        if not model_name.startswith(("deeplab", "fcn", "lraspp")):
            # Format all inputs for eval
            model_call = f"smp.{model_name}"
            enc_call = f'encoder_name="{encoder_name}"'
            weights_call = f'encoder_weights="{encoder_weights}"'
            channel_call = f"in_channels={n_channels}"
            class_call = f"classes={n_classes}"
            dec_att_call = f'decoder_attention_type="{decoder_attention}"'

            # Join all arguments and replace all "None" with None
            if decoder_attention is not None:
                args_call = ", ".join(
                    [enc_call, channel_call, class_call, weights_call, dec_att_call]
                )
            else:
                args_call = ", ".join(
                    [enc_call, class_call, channel_call, weights_call]
                )
            args_call = re.sub(r'"None"', "None", args_call)

            # Construct function call for eval and create model
            model_call = f"{model_call}({args_call})"
            self.model = eval(model_call)

            # Change network architecture for intermediate or late fusion
            if self.fusion == "intermediate":
                if not encoder_name.startswith("mit"):
                    # Extract encoder and deepcopy encoder
                    self.encoder1 = self.model.encoder
                    self.encoder2 = deepcopy(self.encoder1)

                    # Tweak number of channels of encoders if not correct already
                    first_conv1 = self.encoder1.conv1
                    first_conv2 = self.encoder2.conv1
                    if first_conv1.weight.shape[1] != self.n_channels_med1:
                        first_conv1.weight = nn.Parameter(
                            first_conv1.weight[:, : self.n_channels_med1, :, :]
                        )
                    if first_conv2.weight.shape[1] != self.n_channels_med2:
                        first_conv2.weight = nn.Parameter(
                            first_conv2.weight[:, : self.n_channels_med2, :, :]
                        )
                else:
                    raise (
                        "[INFO] Mixed vision transformer does not support intermediate fusion!"
                    )

                # Initialize intermediate fusion modules
                if encoder_name == "timm-res2net50_14w_8s":
                    kim_gated_fusion = False
                    if kim_gated_fusion:
                        pass
                    else:
                        # For first feature map at original number of channels
                        self.halver0 = conv_channel_changer(
                            in_channels=self.n_channels_med1 + self.n_channels_med2,
                            out_channels=self.n_channels_med1,
                        )

                        # For subsequent feature maps
                        self.halver1 = se_halver(channels=128)
                        self.halver2 = se_halver(channels=512)
                        self.halver3 = se_halver(channels=1024)
                        self.halver4 = se_halver(channels=2048)
                        self.halver5 = se_halver(channels=4096)

                        # Compile all halvers to be used for each corresponding feature map
                        self.halvers = nn.ModuleList(
                            [
                                self.halver0,
                                self.halver1,
                                self.halver2,
                                self.halver3,
                                self.halver4,
                                self.halver5,
                            ]
                        )
                else:
                    raise (
                        "[INFO] This encoder does not yet support intermediate fusion!"
                    )

                # Separate decoder and segmentation head from encoders
                self.decoder = self.model.decoder
                self.seghead = self.model.segmentation_head

            # Freeze weights in encoder if desired
            if (self.fusion == None) or (self.fusion == "early"):
                for param in self.model.encoder.parameters():
                    param.requires_grad = not encoder_freeze
            if (self.fusion == "intermediate") or (self.fusion == "late"):
                for param in self.encoder1.parameters():
                    param.requires_grad = not encoder_freeze
                for param in self.encoder2.parameters():
                    param.requires_grad = not encoder_freeze

        # Use backbone from torchvision
        else:
            pass

    def forward(self, x):
        if (self.fusion == None) or (self.fusion == "early"):
            pred_logits = self.model(x)
        if self.fusion == "intermediate":
            # Separate different inputs
            input1 = x[:, : self.n_channels_med1, :, :]
            input2 = x[:, : self.n_channels_med2, :, :]

            # Run inputs through encoders
            features1 = self.encoder1(input1)
            features2 = self.encoder2(input2)

            # Loop through lists of feature maps
            features = []
            for index, feature12 in enumerate(zip(features1, features2)):
                # Concatenate feature maps of different encoders
                feature1, feature2 = feature12
                feature_cat = torch.concatenate((feature1, feature2), dim=1)

                # Squeeze-and-excite and halve each feature map of different encoders
                halver = self.halvers[index]
                feature = halver(feature_cat)
                features.append(feature)

            # Create segmentation predictions
            decoded = self.decoder(*features)
            pred_logits = self.seghead(decoded)

        return pred_logits


class TipburnClassifier(nn.Module):
    def __init__(
        self,
        bb_name,
        n_classes,
        n_channels=3,
        bb_weights="IMAGENET1K_V1",
        bb_freeze=True,
    ):
        """Creates tipburn classifier as PyTorch nn.Module class.

        Creates a CNN with a backbone. Backbone can be pretrained and frozen.
        Names of networks and weights should follow the torchvision API:
            https://pytorch.org/vision/stable/models.html#listing-and-retrieving-available-models
        Only supports the following backbones:
            resnet50, wide_resnet50_2

        Args:
            bb_name (str): Name of backbone network as in torchvision.models.
            n_classes (int): Numbers of classes to predict.
            n_channels (int, optional): Number of input channels from data. Defaults to 3.
            bb_weights (str, optional): Name of pretrained weights. Defaults to IMAGENET_1K_V1.
            bb_freeze (bool, optional): If true, freezes weights in backbone. Defaults to True.

        Raises:
            Exception: _description_
        """
        super(TipburnClassifier, self).__init__()
        self.n_classes = n_classes

        # Set backbone
        # Allows for different models and pretraining
        if bb_weights is not None:
            backbone_call = f'torchvision.models.{bb_name}(weights="{bb_weights}")'
        else:
            backbone_call = f"torchvision.models.{bb_name}(weights={bb_weights})"
        self.backbone = eval(backbone_call)

        # Freeze weights in backbone
        for param in self.backbone.parameters():
            param.requires_grad = not bb_freeze

        # For ResNets
        if bb_name in ["resnet50", "wide_resnet50_2"]:
            # Remember number of input features of second layer
            out_features_bn1 = self.backbone.bn1.num_features

            # Change input channels of first conv layer
            self.backbone.conv1 = nn.Conv2d(
                n_channels,
                out_features_bn1,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )

            # Remember number of output features of backbone
            out_features_bb = list(self.backbone.children())[-1].in_features

            # Remove final layer of backbone
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

            # Classifier with output features of backbone as input
            # TODO: make more sophisticated -> dropout, batch normalization?
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Linear(out_features_bb, self.n_classes)
            )
        else:
            raise Exception("Selected backbone model is unsupported")

    def forward(self, x):
        features = self.backbone(x)
        pred_logits = self.classifier(features)
        return pred_logits
