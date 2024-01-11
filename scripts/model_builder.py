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
    def __init__(self, in_channels):
        super(kim_wg, self).__init__()
        self.weight_generator = nn.Sequential(
            nn.Conv2d(
                in_channels,
                1,
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
            fusion (str, optional): Fusion of RGB with fluor data: early, intermediate, intermediate_kim, or late. Defaults to None.
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
            if self.fusion.startswith("intermediate"):
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
                    if self.fusion == "intermediate_kim":
                        # Initialize weight generators to create weighted sum of features
                        self.wg1_1 = kim_wg(in_channels=128)
                        self.wg1_2 = kim_wg(in_channels=128)
                        self.wg2_1 = kim_wg(in_channels=512)
                        self.wg2_2 = kim_wg(in_channels=512)
                        self.wg3_1 = kim_wg(in_channels=1024)
                        self.wg3_2 = kim_wg(in_channels=1024)
                        self.wg4_1 = kim_wg(in_channels=2048)
                        self.wg4_2 = kim_wg(in_channels=2048)
                        self.wg5_1 = kim_wg(in_channels=4096)
                        self.wg5_2 = kim_wg(in_channels=4096)

                        # Reduce number of channels by half after fusion
                        self.halver0 = conv_channel_changer(
                            in_channels=self.n_channels_med1 + self.n_channels_med2,
                            out_channels=self.n_channels_med1,
                        )
                        self.halver1 = conv_channel_changer(
                            in_channels=128, out_channels=64
                        )
                        self.halver2 = conv_channel_changer(
                            in_channels=512, out_channels=256
                        )
                        self.halver3 = conv_channel_changer(
                            in_channels=1024, out_channels=512
                        )
                        self.halver4 = conv_channel_changer(
                            in_channels=2048, out_channels=1024
                        )
                        self.halver5 = conv_channel_changer(
                            in_channels=4096, out_channels=2048
                        )

                        # Compile all weight generators and halvers
                        self.wgs = nn.ModuleList(
                            [
                                self.wg1_1,
                                self.wg1_2,
                                self.wg2_1,
                                self.wg2_2,
                                self.wg3_1,
                                self.wg3_2,
                                self.wg4_1,
                                self.wg4_2,
                                self.wg5_1,
                                self.wg5_2,
                            ]
                        )
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
        if self.fusion.startswith("intermediate"):
            # Separate different inputs
            input1 = x[:, : self.n_channels_med1, :, :]
            input2 = x[
                :,
                self.n_channels_med1 : self.n_channels_med1 + self.n_channels_med2,
                :,
                :,
            ]

            # Run inputs through encoders
            features1 = self.encoder1(input1)
            features2 = self.encoder2(input2)

            # Loop through lists of feature maps
            features = []
            for index, feature12 in enumerate(zip(features1, features2)):
                # Concatenate feature maps of different encoders
                feature1, feature2 = feature12
                feature_cat = torch.concatenate((feature1, feature2), dim=1)

                if self.fusion == "intermediate_kim":
                    # Don't use weight gate for first feature maps as it has original resolutions
                    if index != 0:
                        # Calculate weights for each feature map of different encoders
                        wg1 = self.wgs[(index - 1) * 2]
                        wg2 = self.wgs[(index - 1) * 2 + 1]
                        weight1 = wg1(feature_cat)
                        weight2 = wg2(feature_cat)

                        # Calculate weighted sum of feature maps of different encoders
                        weighted_feature1 = feature1 * weight1
                        weighted_feature2 = feature2 * weight2
                        feature = torch.concatenate(
                            (weighted_feature1, weighted_feature2), dim=1
                        )
                    else:
                        feature = feature_cat

                    # Half number of channels after fusion
                    halver = self.halvers[index]
                    fused_feature = halver(feature)
                    features.append(fused_feature)

                else:
                    # Squeeze-and-excite and halve each feature map of different encoders
                    halver = self.halvers[index]
                    fused_feature = halver(feature_cat)
                    features.append(fused_feature)

            # Create segmentation predictions
            decoded = self.decoder(*features)
            pred_logits = self.seghead(decoded)

        return pred_logits


class Classifier(nn.Module):
    def __init__(
        self,
        encoder_name,
        encoder_weights,
        n_channels,
        n_classes,
        encoder_freeze,
        fusion=None,
        n_channels_med1=3,
        n_channels_med2=2,
        model_name=None,
        decoder_attention=None,
    ):
        """Creates a classification model as PyTorch nn.Module class.

        Creates a CNN which can be created pretrained and with its parameters in early layers frozen.
        Names of networks and weights should follow the torchvision API:
            https://pytorch.org/vision/stable/models.html#listing-and-retrieving-available-models
        Only supports the following backbones:
            All ResNet, all ResNeXt

        Intermediate fusion is performed by copying the encoder, feeding the input of different
        modalities in the different encoders and concatenating the resulting feature maps before
        the decoder.
        n_channels_med1 and n_channels_med2 are only used with intermediate fusion. They are the input
        channels of the first and second encoder, respectively.
        n_channels should always be equal to or larger than both n_channels_med1 and n_channels_med2.

        Args:
            encoder_name (str): Name of encoder for segmentation mdoel as in torchvision.
            encoder_weights (str): Weights for encoder pretraining as in torchvision. Usually "IMAGENET1K_V1".
            n_channels (int): Number of input channels.
            n_classes (int): Number of output classes for segmentation.
            encoder_freeze (bool): If true, freezes parameters of the encoder.
            fusion (str, optional): Fusion of RGB with fluor data: early, intermediate, intermediate_kim, or late. Defaults to None.
            n_channels_med1(int, optional:) Number of input channels for first encoder in intermediate fusion. Defaults to 3.
            n_channels_med2(int, optional:) Number of input channels for second encoder in intermediate fusion. Defaults to 2.
            model_name (str): Depracated.
            decoder_attention (str): Depracated.
        """
        super(Classifier, self).__init__()
        self.fusion = fusion
        self.n_channels_med1 = n_channels_med1
        self.n_channels_med2 = n_channels_med2

        # Initialize model model, allow for different models and pretraining
        if encoder_name.startswith("resn"):
            # Allows for different models and pretraining
            if encoder_weights is not None:
                model_call = (
                    f'torchvision.models.{encoder_name}(weights="{encoder_weights}")'
                )
            else:
                model_call = (
                    f"torchvision.models.{encoder_name}(weights={encoder_weights})"
                )
            self.model = eval(model_call)

            # Only tweak input and output channels for no or early fusion
            if (self.fusion == None) or (self.fusion == "early"):
                # Change input channels of first conv layer
                old_conv1 = self.model.conv1
                new_conv1 = nn.Conv2d(
                    n_channels,
                    old_conv1.out_channels,
                    kernel_size=old_conv1.kernel_size,
                    stride=old_conv1.stride,
                    padding=old_conv1.padding,
                    bias=old_conv1.bias,
                )
                self.model.conv1 = new_conv1

                # Subset encoder from model and make own decoder with desired output channels
                self.encoder = nn.Sequential(*list(self.model.children())[:-2])
                self.decoder = nn.Sequential(
                    nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                    nn.Linear(in_features=2048, out_features=n_classes, bias=True),
                )

                # Freeze weights in non-fully connected layers if desired
                for name, param in self.model.named_parameters():
                    if not name.startswith("fc"):
                        param.requires_grad = not encoder_freeze

            # Change network architecture for intermediate or late fusion
            elif self.fusion.startswith("intermediate") or (self.fusion == "late"):
                # Extract non-fully connected layers of model to create encoders
                self.encoder1 = nn.Sequential(*list(self.model.children())[:-2])
                self.encoder2 = deepcopy(self.encoder1)

                # Tweak number of channels of first layers if not correct already
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

                # Initialize intermediate fusion modules
                if self.fusion == "intermediate_kim":
                    # Initialize weight generators to create weighted sum of features
                    self.wg1 = kim_wg(in_channels=4096)
                    self.wg2 = kim_wg(in_channels=4096)

                    # Reduce number of channels by half after fusion
                    self.halver = conv_channel_changer(
                        in_channels=4096, out_channels=2048
                    )

                else:
                    # For first feature map at original number of channels
                    self.halver = se_halver(channels=4096)

                # Add decoder after fusion to classify for desired number of classes
                self.decoder = nn.Sequential(
                    nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                    nn.Linear(in_features=2048, out_features=n_classes, bias=True),
                )

                # Freeze weights in non-fully connected layers if desired
                for param in self.encoder1.parameters():
                    param.requires_grad = not encoder_freeze
                for param in self.encoder2.parameters():
                    param.requires_grad = not encoder_freeze
                    
        else:
            if self.fusion.startswith("intermediate"):
                raise Exception(
                    "Selected model is unsupported for intermediate fusion!"
                )

    def forward(self, x):
        # No or early fusion, with unchanged network architecture
        if (self.fusion == None) or (self.fusion == "early"):
            features = self.encoder(x)
            pred_logits = self.decoder(features)

        # Intermediate fusion, with fusion module in architecture
        elif self.fusion.startswith("intermediate"):
            # Separate different inputs
            input1 = x[:, : self.n_channels_med1, :, :]
            input2 = x[
                :,
                self.n_channels_med1 : self.n_channels_med1 + self.n_channels_med2,
                :,
                :,
            ]

            # Run inputs through encoders
            feature1 = self.encoder1(input1)
            feature2 = self.encoder2(input2)

            # Concatenate feature maps of different encoders
            feature_cat = torch.concatenate((feature1, feature2), dim=1)

            if self.fusion == "intermediate_kim":
                weight1 = self.wg1(feature_cat)
                weight2 = self.wg2(feature_cat)

                # Calculate weighted sum of feature maps of different encoders
                weighted_feature1 = feature1 * weight1
                weighted_feature2 = feature2 * weight2
                feature = torch.concatenate(
                    (weighted_feature1, weighted_feature2), dim=1
                )

                # Half number of channels after fusion
                fused_feature = self.halver(feature)

            else:
                # Squeeze-and-excite and halve feature maps of different encoders
                fused_feature = self.halver(feature_cat)

            # Create segmentation predictions
            pred_logits = self.decoder(fused_feature)

        return pred_logits


from torchinfo import summary

model = torchvision.models.resnext101_64x4d(weights="IMAGENET1K_V1")
summary(model)
print(model)
encoder1 = nn.Sequential(*list(model.children())[:-2])
print(encoder1)
decoder = nn.Sequential(*list(model.children())[-2:])
print(decoder)

model = Classifier(
    model_name=None,
    encoder_name=cp.ENCODER_NAME,
    encoder_weights="IMAGENET1K_V1",
    n_channels=cp.N_CHANNELS,
    n_classes=cp.N_CLASSES,
    decoder_attention=cp.DECODER_ATTENTION,
    encoder_freeze=cp.ENCODER_FREEZE,
    fusion=cp.FUSION,
    n_channels_med1=cp.N_CHANNELS_MED1,
    n_channels_med2=cp.N_CHANNELS_MED2,
)


class Classifier(nn.Module):
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
