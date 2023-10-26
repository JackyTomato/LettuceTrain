#!/usr/bin/env python3
"""
Functionality for building models as nn.Module class.

TODO:
    - Provide torchvision support in Segmenter
    - Implement dict with **args instead of multiple if else statements
    - Think about what backbones to use for semantic segmentation
"""
# Import statements
import torch
import torch.nn as nn
import torchvision
import segmentation_models_pytorch as smp
import re


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
    ):
        """Creates a semantic segmentation model as PyTorch nn.Module class.

        Creates a segmentation model with an encoder. Encoder can be pretrained and frozen.
        Names of networks and weights should follow the Segmentation Models Pytorch (smp) API:
            https://github.com/qubvel/segmentation_models.pytorch
        Or the torchvision API:
            https://pytorch.org/vision/stable/models.html#listing-and-retrieving-available-models
        Only supports the following torchvision models:
            -

        Args:
            model_name (str): Name of segmentation model as in smp or torchvision.models.
            encoder_name (str): Name of encoder for segmentation mdoel as in smp.
            encoder_weights (str): Weights for encoder pretraining as in smp. Usually "imagenet".
            n_channels (int): Number of input channels.
            n_classes (int): Number of output classes for segmentation.
            decoder_attention (str): Attention type for decoder. Available for Unet: None or "scse".
            encoder_freeze (bool): If true, freezes parameters of the encoder.
        """
        super(Segmenter, self).__init__()

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
                args_call = ", ".join(enc_call, class_call, channel_call, weights_call)
            args_call = re.sub(r'"None"', "None", args_call)

            # Construct function call for eval and create model
            model_call = f"{model_call}({args_call})"
            self.model = eval(model_call)

            # Freeze weights in encoder if desired
            for param in self.model.encoder.parameters():
                param.requires_grad = not encoder_freeze

        # Use backbone from torchvision
        else:
            pass

    def forward(self, x):
        pred_logits = self.model(x)
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
