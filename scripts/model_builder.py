"""
Functionality for building models as nn.Module class.

TODO:
    - Come up with what kinda backbones/models I want
    - Allow selection for different backbones
    - Allow selection for different subsequent layers (or make preset for each backbone)
"""
# Import statements
import torch
import torch.nn as nn
from torchinfo import summary
import torchvision


# Define nn.Module class for model
class TipburnClassifier(nn.Module):
    def __init__(
        self, n_classes, bb_name=None, bb_weights="IMAGENET1K_V1", bb_freeze=True
    ):
        """Creates tipburn classifier as PyTorch nn.Module class.

        Creates a CNN with a backbone. Backbone can be pretrained and frozen.
        Names of networks and weights should follow the torchvision API:
            https://pytorch.org/vision/stable/models.html#listing-and-retrieving-available-models
        Only supports the following backbones:
            resnet50, wide_resnet50_2

        Args:
            n_classes (int): Numbers of classes to predict.
            bb_name (str, optional): Name of backbone network. Defaults to None.
            bb_weights (str, optional): Name of pretrained weights. Defaults to IMAGENET_1K_V1.
            bb_freeze (bool, optional): If true, freezes weights in backbone. Defaults to True.

        Raises:
            Exception: _description_
        """
        super(TipburnClassifier, self).__init__()
        self.n_classes = n_classes

        # Set backbone
        if bb_name is not None:
            # Allows for different models and pretrained weights
            backbone_call = f'torchvision.models.{bb_name}(weights="{bb_weights}")'
            self.backbone = eval(backbone_call)

            # Freeze weights in backbone
            for param in self.backbone.parameters():
                param.requires_grad = not bb_freeze
        else:
            raise Exception("No argument for bb_name, a backbone is mandatory")

        # For ResNets
        if bb_name in ["resnet50", "wide_resnet50_2"]:
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
