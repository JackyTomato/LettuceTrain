"""
Contains functionality for building models as nn.Module class.

TODO:
    - Come up with what kinda backbones/models I want
    - Fill up model class, figure out how to adapt backbone
    - Allow selection for different backbones
    - Allow selection for different model types
    - Allow option to freeze backbone or not
"""
# Import statements
import torch
from torch import nn
from torchinfo import summary
import torchvision


# Define nn.Module class for model
class TipburnClassifier(nn.Module):
    def __init__(self, n_classes, pretrained_bb=True, freeze_bb=True):
        super(TipburnClassifier, self).__init__()

        # We add the background class
        self.n_classes = n_classes + 1

        # Backbone
        backbone = torchvision.models.(pretrained=pretrained_bb)
        for param in backbone.parameters():
            param.requires_grad = not freeze_bb

        # Layer 1

        # Classifier

    def forward(self, x):
        return
