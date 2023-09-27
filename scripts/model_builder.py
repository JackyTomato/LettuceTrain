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
    def __init__(self, n_classes, bb_name=None, pretrained_bb=True, freeze_bb=True):
        super(TipburnClassifier, self).__init__()

        # We add the background class
        self.n_classes = n_classes + 1

        # Backbone
        if bb_name is not None:
            backbone_call = f"torchvision.models.{bb_name}(pretrained={pretrained_bb})"
            backbone = eval(backbone_call)
            for param in backbone.parameters():
                param.requires_grad = not freeze_bb
            # TODO: adjust final layers of backbone to suit model, prolly only support few models
        else:
            raise Exception("No argument for bb_name, a backbone is required")

        # Layer 1

        # Classifier

    def forward(self, x):
        return
