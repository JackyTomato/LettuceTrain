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

a = 3
b = 4
a + b


# Define nn.Module class for model
class TipburnClassifier(nn.Module):
    def __init__(self, n_classes, bb_name=None, weights_bb=True, freeze_bb=True):
        """Creates tipburn classifier as PyTorch nn.Module class

        Creates a CNN with a backbone. Backbone can be pretrained and frozen.
        Only supports the following backbones:

        Args:
            n_classes (int): Numbers of classes to predict.
            bb_name (str, optional): _description_. Defaults to None.
            weights_bb (bool, optional): _description_. Defaults to True.
            freeze_bb (bool, optional): _description_. Defaults to True.

        Raises:
            Exception: _description_
        """
        super(TipburnClassifier, self).__init__()
        # Backbone
        if bb_name is not None:
            backbone_call = f"torchvision.models.{bb_name}(weights={weights_bb})"
            backbone = eval(backbone_call)
            for param in backbone.parameters():
                param.requires_grad = not freeze_bb
            # TODO: adjust final layers of backbone to suit model, prolly only support few models
        else:
            raise Exception("No argument for bb_name, a backbone is mandatory")

        # Layer 1

        # Classifier

    def forward(self, x):
        return
