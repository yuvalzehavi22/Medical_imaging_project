import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import numpy as np

from exceptions.exceptions import InvalidBackboneError
from open_clip import create_model_from_pretrained, get_tokenizer #!pip install open_clip_torch==2.23.0 transformers==4.35.2 matplotlib


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim, pretrained=True):
        super(ResNetSimCLR, self).__init__()
        
        if pretrained:
            self.resnet_dict = {
                "resnet18": models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
                "resnet50": models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
                "biomedclip": self._get_biomedclip_model(pretrained=True, out_dim=out_dim)
            }
        else:
            self.resnet_dict = {
                "resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                "resnet50": models.resnet50(pretrained=False, num_classes=out_dim),
                "biomedclip": self._get_biomedclip_model(pretrained=False, out_dim=out_dim)
            }

        self.backbone = self._get_basemodel(base_model)
        
        if "resnet" in base_model:
            dim_mlp = self.backbone.fc.in_features
        else: 
            dim_mlp = 768  # BioMedCLIP embedding size (adjust as per model variant)

        # MLP projection head
        self.backbone.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, out_dim)
        )

    def _get_basemodel(self, model_name):
        print('Using:', model_name)
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise ValueError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18, resnet50, or biomedclip")
        return model

    def _get_biomedclip_model(self, pretrained, out_dim):
        # Load the BioMedCLIP model using open_clip_torch's `create_model_from_pretrained`
        if pretrained:
            model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        else:
            # Assuming non-pretrained models can also be created similarly.
            model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', pretrained=False)

        # BioMedCLIP's visual encoder is equivalent to the image model
        return model.visual

    def forward(self, x):
        return self.backbone(x)