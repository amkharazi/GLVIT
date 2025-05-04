import torch
import torch.nn as nn
import timm

class PVT(nn.Module):
    def __init__(self, num_classes=1000, pretrained=True, model_name="pvt_small"):
        super(PVT, self).__init__()

        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.reset_classifier(num_classes)

    def forward(self, x):
        return self.model(x)
