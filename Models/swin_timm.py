import torch
import torch.nn as nn
import timm

class SwinTransformer(nn.Module):
    def __init__(self, num_classes=1000, pretrained=True, model_name="swin_tiny_patch4_window7_224"):
        super(SwinTransformer, self).__init__()

        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.reset_classifier(num_classes)

    def forward(self, x):
        return self.model(x)
