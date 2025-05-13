import torch.nn as nn
import timm

class VisionTransformer(nn.Module):
    def __init__(self, num_classes=1000, pretrained=False, freeze=True, model_name="vit_base_patch16_224"):
        super(VisionTransformer, self).__init__()

        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.reset_classifier(num_classes)

        if pretrained and freeze:
            for param in self.model.parameters():
                param.requires_grad = False

            for param in self.model.head.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.model(x)
