import torch.nn as nn
import timm

class VisionTransformer(nn.Module):
    def __init__(self, num_classes=1000, pretrained=False, freeze=True, model_name="vit_base_patch16_224"):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.reset_classifier(num_classes)

        if pretrained and freeze:
            for p in self.model.parameters():
                p.requires_grad = False

            if hasattr(self.model, "head"):
                for p in self.model.head.parameters():
                    p.requires_grad = True

            if hasattr(self.model, "blocks") and len(self.model.blocks) > 0:
                for p in self.model.blocks[-1].parameters():
                    p.requires_grad = True

            if hasattr(self.model, "norm") and self.model.norm is not None:
                for p in self.model.norm.parameters():
                    p.requires_grad = True

            if hasattr(self.model, "fc_norm") and self.model.fc_norm is not None:
                for p in self.model.fc_norm.parameters():
                    p.requires_grad = True

            if hasattr(self.model, "pre_logits") and self.model.pre_logits is not None:
                for p in self.model.pre_logits.parameters():
                    p.requires_grad = True

    def forward(self, x):
        return self.model(x)
