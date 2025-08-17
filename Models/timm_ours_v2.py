import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=10,
        dim=64,
        depth=6,
        heads=8,
        mlp_dim=128,
        dropout=0.1,
        second_path_size=None,
        print_w=False,
        drop_path=0.1,
        attention_method='default'
    ):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        for p in self.vit.parameters():
            p.requires_grad = False
        self.conv = nn.Conv2d(3, 3, kernel_size=4, stride=4)
        self.mlp_intermediate = nn.Linear(768 * 2, 768)
        self.mlp_head = nn.Linear(768, num_classes)

    def forward(self, x):
        cls1 = self.vit(x)
        x2 = self.conv(x)
        x2 = F.interpolate(x2, size=(self.img_size, self.img_size), mode='bicubic', align_corners=False)
        cls2 = self.vit(x2)
        fused = self.mlp_intermediate(torch.cat([cls1, cls2], dim=1))
        logits = self.mlp_head(fused)
        return logits


def count_and_check(model):
    print("Trainable parameters by tensor:")
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(f"{n:60s} {p.numel()}")

    groups = {}

    def group_total(mod):
        return sum(p.numel() for p in mod.parameters() if p.requires_grad)

    groups["conv"] = group_total(model.conv)
    groups["mlp_intermediate"] = group_total(model.mlp_intermediate)
    groups["mlp_head"] = group_total(model.mlp_head)

    vit_backbone_trainable = sum(
        p.numel() for n, p in model.vit.named_parameters()
        if p.requires_grad
    )
    if vit_backbone_trainable > 0:
        groups["vit_backbone"] = vit_backbone_trainable

    print("\nGrouped totals:")
    for k, v in groups.items():
        print(f"{k:20s} {v}")

    grouped_sum = sum(groups.values())
    model_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\nSummaries:")
    print("sum(groups):               ", grouped_sum)
    print("model trainable total:     ", model_trainable)
    print("equal:", grouped_sum == model_trainable)


def shape_sanity(model, batch_size=2, img_size=224):
    x = torch.randn(batch_size, 3, img_size, img_size)
    with torch.no_grad():
        cls1 = model.vit(x)
        x2 = model.conv(x)
        x2 = F.interpolate(x2, size=(model.img_size, model.img_size), mode='bicubic', align_corners=False)
        cls2 = model.vit(x2)
        fused = model.mlp_intermediate(torch.cat([cls1, cls2], dim=1))
        logits = model.mlp_head(fused)

    print("\nShape sanity check:")
    print("input:", tuple(x.shape))
    print("cls1 (vit(x)):", tuple(cls1.shape))
    print("cls2 (vit(x2)):", tuple(cls2.shape))
    print("fused (after mlp_intermediate):", tuple(fused.shape))
    print("logits:", tuple(logits.shape))


if __name__ == "__main__":
    m = VisionTransformer(num_classes=10)
    count_and_check(m)
    shape_sanity(m, batch_size=2, img_size=224)
