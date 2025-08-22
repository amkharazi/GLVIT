import torch
import torch.nn as nn
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

        self.resnet = timm.create_model('resnet50', pretrained=True, num_classes=0, global_pool='avg')
        for p in self.resnet.parameters():
            p.requires_grad = False

        self.mlp_res = nn.Linear(2048, 768)
        self.mlp_intermediate = nn.Linear(768 * 2, 768)
        self.mlp_head = nn.Linear(768, num_classes)

    def forward(self, x):
        cls1 = self.vit(x)
        f2 = self.resnet(x)
        cls2 = self.mlp_res(f2)
        fused = self.mlp_intermediate(torch.cat([cls1, cls2], dim=1))
        logits = self.mlp_head(fused)
        return logits


def count_and_check(model):
    print("trainable tensors:")
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(f"{n:60s} {p.numel()}")

    def total(mod):
        return sum(p.numel() for p in mod.parameters() if p.requires_grad)

    groups = {
        "mlp_res": total(model.mlp_res),
        "mlp_intermediate": total(model.mlp_intermediate),
        "mlp_head": total(model.mlp_head),
        "resnet_trainable": total(model.resnet),
        "vit_backbone_trainable": sum(p.numel() for _, p in model.vit.named_parameters() if p.requires_grad),
    }

    print("\ngrouped totals:")
    for k, v in groups.items():
        print(f"{k:25s} {v}")

    grouped_sum = groups["mlp_res"] + groups["mlp_intermediate"] + groups["mlp_head"]
    model_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\nsummary:")
    print("sum(mlp_res + mlp_intermediate + mlp_head):", grouped_sum)
    print("model trainable total:                    ", model_trainable)
    print("equal:", grouped_sum == model_trainable)


def shape_sanity(model, batch_size=2, img_size=224, num_classes=10):
    x = torch.randn(batch_size, 3, img_size, img_size)
    model.eval()
    with torch.no_grad():
        cls1 = model.vit(x)
        f2 = model.resnet(x)
        cls2 = model.mlp_res(f2)
        fused = model.mlp_intermediate(torch.cat([cls1, cls2], dim=1))
        logits_manual = model.mlp_head(fused)
        logits_model = model(x)

    print("\nshapes:")
    print("input:", tuple(x.shape))
    print("cls1:", tuple(cls1.shape))
    print("resnet feat:", tuple(f2.shape))
    print("cls2 (after mlp_res):", tuple(cls2.shape))
    print("fused (after mlp_intermediate):", tuple(fused.shape))
    print("logits_manual:", tuple(logits_manual.shape))
    print("logits_model:", tuple(logits_model.shape))
    print("same_logits:", torch.allclose(logits_manual, logits_model, atol=1e-6))


if __name__ == "__main__":
    m = VisionTransformer(num_classes=10)
    count_and_check(m)
    shape_sanity(m, batch_size=2, img_size=224)
