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

        vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
        self.vit_blocks = vit.blocks
        self.vit_norm = vit.norm
        self.vit_pos_drop = vit.pos_drop
        self.vit_head = vit.head

        self.resnet = timm.create_model('resnet50', pretrained=True, num_classes=0, global_pool='avg')

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=768, kernel_size=16, stride=16)

        self.cls_token1 = nn.Parameter(torch.zeros(1, 1, 768))
        self.pos_embed1 = nn.Parameter(torch.zeros(1, 197, 768))

        nn.init.trunc_normal_(self.cls_token1, std=0.02)
        nn.init.trunc_normal_(self.pos_embed1, std=0.02)

        self.res_mlp = nn.Sequential(
            nn.LayerNorm(2048),
            nn.Linear(2048, 768),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.mlp_intermediate = nn.Sequential(
            nn.LayerNorm(768 * 2),
            nn.Linear(768 * 2, 768),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        for p in self.parameters():
            p.requires_grad = False
        for p in self.res_mlp.parameters():
            p.requires_grad = True
        for p in self.mlp_intermediate.parameters():
            p.requires_grad = True
        for p in self.vit_head.parameters():
            p.requires_grad = True

    def forward(self, x):
        feat1 = self.conv1(x)
        B = x.size(0)
        feat1 = feat1.flatten(2).transpose(1, 2)
        cls1_tok = self.cls_token1.expand(B, -1, -1)
        feat1 = torch.cat([cls1_tok, feat1], dim=1)
        feat1 = feat1 + self.pos_embed1
        out1 = self.vit_blocks(feat1)
        out1 = self.vit_norm(out1)
        cls1 = out1[:, 0]

        res_feat = self.resnet(x)
        cls2 = self.res_mlp(res_feat)

        combined = torch.cat([cls1, cls2], dim=1)
        fused = self.mlp_intermediate(combined)
        logits = self.vit_head(fused)
        return logits
