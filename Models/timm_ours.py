import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange
import numpy as np
import timm



class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, print_w=False, attention_method = 'default'):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj1 = nn.Linear(dim, dim)
        self.q_proj2 = nn.Linear(dim, dim)

        self.k_proj1 = nn.Linear(dim, dim)
        self.k_proj2 = nn.Linear(dim, dim)

        self.fc_out = nn.Linear(dim, dim)

        self.weights = nn.Parameter(torch.randn(4))

        self.v_proj1 = nn.Linear(dim, dim)
        self.v_proj2 = nn.Linear(dim, dim)

        self.print_w = print_w
        self.attention_method = attention_method

    def forward(self, x):
        B, M, C = x.shape
        N = (M - 2) // 2

        q1 = self.q_proj1(x[:, :N + 1])
        q1 = rearrange(q1, "b n (h d) -> b h n d", h=self.num_heads, d=self.head_dim)
        q2 = self.q_proj2(x[:, N + 1:])
        q2 = rearrange(q2, "b n (h d) -> b h n d", h=self.num_heads, d=self.head_dim)

        k1 = self.k_proj1(x[:, :N + 1])
        k1 = rearrange(k1, "b n (h d) -> b h n d", h=self.num_heads, d=self.head_dim)
        k2 = self.k_proj2(x[:, N + 1:])
        k2 = rearrange(k2, "b n (h d) -> b h n d", h=self.num_heads, d=self.head_dim)

        q = torch.cat((q1, q2), dim=2)
        k = torch.cat((k1, k2), dim=2)

        v1 = self.v_proj1(x[:, :N + 1])
        v1 = rearrange(v1, "b n (h d) -> b h n d", h=self.num_heads, d=self.head_dim)

        v2 = self.v_proj2(x[:, N + 1:])
        v2 = rearrange(v2, "b n (h d) -> b h n d", h=self.num_heads, d=self.head_dim)

        attn = torch.einsum("bhqd, bhkd -> bhqk", q, k) * self.scale

        block1 = attn[:, :, :N + 1, :N + 1]
        block2 = attn[:, :, :N + 1, N + 1:]
        block3 = attn[:, :, N + 1:, :N + 1]
        block4 = attn[:, :, N + 1:, N + 1:]

        block1 = block1.softmax(dim=-1)
        block2 = block2.softmax(dim=-1)
        block3 = block3.softmax(dim=-1)
        block4 = block4.softmax(dim=-1)
        if self.attention_method=='default':
            w = F.gumbel_softmax(self.weights, tau=1.0, hard=True, dim=0)
            
        attn = w[0] * block1 + w[1] * block2 + w[2] * block3 + w[3] * block4

        out1 = torch.einsum("bhqk, bhkd -> bhqd", attn, v1)
        out2 = torch.einsum("bhqk, bhkd -> bhqd", attn, v2)

        out = torch.cat((out1, out2), dim=2)
        out = rearrange(out, "b h n d -> b n (h d)")

        return self.fc_out(out)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim, dropout, print_w=False, drop_path=0.1, attention_method = 'default'):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, print_w=print_w, attention_method = attention_method)
        self.mlp = MLP(dim, mlp_dim, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=10, dim=64, depth=6, heads=8, mlp_dim=128, dropout=0.1, second_path_size=None, print_w=False, drop_path=0.1, attention_method = 'default'):
        super(VisionTransformer, self).__init__()

        vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.blocks = nn.Sequential(*vit.blocks[:11])  # Use first 11 blocks
        for param in self.blocks.parameters():
            param.requires_grad = False  # Freeze the first 11 blocks

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=768, kernel_size=16, stride=16)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=4, stride=4)
        self.conv3 = nn.Conv2d(in_channels=3, out_channels=768, kernel_size=4, stride=4)

        # CLS tokens for out1 and out3
        self.cls_token1 = nn.Parameter(torch.zeros(1, 1, 768))
        self.cls_token3 = nn.Parameter(torch.zeros(1, 1, 768))
        
        # Positional encoding for out1 and out3
        self.pos_embed1 = nn.Parameter(torch.zeros(1, 197, 768))  # 196 + 1 for CLS
        self.pos_embed3 = nn.Parameter(torch.zeros(1, 197, 768))  # 196 + 1 for CLS
        
        ###############################################################
        nn.init.trunc_normal_(self.cls_token1, std=0.02)
        nn.init.trunc_normal_(self.cls_token3, std=0.02)
        nn.init.trunc_normal_(self.pos_embed1, std=0.02)
        nn.init.trunc_normal_(self.pos_embed3, std=0.02)
        ###############################################################
        self.norm = vit.norm 

        self.transformer_block12 = TransformerBlock(
            dim=768, num_heads=12, mlp_dim=3072, dropout=0.1, drop_path=0.1, attention_method = attention_method
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(768 * 2),  # We concatenate both CLS outputs
            nn.Linear(768 * 2, num_classes)  # 10 classes as an example, adjust as needed
        )

    def forward(self, x):
        feat1 = self.conv1(x)
        feat3 = self.conv3(self.conv2(x))

        B = feat1.size(0)

        # Flatten 14x14 to 196 tokens (excluding CLS token)
        feat1 = feat1.flatten(2).transpose(1, 2)  # (B, 196, 768)
        feat3 = feat3.flatten(2).transpose(1, 2)  # (B, 196, 768)

        # Add CLS token to both paths
        cls1 = self.cls_token1.expand(B, -1, -1)  # (B, 1, 768)
        cls3 = self.cls_token3.expand(B, -1, -1)  # (B, 1, 768)
        feat1 = torch.cat([cls1, feat1], dim=1)  # (B, 197, 768)
        feat3 = torch.cat([cls3, feat3], dim=1)  # (B, 197, 768)

        # Add positional encoding
        feat1 = feat1 + self.pos_embed1
        feat3 = feat3 + self.pos_embed3

        # Pass through first 11 transformer blocks
        out1 = self.blocks(feat1)
        out3 = self.blocks(feat3)

        # Final LayerNorm
        out1 = self.norm(out1)
        out3 = self.norm(out3)

        # Concatenate out1 and out3 (B, 197*2, 768)
        combined_output = torch.cat([out1, out3], dim=1)

        # Pass through the final transformer block (12th layer)
        final_out = self.transformer_block12(combined_output)

        # Use the CLS tokens (combined) for classification
        cls_output = final_out[:, 0]  # Get the first CLS token
        cls_output2 = final_out[:, 197]  # Get the second CLS token
        combined_cls_output = torch.cat((cls_output, cls_output2), dim=1)  # (B, 768*2)

        return self.mlp_head(combined_cls_output)
