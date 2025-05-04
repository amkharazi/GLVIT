
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class SpatialReductionAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0, sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class PVTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, sr_ratio=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = SpatialReductionAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                               attn_drop=0.0, proj_drop=drop, sr_ratio=sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp   = MLP(dim, hidden_dim, drop=drop)
    def forward(self, x, H, W):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x, H, W)
        x = shortcut + x
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + x
        return x

class PatchEmbed(nn.Module):
    def __init__(self, in_channels=3, embed_dim=64, patch_size=16):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x)
        H, W = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        return x, H, W

class PVT(nn.Module):
    def __init__(self, img_size=224, num_classes=1000,
                 embed_dims=[64, 128, 320, 512],
                 num_heads=[1, 2, 5, 8],
                 mlp_ratios=[8, 8, 4, 4],
                 depths=[2, 2, 2, 2],
                 patch_sizes=[4, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1],
                 drop_rate=0.0):
        super().__init__()
        self.num_stages = len(embed_dims)
        self.patch_embeds = nn.ModuleList()
        in_ch = 3
        for i in range(self.num_stages):
            self.patch_embeds.append(PatchEmbed(in_ch, embed_dims[i], patch_size=patch_sizes[i]))
            in_ch = embed_dims[i]
        self.pos_embeds = nn.ParameterList()
        cur_dim = img_size
        for i in range(self.num_stages):
            cur_dim = cur_dim // patch_sizes[i]
            self.pos_embeds.append(nn.Parameter(torch.zeros(1, cur_dim*cur_dim, embed_dims[i])))
        for p in self.pos_embeds:
            nn.init.trunc_normal_(p, std=0.02)
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            blocks = nn.ModuleList()
            for _ in range(depths[i]):
                blocks.append(PVTBlock(embed_dims[i], num_heads[i],
                                       mlp_ratio=mlp_ratios[i], qkv_bias=True,
                                       drop=drop_rate, sr_ratio=sr_ratios[i]))
            self.stages.append(blocks)
        self.norm = nn.LayerNorm(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
    def forward(self, x):
        B = x.size(0)
        for i in range(self.num_stages):
            x, H, W = self.patch_embeds[i](x)
            x = x + self.pos_embeds[i]
            for blk in self.stages[i]:
                x = blk(x, H, W)
            if i < self.num_stages - 1:
                B, N, C = x.shape
                x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x
