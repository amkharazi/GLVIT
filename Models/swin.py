import torch
import torch.nn as nn
import math


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_channels=3, embed_dim=96):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.window_size = window_size

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size - 1) ** 2, num_heads))

        coords = torch.stack(torch.meshgrid(torch.arange(window_size), torch.arange(window_size), indexing="ij"))
        coords_flatten = coords.flatten(1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[..., 0] += window_size - 1
        relative_coords[..., 1] += window_size - 1
        relative_coords[..., 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        attn = (Q @ K.transpose(-2, -1)) * (C // self.num_heads) ** -0.5

        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(N, N, -1).permute(2, 0, 1)
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            b = B_ // nW
            attn = attn.view(b, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(0).unsqueeze(2)
            attn = attn.view(B_, self.num_heads, N, N)

        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ V).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        H, W = input_resolution
        if min(H, W) <= window_size:
            shift_size = 0
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(dropout)
        )

        if shift_size > 0:
            pad_h = math.ceil(H / window_size) * window_size - H
            pad_w = math.ceil(W / window_size) * window_size - W
            H_pad, W_pad = H + pad_h, W + pad_w
            mask = torch.zeros((1, H_pad, W_pad, 1))
            h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
            w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
            cnt = 0
            for hs in h_slices:
                for ws in w_slices:
                    mask[:, hs, ws, :] = cnt
                    cnt += 1
            mask = mask.view(1, H_pad // window_size, window_size, W_pad // window_size, window_size, 1)
            mask = mask.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size * window_size)
            attn_mask = mask.unsqueeze(1) - mask.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            self.register_buffer("attn_mask", attn_mask, persistent=False)
        else:
            self.register_buffer("attn_mask", None, persistent=False)

    def forward(self, x):
        B, L, C = x.shape
        H = W = int(math.sqrt(L))
        x = x.view(B, H, W, C)
        shortcut = x
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h or pad_w:
            x = nn.functional.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        B_pad, H_pad, W_pad, _ = x.shape
        x_windows = x.view(B_pad, H_pad // self.window_size, self.window_size,
                           W_pad // self.window_size, self.window_size, C)
        x_windows = x_windows.permute(0, 1, 3, 2, 4, 5).reshape(-1, self.window_size ** 2, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(B_pad, H_pad // self.window_size, W_pad // self.window_size,
                                         self.window_size, self.window_size, C)
        x = attn_windows.permute(0, 1, 3, 2, 4, 5).reshape(B_pad, H_pad, W_pad, C)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = x[:, :H, :W, :].reshape(B, L, C)
        x = shortcut.view(B, H, W, C).reshape(B, L, C) + x
        x = x + self.mlp(self.norm2(x))
        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        if H % 2 == 1 or W % 2 == 1:
            x = nn.functional.pad(x, (0, 0, 0, W % 2, 0, H % 2))
            H, W = x.shape[1], x.shape[2]
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1).view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class SwinTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_channels=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        self.pos_drop = nn.Dropout(dropout)

        self.stages = nn.ModuleList()
        self.patch_merging = nn.ModuleList()
        H = W = img_size // patch_size
        dim = embed_dim
        for i, num_blocks in enumerate(depths):
            stage_blocks = nn.ModuleList([
                SwinTransformerBlock(dim=dim, input_resolution=(H, W),
                                     num_heads=num_heads[i], window_size=window_size,
                                     shift_size=(window_size // 2 if j % 2 == 1 else 0),
                                     mlp_ratio=mlp_ratio, dropout=dropout)
                for j in range(num_blocks)
            ])
            self.stages.append(stage_blocks)
            if i < len(depths) - 1:
                self.patch_merging.append(PatchMerging((H, W), dim))
                dim *= 2
                H = math.ceil(H / 2)
                W = math.ceil(W / 2)

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for i, blocks in enumerate(self.stages):
            for block in blocks:
                x = block(x)
            if i < len(self.patch_merging):
                x = self.patch_merging[i](x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)
