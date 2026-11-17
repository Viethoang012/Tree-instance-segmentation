"""EDTER (Pu et al., CVPR 2022) in plain PyTorch, with an extra crown-extent head on Stage I."""
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def trunc_normal_(t, std=.02):
    return nn.init.trunc_normal_(t, std=std, a=-2 * std, b=2 * std)


class DropPath(nn.Module):
    def __init__(self, p=0.):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0. or not self.training:
            return x
        keep = 1 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop if self.training else 0.)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        assert x.shape[-1] == self.img_size and x.shape[-2] == self.img_size, x.shape
        return self.proj(x)


def _cbr(cin, cout, k):
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False), nn.BatchNorm2d(cout), nn.ReLU())


class Conv_BIMLA(nn.Module):
    """Bidirectional multi-level aggregation."""

    def __init__(self, in_channels=1024, mla_channels=256, local=False):
        super().__init__()
        k = 1 if local else 3
        for i in (2, 3, 4, 5):
            setattr(self, f'mla_p{i}_1x1', _cbr(in_channels, mla_channels, 1))
            setattr(self, f'mla_p{i}', _cbr(mla_channels, mla_channels, k))
            setattr(self, f'mla_b{i}', _cbr(mla_channels, mla_channels, k))

    @staticmethod
    def to_2D(x):
        n, hw, c = x.shape
        h = w = int(math.sqrt(hw))
        return x.transpose(1, 2).reshape(n, c, h, w)

    def forward(self, res2, res3, res4, res5):
        res2, res3, res4, res5 = map(self.to_2D, (res2, res3, res4, res5))
        p5_1 = self.mla_p5_1x1(res5)
        p4_1 = self.mla_p4_1x1(res4)
        p3_1 = self.mla_p3_1x1(res3)
        p2_1 = self.mla_p2_1x1(res2)
        p4_plus = p5_1 + p4_1
        p3_plus = p4_plus + p3_1
        p2_plus = p3_plus + p2_1
        mla_p5, mla_p4, mla_p3, mla_p2 = self.mla_p5(p5_1), self.mla_p4(p4_plus), self.mla_p3(p3_plus), self.mla_p2(p2_plus)
        b2_plus = p2_1
        b3_plus = b2_plus + p3_1
        b4_plus = b3_plus + p4_1
        b5_plus = b4_plus + p5_1
        mla_b2, mla_b3, mla_b4, mla_b5 = self.mla_b2(b2_plus), self.mla_b3(b3_plus), self.mla_b4(b4_plus), self.mla_b5(b5_plus)
        return mla_b2, mla_b3, mla_b4, mla_b5, mla_p2, mla_p3, mla_p4, mla_p5


class VIT_BIMLA(nn.Module):
    def __init__(self, img_size=320, patch_size=16, in_chans=3, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4.,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., mla_channels=256,
                 mla_index=(5, 11, 17, 23), local=False):
        super().__init__()
        self.img_size = img_size
        self.mla_index = mla_index
        self.local = local
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate, dpr[i],
                                           norm_layer) for i in range(depth)])
        self.mla = Conv_BIMLA(embed_dim, mla_channels, local=local)
        self.norm_0 = norm_layer(embed_dim)
        self.norm_1 = norm_layer(embed_dim)
        self.norm_2 = norm_layer(embed_dim)
        self.norm_3 = norm_layer(embed_dim)
        trunc_normal_(self.pos_embed)
        trunc_normal_(self.cls_token)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1) + self.pos_embed
        x = self.pos_drop(x[:, 1:])
        outs = []
        for blk in self.blocks:
            x = blk(x)
            outs.append(x)
        c = [getattr(self, f'norm_{i}')(outs[j]) for i, j in enumerate(self.mla_index)]
        return self.mla(*c)


def _up(cin, cout, factor):
    k2, s2, p2 = (16, 8, 4) if factor == 16 else (8, 4, 2)
    return nn.Sequential(nn.ConvTranspose2d(cin, cout, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(),
                         nn.ConvTranspose2d(cout, cout, k2, stride=s2, padding=p2, bias=False), nn.BatchNorm2d(cout), nn.ReLU())


class BIMLAHead(nn.Module):
    def __init__(self, mla_channels=256, mlahead_channels=128, factor=16):
        super().__init__()
        for n in ('head2', 'head3', 'head4', 'head5', 'head2_1', 'head3_1', 'head4_1', 'head5_1'):
            setattr(self, n, _up(mla_channels, mlahead_channels, factor))

    def forward(self, b2, b3, b4, b5, p2, p3, p4, p5):
        return torch.cat([self.head2(p2), self.head3(p3), self.head4(p4), self.head5(p5),
                          self.head2_1(b2), self.head3_1(b3), self.head4_1(b4), self.head5_1(b5)], dim=1)


def _feat_stack(c):
    return nn.Sequential(
        nn.Conv2d(8 * c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(),
        nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(),
        nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(),
        nn.Conv2d(c, c, 1), nn.BatchNorm2d(c), nn.ReLU())


class VIT_BIMLAHead(nn.Module):
    """Stage I decode head with extent branch."""

    def __init__(self, mla_channels=256, mlahead_channels=128, with_extent=True):
        super().__init__()
        self.mlahead = BIMLAHead(mla_channels, mlahead_channels, 16)
        self.global_features = _feat_stack(mlahead_channels)
        self.edge = nn.Conv2d(mlahead_channels, 1, 1)
        self.with_extent = with_extent
        if with_extent:
            c = mlahead_channels
            self.extent = nn.Sequential(nn.Conv2d(c, c // 2, 3, padding=1, bias=False), nn.BatchNorm2d(c // 2), nn.ReLU(),
                                        nn.Conv2d(c // 2, 1, 1))

    def forward(self, inputs):
        x = self.global_features(self.mlahead(*inputs))
        ext = self.extent(x) if self.with_extent else None
        return self.edge(x), ext, x


class VIT_BIMLA_AUXIHead(nn.Module):
    def __init__(self, in_channels=256, in_index=0, factor=16):
        super().__init__()
        self.in_index = in_index
        k2, s2, p2 = (16, 8, 4) if factor == 16 else (8, 4, 2)
        self.aux = nn.Sequential(nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1, bias=False),
                                 nn.ConvTranspose2d(in_channels, 1, k2, stride=s2, padding=p2, bias=False))

    def forward(self, inputs):
        return self.aux(inputs[self.in_index])


class EDTERGlobal(nn.Module):
    """EDTER Stage I."""

    def __init__(self, img_size=320, with_extent=True):
        super().__init__()
        self.backbone = VIT_BIMLA(img_size=img_size, patch_size=16, embed_dim=1024, depth=24, num_heads=16, drop_rate=0.,
                                  mla_channels=256, mla_index=(5, 11, 17, 23))
        self.decode_head = VIT_BIMLAHead(256, 128, with_extent)
        self.auxiliary_head = nn.ModuleList([VIT_BIMLA_AUXIHead(256, i, 16) for i in range(8)])

    def forward(self, img, with_aux=True):
        feats = self.backbone(img)
        edge, ext, gfeat = self.decode_head(feats)
        aux = [h(feats) for h in self.auxiliary_head] if with_aux else []
        return dict(edge=edge, extent=ext, aux=aux, global_features=gfeat)


class SFTLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.SFT_scale_conv0 = nn.Conv2d(c, c, 1)
        self.SFT_scale_conv1 = nn.Conv2d(c, c, 1)
        self.SFT_shift_conv0 = nn.Conv2d(c, c, 1)
        self.SFT_shift_conv1 = nn.Conv2d(c, c, 1)

    def forward(self, local_f, global_f):
        scale = self.SFT_scale_conv1(F.relu(self.SFT_scale_conv0(global_f), inplace=True))
        shift = self.SFT_shift_conv1(F.relu(self.SFT_shift_conv0(global_f), inplace=True))
        return local_f * (scale + 1) + shift


class Local8x8_fuse_head(nn.Module):
    def __init__(self, c=128):
        super().__init__()
        self.SFT_head = SFTLayer(c)
        self.edge_head = nn.Sequential(nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(),
                                       nn.Conv2d(c, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
                                       nn.Conv2d(64, 1, 1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, local_f, global_f):
        fuse = self.SFT_head(local_f, global_f)
        return self.edge_head(fuse), fuse


class VIT_BIMLAHead_LOCAL8x8(nn.Module):
    def __init__(self, mla_channels=256, mlahead_channels=128):
        super().__init__()
        self.mlahead = BIMLAHead(mla_channels, mlahead_channels, 8)
        self.local_features = _feat_stack(mlahead_channels)
        self.edge = nn.Conv2d(mlahead_channels, 1, 1)

    def forward(self, inputs):
        x = self.local_features(self.mlahead(*inputs))
        return self.edge(x), x


class EDTERLocal(nn.Module):
    """EDTER Stage II (local ViT-B/8 on 160x160 windows, fused with Stage I features)."""

    def __init__(self, global_model, win=160):
        super().__init__()
        self.global_model = global_model
        for p in self.global_model.parameters():
            p.requires_grad = False
        self.win = win
        self.backbone = VIT_BIMLA(img_size=win, patch_size=8, embed_dim=768, depth=12, num_heads=16, drop_rate=0.,
                                  mla_channels=256, mla_index=(2, 5, 8, 11), local=True)
        self.decode_head = VIT_BIMLAHead_LOCAL8x8(256, 128)
        self.auxiliary_head = nn.ModuleList([VIT_BIMLA_AUXIHead(256, i, 8) for i in range(8)])
        self.fuse_head = Local8x8_fuse_head(128)

    def local_feats(self, img):
        B, _, H, W = img.shape
        w = self.win
        nh, nw = H // w, W // w
        assert nh * w == H and nw * w == W
        tiles = img.reshape(B, 3, nh, w, nw, w).permute(0, 2, 4, 1, 3, 5).reshape(B * nh * nw, 3, w, w)
        feats = self.backbone(tiles)
        out = []
        for f in feats:
            c, s = f.shape[1], f.shape[2]
            out.append(f.reshape(B, nh, nw, c, s, s).permute(0, 3, 1, 4, 2, 5).reshape(B, c, nh * s, nw * s))
        return out

    def forward(self, img, global_features=None, with_aux=True):
        if global_features is None:
            with torch.no_grad():
                global_features = self.global_model(img, with_aux=False)['global_features']
        feats = self.local_feats(img)
        edge, lfeat = self.decode_head(feats)
        aux = [h(feats) for h in self.auxiliary_head] if with_aux else []
        fuse_edge, _ = self.fuse_head(lfeat, global_features)
        return dict(edge=edge, aux=aux, fuse_edge=fuse_edge)
