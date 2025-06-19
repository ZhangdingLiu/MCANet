""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929

The official jax code is released and available at https://github.com/google-research/vision_transformer

"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from functools import partial
from .timm_utils import DropPath, to_2tuple, trunc_normal_
from .csra import MHA, CSRA

default_cfgs = {
    'vit_base_patch16_224':  'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
    'vit_large_patch16_224':'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth'
    }


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads # 64
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # qkv (3, B, 12, N, C/12)
        # q (B, 12, N, C/12)
        # k (B, 12, N, C/12)
        # v (B, 12, N, C/12)
        # attn (B, 12, N, N)
        # x (B, 12, N, C/12)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class VIT_CSRA(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, cls_num_heads=1, cls_num_cls=80, lam=0.3):
        super().__init__()
        self.add_w = 0.
        self.normalize = False
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.HW = int(math.sqrt(num_patches))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        #self.repr = nn.Linear(embed_dim, representation_size)
        #self.repr_act = nn.Tanh()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        # We add our MHA (CSRA) beside the orginal VIT structure below
        self.head = nn.Sequential() # delete original classifier
        self.classifier = MHA(input_dim=embed_dim, num_heads=cls_num_heads, num_classes=cls_num_cls, lam=lam)

        self.loss_func = F.binary_cross_entropy_with_logits

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def backbone(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # (B, 1+HW, C)
        # we use all the feature to form the tensor like B C H W
        x = x[:, 1:]
        b, hw, c = x.shape
        x = x.transpose(1, 2)
        x = x.reshape(b, c, self.HW, self.HW)

        return x

    def forward_train(self, x, target):
        x = self.backbone(x)
        logit = self.classifier(x)
        loss = self.loss_func(logit, target, reduction="mean")
        return logit, loss

    def forward_test(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

    def forward(self, x, target=None):
        if target is not None:
            return self.forward_train(x, target)
        else:
            return self.forward_test(x)

def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict


def VIT_B16_224_CSRA(pretrained=True, cls_num_heads=1, cls_num_cls=80, lam=0.3):
    model = VIT_CSRA(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), cls_num_heads=cls_num_heads, cls_num_cls=cls_num_cls, lam=lam)

    model_url = default_cfgs['vit_base_patch16_224']
    if pretrained:
        state_dict = model_zoo.load_url(model_url)
        model.load_state_dict(state_dict, strict=False)
    return model


def VIT_L16_224_CSRA(pretrained=True, cls_num_heads=1, cls_num_cls=80, lam=0.3):
    model = VIT_CSRA(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), cls_num_heads=cls_num_heads, cls_num_cls=cls_num_cls, lam=lam)

    model_url = default_cfgs['vit_large_patch16_224']
    if pretrained:
        state_dict = model_zoo.load_url(model_url)
        model.load_state_dict(state_dict, strict=False)
        # load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model




# i add some vit

def VIT_B16_448_CSRA(pretrained=True, cls_num_heads=1, cls_num_cls=80, lam=0.3):
    model = VIT_CSRA(
        img_size=448,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        cls_num_heads=cls_num_heads,
        cls_num_cls=cls_num_cls,
        lam=lam,
    )

    model_url = default_cfgs['vit_base_patch16_224']
    if pretrained:
        state_dict = model_zoo.load_url(model_url)

        # interpolate position embedding if size mismatch
        if 'pos_embed' in state_dict:
            pos_embed = state_dict['pos_embed']
            cls_token = pos_embed[:, :1]
            old_num_patches = pos_embed.shape[1] - 1
            old_size = int(old_num_patches ** 0.5)
            new_size = model.HW
            if old_size != new_size:
                print(f"Interpolating position embedding from {old_size}x{old_size} to {new_size}x{new_size}")
                pos_tokens = pos_embed[:, 1:].reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2)
                pos_tokens = F.interpolate(pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, new_size * new_size, -1)
                new_pos_embed = torch.cat([cls_token, pos_tokens], dim=1)
                state_dict['pos_embed'] = new_pos_embed

        model.load_state_dict(state_dict, strict=False)
        return model



def VIT_B16_448_BASE(pretrained=True, num_classes=80):
    class VIT_BASE(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = VIT_CSRA(
                img_size=448,
                patch_size=16,
                embed_dim=768,
                depth=12,
                num_heads=12,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                cls_num_heads=1,  # not used
                cls_num_cls=num_classes,
                lam=0.0,          # not used
            )
            # 替换掉 CSRA 的 classifier
            self.model.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(768, num_classes),
            )
            self.loss_func = nn.BCEWithLogitsLoss()

        def forward_train(self, x, target):
            feat = self.model.backbone(x)
            logit = self.model.classifier(feat)
            loss = self.loss_func(logit, target)
            return logit, loss

        def forward_test(self, x):
            feat = self.model.backbone(x)
            return self.model.classifier(feat)

        def forward(self, x, target=None):
            if target is not None:
                return self.forward_train(x, target)
            else:
                return self.forward_test(x)

    model = VIT_BASE()

    model_url = default_cfgs['vit_base_patch16_224']
    if pretrained:
        state_dict = model_zoo.load_url(model_url)

        if 'pos_embed' in state_dict:
            pos_embed = state_dict['pos_embed']
            cls_token = pos_embed[:, :1]
            old_num_patches = pos_embed.shape[1] - 1
            old_size = int(old_num_patches ** 0.5)
            new_size = model.model.HW
            if old_size != new_size:
                print(f"Interpolating position embedding from {old_size}x{old_size} to {new_size}x{new_size}")
                pos_tokens = pos_embed[:, 1:].reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2)
                pos_tokens = F.interpolate(pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, new_size * new_size, -1)
                new_pos_embed = torch.cat([cls_token, pos_tokens], dim=1)
                state_dict['pos_embed'] = new_pos_embed

        model.model.load_state_dict(state_dict, strict=False)

    return model










