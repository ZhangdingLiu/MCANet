import torch
import torch.nn as nn

# =============================================================================
# Class-Agnostic Residual Attention (CSRA_CA)
# -----------------------------------------------------------------------------
# Ablation variant of CSRA where the spatial attention weights are computed
# from a single shared 1×1 conv (output dim=1), then broadcast across all
# classes.  This isolates the contribution of *class-specific* attention:
#
#   CSRA (original):  each class has its own attention score map  [B, C, H, W]
#   CSRA_CA (this):   one shared attention map broadcast to all C  [B, 1, H, W]
#
# The class-specific logit head (`self.head`) is retained for base_logit
# computation so parameter counts are comparable.
# =============================================================================


class CSRA_CA(nn.Module):
    """
    Class-Agnostic Residual Attention block.

    Differs from CSRA in that spatial attention weights come from a single
    shared 1x1 conv (self.attn_head, output channels=1) rather than the
    class-specific head.  The attention map is broadcast across all C classes.

    Parameters
    ----------
    input_dim : int
        Number of input feature channels.
    num_classes : int
        Number of output classes.
    T : float
        Softmax temperature.  T=99 uses max-pooling instead.
    lam : float
        Weight factor for attention-enhanced logits.
    """

    def __init__(self, input_dim, num_classes, T, lam):
        super(CSRA_CA, self).__init__()
        self.T = T
        self.lam = lam

        # Class-specific conv — used only for computing logit scores
        self.head = nn.Conv2d(input_dim, num_classes, kernel_size=1, bias=False)
        # Shared spatial attention conv — one channel, broadcast across classes
        self.attn_head = nn.Conv2d(input_dim, 1, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor  [B, C_in, H, W]

        Returns
        -------
        torch.Tensor  [B, num_classes]
        """
        # --- class-specific logit scores (same as CSRA) ---
        weight_norm = torch.norm(self.head.weight, dim=1, keepdim=True).transpose(0, 1)
        score = self.head(x) / weight_norm          # [B, num_classes, H, W]
        score = score.flatten(2)                    # [B, num_classes, H*W]
        base_logit = torch.mean(score, dim=2)       # GAP

        # --- shared spatial attention weights ---
        attn_raw = self.attn_head(x).flatten(2)     # [B, 1, H*W]

        if self.T == 99:
            # Hard attention: one-hot at max position, broadcast
            idx = torch.argmax(attn_raw, dim=2, keepdim=True)  # [B, 1, 1]
            attn_w = torch.zeros_like(attn_raw).scatter_(2, idx, 1.0)  # [B, 1, H*W]
        else:
            attn_w = self.softmax(attn_raw * self.T)            # [B, 1, H*W]

        # broadcast: [B, num_classes, H*W] * [B, 1, H*W] → [B, num_classes, H*W]
        att_logit = torch.sum(score * attn_w, dim=2)            # [B, num_classes]

        return base_logit + self.lam * att_logit


# =============================================================================
# MHA_CA — Multi-Head wrapper around CSRA_CA
# =============================================================================

class MHA_CA(nn.Module):
    """
    Multi-Head Class-Agnostic Attention.

    Drop-in replacement for MHA using CSRA_CA heads.
    Temperature schedule is identical to MHA.
    """

    temp_settings = {
        1: [1],
        2: [1, 99],
        4: [1, 2, 4, 99],
        8: [1, 2, 3, 4, 5, 6, 7, 99],
    }

    def __init__(self, num_heads, lam, input_dim, num_classes):
        super(MHA_CA, self).__init__()
        assert num_heads in self.temp_settings, "num_heads must be one of [1, 2, 4, 8]"
        self.temp_list = self.temp_settings[num_heads]

        self.multi_head = nn.ModuleList([
            CSRA_CA(input_dim, num_classes, T=self.temp_list[i], lam=lam)
            for i in range(num_heads)
        ])

    def forward(self, x):
        logit = 0.
        for head in self.multi_head:
            logit += head(x)
        return logit
