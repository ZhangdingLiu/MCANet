import torch
import torch.nn as nn

# =============================================================================
# CSRA Module
# -----------------------------------------------------------------------------
# This implementation is based on:
# Zhu, K., and J. Wu. 2021. “Residual Attention: A Simple but Effective Method
# for Multi-Label Recognition.” arXiv. https://arxiv.org/abs/2106.14334
# =============================================================================

class CSRA(nn.Module):
    """
    Class-Specific Residual Attention (CSRA) block.

    This module enhances class-wise prediction by combining global average pooling
    with an attention mechanism. It computes:
    - base_logit: average-pooled logits (global baseline)
    - att_logit: attention-enhanced logits using class-specific softmax

    Parameters:
    -----------
    input_dim : int
        Number of input feature channels.
    num_classes : int
        Number of output classes.
    T : float
        Softmax temperature controlling attention sharpness.
        If T == 99, max pooling is used instead.
    lam : float
        Weight factor for attention-enhanced logits.
    """
    def __init__(self, input_dim, num_classes, T, lam):
        super(CSRA, self).__init__()
        self.T = T
        self.lam = lam

        # 1x1 Conv to generate class-wise feature maps
        self.head = nn.Conv2d(input_dim, num_classes, kernel_size=1, bias=False)
        # Softmax for attention weighting
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        """
        Forward pass for CSRA.

        Parameters:
        -----------
        x : torch.Tensor
            Input feature map of shape [B, C, H, W]

        Returns:
        --------
        torch.Tensor : shape [B, num_classes]
            Final logits combining base and attention-enhanced predictions
        """
        # Normalize classifier weights
        weight_norm = torch.norm(self.head.weight, dim=1, keepdim=True).transpose(0, 1)
        score = self.head(x) / weight_norm  # [B, num_classes, H, W]
        score = score.flatten(2)            # [B, num_classes, H*W]

        base_logit = torch.mean(score, dim=2)  # Global average pooling

        if self.T == 99:
            # Use max pooling for attention
            att_logit = torch.max(score, dim=2)[0]
        else:
            # Use softmax attention weighting
            score_soft = self.softmax(score * self.T)
            att_logit = torch.sum(score * score_soft, dim=2)

        return base_logit + self.lam * att_logit


# =============================================================================
# MHA Module (Multi-Head Attention over CSRA)
# =============================================================================

class MHA(nn.Module):
    """
    Multi-Head CSRA block with diverse attention temperatures.

    Each head uses a different softmax temperature (or max-pooling) to capture
    attention from various perspectives. Their outputs are summed for final prediction.

    Parameters:
    -----------
    num_heads : int
        Number of parallel CSRA heads to apply.
        Must be one of [1, 2, 4, 8].
    lam : float
        Weight factor for each CSRA head.
    input_dim : int
        Number of input feature channels.
    num_classes : int
        Number of output classes.
    """

    # Predefined temperature configurations for different head counts
    temp_settings = {
        1: [1],
        2: [1, 99],
        4: [1, 2, 4, 99],
        8: [1, 2, 3, 4, 5, 6, 7, 99],
    }

    def __init__(self, num_heads, lam, input_dim, num_classes):
        super(MHA, self).__init__()
        assert num_heads in self.temp_settings, "num_heads must be one of [1, 2, 4, 8]"
        self.temp_list = self.temp_settings[num_heads]

        # Create multiple CSRA heads with different temperatures
        self.multi_head = nn.ModuleList([
            CSRA(input_dim, num_classes, T=self.temp_list[i], lam=lam)
            for i in range(num_heads)
        ])

    def forward(self, x):
        """
        Forward pass for MHA.

        Parameters:
        -----------
        x : torch.Tensor
            Input feature map of shape [B, C, H, W]

        Returns:
        --------
        torch.Tensor : shape [B, num_classes]
            Combined class-wise prediction from all CSRA heads
        """
        logit = 0.
        for head in self.multi_head:
            logit += head(x)
        return logit
