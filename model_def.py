"""
MARF + CBAM + MobileNetV3 model definition.

Architecture copied EXACTLY from the research notebook:
    final_skin_cancer_detection_9_07_2026.ipynb  (cell 62)

    MARF (input-level modality-adaptive receptive field) ->
    MobileNetV3-Large backbone ->
    CBAM (channel + spatial attention) ->
    MLP classification head (960 -> 512 -> 256 -> num_classes).

forward() returns: (logits, alpha, gate_logit)
"""

import torch
import torch.nn as nn


class ModalityAdaptiveBlock(nn.Module):
    """
    Branch A : 3x3 conv, no dilation  -> fine texture (histopathology)
    Branch B : 3x3 conv, dilation=3    -> large structure (dermoscopy)
    Gate     : tiny conv-net -> global pool -> FC -> sigmoid -> alpha in [0,1]
    Fusion   : alpha * A + (1 - alpha) * B, projected back to 3ch,
               added as a residual to the original image.
    """

    def __init__(self, in_channels=3, mid_channels=16):
        super().__init__()

        self.branch_a = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.branch_b = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, 1),
        )

        self.project = nn.Conv2d(mid_channels, in_channels, kernel_size=1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, x):
        a = self.branch_a(x)
        b = self.branch_b(x)

        gate_logit = self.gate(x)          # (B, 1) raw logit
        alpha = torch.sigmoid(gate_logit)  # (B, 1) in [0,1]
        alpha_map = alpha.view(-1, 1, 1, 1)

        fused = alpha_map * a + (1 - alpha_map) * b
        out = x + self.project(fused)      # residual, safe at init
        return out, alpha.squeeze(1), gate_logit.squeeze(1)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        avg_out = self.shared_mlp(self.avg_pool(x).view(b, c))
        max_out = self.shared_mlp(self.max_pool(x).view(b, c))
        attn = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(concat))
        return x * attn


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel_attn = ChannelAttention(in_channels, reduction)
        self.spatial_attn = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class MARF_CBAM_MobileNetV3(nn.Module):
    """
    MARF (input-level modality-adaptive receptive field) ->
    MobileNetV3-Large backbone ->
    CBAM (channel + spatial attention) ->
    MLP classification head.
    """

    def __init__(self, num_classes, freeze_backbone=False,
                 reduction=16, spatial_kernel=7, dropout_p=(0.4, 0.3),
                 marf_mid_channels=16):
        super().__init__()

        self.marf = ModalityAdaptiveBlock(in_channels=3, mid_channels=marf_mid_channels)

        # NOTE: weights=None here — we always load the trained .pth,
        # so no ImageNet download is needed at inference time.
        from torchvision.models import mobilenet_v3_large
        base = mobilenet_v3_large(weights=None)
        self.features = base.features
        feat_channels = 960

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

        self.cbam = CBAM(feat_channels, reduction=reduction, spatial_kernel=spatial_kernel)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(feat_channels, 512),
            nn.BatchNorm1d(512),
            nn.Hardswish(),
            nn.Dropout(dropout_p[0]),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Hardswish(),
            nn.Dropout(dropout_p[1]),

            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x, alpha, gate_logit = self.marf(x)
        x = self.features(x)
        x = self.cbam(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits, alpha, gate_logit


def build_model(num_classes=8, weights_path=None, device="cpu"):
    """Build the MARF+CBAM MobileNetV3 model and (optionally) load .pth weights."""
    model = MARF_CBAM_MobileNetV3(num_classes=num_classes)
    if weights_path:
        state = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


# ----------------------------------------------------------------------------
# Class names + constants (from notebook cell 61)
# ----------------------------------------------------------------------------
CLASS_NAMES = [
    "Dermoscopic_BCC",
    "Dermoscopic_Melanoma",
    "Dermoscopic_Nevus",
    "Dermoscopic_SCC",
    "Histopathology_BCC",
    "Histopathology_Melanoma",
    "Histopathology_Nevus",
    "Histopathology_SCC",
]

# Indices of malignant classes (BCC / Melanoma / SCC), Nevus excluded.
# From me+probal_work.ipynb cells 20/23: MALIGNANT_CLASSES = [0,1,3,4,5,7]
MALIGNANT_CLASSES = [0, 1, 3, 4, 5, 7]

# ImageNet-style normalization used in every notebook transform
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224
