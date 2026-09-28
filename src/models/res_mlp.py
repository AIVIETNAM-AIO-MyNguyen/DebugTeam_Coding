import torch
from torch import nn
from torch.nn import functional as F
from src.models.base import BaseClassifier, register_model

class AffineTransform(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, 1, features))
        self.beta = nn.Parameter(torch.zeros(1, 1, features))

    def forward(self, x):
        return self.alpha * x + self.beta

class TokenMixLayer(nn.Module):
    def __init__(self, features, patches):
        super().__init__()

        self.aff1 = AffineTransform(features=features)
        self.fc1 = nn.Linear(patches, patches)
        self.aff2 = AffineTransform(features=features)

    def forward(self, x):
        residual = x
        x = self.aff1(x)
        x = self.fc1(x.transpose(1, 2)).transpose(1, 2)
        x = self.aff2(x)

        return x + residual

class ChannelMixLayer(nn.Module):
    def __init__(self, features, expansion_factor):
        super().__init__()

        num_hidden = features * expansion_factor
        self.aff1 = AffineTransform(features=features)
        self.fc1 = nn.Linear(features, num_hidden)
        self.fc2 = nn.Linear(num_hidden, features)
        self.aff2 = AffineTransform(features=features)

    def forward(self, x):
        residual = x
        x = self.aff1(x)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.aff2(x)

        return x + residual


class ResMLPLayer(nn.Module):
    def __init__(self, num_features, num_patches, expansion_factor):
        super().__init__()
        self.token_mix = TokenMixLayer(num_features, num_patches)
        self.channel_mix = ChannelMixLayer(num_features, expansion_factor)

    def forward(self, x):
        x = self.token_mix(x)
        return self.channel_mix(x)

def check_sizes(image_size, patch_size):
    sqrt_num_patches, remainder = divmod(image_size, patch_size)
    assert remainder == 0, "`image_size` must be divisibe by `patch_size`"
    num_patches = sqrt_num_patches ** 2
    return num_patches

@register_model("res_mlp")
class ResMLP(BaseClassifier):
    def __init__(
        self,
        image_size=32,
        patch_size=4,
        in_channels=3,
        num_features=128,
        expansion_factor=2,
        num_layers=8,
        num_classes=100,
        **kwargs
    ):
        num_patches = check_sizes(image_size, patch_size)
        super().__init__()
        self.patcher = nn.Conv2d(
            in_channels, num_features, kernel_size=patch_size, stride=patch_size
        )
        self.mlps = nn.Sequential(
            *[
                ResMLPLayer(num_features, num_patches, expansion_factor)
                for _ in range(num_layers)
            ]
        )
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        patches = self.patcher(x)
        batch_size, num_features, _, _ = patches.shape
        patches = patches.permute(0, 2, 3, 1)
        patches = patches.view(batch_size, -1, num_features)
        # patches.shape == (batch_size, num_patches, num_features)
        embedding = self.mlps(patches)
        embedding = torch.mean(embedding, dim=1)
        logits = self.classifier(embedding)
        return logits