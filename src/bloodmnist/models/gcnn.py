import torch
import torch.nn.functional as F
from torch import nn

from bloodmnist.models.group_ops import (
    P4GroupConv2d,
    P4LiftConv2d,
)


def group_max_pool(features: torch.Tensor) -> torch.Tensor:
    """Uzamsal max pooling uygular, yönelim boyutunu korur."""
    batch_size, channels, orientations, height, width = features.shape

    flattened = (
        features.permute(0, 2, 1, 3, 4)
        .contiguous()
        .reshape(batch_size, orientations * channels, height, width)
    )
    pooled = F.max_pool2d(flattened, kernel_size=2)

    pooled_height, pooled_width = pooled.shape[-2:]
    pooled = pooled.reshape(
        batch_size,
        orientations,
        channels,
        pooled_height,
        pooled_width,
    )
    return pooled.permute(0, 2, 1, 3, 4).contiguous()


class GroupEquivariantCNN(nn.Module):
    """P4 dönüş grubunda çalışan küçük bir G-CNN sınıflandırıcısı."""

    def __init__(self, num_classes: int = 8):
        super().__init__()

        self.lift = P4LiftConv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
        )
        self.group_conv1 = P4GroupConv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
        )
        self.group_conv2 = P4GroupConv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # [N, 3, 28, 28] -> [N, 16, 4, 28, 28]
        features = F.relu(self.lift(images))
        features = group_max_pool(features)
        # -> [N, 16, 4, 14, 14]

        features = F.relu(self.group_conv1(features))
        features = group_max_pool(features)
        # -> [N, 32, 4, 7, 7]

        features = F.relu(self.group_conv2(features))
        # P4 yönelimleri ve uzamsal konumlar üzerinde ortalama al.
        invariant_features = features.mean(dim=(2, 3, 4))
        # -> [N, 32]

        return self.classifier(invariant_features)