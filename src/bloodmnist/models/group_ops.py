import math

import torch
import torch.nn.functional as F
from torch import nn


NUM_P4_ORIENTATIONS = 4


def apply_p4_action(
    features: torch.Tensor,
    quarter_turns: int,
) -> torch.Tensor:
    """[N, C, 4, H, W] grup özelliklerine P4 dönüşünü uygular."""
    if features.ndim != 5:
        raise ValueError(
            "Beklenen özellik şekli [N, C, 4, H, W]; "
            f"alınan şekil: {tuple(features.shape)}"
        )

    quarter_turns %= NUM_P4_ORIENTATIONS

    rotated = torch.rot90(
        features,
        k=quarter_turns,
        dims=(-2, -1),
    )
    return torch.roll(
        rotated,
        shifts=quarter_turns,
        dims=2,
    )


class P4LiftConv2d(nn.Module):
    """Görüntü özelliklerini dört yönelimli P4 özelliklerine yükseltir."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
    ):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("P4LiftConv2d tek boyutlu kernel gerektirir.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = (
            nn.Parameter(torch.zeros(out_channels)) if bias else None
        )

        nn.init.kaiming_normal_(
            self.weight,
            mode="fan_out",
            nonlinearity="relu",
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        rotated_kernels = torch.cat(
            [
                torch.rot90(self.weight, k=k, dims=(-2, -1))
                for k in range(NUM_P4_ORIENTATIONS)
            ],
            dim=0,
        )
        rotated_bias = (
            self.bias.repeat(NUM_P4_ORIENTATIONS)
            if self.bias is not None
            else None
        )
        output = F.conv2d(
            images,
            rotated_kernels,
            bias=rotated_bias,
            padding=self.padding,
        )
        batch_size, _, height, width = output.shape
        output = output.reshape(
            batch_size,
            NUM_P4_ORIENTATIONS,
            self.out_channels,
            height,
            width,
        )
        return output.permute(0, 2, 1, 3, 4).contiguous()


class P4GroupConv2d(nn.Module):
    """P4 yönelimli özellik haritaları üzerinde group convolution uygular."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
    ):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("P4GroupConv2d tek boyutlu kernel gerektirir.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.empty(
                out_channels,
                in_channels,
                NUM_P4_ORIENTATIONS,
                kernel_size,
                kernel_size,
            )
        )
        self.bias = (
            nn.Parameter(torch.zeros(out_channels)) if bias else None
        )

        fan_in = (
            in_channels
            * NUM_P4_ORIENTATIONS
            * kernel_size
            * kernel_size
        )
        nn.init.normal_(
            self.weight,
            mean=0.0,
            std=math.sqrt(2.0 / fan_in),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 5:
            raise ValueError(
                "Beklenen giriş şekli [N, C, 4, H, W]; "
                f"alınan şekil: {tuple(features.shape)}"
            )

        batch_size, channels, orientations, height, width = features.shape
        if channels != self.in_channels:
            raise ValueError(
                f"Beklenen kanal sayısı {self.in_channels}, alınan {channels}."
            )
        if orientations != NUM_P4_ORIENTATIONS:
            raise ValueError(
                f"Beklenen yönelim sayısı {NUM_P4_ORIENTATIONS}, "
                f"alınan {orientations}."
            )
        # [N, C, G, H, W] -> [N, G*C, H, W].
        flattened = (
            features.permute(0, 2, 1, 3, 4)
            .contiguous()
            .reshape(batch_size, orientations * channels, height, width)
        )
        outputs = []

        for output_orientation in range(NUM_P4_ORIENTATIONS):
            orientation_kernels = []
            for input_orientation in range(NUM_P4_ORIENTATIONS):
                relative_orientation = (
                    input_orientation - output_orientation
                ) % NUM_P4_ORIENTATIONS
                kernel = self.weight[:, :, relative_orientation]
                kernel = torch.rot90(
                    kernel,
                    k=output_orientation,
                    dims=(-2, -1),
                )
                orientation_kernels.append(kernel)

            kernels = torch.cat(orientation_kernels, dim=1)

            output = F.conv2d(
                flattened,
                kernels,
                bias=self.bias,
                padding=self.padding,
            )
            outputs.append(output)

        # [N, C_out, G, H, W]
        return torch.stack(outputs, dim=2)