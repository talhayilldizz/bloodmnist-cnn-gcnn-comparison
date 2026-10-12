import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch

from src.bloodmnist.models.gcnn import GroupEquivariantCNN
from src.bloodmnist.models.group_ops import (
    P4GroupConv2d,
    P4LiftConv2d,
    apply_p4_action,
)


class P4EquivarianceTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_lifting_convolution_is_p4_equivariant(self) -> None:
        layer = P4LiftConv2d(3, 5, kernel_size=3).eval()
        images = torch.randn(2, 3, 17, 17)

        with torch.inference_mode():
            original_output = layer(images)

            for quarter_turns in (1, 2, 3):
                rotated_images = torch.rot90(
                    images,
                    k=quarter_turns,
                    dims=(-2, -1),
                )
                actual = layer(rotated_images)
                expected = apply_p4_action(
                    original_output,
                    quarter_turns,
                )
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=1e-4,
                    atol=1e-5,
                )

    def test_group_convolution_is_p4_equivariant(self) -> None:
        layer = P4GroupConv2d(3, 5, kernel_size=3).eval()
        features = torch.randn(2, 3, 4, 13, 13)

        with torch.inference_mode():
            original_output = layer(features)

            for quarter_turns in (1, 2, 3):
                rotated_features = apply_p4_action(
                    features,
                    quarter_turns,
                )
                actual = layer(rotated_features)
                expected = apply_p4_action(
                    original_output,
                    quarter_turns,
                )
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=1e-4,
                    atol=1e-5,
                )

    def test_classifier_logits_are_rotation_invariant(self) -> None:
        model = GroupEquivariantCNN(num_classes=8).eval()
        images = torch.randn(2, 3, 28, 28)

        with torch.inference_mode():
            original_logits = model(images)

            self.assertEqual(tuple(original_logits.shape), (2, 8))

            for quarter_turns in (1, 2, 3):
                rotated_images = torch.rot90(
                    images,
                    k=quarter_turns,
                    dims=(-2, -1),
                )
                rotated_logits = model(rotated_images)
                torch.testing.assert_close(
                    rotated_logits,
                    original_logits,
                    rtol=1e-4,
                    atol=1e-5,
                )


if __name__ == "__main__":
    unittest.main()