import torch

class RandomP4Rotation:
    def __call__(
        self,
        image: torch.Tensor
    ) -> torch.Tensor:
        if not isinstance(image, torch.Tensor):
            raise TypeError(
                "RandomP4Rotation, ToTensor() sonrasında kullanılmalıdır."
            )

        if image.ndim != 3:
            raise ValueError(
                "Beklenen görüntü şekli (kanal, yükseklik, genişlik); "
                f"alınan şekil: {tuple(image.shape)}"
            )

        quarter_turns = int(torch.randint(low=0, high=4, size=()).item())
        return torch.rot90(image, k=quarter_turns, dims=(-2, -1))