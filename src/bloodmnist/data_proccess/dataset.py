from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from medmnist.dataset import BloodMNIST

class BloodMNISTDataModule:
    def __init__(
       self, 
       config, 
       train_transform=None, 
       eval_transform=None
    ):
        self.config = config
        self.train_transform = train_transform or ToTensor()
        self.eval_transform = eval_transform or ToTensor()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self) -> None:
        common_args = {
            "root" : str(self.config.dataset_root),
            "download": False,
            "as_rgb": True,
        }

        self.train_dataset = BloodMNIST(
            split="train",
            transform=self.train_transform,
            **common_args
        )
        self.val_dataset = BloodMNIST(
            split="val",
            transform=self.eval_transform,
            **common_args
        )


    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Önce setup() çağrılmalıdır.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Önce setup() çağrılmalıdır.")

        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Test kümesini yalnızca final değerlendirmesi istendiğinde yükler."""
        if self.test_dataset is None:
            self.test_dataset = BloodMNIST(
                split="test",
                root=str(self.config.dataset_root),
                download=False,
                as_rgb=True,
                transform=self.eval_transform,
            )

        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )