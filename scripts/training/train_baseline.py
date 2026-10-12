import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from torch import nn

from configs.experiment_config import CONFIG
from src.bloodmnist.dataa.dataset import BloodMNISTDataModule
from src.bloodmnist.models.cnn import BaselineCNN
from src.bloodmnist.training.trainer import Trainer
from src.bloodmnist.utils.device import resolve_device
from src.bloodmnist.utils.reproducibility import set_seed


def main() -> None:
    config = CONFIG
    set_seed(config.seed)

    device = resolve_device(config.device)
    print(f"Cihaz: {device}")

    data = BloodMNISTDataModule(config)
    data.setup()

    model = BaselineCNN(num_classes=config.num_classes)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        selection_metric=config.selection_metric,
    )

    result = trainer.fit(
        train_loader=data.train_dataloader(),
        val_loader=data.val_dataloader(),
        epochs=config.epochs,
        checkpoint_path=(
            config.checkpoint_dir / "baseline_cnn_best.pt"
        ),
        history_path=(
            config.log_dir / "baseline_cnn_history.csv"
        ),
    )

    print("\nBaseline eğitimi tamamlandı.")
    print(f"En iyi epoch: {result['best_epoch']}")
    print(
        f"En iyi validation {result['selection_metric']}: "
        f"{result['best_validation_score']:.4f}"
    )
    print(f"Toplam eğitim süresi: {result['training_seconds']:.1f} sn")
    print(f"Checkpoint: {result['checkpoint_path']}")
    print(f"Epoch geçmişi: {result['history_path']}")


if __name__ == "__main__":
    main()