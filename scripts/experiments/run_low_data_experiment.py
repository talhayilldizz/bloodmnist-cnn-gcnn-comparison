import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import csv
import statistics

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, ToTensor

from configs.experiment_config import CONFIG
from src.bloodmnist.dataa.dataset import BloodMNISTDataModule
from src.bloodmnist.dataa.transforms import RandomP4Rotation
from src.bloodmnist.models.cnn import BaselineCNN
from src.bloodmnist.models.gcnn import GroupEquivariantCNN
from src.bloodmnist.training.trainer import Trainer
from src.bloodmnist.utils.device import resolve_device
from src.bloodmnist.utils.reproducibility import set_seed


SEEDS = (42, 43, 44)
FRACTIONS = (0.10, 0.25, 0.50, 1.00)

MODEL_CLASSES = {
    "baseline_cnn": BaselineCNN,
    "p4_augmented_cnn": BaselineCNN,
    "gcnn_p4": GroupEquivariantCNN,
}


def stratified_subset_indices(
    labels: np.ndarray,
    fraction: float,
    seed: int,
) -> list[int]:
    """Her sınıfın oranını koruyarak tekrarlanabilir alt küme seçer."""
    rng = np.random.default_rng(seed)
    selected = []

    for class_id in range(CONFIG.num_classes):
        class_indices = np.flatnonzero(labels == class_id)
        shuffled = rng.permutation(class_indices)

        if fraction == 1.0:
            count = len(shuffled)
        else:
            count = max(1, int(round(len(shuffled) * fraction)))

        selected.extend(shuffled[:count].tolist())

    # DataLoader'a dengeli ama karışık bir indeks sırası ver.
    return rng.permutation(selected).tolist()


def make_data_modules():
    modules = {}

    for model_name in MODEL_CLASSES:
        if model_name == "p4_augmented_cnn":
            train_transform = Compose(
                [
                    ToTensor(),
                    RandomP4Rotation(),
                ]
            )
        else:
            train_transform = ToTensor()

        data = BloodMNISTDataModule(
            CONFIG,
            train_transform=train_transform,
            eval_transform=ToTensor(),
        )
        data.setup()
        modules[model_name] = data

    return modules


def main() -> None:
    device = resolve_device(CONFIG.device)
    print(f"Cihaz: {device}")

    data_modules = make_data_modules()

    # Eğitim veri etiketleri her yöntem için aynı olmalı.
    reference_labels = np.asarray(
        data_modules["baseline_cnn"].train_dataset.labels
    ).reshape(-1)

    for model_name, data in data_modules.items():
        labels = np.asarray(data.train_dataset.labels).reshape(-1)
        if not np.array_equal(reference_labels, labels):
            raise RuntimeError(
                f"{model_name} eğitim etiketleri diğer modellerle eşleşmiyor."
            )

    output_dir = CONFIG.log_dir / "low_data"
    checkpoint_dir = CONFIG.checkpoint_dir / "low_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "low_data_results.csv"
    result_rows = []

    fieldnames = [
        "model",
        "fraction",
        "train_examples",
        "seed",
        "best_epoch",
        "val_accuracy",
        "val_macro_f1",
        "training_seconds",
    ]

    # Her koşunun sonucu hemen CSV'ye yazılır; uzun koşuda ilerleme kaybolmaz.
    with results_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        csv_file.flush()

        for seed in SEEDS:
            for fraction in FRACTIONS:
                subset_indices = stratified_subset_indices(
                    labels=reference_labels,
                    fraction=fraction,
                    seed=seed,
                )
                percent = int(fraction * 100)

                for model_name, model_class in MODEL_CLASSES.items():
                    set_seed(seed)

                    data = data_modules[model_name]
                    train_subset = Subset(
                        data.train_dataset,
                        subset_indices,
                    )
                    train_loader = DataLoader(
                        train_subset,
                        batch_size=CONFIG.batch_size,
                        shuffle=True,
                        num_workers=CONFIG.num_workers,
                        pin_memory=True,
                    )
                    val_loader = data.val_dataloader()

                    model = model_class(
                        num_classes=CONFIG.num_classes
                    )
                    optimizer = torch.optim.AdamW(
                        model.parameters(),
                        lr=CONFIG.learning_rate,
                        weight_decay=CONFIG.weight_decay,
                    )
                    criterion = nn.CrossEntropyLoss()

                    trainer = Trainer(
                        model=model,
                        optimizer=optimizer,
                        criterion=criterion,
                        device=device,
                        selection_metric="macro_f1",
                    )

                    run_name = (
                        f"{model_name}_{percent}pct_seed{seed}"
                    )
                    checkpoint_path = (
                        checkpoint_dir / f"{run_name}_best.pt"
                    )
                    history_path = (
                        output_dir / f"{run_name}_history.csv"
                    )

                    print(
                        f"\nBaşlıyor: {model_name}, "
                        f"%{percent} veri, seed={seed}, "
                        f"örnek={len(subset_indices)}"
                    )

                    result = trainer.fit(
                        train_loader=train_loader,
                        val_loader=val_loader,
                        epochs=CONFIG.epochs,
                        checkpoint_path=checkpoint_path,
                        history_path=history_path,
                    )

                    # Trainer en iyi checkpoint'i validation macro-F1'e göre seçer.
                    best_row = max(
                        result["history"],
                        key=lambda row: row["val_macro_f1"],
                    )

                    row = {
                        "model": model_name,
                        "fraction": fraction,
                        "train_examples": len(subset_indices),
                        "seed": seed,
                        "best_epoch": result["best_epoch"],
                        "val_accuracy": best_row["val_accuracy"],
                        "val_macro_f1": best_row["val_macro_f1"],
                        "training_seconds": result["training_seconds"],
                    }

                    writer.writerow(row)
                    csv_file.flush()
                    result_rows.append(row)

                    print(
                        f"Tamamlandı: val accuracy="
                        f"{row['val_accuracy']:.4f}, "
                        f"val macro-F1={row['val_macro_f1']:.4f}, "
                        f"süre={row['training_seconds']:.1f} sn"
                    )

    print(f"\nSonuçları kaydedildi: {results_path}")

    print("\nÜç seed ortalaması ve standart sapması")
    print(
        "Model                 Veri    "
        "Accuracy ort±std    Macro-F1 ort±std"
    )

    for fraction in FRACTIONS:
        for model_name in MODEL_CLASSES:
            rows = [
                row
                for row in result_rows
                if row["fraction"] == fraction
                and row["model"] == model_name
            ]

            accuracies = [row["val_accuracy"] for row in rows]
            macro_f1s = [row["val_macro_f1"] for row in rows]

            print(
                f"{model_name:21} "
                f"%{int(fraction * 100):<5} "
                f"{statistics.mean(accuracies):.4f} ± "
                f"{statistics.stdev(accuracies):.4f}       "
                f"{statistics.mean(macro_f1s):.4f} ± "
                f"{statistics.stdev(macro_f1s):.4f}"
            )


if __name__ == "__main__":
    main()