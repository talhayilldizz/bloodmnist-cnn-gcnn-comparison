import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib.pyplot as plt
import torch
from medmnist import INFO

from configs.experiment_config import CONFIG
from src.bloodmnist.dataa.dataset import BloodMNISTDataModule
from src.bloodmnist.models.cnn import BaselineCNN
from src.bloodmnist.training.evaluator import Evaluator
from src.bloodmnist.utils.device import resolve_device


def main() -> None:
    checkpoint_path = CONFIG.checkpoint_dir / "baseline_cnn_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint bulunamadı: {checkpoint_path}"
        )

    device = resolve_device(CONFIG.device)

    data = BloodMNISTDataModule(CONFIG)
    data.setup()

    model = BaselineCNN(num_classes=CONFIG.num_classes)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    info = INFO["bloodmnist"]
    class_names = [
        info["label"][str(class_id)]
        for class_id in range(CONFIG.num_classes)
    ]

    evaluator = Evaluator(
        model=model,
        device=device,
        class_names=class_names,
    )
    results = evaluator.evaluate(data.val_dataloader())

    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print("\nValidation metrikleri:")
    for metric in (
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    ):
        print(f"{metric}: {results[metric]:.4f}")

    print("\nSınıf raporu:")
    print(results["classification_report"])

    CONFIG.log_dir.mkdir(parents=True, exist_ok=True)
    report_path = CONFIG.log_dir / "baseline_val_classification_report.txt"
    report_path.write_text(
        "Validation metrikleri\n"
        + "".join(
            f"{metric}: {results[metric]:.4f}\n"
            for metric in (
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
            )
        )
        + "\nSınıf raporu\n"
        + results["classification_report"],
        encoding="utf-8",
    )

    # Uzun sınıf adlarını grafikte okunur tutmak için parantez içini kısalt.
    display_names = [name.split("(")[0].strip() for name in class_names]

    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(
        results["confusion_matrix"],
        interpolation="nearest",
        cmap="Blues",
    )
    figure.colorbar(image, ax=axis)

    axis.set(
        xticks=range(CONFIG.num_classes),
        yticks=range(CONFIG.num_classes),
        xticklabels=display_names,
        yticklabels=display_names,
        xlabel="Tahmin edilen sınıf",
        ylabel="Gerçek sınıf",
        title="Baseline CNN — validation confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")

    matrix = results["confusion_matrix"]
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )

    figure.tight_layout()
    CONFIG.plot_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = CONFIG.plot_dir / "baseline_val_confusion_matrix.png"
    figure.savefig(matrix_path, dpi=160, bbox_inches="tight")

    print(f"\nRapor kaydedildi: {report_path}")
    print(f"Confusion matrix kaydedildi: {matrix_path}")
    plt.show()


if __name__ == "__main__":
    main()