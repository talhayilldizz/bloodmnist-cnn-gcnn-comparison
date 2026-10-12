import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt

from configs.experiment_config import CONFIG


def main() -> None:
    history_path = CONFIG.log_dir / "baseline_cnn_history.csv"

    if not history_path.exists():
        raise FileNotFoundError(
            f"Eğitim geçmişi bulunamadı: {history_path}"
        )

    with history_path.open("r", newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    if not rows:
        raise RuntimeError(f"Eğitim geçmişi boş: {history_path}")

    epochs = [int(row["epoch"]) for row in rows]
    metrics = {
        "Loss": ("train_loss", "val_loss"),
        "Accuracy": ("train_accuracy", "val_accuracy"),
        "Macro-F1": ("train_macro_f1", "val_macro_f1"),
    }

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for axis, (title, (train_key, val_key)) in zip(axes, metrics.items()):
        train_values = [float(row[train_key]) for row in rows]
        val_values = [float(row[val_key]) for row in rows]

        axis.plot(epochs, train_values, marker="o", label="Train")
        axis.plot(epochs, val_values, marker="o", label="Validation")
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_xticks(epochs)
        axis.grid(alpha=0.3)
        axis.legend()

    figure.suptitle("Baseline CNN — eğitim ve validation eğrileri")
    figure.tight_layout()

    CONFIG.plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = CONFIG.plot_dir / "baseline_training_curves.png"
    figure.savefig(plot_path, dpi=160, bbox_inches="tight")

    print(f"Grafik kaydedildi: {plot_path}")
    plt.show()


if __name__ == "__main__":
    main()