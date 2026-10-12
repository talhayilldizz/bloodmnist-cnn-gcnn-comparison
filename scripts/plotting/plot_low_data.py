import csv
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np


RESULTS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "logs"
    / "low_data"
    / "low_data_results.csv"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "plots"
    / "low_data_performance.png"
)

MODEL_LABELS = {
    "baseline_cnn": "Baseline CNN",
    "p4_augmented_cnn": "P4 augmentasyonlu CNN",
    "gcnn_p4": "P4 G-CNN",
}
FRACTIONS = (0.10, 0.25, 0.50, 1.00)
METRICS = (
    ("val_macro_f1", "Validation macro-F1"),
    ("val_accuracy", "Validation accuracy"),
)


def main() -> None:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Az veri sonuç CSV'si bulunamadı: {RESULTS_PATH}"
        )

    grouped = defaultdict(lambda: defaultdict(list))
    with RESULTS_PATH.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        for row in csv.DictReader(csv_file):
            model = row["model"]
            fraction = float(row["fraction"])
            grouped[model][fraction].append(row)

    expected_models = set(MODEL_LABELS)
    if set(grouped) != expected_models:
        raise ValueError(
            "CSV'deki model adları beklenen modellerle eşleşmiyor: "
            f"{sorted(grouped)}"
        )

    for model in MODEL_LABELS:
        for fraction in FRACTIONS:
            if len(grouped[model][fraction]) < 2:
                raise ValueError(
                    f"{MODEL_LABELS[model]} için %{fraction * 100:g} "
                    "sonucunda standart sapma hesaplamak için en az iki "
                    "seed sonucu gerekir."
                )

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12, 5.4),
        sharex=True,
        sharey=True,
    )

    x_values = np.asarray(FRACTIONS) * 100
    markers = {
        "baseline_cnn": "o",
        "p4_augmented_cnn": "s",
        "gcnn_p4": "^",
    }

    for axis, (metric, title) in zip(axes, METRICS):
        for model, label in MODEL_LABELS.items():
            means = []
            standard_deviations = []

            for fraction in FRACTIONS:
                values = np.asarray(
                    [
                        float(row[metric])
                        for row in grouped[model][fraction]
                    ],
                    dtype=float,
                )
                means.append(values.mean())
                standard_deviations.append(values.std(ddof=1))

            axis.errorbar(
                x_values,
                means,
                yerr=standard_deviations,
                marker=markers[model],
                linewidth=2,
                markersize=6,
                capsize=4,
                label=label,
            )

        axis.set_title(title)
        axis.set_xlabel("Eğitim verisi (%)")
        axis.set_xticks(x_values, labels=["10", "25", "50", "100"])
        axis.set_ylim(0.55, 1.00)
        axis.grid(True, linestyle="--", alpha=0.35)

    axes[0].set_ylabel("Skor")
    axes[1].legend(loc="lower right", frameon=True)
    figure.suptitle(
        "BloodMNIST az veri deneyi — validation başarımı\n"
        "Noktalar: 3 seed ortalaması; hata çubukları: ±1 standart sapma",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)

    print(f"Grafik kaydedildi: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
