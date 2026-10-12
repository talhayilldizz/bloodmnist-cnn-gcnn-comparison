import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import csv
import torch
from sklearn.metrics import accuracy_score, f1_score

from configs.experiment_config import CONFIG
from src.bloodmnist.dataa.dataset import BloodMNISTDataModule
from src.bloodmnist.models.cnn import BaselineCNN
from src.bloodmnist.models.gcnn import GroupEquivariantCNN
from src.bloodmnist.utils.device import resolve_device


MODELS = {
    "baseline_cnn": (
        BaselineCNN,
        "baseline_cnn_best.pt",
    ),
    "p4_augmented_cnn": (
        BaselineCNN,
        "p4_augmented_cnn_best.pt",
    ),
    "gcnn_p4": (
        GroupEquivariantCNN,
        "gcnn_p4_best.pt",
    ),
}

ROTATIONS = (0, 1, 2, 3)  # torch.rot90 için 0°, 90°, 180°, 270°


def main() -> None:
    device = resolve_device(CONFIG.device)
    print(f"Cihaz: {device}")

    data = BloodMNISTDataModule(CONFIG)
    data.setup()
    loader = data.val_dataloader()

    results = []

    for model_name, (model_class, checkpoint_name) in MODELS.items():
        checkpoint_path = CONFIG.checkpoint_dir / checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"{model_name} checkpoint'i bulunamadı: {checkpoint_path}"
            )

        model = model_class(num_classes=CONFIG.num_classes).to(device)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Her açıda tüm validation örneklerinin tahminlerini sakla.
        predictions_by_rotation = {k: [] for k in ROTATIONS}
        all_targets = []

        with torch.inference_mode():
            for images, labels in loader:
                images = images.to(
                    device,
                    non_blocking=(device.type == "cuda"),
                )
                labels = labels.view(-1).long()

                for k in ROTATIONS:
                    rotated_images = torch.rot90(
                        images,
                        k=k,
                        dims=(-2, -1),
                    )
                    logits = model(rotated_images)
                    predictions = logits.argmax(dim=1)

                    predictions_by_rotation[k].extend(
                        predictions.cpu().tolist()
                    )

                all_targets.extend(labels.tolist())

        original_predictions = predictions_by_rotation[0]

        for k in ROTATIONS:
            angle_predictions = predictions_by_rotation[k]

            accuracy = accuracy_score(
                all_targets,
                angle_predictions,
            )
            macro_f1 = f1_score(
                all_targets,
                angle_predictions,
                labels=list(range(CONFIG.num_classes)),
                average="macro",
                zero_division=0,
            )

            # Döndürülmüş görüntünün tahmini, aynı görüntünün 0° tahminiyle
            # ne sıklıkta aynı?
            prediction_agreement = sum(
                original == rotated
                for original, rotated in zip(
                    original_predictions,
                    angle_predictions,
                )
            ) / len(all_targets)

            results.append(
                {
                    "model": model_name,
                    "rotation_degrees": k * 90,
                    "accuracy": accuracy,
                    "macro_f1": macro_f1,
                    "prediction_agreement_with_0_degree":
                        prediction_agreement,
                }
            )

        # Dört açıdaki tahminin de aynı olduğu örneklerin oranı.
        all_angle_agreement = sum(
            len(
                {
                    predictions_by_rotation[k][sample_index]
                    for k in ROTATIONS
                }
            ) == 1
            for sample_index in range(len(all_targets))
        ) / len(all_targets)

        print(f"\n{model_name}")
        print("Açı   Accuracy   Macro-F1   0° tahminiyle uyum")
        for row in results:
            if row["model"] == model_name:
                print(
                    f"{row['rotation_degrees']:>3}°   "
                    f"{row['accuracy']:.4f}     "
                    f"{row['macro_f1']:.4f}      "
                    f"{row['prediction_agreement_with_0_degree']:.4f}"
                )

        print(
            "Dört açıda da tahmini aynı kalan örnek oranı: "
            f"{all_angle_agreement:.4f}"
        )

    output_path = CONFIG.log_dir / "rotation_consistency.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        fieldnames = [
            "model",
            "rotation_degrees",
            "accuracy",
            "macro_f1",
            "prediction_agreement_with_0_degree",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSonuç tablosu kaydedildi: {output_path}")


if __name__ == "__main__":
    main()