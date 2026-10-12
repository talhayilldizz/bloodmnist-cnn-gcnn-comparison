import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


class Evaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        class_names: list[str],
    ):
        self.model = model.to(device)
        self.device = device
        self.class_names = class_names

    def evaluate(self, data_loader) -> dict:
        self.model.eval()
        all_targets = []
        all_predictions = []

        with torch.inference_mode():
            for images, labels in data_loader:
                images = images.to(
                    self.device,
                    non_blocking=(self.device.type == "cuda"),
                )
                labels = labels.view(-1).long().to(self.device)

                logits = self.model(images)
                predictions = logits.argmax(dim=1)

                all_targets.extend(labels.cpu().tolist())
                all_predictions.extend(predictions.cpu().tolist())

        class_ids = list(range(len(self.class_names)))

        precision, recall, macro_f1, _ = precision_recall_fscore_support(
            all_targets,
            all_predictions,
            labels=class_ids,
            average="macro",
            zero_division=0,
        )

        report = classification_report(
            all_targets,
            all_predictions,
            labels=class_ids,
            target_names=self.class_names,
            zero_division=0,
            digits=4,
        )

        matrix = confusion_matrix(
            all_targets,
            all_predictions,
            labels=class_ids,
        )

        return {
            "accuracy": float(accuracy_score(all_targets, all_predictions)),
            "macro_precision": float(precision),
            "macro_recall": float(recall),
            "macro_f1": float(macro_f1),
            "confusion_matrix": matrix,
            "classification_report": report,
        }