import csv
import time
from pathlib import Path
import torch
from tqdm import tqdm
from bloodmnist.training.metrics import classification_metrics

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: torch.device,
        selection_metric: str = "macro_f1",
    ):
        if selection_metric not in {"macro_f1", "accuracy"}:
            raise ValueError(
                "selection_metric 'macro_f1' veya 'accuracy' olmalıdır."
            )

        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.selection_metric = selection_metric


    def _run_epoch(self, loader, training: bool) -> dict[str, float]:
        self.model.train(training)

        total_loss = 0.0
        total_examples = 0
        all_targets = []
        all_predictions = []

        phase = "train" if training else "validation"

        for images, labels in tqdm(loader, desc=phase, leave=False):
            images = images.to(
                self.device,
                non_blocking=(self.device.type == "cuda"),
            )
            labels = labels.view(-1).long().to(self.device)

            if training:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(training):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

                if training:
                    loss.backward()
                    self.optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

            predictions = logits.argmax(dim=1)
            all_targets.extend(labels.detach().cpu().tolist())
            all_predictions.extend(predictions.detach().cpu().tolist())

        if total_examples == 0:
            raise RuntimeError(f"{phase} DataLoader boş.")

        metrics = classification_metrics(all_targets, all_predictions)
        metrics["loss"] = total_loss / total_examples
        return metrics

    @staticmethod
    def _write_history(
        history: list[dict],
        path: Path
    )-> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(history[0].keys())

        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int,
        checkpoint_path: Path,
        history_path: Path,
    ) -> dict:
        if epochs < 1:
            raise ValueError("epochs en az 1 olmalıdır.")

        checkpoint_path = Path(checkpoint_path)
        history_path = Path(history_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        history = []
        best_score = float("-inf")
        best_epoch = 0
        start_time = time.perf_counter()

        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()

            train_metrics = self._run_epoch(train_loader, training=True)
            val_metrics = self._run_epoch(val_loader, training=False)

            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "epoch_seconds": time.perf_counter() - epoch_start,
            }
            history.append(row)
            self._write_history(history, history_path)

            score = val_metrics[self.selection_metric]
            if score > best_score:
                best_score = score
                best_epoch = epoch

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "selection_metric": self.selection_metric,
                        "selection_score": best_score,
                    },
                    checkpoint_path,
                )

            print(
                f"Epoch {epoch:02d}/{epochs} | "
                f"train loss {row['train_loss']:.4f}, "
                f"acc {row['train_accuracy']:.4f}, "
                f"macro-F1 {row['train_macro_f1']:.4f} | "
                f"val loss {row['val_loss']:.4f}, "
                f"acc {row['val_accuracy']:.4f}, "
                f"macro-F1 {row['val_macro_f1']:.4f} | "
                f"{row['epoch_seconds']:.1f} sn"
            )

        total_seconds = time.perf_counter() - start_time

        return {
            "best_epoch": best_epoch,
            "best_validation_score": best_score,
            "selection_metric": self.selection_metric,
            "training_seconds": total_seconds,
            "checkpoint_path": checkpoint_path,
            "history_path": history_path,
            "history": history,
        }
