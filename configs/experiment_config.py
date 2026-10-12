from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

@dataclass
class ExperimentConfig:
    """Üç deneyde ortak kullanılacak başlangıç ayarları."""

    seed: int = 42
    batch_size: int = 32
    epochs: int = 20
    optimizer_name: str = "AdamW"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    loss_name: str = "CrossEntropyLoss"
    device: str = "auto"
    num_workers: int = 0
    dataset_name: str = "bloodmnist"
    dataset_root: Path = PROJECT_ROOT / "data"
    image_channels: int = 3
    image_size: int = 28
    num_classes: int = 8
    selection_metric: str = "macro_f1"
    output_root: Path = PROJECT_ROOT / "outputs"
    checkpoint_dir: Path = PROJECT_ROOT / "outputs" / "checkpoints"
    plot_dir: Path = PROJECT_ROOT / "outputs" / "plots"
    log_dir: Path = PROJECT_ROOT / "outputs" / "logs"

CONFIG = ExperimentConfig()