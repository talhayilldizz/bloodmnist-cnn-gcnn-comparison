from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from medmnist import INFO
from medmnist.dataset import BloodMNIST

def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    dataset_info = INFO["bloodmnist"]
    class_names = dataset_info["label"]
    print("Veri kümesi:", dataset_info["python_class"])
    print("Görev:", dataset_info["task"])
    print("Kanal sayısı:", dataset_info["n_channels"])
    print("Sınıf eşlemesi:")
    for class_id, class_name in class_names.items():
        print(f"  {class_id}: {class_name}")

    datasets = {}
    for split in ("train", "val", "test"):
        dataset = BloodMNIST(
            split=split,
            root= str(data_dir),
            download=(split == "train"),
            transform=ToTensor(),
            as_rgb=True
        )

        datasets[split] = dataset
        labels = np.asarray(dataset.labels).reshape(-1)
        distribution = Counter(int(label) for label in labels)

        print(f"\n{split.upper()} bölünmesi")
        print("Örnek sayısı:", len(dataset))
        print("Etiket dizisi şekli:", np.asarray(dataset.labels).shape)
        print("Sınıf dağılımı:")
        for class_id in sorted(class_names, key=int):
            count = distribution.get(int(class_id), 0)
            print(f"  {class_id} ({class_names[class_id]}): {count}")

        image, label = dataset[0]
        print("Bir görüntünün tensör şekli:", tuple(image.shape))
        print("İlk örneğin etiketi:", int(np.asarray(label).reshape(-1)[0]))
        print(
            "Dönüştürülmüş piksel aralığı:",
            f"{image.min().item():.3f}–{image.max().item():.3f}",
        )


    train_loader = DataLoader(
        datasets["train"],
        batch_size=32,
        shuffle=False,
        num_workers=0
    )
    images, labels = next(iter(train_loader))
    print("\nİlk train batch’i")
    print("Görüntü tensörü:", tuple(images.shape))
    print("Etiket tensörü:", tuple(labels.shape))
    print(
        "Batch piksel aralığı:",
        f"{images.min().item():.3f}–{images.max().item():.3f}",
    )

    selected_images = {}
    train_dataset = datasets["train"]
    for index in range(len(train_dataset)):
        image, label = train_dataset[index]
        class_id = int(np.asarray(label).reshape(-1)[0])
        if class_id not in selected_images:
            selected_images[class_id] = image
        if len(selected_images) == len(class_names):
            break

    figure, axes = plt.subplots(2, 4, figsize=(10, 5))
    for class_id, axis in enumerate(axes.flat):
        image = selected_images[class_id].permute(1, 2, 0).numpy()
        axis.imshow(image)
        axis.set_title(f"{class_id}: {class_names[str(class_id)]}")
        axis.axis("off")

    figure.suptitle("BloodMNIST: eğitim kümesinden bir örnek / sınıf")
    figure.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()