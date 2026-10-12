import torch
def resolve_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(preference)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA istendi ancak PyTorch CUDA'yı kullanamıyor.")

    return device