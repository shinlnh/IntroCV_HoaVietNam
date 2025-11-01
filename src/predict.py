import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from .config import Config
from .data import IMAGENET_STATS
from .model import build_model


def load_image(path: Path, img_size: int):
    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_STATS["mean"], IMAGENET_STATS["std"]),
    ])
    with Image.open(path) as handle:
        return tfm(handle.convert("RGB")).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--images", nargs="+", required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    cfg = Config.from_file(args.config)
    if "config" in checkpoint:
        cfg.update(checkpoint["config"])

    class_names = checkpoint["class_names"]
    model = build_model(cfg.model["backbone"], len(class_names), cfg.model["dropout"])
    model.load_state_dict(checkpoint["model_state"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    for img_path in args.images:
        tensor = load_image(Path(img_path), cfg.data["img_size"]).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0]
        top_prob, top_idx = probs.max(dim=0)
        print(f"{img_path}: {class_names[top_idx]} ({top_prob:.2%})")


if __name__ == "__main__":
    main()
