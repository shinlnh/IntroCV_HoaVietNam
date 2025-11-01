import argparse
from pathlib import Path

import torch

from .config import Config
from .data import build_dataloaders
from .metrics import classification_stats
from .model import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    cfg = Config.from_file(args.config)
    if "config" in checkpoint:
        cfg.update(checkpoint["config"])

    _, val_loader, classes = build_dataloaders(
        cfg.data["root"],
        cfg.data["img_size"],
        cfg.data["batch_size"],
        cfg.data["num_workers"],
        cfg.data.get("augment", {}),
    )

    model = build_model(cfg.model["backbone"], len(classes), cfg.model["dropout"])
    model.load_state_dict(checkpoint["model_state"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    logits_list = []
    labels_list = []
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images)
            logits_list.append(logits.cpu())
            labels_list.append(labels)
    logits = torch.cat(logits_list)
    targets = torch.cat(labels_list)

    report, cm = classification_stats(logits, targets, classes)
    print(report)
    print("Confusion matrix:")
    print(cm)


if __name__ == "__main__":
    main()
