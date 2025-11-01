import argparse
from pathlib import Path

import torch

from .config import Config
from .data import build_dataloaders
from .engine import train_loop
from .model import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.from_file(args.config)
    overrides: dict = {}
    if args.data_root:
        overrides.setdefault("data", {})["root"] = args.data_root
    if args.epochs:
        overrides.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size:
        overrides.setdefault("data", {})["batch_size"] = args.batch_size
    if args.lr:
        overrides.setdefault("training", {})["lr"] = args.lr
    if args.backbone:
        overrides.setdefault("model", {})["backbone"] = args.backbone
    if overrides:
        cfg.update(overrides)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, classes = build_dataloaders(
        cfg.data["root"],
        cfg.data["img_size"],
        cfg.data["batch_size"],
        cfg.data["num_workers"],
        cfg.data.get("augment", {}),
    )

    model = build_model(cfg.model["backbone"], len(classes), cfg.model["dropout"]).to(device)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])

    best_state, best_acc = train_loop(
        model,
        (train_loader, val_loader),
        {
            **cfg.training,
            **cfg.logging,
            "experiment_name": cfg.experiment_name,
        },
        device,
    )

    output_dir = Path(cfg.logging["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if best_state:
        torch.save(
            {
                "model_state": best_state,
                "class_names": classes,
                "config": cfg.to_dict(),
                "val_acc": best_acc,
            },
            output_dir / f"{cfg.experiment_name}.pt",
        )
        print(f"Saved best checkpoint with val_acc={best_acc:.4f}")


if __name__ == "__main__":
    main()
