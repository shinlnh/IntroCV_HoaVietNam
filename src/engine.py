from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .metrics import topk_accuracy


def create_optimizer(params, cfg: Dict) -> Optimizer:
    name = cfg["optimizer"].lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=cfg["lr"],
            momentum=0.9,
            weight_decay=cfg["weight_decay"],
            nesterov=True,
        )
    raise ValueError(f"Unsupported optimizer {name}")


def train_loop(
    model: nn.Module,
    loaders: Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader],
    train_cfg: Dict,
    device: torch.device,
):
    train_loader, val_loader = loaders
    criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg["label_smoothing"])
    optimizer = create_optimizer(model.parameters(), train_cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_cfg["epochs"] - 1,
        eta_min=1e-6,
    )

    writer = None
    if train_cfg.get("tensorboard", False):
        log_dir = Path("runs") / train_cfg.get("experiment_name", "run")
        writer = SummaryWriter(log_dir=log_dir)

    best_acc = 0.0
    best_state = None
    patience = max(int(train_cfg.get("patience", 0)), 0)
    use_early_stopping = patience > 0
    patience_left = patience
    for epoch in range(train_cfg["epochs"]):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{train_cfg['epochs']}")
        for step, (imgs, labels) in enumerate(progress, start=1):
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            top1 = topk_accuracy(logits, labels, topk=(1,))[0]
            running_loss += loss.item() * imgs.size(0)
            running_acc += top1.item() * imgs.size(0)
            if step % train_cfg["log_interval"] == 0:
                progress.set_postfix(loss=loss.item(), acc=top1.item())
        scheduler.step()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_acc / len(train_loader.dataset)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if writer:
            writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
            writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, epoch)

        print(
            f"[Epoch {epoch + 1}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict()
            patience_left = patience
        elif use_early_stopping:
            patience_left -= 1
            if patience_left == 0:
                print("Early stopping triggered")
                break

    if writer:
        writer.close()
    return best_state, best_acc


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion: nn.Module, device: torch.device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits = model(imgs)
        running_loss += criterion(logits, labels).item() * imgs.size(0)
        running_acc += topk_accuracy(logits, labels, topk=(1,))[0].item() * imgs.size(0)
    dataset_size = len(loader.dataset)
    return running_loss / dataset_size, running_acc / dataset_size
