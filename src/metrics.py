from typing import Iterable

import torch
from sklearn.metrics import classification_report, confusion_matrix


def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, topk: Iterable[int] = (1,)):
    maxk = max(topk)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    accs = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        accs.append(correct_k * (1.0 / targets.size(0)))
    return accs


def classification_stats(logits: torch.Tensor, targets: torch.Tensor, class_names: list[str]):
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    truth = targets.cpu().numpy()
    report = classification_report(truth, preds, target_names=class_names, digits=3)
    cm = confusion_matrix(truth, preds)
    return report, cm
