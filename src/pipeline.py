"""
PlatesV2 — классификация тарелок cleaned/dirty.
ResNet18 (transfer learning) + 5-fold StratifiedKFold + TTA.

Запуск:
    cd ~/projects/platesv2
    uv run python src/pipeline.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "plates"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 8
NUM_EPOCHS = 25
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
N_FOLDS = 5
N_TTA = 5

CLASSES = ["cleaned", "dirty"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --- Данные ----------------------------------------------------------------
def collect_train() -> pd.DataFrame:
    rows = []
    for cls in CLASSES:
        for p in sorted((DATA_DIR / "train" / cls).iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                rows.append({"path": str(p), "class": cls, "label": CLASS_TO_IDX[cls]})
    return pd.DataFrame(rows)


def collect_test() -> pd.DataFrame:
    rows = []
    for p in sorted((DATA_DIR / "test").iterdir()):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            rows.append({"path": str(p), "id": p.stem})
    return pd.DataFrame(rows)


class PlatesDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform, has_labels: bool = True):
        self.paths = df["path"].tolist()
        self.transform = transform
        self.has_labels = has_labels
        self.labels = df["label"].tolist() if has_labels else None

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)
        if self.has_labels:
            return img, self.labels[idx]
        return img, self.paths[idx]


# --- Аугментации -----------------------------------------------------------
def build_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


# --- EDA -------------------------------------------------------------------
def run_eda(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    sizes = [Image.open(p).size for p in train_df["path"]]
    widths, heights = zip(*sizes)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    train_df["class"].value_counts().plot.bar(ax=axes[0], color=["steelblue", "tomato"])
    axes[0].set_title("Class balance (train)")
    axes[1].hist(widths, bins=15, color="steelblue", alpha=0.7, label="width")
    axes[1].hist(heights, bins=15, color="tomato", alpha=0.7, label="height")
    axes[1].set_title("Image dimensions"); axes[1].legend()
    axes[2].scatter(widths, heights, alpha=0.6)
    axes[2].set_xlabel("width"); axes[2].set_ylabel("height"); axes[2].set_title("W x H")
    plt.tight_layout(); plt.savefig(OUTPUTS_DIR / "eda_distributions.png", dpi=100); plt.close()

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for row, cls in enumerate(CLASSES):
        paths = train_df[train_df["class"] == cls]["path"].sample(5, random_state=SEED).tolist()
        for col, p in enumerate(paths):
            axes[row, col].imshow(Image.open(p))
            axes[row, col].set_title(cls); axes[row, col].axis("off")
    plt.tight_layout(); plt.savefig(OUTPUTS_DIR / "eda_samples.png", dpi=100); plt.close()

    return {
        "n_train": len(train_df), "n_test": len(test_df),
        "n_cleaned": int((train_df["class"] == "cleaned").sum()),
        "n_dirty": int((train_df["class"] == "dirty").sum()),
        "width_min": int(min(widths)), "width_max": int(max(widths)),
        "width_mean": float(np.mean(widths)),
        "height_min": int(min(heights)), "height_max": int(max(heights)),
        "height_mean": float(np.mean(heights)),
    }


# --- Модель ----------------------------------------------------------------
def build_model() -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Sequential(
        nn.Dropout(DROPOUT),
        nn.Linear(model.fc.in_features, 2),
    )
    return model


# --- Обучение --------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses, preds, targets = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward(); optimizer.step()
        losses.append(loss.item())
        preds.extend(out.argmax(1).cpu().numpy()); targets.extend(y.cpu().numpy())
    return float(np.mean(losses)), accuracy_score(targets, preds)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    losses, preds, targets = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x); loss = criterion(out, y)
        losses.append(loss.item())
        preds.extend(out.argmax(1).cpu().numpy()); targets.extend(y.cpu().numpy())
    return float(np.mean(losses)), accuracy_score(targets, preds), preds, targets


def train_fold(model, train_loader, val_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3,
    )
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc, best_state = 0.0, None
    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_acc)
        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss); history["val_acc"].append(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"  ep {epoch:02d} | tr_loss {tr_loss:.3f} tr_acc {tr_acc:.3f} | "
              f"val_loss {val_loss:.3f} val_acc {val_acc:.3f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_acc


# --- TTA + предсказания ----------------------------------------------------
@torch.no_grad()
def predict_with_tta(model, test_df, eval_tf, device, n_tta=N_TTA):
    tta_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    model.eval()
    probs_sum = np.zeros((len(test_df), 2), dtype=np.float32)

    def run(tf):
        ds = PlatesDataset(test_df, tf, has_labels=False)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
        out = []
        for x, _ in loader:
            out.append(torch.softmax(model(x.to(device)), dim=1).cpu().numpy())
        return np.vstack(out)

    probs_sum += run(eval_tf)  # базовый прогон
    for _ in range(n_tta):
        probs_sum += run(tta_tf)
    return probs_sum / (n_tta + 1)


# --- Отчётные графики ------------------------------------------------------
def plot_history(history: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend()
    plt.tight_layout(); plt.savefig(OUTPUTS_DIR / "training_history.png", dpi=100); plt.close()


def plot_confusion(cm: np.ndarray, acc: float) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xticks([0, 1]); ax.set_xticklabels(CLASSES)
    ax.set_yticks([0, 1]); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"OOF confusion matrix (acc={acc:.2f})")
    plt.tight_layout(); plt.savefig(OUTPUTS_DIR / "confusion_matrix.png", dpi=100); plt.close()


def plot_errors(train_df, err_mask, preds) -> None:
    err_df = train_df.iloc[np.where(err_mask)[0]].reset_index(drop=True)
    err_df["pred"] = preds[err_mask]
    err_df["true"] = train_df["label"].values[err_mask]
    n_show = min(8, len(err_df))
    if n_show == 0:
        return
    fig, axes = plt.subplots(1, n_show, figsize=(3 * n_show, 3))
    if n_show == 1:
        axes = [axes]
    for ax, (_, row) in zip(axes, err_df.head(n_show).iterrows()):
        ax.imshow(Image.open(row["path"]))
        ax.set_title(f"T:{IDX_TO_CLASS[row['true']]}\nP:{IDX_TO_CLASS[row['pred']]}")
        ax.axis("off")
    plt.tight_layout(); plt.savefig(OUTPUTS_DIR / "validation_errors.png", dpi=100); plt.close()


def make_submission(test_df, probs) -> Path:
    pred_labels = [IDX_TO_CLASS[i] for i in probs.argmax(axis=1)]
    sub = pd.DataFrame({"id": test_df["id"].values, "label": pred_labels})
    sub = sub.sort_values("id").reset_index(drop=True)
    path = OUTPUTS_DIR / "submission.csv"
    sub.to_csv(path, index=False)
    return path


# --- Main ------------------------------------------------------------------
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_df = collect_train().reset_index(drop=True)
    test_df = collect_test().reset_index(drop=True)
    eda = run_eda(train_df, test_df)
    print(f"Train: {len(train_df)} | Test: {len(test_df)} | EDA saved")

    train_tf, eval_tf = build_transforms()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_accs, histories = [], []
    test_probs_sum = np.zeros((len(test_df), 2), dtype=np.float32)
    oof_preds = np.zeros(len(train_df), dtype=np.int64)
    oof_true = train_df["label"].values

    for fold, (tr_idx, vl_idx) in enumerate(skf.split(train_df, train_df["label"]), start=1):
        print(f"\n=== FOLD {fold}/{N_FOLDS} ===")
        tr = train_df.iloc[tr_idx].reset_index(drop=True)
        vl = train_df.iloc[vl_idx].reset_index(drop=True)
        train_loader = DataLoader(PlatesDataset(tr, train_tf), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader = DataLoader(PlatesDataset(vl, eval_tf), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        model = build_model().to(device)
        model, hist, best_acc = train_fold(model, train_loader, val_loader, device)
        fold_accs.append(best_acc); histories.append(hist)

        _, _, vp, _ = evaluate(model, val_loader, nn.CrossEntropyLoss(), device)
        oof_preds[vl_idx] = vp
        test_probs_sum += predict_with_tta(model, test_df, eval_tf, device)
        torch.save(model.state_dict(), OUTPUTS_DIR / f"model_fold{fold}.pt")

    test_probs = test_probs_sum / N_FOLDS

    print(f"\nFold val accs: {[round(a, 3) for a in fold_accs]}")
    print(f"Mean: {np.mean(fold_accs):.3f} +- {np.std(fold_accs):.3f}")

    plot_history(histories[0])

    oof_acc = accuracy_score(oof_true, oof_preds)
    oof_f1 = f1_score(oof_true, oof_preds, average="macro")
    cm = confusion_matrix(oof_true, oof_preds, labels=[0, 1])
    print(f"\nOOF acc {oof_acc:.3f}, macro-F1 {oof_f1:.3f}")
    print(classification_report(oof_true, oof_preds, target_names=CLASSES, zero_division=0))
    plot_confusion(cm, oof_acc)
    plot_errors(train_df, oof_preds != oof_true, oof_preds)

    sub_path = make_submission(test_df, test_probs)
    print(f"Submission saved: {sub_path}")

    summary = {
        "device": str(device),
        "config": {
            "img_size": IMG_SIZE, "batch_size": BATCH_SIZE, "epochs": NUM_EPOCHS,
            "lr": LR, "weight_decay": WEIGHT_DECAY, "dropout": DROPOUT,
            "n_folds": N_FOLDS, "n_tta": N_TTA, "seed": SEED,
            "architecture": "ResNet18 (ImageNet), backbone frozen; Dropout(0.3) + Linear(512, 2); "
                            "5-fold StratifiedKFold + TTA(x6); Adam + ReduceLROnPlateau",
        },
        "eda": eda,
        "fold_val_accs": [float(a) for a in fold_accs],
        "mean_val_acc": float(np.mean(fold_accs)),
        "std_val_acc": float(np.std(fold_accs)),
        "oof_acc": float(oof_acc), "oof_macro_f1": float(oof_f1),
        "oof_confusion_matrix": cm.tolist(),
        "submission": str(sub_path),
    }
    (OUTPUTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Done. Artifacts in {OUTPUTS_DIR}/")


if __name__ == "__main__":
    main()
