"""
Rice Grain Defect Classifier — ResNet-18
==========================================
Classifies individual rice grain images into 6 defect categories:
whole, broken, chalky, damaged, discolored, foreign.

Uses a pretrained ResNet-18 backbone with fine-tuning.
"""

import sys
import os
import argparse
import json
import glob
from collections import Counter

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    RESNET_INPUT_SIZE,
    RESNET_NUM_CLASSES,
    RESNET_BATCH_SIZE,
    RESNET_EPOCHS,
    RESNET_LEARNING_RATE,
    RESNET_WEIGHT_DECAY,
    RESNET_TRAIN_SPLIT,
    RESNET_FREEZE_UNTIL,
    DEFECT_LABELS,
    DEFECT_CATEGORIES,
    EXTRACTED_DEFECT_CATEGORIES,
    MODEL_DIR,
)

# ImageNet normalisation constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Supported image extensions
_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ===================================================================
# 1. Dataset
# ===================================================================
class RiceDefectDataset(Dataset):
    """Custom dataset that loads rice grain images organised by defect category.

    Directory layout expected (mirrors ``DEFECT_CATEGORIES`` from config):
        whole_grain/
            <variety_subfolder>/
                img1.jpg …
        Broken_grain/
            img1.jpg …
        chalky_grain/ …
        Damaged_grain/ …
        Discolored_grain/ …
        Foreign_matter/ …

    For the *whole* category the loader recurses into variety sub-folders;
    for every other category images are collected directly from the directory.
    """

    def __init__(self, transform=None):
        """
        Args:
            transform: torchvision transforms to apply to each image.
        """
        self.transform = transform
        self.samples: list[tuple[str, int]] = []  # (image_path, label_idx)
        self.label_to_idx: dict[str, int] = {
            label: idx for idx, label in enumerate(DEFECT_LABELS)
        }
        self.idx_to_label: dict[int, str] = {
            idx: label for label, idx in self.label_to_idx.items()
        }
        self._collect_samples()

    # ------------------------------------------------------------------
    def _is_image(self, filename: str) -> bool:
        return os.path.splitext(filename)[1].lower() in _IMG_EXTENSIONS

    def _collect_samples(self):
        """Walk through extracted grain crop directories and build (path, label) pairs."""
        # Use extracted individual grain crops (created by extract_grains.py)
        categories = EXTRACTED_DEFECT_CATEGORIES

        for label, dir_path in categories.items():
            if not os.path.isdir(dir_path):
                print(f"[WARN] Directory not found for '{label}': {dir_path}")
                print(f"       Run 'python src/extract_grains.py' first.")
                continue

            idx = self.label_to_idx[label]

            # Collect individual grain crop images
            for fname in os.listdir(dir_path):
                fpath = os.path.join(dir_path, fname)
                if os.path.isfile(fpath) and self._is_image(fname):
                    self.samples.append((fpath, idx))

        print(f"[INFO] RiceDefectDataset: {len(self.samples)} images across "
              f"{len(DEFECT_LABELS)} classes")
        # Per-class counts
        counts = Counter(label for _, label in self.samples)
        for idx in range(len(DEFECT_LABELS)):
            print(f"  {DEFECT_LABELS[idx]:>12s}: {counts.get(idx, 0)}")

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label_idx = self.samples[index]
        image = cv2.imread(path)
        if image is None:
            raise IOError(f"Failed to read image: {path}")
        # BGR → RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        from PIL import Image
        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)
        return image, label_idx

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for balanced loss."""
        counts = Counter(label for _, label in self.samples)
        total = len(self.samples)
        weights = []
        for idx in range(len(DEFECT_LABELS)):
            c = counts.get(idx, 1)  # avoid division by zero
            weights.append(total / (len(DEFECT_LABELS) * c))
        return torch.tensor(weights, dtype=torch.float32)


# ===================================================================
# 2. Transforms
# ===================================================================
def get_transforms(is_training: bool = True) -> transforms.Compose:
    """Return image transforms for training or validation/inference.

    Training augmentations include random crop, flips, rotation and colour
    jitter.  Validation uses a simple resize + centre-crop pipeline.
    """
    if is_training:
        return transforms.Compose([
            transforms.RandomResizedCrop(RESNET_INPUT_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(RESNET_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


# ===================================================================
# 3. Model creation
# ===================================================================
def create_model(
    num_classes: int = RESNET_NUM_CLASSES,
    pretrained: bool = True,
) -> nn.Module:
    """Create a ResNet-18 model for defect classification.

    * Loads ImageNet-pretrained weights (if *pretrained* is True).
    * Replaces the final fully-connected layer to output *num_classes*.
    * Freezes all layers **before** ``RESNET_FREEZE_UNTIL`` (from config).
    """
    if pretrained:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    else:
        model = models.resnet18(weights=None)

    # Replace final FC layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # Freeze layers before RESNET_FREEZE_UNTIL
    freeze_until = RESNET_FREEZE_UNTIL  # e.g. "layer3"
    freeze = True
    for name, param in model.named_parameters():
        if freeze_until in name:
            freeze = False
        if freeze:
            param.requires_grad = False

    # Always keep FC trainable
    for param in model.fc.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[INFO] ResNet-18 created — trainable: {trainable:,} / {total:,} params")
    return model


# ===================================================================
# 4. Training
# ===================================================================
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = RESNET_EPOCHS,
    lr: float = RESNET_LEARNING_RATE,
    device: str = "cpu",
) -> dict:
    """Full training loop with early stopping and best-model checkpointing.

    Returns:
        dict with keys ``train_loss``, ``val_loss``, ``train_acc``,
        ``val_acc`` (each a list of per-epoch values).
    """
    model = model.to(device)

    # Class-weighted loss
    class_weights = train_loader.dataset.get_class_weights().to(device) \
        if hasattr(train_loader.dataset, "get_class_weights") else None
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=RESNET_WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    patience = 10  # early-stopping patience
    best_state = None

    for epoch in range(1, epochs + 1):
        # --- train phase ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # --- val phase ---
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)

        val_loss = running_loss / total
        val_acc = correct / total

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} | "
            f"LR: {lr_now:.2e}"
        )

        # Checkpoint best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Early stopping
        if epoch - best_epoch >= patience:
            print(f"[INFO] Early stopping at epoch {epoch} "
                  f"(best val_loss={best_val_loss:.4f} at epoch {best_epoch})")
            break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[INFO] Restored best model from epoch {best_epoch}")

    # Save checkpoint
    os.makedirs(MODEL_DIR, exist_ok=True)
    ckpt_path = os.path.join(MODEL_DIR, "defect_resnet18_best.pth")
    save_model(model, ckpt_path)
    print(f"[INFO] Best model saved to {ckpt_path}")

    return history


# ===================================================================
# 5. Single & batch prediction
# ===================================================================
def predict_grain(
    model: nn.Module,
    grain_image: np.ndarray,
    device: str = "cpu",
) -> tuple[str, dict[str, float]]:
    """Classify a single grain image (BGR numpy array).

    Returns:
        (predicted_label, confidence_dict)  where confidence_dict maps
        each defect label to its softmax probability.
    """
    model = model.to(device)
    model.eval()

    # BGR → RGB → PIL
    rgb = cv2.cvtColor(grain_image, cv2.COLOR_BGR2RGB)
    from PIL import Image
    pil_img = Image.fromarray(rgb)

    transform = get_transforms(is_training=False)
    tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    confidence = {DEFECT_LABELS[i]: float(probs[i]) for i in range(len(DEFECT_LABELS))}
    predicted_label = DEFECT_LABELS[int(np.argmax(probs))]
    return predicted_label, confidence


def predict_batch(
    model: nn.Module,
    grain_images: list[np.ndarray],
    device: str = "cpu",
) -> list[tuple[str, dict[str, float]]]:
    """Classify a list of grain images (BGR numpy arrays).

    Returns:
        List of (predicted_label, confidence_dict) tuples.
    """
    model = model.to(device)
    model.eval()

    transform = get_transforms(is_training=False)
    from PIL import Image

    tensors = []
    for img in grain_images:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tensors.append(transform(pil_img))

    if not tensors:
        return []

    batch = torch.stack(tensors).to(device)

    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    results = []
    for i in range(len(grain_images)):
        confidence = {
            DEFECT_LABELS[j]: float(probs[i, j]) for j in range(len(DEFECT_LABELS))
        }
        max_idx = int(np.argmax(probs[i]))
        max_prob = float(probs[i, max_idx])
        max_label = DEFECT_LABELS[max_idx]
        
        # Asymmetric confidence: accept defect predictions at lower
        # confidence than "whole" predictions, to catch more defects
        if max_label != "whole" and max_prob >= 0.3:
            # ML thinks it's a defect with moderate confidence → trust it
            label = max_label
        elif max_label == "whole" and max_prob >= 0.5:
            # ML is confident it's whole
            label = "whole"
        else:
            # Not confident enough either way → let hybrid logic decide
            # Still return the best guess so hybrid can use the confidence dict
            label = "unknown"
        results.append((label, confidence))
    return results


# ===================================================================
# 6. Evaluation
# ===================================================================
def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "cpu",
) -> dict:
    """Evaluate model on a test/validation DataLoader.

    Returns:
        dict with keys: ``accuracy``, ``per_class`` (dict of dicts with
        precision / recall / f1 per label), ``confusion_matrix`` (list of
        lists).
    """
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(len(DEFECT_LABELS))), zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(DEFECT_LABELS))))

    per_class = {}
    for i, label in enumerate(DEFECT_LABELS):
        per_class[label] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
        }

    print(f"\n{'='*60}")
    print(f"  Overall Accuracy: {acc:.4f}")
    print(f"{'='*60}")
    print(classification_report(
        all_labels, all_preds,
        target_names=DEFECT_LABELS,
        zero_division=0,
    ))
    print("Confusion Matrix:")
    print(cm)

    return {
        "accuracy": float(acc),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


# ===================================================================
# 7. Save / Load
# ===================================================================
def save_model(model: nn.Module, path: str):
    """Save model state dict to *path*."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[INFO] Model saved to {path}")


def load_model(path: str, num_classes: int = RESNET_NUM_CLASSES) -> nn.Module:
    """Load a saved defect classifier from *path*.

    Creates a fresh ResNet-18 (without pretrained weights) and loads the
    state dict from disk.
    """
    model = create_model(num_classes=num_classes, pretrained=False)
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[INFO] Model loaded from {path}")
    return model


# ===================================================================
# CLI entry point
# ===================================================================
def _parse_args():
    parser = argparse.ArgumentParser(
        description="Rice Grain Defect Classifier (ResNet-18)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--train",
        action="store_true",
        help="Train the defect classifier from scratch",
    )
    group.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate a trained model on the validation set",
    )
    group.add_argument(
        "--predict",
        type=str,
        metavar="IMAGE_PATH",
        help="Predict defect class for a single grain image",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.path.join(MODEL_DIR, "defect_resnet18_best.pth"),
        help="Path to model checkpoint (default: MODEL_DIR/defect_resnet18_best.pth)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=RESNET_EPOCHS,
        help=f"Number of training epochs (default: {RESNET_EPOCHS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=RESNET_BATCH_SIZE,
        help=f"Batch size (default: {RESNET_BATCH_SIZE})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=RESNET_LEARNING_RATE,
        help=f"Learning rate (default: {RESNET_LEARNING_RATE})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (default: cuda if available, else cpu)",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    if args.train:
        # ── Train mode ──────────────────────────────────────────
        print("=" * 60)
        print("  Rice Defect Classifier — Training")
        print("=" * 60)

        # Build full dataset, split into train/val
        full_dataset = RiceDefectDataset(transform=None)
        n_total = len(full_dataset)
        n_train = int(n_total * RESNET_TRAIN_SPLIT)
        n_val = n_total - n_train

        train_ds, val_ds = random_split(
            full_dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

        # Wrap subsets with appropriate transforms
        train_ds.dataset = RiceDefectDataset(transform=get_transforms(is_training=True))
        val_ds_full = RiceDefectDataset(transform=get_transforms(is_training=False))

        # Re-split with same seed so indices match
        train_ds2, val_ds2 = random_split(
            RiceDefectDataset(transform=get_transforms(is_training=True)),
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

        # For validation, use val transforms
        val_ds_with_transform = RiceDefectDataset(transform=get_transforms(is_training=False))
        _, val_ds_final = random_split(
            val_ds_with_transform,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

        train_loader = DataLoader(
            train_ds2,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds_final,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        print(f"[INFO] Train: {n_train} | Val: {n_val}")
        print(f"[INFO] Device: {args.device}")

        model = create_model(pretrained=True)
        history = train_model(
            model, train_loader, val_loader,
            epochs=args.epochs, lr=args.lr, device=args.device,
        )

        # Save training history
        hist_path = os.path.join(MODEL_DIR, "training_history.json")
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"[INFO] Training history saved to {hist_path}")

    elif args.evaluate:
        # ── Evaluate mode ───────────────────────────────────────
        print("=" * 60)
        print("  Rice Defect Classifier — Evaluation")
        print("=" * 60)

        if not os.path.isfile(args.model_path):
            print(f"[ERROR] Model not found at {args.model_path}")
            sys.exit(1)

        model = load_model(args.model_path)
        val_dataset = RiceDefectDataset(transform=get_transforms(is_training=False))

        # Use the val split (same seed as training)
        n_total = len(val_dataset)
        n_train = int(n_total * RESNET_TRAIN_SPLIT)
        n_val = n_total - n_train
        _, val_subset = random_split(
            val_dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )

        results = evaluate_model(model, val_loader, device=args.device)

        # Save results
        eval_path = os.path.join(MODEL_DIR, "evaluation_results.json")
        with open(eval_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[INFO] Evaluation results saved to {eval_path}")

    elif args.predict:
        # ── Predict mode ────────────────────────────────────────
        if not os.path.isfile(args.predict):
            print(f"[ERROR] Image not found: {args.predict}")
            sys.exit(1)
        if not os.path.isfile(args.model_path):
            print(f"[ERROR] Model not found at {args.model_path}")
            sys.exit(1)

        model = load_model(args.model_path)
        image = cv2.imread(args.predict)
        if image is None:
            print(f"[ERROR] Failed to read image: {args.predict}")
            sys.exit(1)

        label, confidence = predict_grain(model, image, device=args.device)

        print(f"\n{'='*60}")
        print(f"  Prediction for: {args.predict}")
        print(f"{'='*60}")
        print(f"  Predicted class : {label}")
        print(f"  Confidence      : {confidence[label]:.4f}")
        print(f"\n  All probabilities:")
        for lbl in DEFECT_LABELS:
            bar = "█" * int(confidence[lbl] * 40)
            print(f"    {lbl:>12s}: {confidence[lbl]:.4f}  {bar}")
        print()


if __name__ == "__main__":
    main()
