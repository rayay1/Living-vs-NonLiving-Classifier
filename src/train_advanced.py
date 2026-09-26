import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import numpy as np

from .data_aug import get_data_transforms
from .models import build_classifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 10
MODEL_NAME = "resnet18"
SAVE_PATH = "living_vs_nonliving.pth"
REPORTS_DIR = "./reports"


def plot_confusion_matrix(cm, classes, output_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix (Living vs Non-Living)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, output_path):
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, 'b-o', label='Train Loss')
    plt.plot(epochs_range, val_losses, 'r-o', label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accs, 'b-o', label='Train Acc (%)')
    plt.plot(epochs_range, val_accs, 'r-o', label='Val Acc (%)')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def train_model(data_dir="./dataset", model_name=MODEL_NAME, epochs=EPOCHS, batch_size=BATCH_SIZE):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    print(f"[*] Training Living-vs-NonLiving Classifier using {DEVICE} | Architecture: {model_name}")

    train_transform, val_transform = get_data_transforms()

    # Load dataset structure
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    if os.path.exists(train_dir) and os.path.exists(val_dir):
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    elif os.path.exists(data_dir):
        full_dataset = datasets.ImageFolder(data_dir, transform=train_transform)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    else:
        print("[!] Warning: Dataset directory not found. Please run dataset_builder.py first!")
        return

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"[v] Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    model = build_classifier(model_name=model_name, num_classes=2, pretrained=True)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_train_loss = running_loss / total
        epoch_train_acc = 100.0 * correct / total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        # Validation
        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_val_loss = val_running_loss / val_total
        epoch_val_acc = 100.0 * val_correct / val_total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)

        scheduler.step()

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}%")

        if epoch_val_acc >= best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            torch.save(model.state_dict(), f"{model_name}_best.pth")

    print(f"\n[v] Training complete! Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"[v] Best weights saved to '{SAVE_PATH}'")

    # Save training curves & confusion matrix
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, os.path.join(REPORTS_DIR, "training_curves.png"))
    
    classes = ['living', 'non_living']
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, classes, os.path.join(REPORTS_DIR, "confusion_matrix.png"))

    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    with open(os.path.join(REPORTS_DIR, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("[v] Saved metrics report, confusion matrix, and training curves to ./reports/")


if __name__ == "__main__":
    train_model()
