import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import os

# 1. 設定參數
BATCH_SIZE = 8
LEARNING_RATE = 0.001
EPOCHS = 10  # 訓練幾輪，越多通常越準但越久
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model():
    print(f"使用裝置: {DEVICE}")

    # 2. 圖片預處理 (縮放、轉成Tensor、正規化)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. 讀取資料
    if not os.path.exists('./dataset'):
        print("錯誤：找不到 dataset 資料夾！請確認資料夾結構。")
        return

    dataset = datasets.ImageFolder('./dataset', transform=transform)

    # 分割 80% 訓練, 20% 驗證
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    print(f"生物類別: {dataset.class_to_idx}")
    print(f"訓練圖片: {len(train_data)} 張, 驗證圖片: {len(val_data)} 張")

    # 4. 建立模型 (使用預訓練的 ResNet18)
    model = models.resnet18(pretrained=True)
    # 修改最後一層，因為我們只有 2 類 (生物 vs 非生物)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    # 5. 開始訓練
    acc_history = []
    loss_history = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_acc = 100 * correct / total
        epoch_loss = running_loss / len(train_loader)
        acc_history.append(epoch_acc)
        loss_history.append(epoch_loss)

        print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}%")

    # 6. 儲存模型
    torch.save(model.state_dict(), 'living_vs_nonliving.pth')
    print("模型已儲存為 living_vs_nonliving.pth")

    # 7. 畫圖 (給報告用)
    plt.plot(acc_history)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.savefig('result_chart.png')
    print("訓練圖表已儲存為 result_chart.png")


if __name__ == '__main__':
    train_model()