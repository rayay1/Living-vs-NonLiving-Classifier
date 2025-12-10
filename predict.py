import torch
from torchvision import models, transforms
from PIL import Image
import sys

# 設定類別名稱 (要跟 dataset 資料夾順序一樣，通常是照字母順序)
CLASSES = ['living', 'non_living']


def predict_image(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 載入模型架構
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)

    # 2. 載入剛剛訓練好的權重
    try:
        model.load_state_dict(torch.load('living_vs_nonliving.pth', map_location=device))
    except FileNotFoundError:
        print("錯誤：找不到 model.pth，請先執行 train.py！")
        return

    model = model.to(device)
    model.eval()

    # 3. 圖片處理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        image = Image.open(image_path)
        image_tensor = transform(image).unsqueeze(0).to(device)
    except:
        print("錯誤：找不到圖片或是格式不對")
        return

    # 4. 預測
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted].item()

    result = CLASSES[predicted.item()]
    print(f"這張圖是: 【{result}】 (信心水準: {confidence * 100:.1f}%)")


if __name__ == '__main__':
    # 這裡換成你想測試的圖片檔名
    test_img = 'test.jpg'
    predict_image(test_img)