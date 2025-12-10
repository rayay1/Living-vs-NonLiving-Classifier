import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib

# 1. 設定參數
CLASSES = ['living', 'non_living']
Chinese_Labels = {'living': '生物 (Living)', 'non_living': '非生物 (Non-Living)'}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 設定中文字型 (Windows 專用) ---
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
matplotlib.rcParams['axes.unicode_minus'] = False


def predict_and_visualize(image_path):
    print("正在載入模型...")
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)

    try:
        model.load_state_dict(torch.load('living_vs_nonliving.pth', map_location=DEVICE))
    except FileNotFoundError:
        print("錯誤：找不到模型檔，請先執行 train.py")
        return

    model = model.to(DEVICE)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        original_image = Image.open(image_path).convert('RGB')
        image_tensor = transform(original_image).unsqueeze(0).to(DEVICE)
    except Exception as e:
        print(f"無法讀取圖片: {e}")
        return

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        prediction = torch.argmax(probabilities).item()

    class_name = CLASSES[prediction]
    score = probabilities[prediction].item() * 100

    print(f"分析完成！結果: {class_name}, 信心水準: {score:.2f}%")

    # --- 繪圖區 ---
    plt.figure(figsize=(8, 6))
    plt.imshow(original_image)
    ax = plt.gca()

    if class_name == 'living':
        box_color = '#00FF00'
        text_bg = 'green'
    else:
        box_color = '#FF0000'
        text_bg = 'red'

    img_w, img_h = original_image.size
    rect = patches.Rectangle((img_w * 0.1, img_h * 0.1), img_w * 0.8, img_h * 0.8,
                             linewidth=4, edgecolor=box_color, facecolor='none', linestyle='--')
    ax.add_patch(rect)

    text_label = f"{Chinese_Labels[class_name]}\n信心水準: {score:.2f}%"

    plt.text(img_w * 0.05, img_h * 0.15, text_label,
             fontsize=15, color='white', fontweight='bold',
             bbox=dict(facecolor=text_bg, alpha=0.7, edgecolor='white', boxstyle='round,pad=0.5'))

    plt.axis('off')
    plt.title(f"AI Prediction Result", fontsize=14)

    save_name = 'prediction_result_visual.png'
    plt.savefig(save_name, bbox_inches='tight', dpi=150)
    print(f"圖片已儲存為: {save_name}")
    plt.show()


if __name__ == '__main__':
    target_img = 'test.jpg'
    predict_and_visualize(target_img)