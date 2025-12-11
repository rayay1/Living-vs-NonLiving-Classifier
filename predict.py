import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import os

# --- 設定參數 ---
CLASSES = ['living', 'non_living']
Chinese_Labels = {'living': '生物 (Living)', 'non_living': '非生物 (Non-Living)'}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOURCE_FOLDER = 'test_images'
OUTPUT_FOLDER = 'predict_results'

# --- 設定中文字型 ---
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# --- 1. 載入你的分類模型 (負責判斷是誰) ---
def load_classifier():
    print("正在載入你的分類模型 (ResNet18)...")
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)

    try:
        model.load_state_dict(torch.load('living_vs_nonliving.pth', map_location=DEVICE))
    except FileNotFoundError:
        print("錯誤：找不到 model.pth")
        return None
    model = model.to(DEVICE)
    model.eval()
    return model


# --- 2. 載入定位助手 (負責找位置) ---
# 使用 PyTorch 內建的 Faster R-CNN，它看過幾百萬張圖，很會找東西
def load_detector():
    print("正在載入定位助手 (Faster R-CNN)...")
    # 使用預設權重 (COCO dataset)
    detector = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    detector = detector.to(DEVICE)
    detector.eval()
    return detector


# --- 3. 核心功能：找出最明顯的物體框框 ---
def get_dynamic_box(detector, image_tensor, img_w, img_h):
    with torch.no_grad():
        predictions = detector(image_tensor)[0]

    # 過濾：只留信心分數 > 0.25 的框框
    keep = predictions['scores'] > 0.25
    boxes = predictions['boxes'][keep].cpu().numpy()

    if len(boxes) > 0:
        # 策略：如果有好幾個物體，我們選「面積最大」的那一個
        max_area = 0
        best_box = None

        for box in boxes:
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                best_box = (x1, y1, x2 - x1, y2 - y1)  # 轉成 x, y, w, h

        return best_box  # 回傳抓到的框框
    else:
        # 如果助手眼殘沒看到東西，就退回原本的「中間 80%」方案
        return (img_w * 0.1, img_h * 0.1, img_w * 0.8, img_h * 0.8)


def process_one_image(classifier, detector, image_path, filename):
    # 預處理 (給分類器用)
    transform_cls = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 預處理 (給定位助手用 - 只要轉 Tensor)
    transform_det = transforms.ToTensor()

    try:
        original_image = Image.open(image_path).convert('RGB')
        img_w, img_h = original_image.size

        # 準備資料
        input_tensor_cls = transform_cls(original_image).unsqueeze(0).to(DEVICE)
        input_tensor_det = transform_det(original_image).unsqueeze(0).to(DEVICE)

    except Exception as e:
        print(f"無法讀取 {filename}: {e}")
        return

    # A. 你的模型判斷類別
    with torch.no_grad():
        outputs = classifier(input_tensor_cls)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        prediction = torch.argmax(probabilities).item()

    class_name = CLASSES[prediction]
    score = probabilities[prediction].item() * 100

    # B. 定位助手找框框
    print(f"正在偵測 {filename} 的物體位置...")
    box_x, box_y, box_w, box_h = get_dynamic_box(detector, input_tensor_det, img_w, img_h)

    # --- 繪圖 ---
    plt.figure(figsize=(8, 6))
    plt.imshow(original_image)
    ax = plt.gca()

    if class_name == 'living':
        box_color = '#00FF00'
        text_bg = 'green'
    else:
        box_color = '#FF0000'
        text_bg = 'red'

    # 畫出動態框框
    rect = patches.Rectangle((box_x, box_y), box_w, box_h,
                             linewidth=3, edgecolor=box_color, facecolor='none', linestyle='--')
    ax.add_patch(rect)

    # 文字標籤 (放在框框的左上角)
    text_label = f"{Chinese_Labels[class_name]}\n信心: {score:.1f}%"
    plt.text(box_x, max(box_y - 20, 10), text_label,
             fontsize=12, color='white', fontweight='bold',
             bbox=dict(facecolor=text_bg, alpha=0.7, edgecolor='white', boxstyle='round,pad=0.3'))

    plt.axis('off')
    #plt.title(f"Result: {filename}", fontsize=12)

    save_path = os.path.join(OUTPUT_FOLDER, f"result_{filename}")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"✅ 完成: {filename}")

    print("👉 請關閉圖片視窗繼續...")
    plt.show()


def run():
    if not os.path.exists(SOURCE_FOLDER):
        print(f"錯誤：找不到 '{SOURCE_FOLDER}' 資料夾！")
        return
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # 載入兩個模型
    classifier = load_classifier()
    detector = load_detector()  # 這是新的

    if classifier is None: return

    image_files = [f for f in os.listdir(SOURCE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print("沒有圖片！")
        return

    for img_name in image_files:
        img_path = os.path.join(SOURCE_FOLDER, img_name)
        process_one_image(classifier, detector, img_path, img_name)


if __name__ == '__main__':
    run()