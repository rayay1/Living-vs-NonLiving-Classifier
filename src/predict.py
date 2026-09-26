import os
import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib

# Matplotlib Chinese font support
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

CLASSES = ['living', 'non_living']
CHINESE_LABELS = {'living': '生物 (Living)', 'non_living': '非生物 (Non-Living)'}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOURCE_FOLDER = 'test_images'
OUTPUT_FOLDER = 'predict_results'


def load_classifier(model_path='living_vs_nonliving.pth'):
    """Loads the ResNet-18 Living vs Non-Living classifier."""
    print("[*] Loading ResNet18 Classifier...")
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        print(f"[!] Warning: '{model_path}' not found. Using pretrained backbone for inference.")
        
    model = model.to(DEVICE)
    model.eval()
    return model


def load_detector():
    """Loads Faster R-CNN Object Detector for bounding box region proposal."""
    print("[*] Loading Faster R-CNN Object Detector...")
    detector = models.detection.fasterrcnn_resnet50_fpn(weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    detector = detector.to(DEVICE)
    detector.eval()
    return detector


def get_bounding_box(detector, image_tensor):
    """Finds the most prominent bounding box in the input image."""
    with torch.no_grad():
        predictions = detector(image_tensor)[0]

    keep = predictions['scores'] > 0.3
    boxes = predictions['boxes'][keep].cpu().numpy()

    if len(boxes) > 0:
        max_area = 0
        best_box = None
        for box in boxes:
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                best_box = (x1, y1, x2, y2)
        return best_box
    return None


def run_prediction():
    if not os.path.exists(SOURCE_FOLDER):
        os.makedirs(SOURCE_FOLDER, exist_ok=True)
        print(f"[!] Warning: '{SOURCE_FOLDER}' directory was empty. Please place test images there.")
        return

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    classifier = load_classifier()
    detector = load_detector()

    classifier_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    detector_transform = transforms.ToTensor()
    image_files = [f for f in os.listdir(SOURCE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"[!] No image files found in '{SOURCE_FOLDER}'. Please add images to test prediction.")
        return

    print(f"[*] Found {len(image_files)} test images. Processing inference...")

    for img_name in image_files:
        img_path = os.path.join(SOURCE_FOLDER, img_name)
        raw_img = Image.open(img_path).convert("RGB")
        img_w, img_h = raw_img.size

        det_tensor = detector_transform(raw_img).unsqueeze(0).to(DEVICE)
        box = get_bounding_box(detector, det_tensor)

        if box is not None:
            x1, y1, x2, y2 = map(int, box)
            cropped_img = raw_img.crop((x1, y1, x2, y2))
        else:
            cropped_img = raw_img
            x1, y1, x2, y2 = 0, 0, img_w, img_h

        cls_tensor = classifier_transform(cropped_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = classifier(cls_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, 0)

        pred_class = CLASSES[predicted_idx.item()]
        zh_label = CHINESE_LABELS[pred_class]
        conf_percent = confidence.item() * 100

        # Draw and Save Output Visualization
        fig, ax = plt.subplots(1, figsize=(8, 6))
        ax.imshow(raw_img)

        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                 edgecolor='r' if pred_class == 'living' else 'b',
                                 facecolor='none')
        ax.add_patch(rect)

        label_text = f"{zh_label}: {conf_percent:.1f}%"
        ax.text(x1, max(y1 - 10, 15), label_text, color='white',
                fontsize=12, weight='bold',
                bbox=dict(facecolor='red' if pred_class == 'living' else 'blue', alpha=0.7, pad=2))

        plt.axis('off')
        plt.tight_layout()

        output_path = os.path.join(OUTPUT_FOLDER, f"result_{img_name}")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

        print(f"[v] Processed: {img_name} => Prediction: {zh_label} ({conf_percent:.1f}%) | Saved to {output_path}")

    print(f"\n[v] All prediction results saved to '{OUTPUT_FOLDER}/'")


if __name__ == "__main__":
    run_prediction()
