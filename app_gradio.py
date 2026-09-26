import os
import torch
import gradio as gr
from PIL import Image
from torchvision import models, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['Living (生物)', 'Non-Living (非生物)']

# Build & Load Classifier
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)

model_weights = "living_vs_nonliving.pth"
if os.path.exists(model_weights):
    model.load_state_dict(torch.load(model_weights, map_location=DEVICE))
    print("[v] Loaded trained classifier weights.")
else:
    print("[!] Model weights file not found. Running with default backbone.")

model = model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def classify_image(input_img):
    if input_img is None:
        return None
        
    pil_img = Image.fromarray(input_img).convert("RGB")
    tensor_img = transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor_img)
        probabilities = torch.softmax(outputs, dim=1)[0]

    return {CLASSES[i]: float(probabilities[i]) for i in range(2)}


demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="numpy", label="上傳待測圖片 (Upload Image)"),
    outputs=gr.Label(num_top_classes=2, label="預測結果與機率分析 (Prediction Probabilities)"),
    title="🧬 生物 vs 非生物 AI 影像分類系統 (Living vs Non-Living Classifier)",
    description="結合 ResNet-18 遷移學習與微調，自動辨識圖片主體屬於生物 (Living) 或非生物 (Non-Living) 物件。"
)

if __name__ == "__main__":
    demo.launch()
