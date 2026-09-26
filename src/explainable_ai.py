import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image
import os

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
            
        self.model.zero_grad()
        output[0, target_class].backward()
        
        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        return cam, target_class


def generate_gradcam_visualization(image_path, model_path="living_vs_nonliving.pth", output_path="./reports/gradcam_result.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load ResNet-18 model
    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        
    model = model.to(device)

    cam_extractor = GradCAM(model, model.layer4[-1])
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    raw_img = Image.open(image_path).convert('RGB')
    input_tensor = transform(raw_img).unsqueeze(0).to(device)
    
    cam, pred_class = cam_extractor.generate(input_tensor)
    
    img_np = np.array(raw_img.resize((224, 224))) / 255.0
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam_overlay = heatmap[..., ::-1] * 0.4 + img_np * 0.6
    
    labels = ['Living (生物)', 'Non-Living (非生物)']
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Input Image")
    plt.imshow(img_np)
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.title(f"Grad-CAM Attention ({labels[pred_class]})")
    plt.imshow(cam_overlay)
    plt.axis("off")
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[v] Saved Grad-CAM visualization to '{output_path}'")


if __name__ == "__main__":
    pass
