# 🧬 AI 影像辨識系統：生物與非生物的分類應用

## 1. 專題簡介
本專題是影像處理概論的期末報告。目標是設計一個深度學習模型 (CNN)，用以自動判斷影像中的主體屬於「生物」還是「非生物」。本系統採用雙模型架構，解決了通用層級的分類問題，並結合物件偵測技術，探討其在環境監控與智慧裝置中的潛在應用。

## 2. 環境需求
- **語言**: Python 3.10
- **框架**: PyTorch (Torchvision)
- **其他套件**: Matplotlib, Pillow, Scikit-learn
- **開發工具**: PyCharm Community Edition

## 3. 資料集 (Dataset)
本研究使用來自 **同學提供** 與 **Google Image Dataset** 的公開圖片資源，經人工篩選並標註為以下兩類：
- **生物類 (Living)**: 包含動物、植物、人類。
- **非生物類 (Non-Living)**: 包含交通工具、家具、建築物。

*(資料集已進行縮放、正規化與資料擴增處理以提升模型效能。)*

## 4. 模型架構
本專案採用 **雙模型協作** 方式：
1.  **分類模型 (Classifier)**: 使用 **ResNet18** 預訓練模型進行遷移學習 (Transfer Learning)，負責區分生物與非生物。
2.  **定位模型 (Detector)**: 引入 **Faster R-CNN** 進行物件偵測，負責在影像中標示物體位置 (Bounding Box)。

## 5. 使用說明 (How to Run)
請依照以下順序執行程式：
1.  **訓練模型**: 執行 python train.py
2.  **下載測試圖**: 執行 python download_random.py
3.  **進行預測**: 執行 python predict.py

## 6. 參考文獻
- Krizhevsky, A., et al. (2012). ImageNet Classification with Deep Convolutional Neural Networks.
- He, K., et al. (2016). Deep Residual Learning for Image Recognition.

## 7. 專案檔案結構
```text
AI_Final_Project/
├── dataset/                 # 訓練資料集
├── predict_results/         # 預測結果輸出
├── test_images/             # 測試圖片
├── train.py                 # 訓練程式
├── download_random.py       # 下載測試圖程式
├── predict.py               # 預測與繪圖程式
└── living_vs_nonliving.pth  # 模型權重檔

