import os
import urllib.request
import random
import time

# --- 設定參數 ---
SAVE_FOLDER = 'test_images'
DOWNLOAD_COUNT = 20  # 這裡可以改成你想下載的總張數

# 關鍵字池 (各約 60 個單字)
KEYWORDS_LIVING = [
    # 動物 (陸地)
    'dog', 'cat', 'lion', 'tiger', 'elephant', 'giraffe', 'zebra', 'monkey', 'panda', 'kangaroo',
    'koala', 'bear', 'wolf', 'fox', 'rabbit', 'squirrel', 'hamster', 'mouse', 'horse', 'cow',
    'pig', 'sheep', 'goat', 'deer', 'camel', 'rhino', 'hippo', 'leopard', 'cheetah', 'buffalo',
    # 禽鳥類
    'chicken', 'duck', 'goose', 'eagle', 'parrot', 'owl', 'penguin', 'swan', 'flamingo', 'peacock',
    # 爬蟲與兩棲
    'frog', 'snake', 'turtle', 'lizard', 'crocodile', 'chameleon',
    # 水生生物
    'fish', 'shark', 'whale', 'dolphin', 'octopus', 'jellyfish', 'crab', 'shrimp', 'starfish',
    # 昆蟲
    'butterfly', 'bee', 'dragonfly', 'ant', 'spider',
    # 植物與人類
    'flower', 'tree', 'grass', 'rose', 'sunflower', 'cactus', 'mushroom', 'boy', 'girl', 'baby'
]

KEYWORDS_NON_LIVING = [
    # 交通工具
    'car', 'bus', 'truck', 'bicycle', 'motorcycle', 'airplane', 'helicopter', 'boat', 'ship', 'train',
    'scooter', 'van', 'taxi', 'ambulance', 'fire truck', 'police car', 'tractor', 'subway', 'yacht',
    # 家具與家電
    'chair', 'sofa', 'table', 'bed', 'desk', 'lamp', 'clock', 'mirror', 'cabinet', 'shelf',
    'oven', 'microwave', 'fridge', 'toaster', 'washing machine', 'fan', 'heater',
    # 3C 電子產品
    'laptop', 'phone', 'computer', 'camera', 'television', 'keyboard', 'mouse', 'headphones', 'speaker', 'robot',
    # 建築與設施
    'house', 'building', 'skyscraper', 'castle', 'bridge', 'tower', 'stadium', 'factory', 'tent', 'statue',
    # 日常用品
    'book', 'pen', 'cup', 'bottle', 'bag', 'backpack', 'shoe', 'hat', 'glasses', 'umbrella',
    'guitar', 'piano', 'ball', 'toy', 'rock', 'stone'
]


def download_random_images():
    # 1. 建立資料夾
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)

    # 清空舊的圖片 (選擇性：如果你想保留舊圖，把這兩行刪掉)
    for f in os.listdir(SAVE_FOLDER):
        os.remove(os.path.join(SAVE_FOLDER, f))
    print("🧹 已清空舊的測試圖片...")

    print(f"🎲 正在隨機抽取 {DOWNLOAD_COUNT} 張新圖片...\n" + "-" * 30)

    # 2. 開始隨機下載
    for i in range(1, DOWNLOAD_COUNT + 1):
        # 決定這一張要是生物還是非生物 (50% 機率)
        is_living = random.choice([True, False])

        if is_living:
            category = 'living'
            keyword = random.choice(KEYWORDS_LIVING)
        else:
            category = 'non_living'
            keyword = random.choice(KEYWORDS_NON_LIVING)

        # 產生隨機網址 (加個 random 數字避免抓到重複的)
        # 使用 loremflickr 服務
        rand_id = random.randint(1, 100000)
        url = f"https://loremflickr.com/600/600/{keyword}?lock={rand_id}"

        # 檔名：例如 test_01_living_cat.jpg
        filename = f"test_{i:02d}_{category}_{keyword}.jpg"
        save_path = os.path.join(SAVE_FOLDER, filename)

        try:
            print(f"[{i}/{DOWNLOAD_COUNT}] 正在抓一張「{keyword}」的照片...", end="")

            # 偽裝瀏覽器下載
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as response, open(save_path, 'wb') as out_file:
                out_file.write(response.read())

            print(" ✅ 成功！")

        except Exception as e:
            print(f" ❌ 失敗 ({e})")

        # 休息一下，對伺服器有禮貌
        time.sleep(1)

    print("-" * 30)
    print("🎉 下載完成！快去執行 predict.py 看看 AI 這次考幾分！")


if __name__ == '__main__':
    download_random_images()