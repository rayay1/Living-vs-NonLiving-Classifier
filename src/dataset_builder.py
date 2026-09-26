import os
import shutil
import random
import time
import urllib.request
from PIL import Image

# Expanded Keyword Lists for Diversity
LIVING_KEYWORDS = [
    # Animals (Mammals, Birds, Reptiles, Aquatic, Insects)
    'dog', 'cat', 'lion', 'tiger', 'elephant', 'giraffe', 'zebra', 'monkey', 'panda', 'bear',
    'wolf', 'fox', 'rabbit', 'squirrel', 'horse', 'cow', 'pig', 'sheep', 'goat', 'deer',
    'eagle', 'parrot', 'owl', 'penguin', 'swan', 'flamingo', 'peacock', 'duck',
    'frog', 'snake', 'turtle', 'lizard', 'crocodile', 'chameleon',
    'shark', 'whale', 'dolphin', 'octopus', 'jellyfish', 'crab',
    'butterfly', 'bee', 'dragonfly', 'ant', 'spider',
    # Plants & Humans
    'sunflower', 'oak tree', 'rose flower', 'cactus', 'mushroom', 'fern plant', 'human face'
]

NON_LIVING_KEYWORDS = [
    # Vehicles & Transportation
    'car', 'bus', 'truck', 'bicycle', 'motorcycle', 'airplane', 'helicopter', 'boat', 'ship', 'train',
    # Furniture & Appliances
    'chair', 'sofa', 'table', 'bed', 'desk', 'lamp', 'clock', 'mirror', 'cabinet', 'shelf',
    'microwave', 'fridge', 'toaster', 'washing machine', 'fan', 'heater',
    # Electronics & 3C
    'laptop', 'phone', 'computer', 'camera', 'television', 'keyboard', 'headphones', 'speaker', 'robot',
    # Buildings & Structures
    'house', 'building', 'skyscraper', 'castle', 'bridge', 'tower', 'stadium', 'factory', 'statue',
    # Daily Objects
    'book', 'pen', 'cup', 'bottle', 'backpack', 'shoe', 'hat', 'glasses', 'umbrella', 'guitar', 'piano', 'rock'
]

DATASET_DIR = './dataset'


def download_fallback_images(category_name, keywords, num_per_kw=5):
    """Fallback automated downloader using standard HTTPS image APIs."""
    raw_dir = os.path.join(DATASET_DIR, 'raw', category_name)
    os.makedirs(raw_dir, exist_ok=True)
    
    print(f"[*] Downloading {category_name} images from online resources...")
    count = 0
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    for kw in keywords[:15]:  # Select diverse keywords
        for i in range(num_per_kw):
            url = f"https://source.unsplash.com/featured/300x300/?{kw.replace(' ', ',')}&sig={count}"
            dest_file = os.path.join(raw_dir, f"{kw.replace(' ', '_')}_{i}.jpg")
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=5) as resp:
                    with open(dest_file, 'wb') as f:
                        f.write(resp.read())
                count += 1
            except Exception as e:
                # Alternate fallback: picsum
                try:
                    alt_url = f"https://picsum.photos/300/300?random={count}"
                    req = urllib.request.Request(alt_url, headers=headers)
                    with urllib.request.urlopen(req, timeout=5) as resp:
                        with open(dest_file, 'wb') as f:
                            f.write(resp.read())
                    count += 1
                except Exception:
                    pass
            time.sleep(0.1)
            
    print(f"[v] Downloaded {count} images for {category_name}.")


def clean_and_split_dataset(split_ratio=(0.7, 0.15, 0.15)):
    """Validate images, filter out corrupted files, and partition into train/val/test splits."""
    print("[*] Validating and partition dataset into train/val/test splits...")
    
    for category in ['living', 'non_living']:
        raw_cat_dir = os.path.join(DATASET_DIR, 'raw', category)
        if not os.path.exists(raw_cat_dir):
            continue
            
        valid_files = []
        for root, _, files in os.walk(raw_cat_dir):
            for f in files:
                filepath = os.path.join(root, f)
                try:
                    with Image.open(filepath) as img:
                        img.verify()
                    valid_files.append(filepath)
                except Exception:
                    # Skip corrupt image
                    continue
                    
        random.shuffle(valid_files)
        total = len(valid_files)
        if total == 0:
            print(f"[!] Warning: No images found for {category}")
            continue
            
        n_train = int(total * split_ratio[0])
        n_val = int(total * split_ratio[1])
        
        splits = {
            'train': valid_files[:n_train],
            'val': valid_files[n_train:n_train + n_val],
            'test': valid_files[n_train + n_val:]
        }
        
        for split_name, files in splits.items():
            target_dir = os.path.join(DATASET_DIR, split_name, category)
            os.makedirs(target_dir, exist_ok=True)
            for idx, src_path in enumerate(files):
                ext = os.path.splitext(src_path)[1]
                if not ext:
                    ext = '.jpg'
                dest_path = os.path.join(target_dir, f"{category}_{idx:05d}{ext}")
                shutil.copy(src_path, dest_path)
                
        print(f"[v] {category}: {n_train} train, {n_val} val, {total - n_train - n_val} test samples prepared.")


def build_dataset():
    download_fallback_images('living', LIVING_KEYWORDS, num_per_kw=5)
    download_fallback_images('non_living', NON_LIVING_KEYWORDS, num_per_kw=5)
    clean_and_split_dataset()


if __name__ == '__main__':
    build_dataset()
