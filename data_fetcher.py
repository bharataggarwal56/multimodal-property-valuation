import pandas as pd
import requests
import os
from tqdm import tqdm
import time
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

MAPBOX_ACCESS_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN")

TRAIN_FILE = os.path.join(BASE_DIR, "data", "train.csv")
TEST_FILE = os.path.join(BASE_DIR, "data", "test.csv")
IMAGE_DIR = os.path.join(BASE_DIR, "data", "images")

IMG_WIDTH = 600
IMG_HEIGHT = 600
ZOOM_LEVEL = 18 
STYLE_ID = "satellite-v9"
MAX_WORKERS = 20  

def download_image(args):
    lat, lon, img_id = args
    
    if not MAPBOX_ACCESS_TOKEN:
        return 

    file_path = os.path.join(IMAGE_DIR, f"{img_id}.jpg")
    
    if os.path.exists(file_path):
        return

    url = f"https://api.mapbox.com/styles/v1/mapbox/{STYLE_ID}/static/{lon},{lat},{ZOOM_LEVEL},0/{IMG_WIDTH}x{IMG_HEIGHT}?access_token={MAPBOX_ACCESS_TOKEN}"
    
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
        else:
            pass 
    except Exception:
        pass

def main():
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
    
    if not MAPBOX_ACCESS_TOKEN:
        print("ERROR: Please set MAPBOX_ACCESS_TOKEN in your .env file")
        return

    print(f"Loading datasets from {BASE_DIR}...")
    
    if not os.path.exists(TRAIN_FILE) or not os.path.exists(TEST_FILE):
        print("ERROR: train.csv or test.csv not found.")
        return

    df_train = pd.read_csv(TRAIN_FILE)
    df_test = pd.read_csv(TEST_FILE)
    
    tasks = []
    
    print("Preparing download queue...")
    for df in [df_train, df_test]:
        if {'id', 'lat', 'long'}.issubset(df.columns):
            for _, row in df.iterrows():
                tasks.append((row['lat'], row['long'], str(row['id'])))
    
    print(f"Starting parallel download for {len(tasks)} images with {MAX_WORKERS} workers...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(download_image, task) for task in tasks]
        
        for _ in tqdm(as_completed(futures), total=len(tasks), unit="img"):
            pass

    print(f"\nDone! Images saved in '{IMAGE_DIR}'")

if __name__ == "__main__":
    main()