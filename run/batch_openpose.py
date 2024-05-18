import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from preprocess.openpose.run_openpose import OpenPose

# Set the source and destination directories
src_dir = "/home/ubuntu/dataset/aigc-app-vto/shenin/shein_data/image"
img_dst_dir = "/home/ubuntu/dataset/aigc-app-vto/shenin/shein_data/openpose_img"
json_dst_dir = "/home/ubuntu/dataset/aigc-app-vto/shenin/shein_data/openpose_json"

# Create the destination directory if it doesn't exist
os.makedirs(img_dst_dir, exist_ok=True)
os.makedirs(json_dst_dir, exist_ok=True)

openpose_model = OpenPose(0)

# Loop through all files in the source directory
for filename in tqdm(os.listdir(src_dir)):
    try:
        model_img = Image.open(os.path.join(src_dir, filename)).resize((768, 1024))
        save_json_filename = os.path.join(json_dst_dir, filename.split('.')[0] + '_keypoints.json')
        save_image_filename = os.path.join(img_dst_dir, filename.split('.')[0] + '_rendered.png')
        keypoints = openpose_model(model_img.resize((384, 512)), 384, save_image_filename, save_json_filename)
    except:
        print(f"fail on image {filename}")