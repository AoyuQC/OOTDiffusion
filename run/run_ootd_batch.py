from pathlib import Path
import sys
import os
import torch
from PIL import Image
from utils_ootd import get_mask_location
from tqdm import tqdm
import random

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from preprocess.openpose.run_openpose import OpenPose
from preprocess.humanparsing.run_parsing import Parsing
from ootd.inference_ootd_hd import OOTDiffusionHD
from ootd.inference_ootd_dc import OOTDiffusionDC

device_name = torch.cuda.get_device_name()
if device_name == 'NVIDIA A10G':
    # g5 instance
    try:
        import debugpy

        debugpy.listen(5889)  # 5678 is port
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        debugpy.breakpoint()
        print('break on this line')
    except:
        print("non debug mode")
    output_path = "/home/ubuntu/pytorch_gpu_base_ubuntu_uw2_workplace/aws-gcr-csdc-atl/aigc-vto-models/aigc-vto-models-ootd/reference/OOTDiffusion/run"
elif device_name == 'NVIDIA A100-SXM4-40GB':
    # a100 instance
    output_path = "/home/ec2-user/SageMaker/vto/OOTDiffusion/run"
else:
    raise Exception("only for a10 and a100 instance")


import argparse
parser = argparse.ArgumentParser(description='run ootd')
parser.add_argument('--gpu_id', '-g', type=int, default=0, required=False)
parser.add_argument('--checkpoint_id', type=str, default="checkpoint-36000", required=False)
parser.add_argument('--model_path', type=str, default="", required=False)
parser.add_argument('--cloth_path', type=str, default="", required=False)
parser.add_argument('--model_type', type=str, default="hd", required=False)
parser.add_argument('--base_path', type=str, default="", required=True)
parser.add_argument('--txt_file', type=str, default="hd", required=False)
parser.add_argument('--category', '-c', type=int, default=0, required=False)
parser.add_argument('--scale', type=float, default=2.0, required=False)
parser.add_argument('--step', type=int, default=20, required=False)
parser.add_argument('--sample', type=int, default=4, required=False)
parser.add_argument('--seed', type=int, default=-1, required=False)
parser.add_argument('--unpair_seed', type=int, default=0, required=False)
args = parser.parse_args()


openpose_model = OpenPose(args.gpu_id)
parsing_model = Parsing(args.gpu_id)


category_dict = ['upperbody', 'lowerbody', 'dress']
category_dict_utils = ['upper_body', 'lower_body', 'dresses']

model_type = args.model_type # "hd" or "dc"
category = args.category # 0:upperbody; 1:lowerbody; 2:dress
# cloth_path = args.cloth_path
# model_path = args.model_path
txt_file = args.txt_file
base_path = args.base_path
checkpoint_id = args.checkpoint_id

image_scale = args.scale
n_steps = args.step
n_samples = args.sample
seed = args.seed
unpair_seed = args.unpair_seed


if model_type == "hd":
    model = OOTDiffusionHD(args.gpu_id, args.checkpoint_id)
elif model_type == "dc":
    model = OOTDiffusionDC(args.gpu_id)
else:
    raise ValueError("model_type must be \'hd\' or \'dc\'!")

if __name__ == '__main__':
    # mkdirs
    txt_name = txt_file.split('/')[-1].split('.')[0]
    
    print(unpair_seed)
    if unpair_seed != 0:
        complete_output_path = f'{output_path}/unpair_seed{unpair_seed}_{checkpoint_id}_{txt_name}'
    else:
        complete_output_path = f'{output_path}/{checkpoint_id}_{txt_name}'

    os.makedirs(complete_output_path, exist_ok=True)

    model_name_lists = []
    cloth_name_lists = []
    with open(txt_file, 'r') as file:
        # Read the file line by line
        for line in tqdm(file):
            # check exists
            model_name = line.strip().split(' ')[0]
            cloth_name = line.strip().split(' ')[1]
            model_name_lists.append(model_name)
            cloth_name_lists.append(cloth_name)
    
    if unpair_seed != 0:
        random.seed(unpair_seed)
        # Shuffle the list using the seed value
        random.shuffle(cloth_name_lists)

    for model_name, cloth_name in zip(model_name_lists, cloth_name_lists):
        # check exists
        save_name = model_name.split('.')[0]+'_'+cloth_name

        if os.path.isfile(f"{complete_output_path}/{save_name}"):
            print(f"{save_name} exists. Bypassing logic...")
            continue
        else:
            print(f"{save_name} to be processed...")
            try:
                cloth_path = f"{base_path}/cloth/{cloth_name}" 
                model_path = f"{base_path}/image/{model_name}"

                if model_type == 'hd' and category != 0:
                    raise ValueError("model_type \'hd\' requires category == 0 (upperbody)!")

                cloth_img = Image.open(cloth_path).resize((768, 1024))
                model_img = Image.open(model_path).resize((768, 1024))
                keypoints = openpose_model(model_img.resize((384, 512)))
                model_parse, _ = parsing_model(model_img.resize((384, 512)))

                mask, mask_gray = get_mask_location(model_type, category_dict_utils[category], model_parse, keypoints)
                mask = mask.resize((768, 1024), Image.NEAREST)
                mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)

                masked_vton_img = Image.composite(mask_gray, model_img, mask)

                images = model(
                    model_type=model_type,
                    category=category_dict[category],
                    image_garm=cloth_img,
                    image_vton=masked_vton_img,
                    mask=mask,
                    image_ori=model_img,
                    num_samples=n_samples,
                    num_steps=n_steps,
                    image_scale=image_scale,
                    seed=seed,
                )
                image_idx = 0
                for image in images:
                    save_name = model_name.split('.')[0]+'_'+cloth_name
                    image.save(f'{complete_output_path}/{save_name}')
                    image_idx += 1
            except Exception as e:
                print(f"by pass {save_name} ... with error message \n {e}")
