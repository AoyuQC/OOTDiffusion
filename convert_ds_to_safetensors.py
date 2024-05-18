import torch

device_name = torch.cuda.get_device_name()
epoch_num = 0
if device_name == 'NVIDIA A10G':
    try:
        import debugpy

        debugpy.listen(5889)  # 5678 is port
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        debugpy.breakpoint()
        print('break on this line')
    except:
        print("non debug mode")
    
    base_path = "/home/ubuntu/pytorch_gpu_base_ubuntu_uw2_workplace/aws-gcr-csdc-atl/aigc-vto-models/aigc-vto-models-ootd/ootd_train_ds_checkpoints/checkpoint-epoch0/"
    infer_base_path = "/home/ubuntu/pytorch_gpu_base_ubuntu_uw2_workplace/aws-gcr-csdc-atl/aigc-vto-models/aigc-vto-models-ootd/reference/OOTDiffusion"
elif device_name == 'NVIDIA A100-SXM4-40GB':
    # a100 instance
    base_path = f"/home/ec2-user/SageMaker/vto/OOTDiffusion-train/ootd_train_ds_checkpoints/checkpoint-split-epoch{epoch_num}"
    infer_base_path = "/home/ec2-user/SageMaker/vto/OOTDiffusion"
else:
    raise Exception("only for a10 and a100 instance")


## use GPU or CPU

# import re
# import os
# import argparse
# import torch;
# from safetensors.torch import save_file

# if torch.cuda.is_available():
#         device = 'cuda'
#         checkpoint = torch.load(f"{base_path}/pytorch_model.bin", map_location=torch.device('cuda'))
# else:
#         device = 'cpu'
#         # if on CPU or want to have maximum precision on GPU, use default full-precision setting
#         checkpoint = torch.load(f"{base_path}/pytorch_model.bin", map_location=torch.device('cpu'))

# print(f'device is {device}')

# unet_vton_dict = dict()
# unet_garm_dict = dict()
# for idx, key in enumerate(checkpoint):
#         name = key.split('.')[0]
#         if name == 'unet_vton':
#                 new_key = key[10:]
#                 unet_vton_dict[new_key] = checkpoint[key]
#         elif name == 'unet_garm':
#                 new_key = key[10:]
#                 unet_garm_dict[new_key] = checkpoint[key]
# save_file(unet_vton_dict, "unet_vton.safetensors")
# save_file(unet_garm_dict, "unet_garm.safetensors")

from safetensors.torch import save_file
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

state_dict = get_fp32_state_dict_from_zero_checkpoint(base_path)

unet_vton_dict = dict()
unet_garm_dict = dict()
for idx, key in enumerate(state_dict):
        name = key.split('.')[0]
        if name == 'unet_vton':
                new_key = key[10:]
                unet_vton_dict[new_key] = state_dict[key]
        elif name == 'unet_garm':
                new_key = key[10:]
                unet_garm_dict[new_key] = state_dict[key]

save_file(unet_vton_dict, "unet_vton.safetensors")
save_file(unet_garm_dict, "unet_garm.safetensors")

# make upper body folder and soft link
import os
import shutil
from pathlib import Path

# Create a new folder
folder_name = f"checkpoints/ootd/ootd_hd/checkpoint-epoch{epoch_num}"
os.makedirs(os.path.join(infer_base_path,folder_name), exist_ok=True)
unet_garm_folder_name = f"checkpoints/ootd/ootd_hd/checkpoint-epoch{epoch_num}/unet_garm"
os.makedirs(os.path.join(infer_base_path,unet_garm_folder_name), exist_ok=True)
unet_vton_folder_name = f"checkpoints/ootd/ootd_hd/checkpoint-epoch{epoch_num}/unet_vton"
os.makedirs(os.path.join(infer_base_path,unet_vton_folder_name), exist_ok=True)

source_path = "checkpoints/ootd/ootd_hd/checkpoint-36000/unet_garm/config.json"
shutil.copy(os.path.join(infer_base_path, source_path), os.path.join(infer_base_path, unet_garm_folder_name,"config.json"))
source_path = "checkpoints/ootd/ootd_hd/checkpoint-36000/unet_vton/config.json"
shutil.copy(os.path.join(infer_base_path, source_path), os.path.join(infer_base_path, unet_vton_folder_name,"config.json"))

dst_path = os.path.join(infer_base_path, unet_garm_folder_name, "diffusion_pytorch_model.safetensors")
shutil.move("unet_garm.safetensors", dst_path)
dst_path = os.path.join(infer_base_path, unet_vton_folder_name, "diffusion_pytorch_model.safetensors")
shutil.move("unet_vton.safetensors", dst_path)
