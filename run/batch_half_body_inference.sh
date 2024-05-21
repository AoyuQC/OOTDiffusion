#!/bin/bash
gpu_info=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)

txt_file="$1"
checkpoint_id="$2"
gpu_id="$3"

if [[ "$gpu_info" == *"A100"* ]]; then
    echo "The GPU is an A100 GPU."
    python /home/ec2-user/SageMaker/vto/OOTDiffusion/run/run_ootd_batch.py --base_path /home/ec2-user/SageMaker/data/dataset/vto/shenin/shein_data \
        --txt_file $txt_file \
        --scale 4.0 \
        --checkpoint_id $checkpoint_id \
        --gpu_id $gpu_id \
        --unpair_seed 800 \
        --sample 1
elif [[ "$gpu_info" == *"A10"* ]]; then
    echo "The GPU is an A10 GPU."
    python /home/ubuntu/pytorch_gpu_base_ubuntu_uw2_workplace/aws-gcr-csdc-atl/aigc-vto-models/aigc-vto-models-ootd/reference/OOTDiffusion/run/run_ootd_batch.py --base_path $base_path \
        --txt_file $txt_file \
        --scale 2.0 \
        --checkpoint_id $checkpoint_id \
        --gpu_id $gpu_id \
        --sample 1
else
    echo "The GPU is neither A10 nor A100."
    echo "GPU name: $gpu_info"
fi
    
        # --model_path /home/ubuntu/pytorch_gpu_base_ubuntu_uw2_workplace/aws-gcr-csdc-atl/aigc-vto-models/aigc-vto-models-ootd/reference/OOTDiffusion/run/examples/model/01008_00.jpg \
