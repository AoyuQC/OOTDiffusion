#!/bin/bash
gpu_info=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)

echo "GPU name: $gpu_info"

if [[ "$gpu_info" == *"A100"* ]]; then
    echo "The GPU is an A100 GPU."
    python /home/ec2-user/SageMaker/vto/OOTDiffusion/run/run_ootd_batch.py --base_path /home/ec2-user/SageMaker/data/dataset/vto/shenin/shein_data \
        --txt_file /home/ec2-user/SageMaker/data/dataset/vto/shenin/remaining_test_pairs_shein.txt \
        --scale 4.0 \
        --checkpoint_id checkpoint-36000 \
	--gpu_id 1 \
        --sample 1
elif [[ "$gpu_info" == *"A10"* ]]; then
    echo "The GPU is an A10 GPU."
    python /home/ubuntu/pytorch_gpu_base_ubuntu_uw2_workplace/aws-gcr-csdc-atl/aigc-vto-models/aigc-vto-models-ootd/reference/OOTDiffusion/run/run_ootd_batch.py --base_path /home/ubuntu/dataset/aigc-app-vto/shenin/shein_data \
        --txt_file /home/ubuntu/dataset/aigc-app-vto/shenin/test_pairs_shein.txt \
        --scale 2.0 \
        --checkpoint_id checkpoint-epoch0 \
        --sample 1
else
    echo "The GPU is neither A10 nor A100."
    echo "GPU name: $gpu_info"
fi
    
        # --model_path /home/ubuntu/pytorch_gpu_base_ubuntu_uw2_workplace/aws-gcr-csdc-atl/aigc-vto-models/aigc-vto-models-ootd/reference/OOTDiffusion/run/examples/model/01008_00.jpg \
