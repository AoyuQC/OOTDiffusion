#!/bin/bash
gpu_info=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)

echo "GPU name: $gpu_info"

if [[ "$gpu_info" == *"A100"* ]]; then
    echo "The GPU is an A100 GPU."
    python run/run_ootd.py --model_path run/examples/garment/00055_00.jpg \
        --cloth_path run/examples/model/i01008_00.jpg \
        --scale 2.0 \
        --checkpoint_id checkpoint-36000 \
        --sample 1
elif [[ "$gpu_info" == *"A10"* ]]; then
    echo "The GPU is an A10 GPU."
    python run/run_ootd.py --model_path run/examples/garment/00055_00.jpg \
        --cloth_path run/examples/model/i01008_00.jpg \
        --scale 2.0 \
        --checkpoint_id checkpoint-epoch0 \
        --sample 1
else
    echo "The GPU is neither A10 nor A100."
    echo "GPU name: $gpu_info"
fi
    
