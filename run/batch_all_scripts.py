import multiprocessing
import logging
import sys
from datetime import datetime

def run_script(script_path, args, log_file):
    import subprocess

    # Configure logging for the process
    logger = logging.getLogger(f'Process {args}')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create a file handler for log output
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Create a stream handler for console output
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    cmd = [script_path] + args
    logger.info(f'Running command: {" ".join(cmd)}')
    subprocess.run(cmd)

if __name__ == '__main__':
    script_path = '/home/ubuntu/pytorch_gpu_base_ubuntu_uw2_workplace/aws-gcr-csdc-atl/aigc-vto-models/aigc-vto-models-ootd/reference/OOTDiffusion/run/batch_half_body_inference.sh' 

    # Define the arguments for each process
    txt_base = "/home/ec2-user/SageMaker/data/dataset/vto/shenin"
    args_list = [
        [f"{txt_base}/remaining_test_pairs_shein.txt", "checkpoint-36000", "0"],
        [f"{txt_base}/remaining_test_pairs_shein.txt", "checkpoint-36000", "1"],
        [f"{txt_base}/remaining_test_pairs_shein.txt", "checkpoint-36000", "2"],
        [f"{txt_base}/remaining_test_pairs_shein.txt", "checkpoint-36000", "3"]
    ]

    processes = []
    for i, args in enumerate(args_list):
        log_file = f'process_{i}_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        p = multiprocessing.Process(target=run_script, args=(script_path, args, log_file))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()