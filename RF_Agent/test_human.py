import hydra
import numpy as np 
import json
import logging 
import matplotlib.pyplot as plt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import openai
import re
import subprocess
from pathlib import Path
import shutil
import time 

from utils.misc import * 
from utils.file_utils import find_files_with_substring, load_tensorboard_logs
from utils.create_task import create_task
from utils.extract_task_code import *

EUREKA_ROOT_DIR = os.getcwd()
ISAAC_ROOT_DIR = f"{EUREKA_ROOT_DIR}/../isaacgymenvs/isaacgymenvs"

@hydra.main(config_path="cfg", config_name="config_test", version_base="1.1")
def main(cfg):
    env = os.environ.copy()
    task = cfg.env.task

    # Evaluate the best reward code many times
    logging.info(f"Task: {task}, GT_Reward")
    logging.info(f"Evaluating best reward code {cfg.num_eval} times")
    
    eval_runs = []
    for i in range(cfg.num_eval):
        
        # Execute the python file with flags
        rl_filepath = f"reward_code_eval{i}.txt"
        with open(rl_filepath, 'w') as f:
            process = subprocess.Popen(['python', '-u', f'{ISAAC_ROOT_DIR}/train.py',  
                                        'hydra/output=subprocess',
                                        f'task={task}', f'wandb_activate={cfg.use_wandb}',
                                        f'wandb_entity={cfg.wandb_username}', f'wandb_project={cfg.wandb_project}',
                                        f'headless={not cfg.capture_video}', f'capture_video={cfg.capture_video}', 'force_render=False', f'seed={i}',
                                        ],
                                        stdout=f, stderr=f, env=env)

        block_until_training(rl_filepath)
        eval_runs.append(process)

    reward_code_final_successes = []
    for i, rl_run in enumerate(eval_runs):
        rl_run.communicate()
        rl_filepath = f"reward_code_eval{i}.txt"
        with open(rl_filepath, 'r') as f:
            stdout_str = f.read() 
        lines = stdout_str.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('Tensorboard Directory:'):
                break 
        tensorboard_logdir = line.split(':')[-1].strip() 
        tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)
        max_success = max(tensorboard_logs['consecutive_successes'])
        reward_code_final_successes.append(max_success)

    logging.info(f"Final Success Mean: {np.mean(reward_code_final_successes)}, Std: {np.std(reward_code_final_successes)}, Raw: {reward_code_final_successes}")


if __name__ == "__main__":
    main()