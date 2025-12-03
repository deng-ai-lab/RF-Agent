# main.py
import hydra
import numpy as np
import json
import logging
import matplotlib.pyplot as plt
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import shutil
import time
import openai
# from openai import OpenAI

from pathlib import Path

from utils.misc import *
from utils.file_utils import find_files_with_substring, load_tensorboard_logs
from utils.create_task import create_task
from utils.extract_task_code import *

ROOT_DIR = os.getcwd()
ISAAC_ROOT_DIR = f"{ROOT_DIR}/../isaacgymenvs/isaacgymenvs"

@hydra.main(config_path="cfg", config_name="config_test", version_base="1.1")
def main(cfg):
    task = cfg.env.task
    suffix = cfg.suffix
    env_name = cfg.env.env_name.lower()
    output_file = f"{ISAAC_ROOT_DIR}/tasks/{env_name}{suffix.lower()}.py"
    max_reward_code_path = ROOT_DIR + cfg.test_reward_function
    # Evaluate the best reward code
    if max_reward_code_path is None:
        logging.info("All iterations of code generation failed, aborting...")
        logging.info("Please double check the output env_iter*_response*.txt files for repeating errors!")
        exit()
    logging.info(f"Task: {task}, Best Reward Code Path: {max_reward_code_path}")
    logging.info(f"Evaluating best reward code {cfg.num_eval} times")
    shutil.copy(max_reward_code_path, output_file)

    cur_eval_runs = []
    for i in range(cfg.num_eval):
        set_freest_gpu()
        env = os.environ.copy()
        seed = i
        rl_filepath = f"reward_code_cur_eval{seed}.txt"
        with open(rl_filepath, 'w') as f:
            process = subprocess.Popen(['python', '-u', f'{ISAAC_ROOT_DIR}/train_with_seed.py',
                                        'hydra/output=subprocess',
                                        f'task={task}{suffix}', f'wandb_activate={cfg.use_wandb}',
                                        f'wandb_entity={cfg.wandb_username}',
                                        f'wandb_project={cfg.wandb_project}',
                                        f'headless={not cfg.capture_video}',
                                        f'capture_video={cfg.capture_video}',
                                        'force_render=False', f'seed={seed}',
                                        f'max_iterations={cfg.test_max_iterations}'],
                                       stdout=f, stderr=f, env=env)
        block_until_training(rl_filepath)
        cur_eval_runs.append(process)

    cur_reward_code_final_successes = []
    for i, rl_run in enumerate(cur_eval_runs):
        rl_run.communicate()
        seed = i
        rl_filepath = f"reward_code_cur_eval{seed}.txt"
        with open(rl_filepath, 'r') as f:
            stdout_str = f.read()
        lines = stdout_str.split('\n')
        for k, line in enumerate(lines):
            if line.startswith('Tensorboard Directory:'):
                break
        tensorboard_logdir = line.split(':')[-1].strip()
        tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)
        max_success = max(tensorboard_logs['consecutive_successes'])
        cur_reward_code_final_successes.append(max_success)

    logging.info(f"Current Reward Code Final Success Mean: {np.mean(cur_reward_code_final_successes)}, "
                 f"Std: {np.std(cur_reward_code_final_successes)}, Raw: {cur_reward_code_final_successes}")

if __name__ == "__main__":
    main()