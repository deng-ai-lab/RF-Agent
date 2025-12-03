# main.py
import hydra
import numpy as np
import json
import logging
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
from rf_agent_algo.rfagent_algo import RFAgent

RFAGENT_ROOT_DIR = os.getcwd()
ISAAC_ROOT_DIR = f"{RFAGENT_ROOT_DIR}/../isaacgymenvs/isaacgymenvs"
API_KEY = os.getenv("OPENAI_API_KEY") # get openai API key from environment variable or just put your API string here "..."

@hydra.main(config_path="cfg", config_name="config_rf_agent", version_base="1.1")
def main(cfg):
    env = os.environ.copy()
    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {RFAGENT_ROOT_DIR}")

    openai.api_key = API_KEY
    client = None

    task = cfg.env.task
    task_description = cfg.env.description
    suffix = cfg.suffix
    model = cfg.model
    logging.info(f"Using LLM: {model}")
    logging.info("Task: " + task)
    logging.info("Task description: " + task_description)

    env_name = cfg.env.env_name.lower()
    env_parent = 'isaac' if f'{env_name}.py' in os.listdir(f'{RFAGENT_ROOT_DIR}/envs/isaac') else 'bidex'
    task_file = f'{RFAGENT_ROOT_DIR}/envs/{env_parent}/{env_name}.py'
    task_obs_file = f'{RFAGENT_ROOT_DIR}/envs/{env_parent}/{env_name}_obs.py'
    shutil.copy(task_obs_file, f"env_init_obs.py")
    task_code_string = file_to_string(task_file)
    task_obs_code_string = file_to_string(task_obs_file)
    output_file = f"{ISAAC_ROOT_DIR}/tasks/{env_name}{suffix.lower()}.py"

    # load prompts
    prompt_dir = f'{RFAGENT_ROOT_DIR}/utils/prompts_rfagent'
    initial_system = file_to_string(f'{prompt_dir}/initial_system.txt')
    initial_tip = file_to_string(f'{prompt_dir}/code_output_tip.txt')
    initial_user = file_to_string(f'{prompt_dir}/initial_user.txt')
    reward_signature = file_to_string(f'{prompt_dir}/reward_signature.txt')
    execution_error_feedback = file_to_string(f'{prompt_dir}/execution_error_feedback.txt')
    initial_action = file_to_string(f'{prompt_dir}/thought_code_output.txt')
    initial_failed_feedback = file_to_string(f'{prompt_dir}/initial_failed_feedback.txt')
    initial_thought_alignment = file_to_string(f'{prompt_dir}/initial_thought_alignment.txt')
    trained_result_analysis_tip = file_to_string(f'{prompt_dir}/trained_result_analysis_tip.txt')

    action_mutation_mechanism = file_to_string(f'{prompt_dir}/action_1_mutation_mechanism.txt')
    action_mutation_param = file_to_string(f'{prompt_dir}/action_2_mutation_param.txt')
    action_crossover_elite = file_to_string(f'{prompt_dir}/action_3_crossover_elite.txt')
    action_tree_reasoning = file_to_string(f'{prompt_dir}/action_4_tree_reasoning.txt')
    action_different_thought = file_to_string(f'{prompt_dir}/action_5_different_thought.txt')
    base_thought_code = file_to_string(f'{prompt_dir}/base_thought_code.txt')

    initial_system_verify = file_to_string(f'{prompt_dir}/initial_system_verify.txt')
    self_node_value_verify_single = file_to_string(f'{prompt_dir}/self_node_value_verify_single.txt')

    initial_system = initial_system.format(task_reward_signature_string=reward_signature)
    initial_user = initial_user.format(task_obs_code_string=task_obs_code_string, task_description=task_description)
    initial_action = initial_action
    prompts_dict = {'initial_system': initial_system, 'initial_user': initial_user, 'initial_tip': initial_tip,
                    'initial_action': initial_action, 'initial_failed_feedback': initial_failed_feedback, 'initial_thought_alignment': initial_thought_alignment,
                    'action_mutation_mechanism': action_mutation_mechanism, 'action_mutation_param': action_mutation_param, 'action_crossover_elite':action_crossover_elite, 'action_tree_reasoning': action_tree_reasoning, 'action_different_thought': action_different_thought, 'base_thought_code': base_thought_code, 'trained_result_analysis_tip': trained_result_analysis_tip,
                    'initial_system_verify':initial_system_verify, 'self_node_value_verify_single':self_node_value_verify_single}

    task_code_string = task_code_string.replace(task, task + suffix)
    # Create the task YAML file
    create_task(ISAAC_ROOT_DIR, cfg.env.task, cfg.env.env_name, suffix)

    # Initialize RF-Agent
    rfagent = RFAgent(cfg=cfg, prompts_dict=prompts_dict, execution_error_feedback=execution_error_feedback, task_code_string=task_code_string, output_file=output_file, issac_root_dir=ISAAC_ROOT_DIR, client=client)

    logging.info(f"Starting RFAgent search")

    # Perform a search to select the optimal reward function
    best_cur_node, best_avg_node, elite_set = rfagent.run(env=env, train_seed=cfg.train_seed)
    bset_cur_code_py = best_cur_node.exec_file_path

    logging.info(f"RFAgent search completed")

    # Evaluate the best current reward code
    if bset_cur_code_py == "":
        logging.info("All iterations of code generation failed, aborting...")
        logging.info("Please double check the output env_iter*_response*.txt files for repeating errors!")
        exit()

    logging.info(f"Evaluating best current reward code {cfg.num_eval} times")
    shutil.copy(bset_cur_code_py, output_file)

    cur_eval_runs = []
    for i in range(cfg.num_eval):
        set_freest_gpu()
        env = os.environ.copy()
        seed = i
        # execute python files and save under the rl_filepath
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
