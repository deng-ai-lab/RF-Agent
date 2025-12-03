import re
import logging
import time
import subprocess
import os
import json
import logging
import ast
from .extract_task_code import file_to_string
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def set_freest_gpu():
    freest_gpu = get_freest_gpu()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(freest_gpu)

def set_freest_gpu_gai():
    freest_gpu = get_freest_gpu_gai()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(freest_gpu)

def get_freest_gpu_gai(ban_id = 1):
    sp = subprocess.Popen(['gpustat', '--json'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str, _ = sp.communicate()
    gpustats = json.loads(out_str.decode('utf-8'))
    list_gpu = gpustats['gpus']
    list_gpu.sort(key=lambda x: x['memory.used'])
    # Find GPU with most free memory
    for gpu in list_gpu:
        if gpu['index'] == ban_id:
            continue
        else:
            break

    return gpu['index']

def get_freest_gpu():
    sp = subprocess.Popen(['gpustat', '--json'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str, _ = sp.communicate()
    gpustats = json.loads(out_str.decode('utf-8'))
    # Find GPU with most free memory
    freest_gpu = min(gpustats['gpus'], key=lambda x: x['memory.used'])

    return freest_gpu['index']

def filter_traceback(s):
    lines = s.split('\n')
    filtered_lines = []
    for i, line in enumerate(lines):
        if line.startswith('Traceback'):
            for j in range(i, len(lines)):
                if "Set the environment variable HYDRA_FULL_ERROR=1" in lines[j]:
                    break
                filtered_lines.append(lines[j])
            return '\n'.join(filtered_lines)
    return ''  # Return an empty string if no Traceback is found

def extract_code_from_response(response):
    content = response["choices"][0]["message"]["content"]
    patterns = [
        r'```python(.*?)```',
        r'```(.*?)```',
        r'"""(.*?)"""',
        r'""(.*?)""',
        r'"(.*?)"',
    ]
    for pattern in patterns:
        code_match = re.search(pattern, content, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
    return content.strip()

def validate_code(code_string):
    try:
        compile(code_string, '<string>', 'exec')
        return True
    except Exception as e:
        logging.error(f"Code validation failed: {e}")
        return False

def get_function_signature(code_string):
    # Parse the code string into an AST
    module = ast.parse(code_string)

    # Find the function definitions
    function_defs = [node for node in module.body if isinstance(node, ast.FunctionDef)]

    # If there are no function definitions, return None
    if not function_defs:
        return None

    # For simplicity, we'll just return the signature of the first function definition
    function_def = function_defs[0]

    input_lst = []
    # Construct the function signature (within object class)
    signature = function_def.name + '(self.' + ', self.'.join(arg.arg for arg in function_def.args.args) + ')'
    for arg in function_def.args.args:
        input_lst.append(arg.arg)
    return signature, input_lst

def inject_reward_function(task_code_string, reward_func_code):
    if "def compute_reward(self)" in task_code_string:
        return task_code_string.replace(
            "def compute_reward(self):",
            f"def compute_reward(self):\n    {reward_func_code}"
        )
    elif "def compute_reward(self, actions)" in task_code_string:
        return task_code_string.replace(
            "def compute_reward(self, actions):",
            f"def compute_reward(self, actions):\n    {reward_func_code}"
        )
    else:
        raise NotImplementedError("compute_reward function not found in task code.")

def block_until_training(rl_filepath, log_status=False, path="", response_id=-1):
    # Ensure that the RL training has started before moving on
    while True:
        rl_log = file_to_string(rl_filepath)
        if "fps step:" in rl_log or "Traceback" in rl_log:
            if log_status and "fps step:" in rl_log:
                logging.info(f"Node {path}: Code Run {response_id} successfully training!")
            if log_status and "Traceback" in rl_log:
                logging.info(f"Node {path}: Code Run {response_id} execution error!")
            break

def block_until_training_parallel(rl_filepath, log_status=False, path="", response_id=-1, max_wait_count=1000):
    """
    Wait for training to start, try max_wait_count at most to prevent a dead loop.
    """
    wait_count = 0
    while True:
        rl_log = file_to_string(rl_filepath)
        if "fps step:" in rl_log or "Traceback" in rl_log:
            if log_status and "fps step:" in rl_log:
                logging.info(f"Node {path}: Code Run {response_id} successfully training!")
                return [True, False]
            if log_status and "Traceback" in rl_log:
                logging.info(f"Node {path}: Code Run {response_id} execution error!")
                return [False, False]
        if wait_count >= max_wait_count:
            logging.error(f"Node {path}: Reached maximum wait count without detecting training stability!")
            return [False, True]
        wait_count += 1
        time.sleep(1)

def parse_success_rate(rl_filepath):
    with open(rl_filepath, 'r') as f:
        stdout_str = f.read()
    match = re.search(r'Success Rate:\s*([0-9.]+)', stdout_str)
    if match:
        return float(match.group(1))
    else:
        return -10000.0  # DUMMY_FAILURE

def extract_tensorboard_logdir(stdout_str):
    match = re.search(r'Tensorboard Directory:\s*(\S+)', stdout_str)
    if match:
        return match.group(1).strip()
    else:
        return ""


def load_tensorboard_logs(path):
    data = defaultdict(list)
    event_acc = EventAccumulator(path)
    event_acc.Reload()  # Load all data written so far

    for tag in event_acc.Tags()["scalars"]:
        events = event_acc.Scalars(tag)
        for event in events:
            data[tag].append(event.value)

    return data
