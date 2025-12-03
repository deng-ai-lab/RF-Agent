import logging
import re
import subprocess
import time
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import openai
# from openai import OpenAI
import numpy as np
from .extract_task_code import *
from .utils import *
import math
import random
import threading
import copy

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return list(e_x / e_x.sum())

class MCTSNode:
    def __init__(self, reward_func=None, design_thought=None, action=None, parent=None, history_exec_state="", path="", action_total_num=5):
        self.success_response = None
        self.reward_func = reward_func  # Reward function code for the current node
        self.design_thought = design_thought # The design thought based on the reward function code of the current node
        self.action = action  # Actions selected by the current node action e.g. {'type':'0i', 'index':'0'} {'type':'1m', 'index':'1'}
        self.action_based_nodes_path = None # The action selected by the current node constitutes the paths of other nodes required by the propt
        self.parent = parent  # Parent node
        self.children = []  # List of expanded child nodes
        self.visits = 0  # Number of visits
        self.total_reward = 0.0  # Cumulative Rewards
        self.Q_value = 0.0  # Q-value for selection
        self.reward_cur = 0.0  # Current Rewards for each node
        self.is_fully_expanded = False
        self.history_exec_state = history_exec_state
        self.path = path
        self.exec_file_path = ""
        self.summary_path = ""
        self.prompt = ""
        self.depth = 0
        self.update_best_child_gamma = 0.7
        self.update_mean_gamma = 0.15
        self.epoch_freq = None
        self.sim_time = -1
        self.self_verify_score = -1

    def expand(self, action):
        """
        Extend the current node, create a new child node and add it to the children list.
        """
        child_path = f"{self.path}-{action['type'] + action['index']}" if self.path else str(action['template_id'])
        child = MCTSNode(action=action, parent=self, path=child_path)
        self.children.append(child)
        logging.info(f"Expanded node {self.path} with action {action['type'] + action['index']}, new child {child.path}")
        return child

    def update(self, reward, decay):
        """
        Update the number of visits and cumulative rewards for nodes.
        """
        self.visits += 1
        self.total_reward += reward
        self.Q_mean_value = self.total_reward / self.visits
        best_child_Q = max(child.Q_value for child in self.children)
        update_mean_gamma = self.update_mean_gamma * decay
        self.Q_value = (1 - self.update_best_child_gamma - update_mean_gamma) * self.Q_value + self.update_best_child_gamma * best_child_Q + update_mean_gamma * self.Q_mean_value

    def uct_select_with_verify(self, c_param, q_max, q_min):
        """
        Input: self(node)
        Output: one child of this node
        """
        eps = 1e-8
        children_Q_norm = [((child.Q_value - q_min) / (q_max - q_min + eps)) for child in self.children]
        children_visit_part = [math.sqrt((2 * math.log(self.visits + 1) / child.visits)) for child in self.children]
        children_verify_part = softmax(np.array([child.self_verify_score for child in self.children]))
        children_weights_only_visit = [
            ((child.Q_value - q_min) / (q_max - q_min + eps) + c_param * math.sqrt((2 * math.log(self.visits + 1) / child.visits)))
            for child in self.children
        ]
        children_weights = [child_weight + c_param * self_score for (child_weight, self_score) in zip(children_weights_only_visit, children_verify_part)]
        logging.info(f"Selection children_Q_norm: " + str(children_Q_norm) + "\n")
        logging.info(f"Selection children_visit_part: " + str(children_visit_part) + "\n")
        logging.info(f"Selection children_verify_part: " + str(children_verify_part) + "\n")
        logging.info(f"Selection balancing with c_param {c_param}: " + str(children_weights_only_visit) + "\n")
        logging.info(f"Selection combine the verify with c_param {c_param}: " + str(children_weights) + "\n")
        return self.children[np.argmax(children_weights)]

    def best_child_step_reward(self):
        """
        Select the child node with the highest current reward.
        """
        if not self.children:
            return None
        return max(self.children, key=lambda child: child.reward_cur)


class RFAgent:
    def __init__(self, cfg, prompts_dict, execution_error_feedback, task_code_string, output_file, issac_root_dir, client=None, test_mode=False):
        self.cfg = cfg
        self.prompts_dict = prompts_dict
        self.execution_error_feedback = execution_error_feedback
        self.task_code_string = task_code_string
        self.output_file = output_file
        self.issac_root_dir = issac_root_dir
        self.root = MCTSNode(path="root")
        self.DUMMY_FAILURE = -10000.0
        self.test_mode = test_mode  # Whether it is in debug mode
        self.client = client

        self.max_Q = 0
        self.min_Q = 0
        self.max_depth = cfg.tree_max_depth
        self.c_param_init = 0.4 # lambda for balancing exploration and exploitation
        self.c_param_final = 0.1
        self.max_times = cfg.simulations
        self.sim_times = 0

        # param about "actions"
        self.action_list = ['0i', '1m', '2m', '3e', '4r', '5d']
        self.action_weight_num = [0, 2, 2, 2, 1, 1] # weight_num大于1可以用bs
        self.max_children = sum(self.action_weight_num)

        # param about "action with global message"
        self.elite_set = []
        self.elite_max_length = 10
        self.elite_max_control_num = 4
        self.elite_weight_bias = 1 # e.g.1/(1+index+bias) to form the weight in elite set
        self.tree_reasoning_max_length = 4 # k num
        self.different_max_control_num = 4

        # param about initialization and re-debug for error
        self.initial_size = 6
        self.max_try_num = 9
        self.max_same_try_cnt = 3

        # param about parallel
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=self.max_children)
        self.stable_event = threading.Event()

    def selection(self, node, c_param):
        """
        Use the UCT policy to select a node until a node that is not fully expanded or terminal is found.
        Returns the node and action dictionary (if an unexpanded action is selected).
        """
        while len(node.children) > 0 and node.depth < self.max_depth:
            node = node.uct_select_with_verify(c_param, self.max_Q, self.min_Q)
            logging.info(f"Selection: Moving to best child {node.path}")
        return node


    def get_cur_depth(self, node):
        """
        Returns the depth at which the given node is located from the root node.
        """
        depth = 0
        # Iterate through the path from the current node to the root node and calculate the depth
        while node.parent is not None:
            depth += 1
            node = node.parent
        return depth

    def get_max_depth(self, node):
        """
        Returns the depth from the given node to the bottom.
        """
        # If there are no children, return the depth of the current node (depth of 0)
        if not node.children:
            return 0

        # If there are child nodes, recursively calculate the maximum depth of the subtree
        max_child_depth = max(self.get_max_depth(child) for child in node.children)

        return max_child_depth + 1

    def expansion(self, node, action):
        """
        Extend the node to create a child node by selecting an action that was not tried.
        Logically, 0i must be executed completely at depth = 1 before other actions can be unlocked.
        """
        message_predefined = ""
        child = node.expand(action) # node here will be child's parent
        child.depth = self.get_cur_depth(child)
        if action['type'] == '0i':
            message_predefined = self.action_0_initialize()
            action_based_nodes_path = []
            logging.info(f"Initializing action 0i with Node {child.path} expanded")
        elif action['type'] == '1m':
            message_predefined = self.action_1_mutation_mechanism(child)
            action_based_nodes_path = [child.parent.path]
            logging.info(f"Mutation action 1m with Node {child.path} expanded, using parent message: " + str(action_based_nodes_path) + "\n")
        elif action['type'] == '2m':
            message_predefined = self.action_2_mutation_param(child)
            action_based_nodes_path = [child.parent.path]
            logging.info(f"Mutation action 2m with Node {child.path} expanded, using parent message: " + str(action_based_nodes_path) + "\n")
        elif action['type'] == '3e':
            # take part of nodes from elite set
            ranks = [i for i in range(len(self.elite_set))]
            probs = [1 / (rank + 1 + self.elite_weight_bias) for rank in ranks] # select_weight by num control
            nums = random.randint(2, self.elite_max_control_num) - 1
            nodes = random.choices(self.elite_set, weights=probs, k=nums)
            nodes.append(child.parent)
            message_predefined = self.action_3_crossover_elite(nodes)
            action_based_nodes_path = [node.path for node in nodes]
            logging.info(f"Crossover action 3e with Node {child.path} expanded, using elite set and parent message: " + str(action_based_nodes_path) + "\n")
        elif action['type'] == '4r':
            nodes = []
            now_node = child.parent
            while now_node.path != "root":
                nodes.append(now_node)
                now_node = now_node.parent
            nodes.reverse()
            if len(nodes) > self.tree_reasoning_max_length:
                nodes = nodes[-self.tree_reasoning_max_length:]
            message_predefined = self.action_4_tree_reasoning(nodes)
            action_based_nodes_path = [node.path for node in nodes]
            logging.info(f"Path Reasoning action 4r with Node {child.path} expanded, using old parents message: " + str(action_based_nodes_path) + "\n")
        elif action['type'] == '5d':
            nodes = []
            nums = random.randint(2, self.different_max_control_num) - 1

            # subtree a.k.a node with i0
            child_sub_tree_root = child
            while child_sub_tree_root.action['type'] != '0i':
                child_sub_tree_root = child_sub_tree_root.parent
            # Traversal, randomly select the subtree caused by other i0 actions (excluding the subtree where the parent of the current node is located)
            available_subtrees = [subtree for subtree in self.root.children if
                                  subtree.action['index'] != child_sub_tree_root.action['index']]  # Get all subtrees, excluding the subtrees of the current parent

            selected_subtrees = random.sample(available_subtrees,
                                              min(nums, len(available_subtrees)))  # Randomly select nums subtree from the remaining subtrees

            # From each selected subtree, select the reward nodes based on the reward value and explore the number of random layers
            for subtree in selected_subtrees:
                max_depth = self.get_max_depth(subtree)  # Get the maximum depth of this subtree
                explore_depth = random.randint(0, max_depth)  # Randomly select the depth of exploration

                # Select a node with a higher reward from the depth range of this subtree to explore downward
                d_node = subtree
                for i in range(explore_depth):
                    if d_node.children:  # Make sure that the node has children to continue exploring downwards
                        d_node = d_node.best_child_step_reward()  # Select the node with the largest reward value
                    else:
                        break
                nodes.append(d_node)
            nodes.append(child.parent)
            message_predefined = self.action_5_different_thought(nodes)
            action_based_nodes_path = [node.path for node in nodes]
            logging.info(f"Different thought action 5d with Node {child.path} expanded, using random set and parent message: " + str(action_based_nodes_path) + "\n")

        child.prompt = message_predefined
        child.action_based_nodes_path = action_based_nodes_path

        if len(node.children) >= self.max_children:
            node.is_fully_expanded = True
        return child

    def action_0_initialize(self):
        messages_temp = [
            {"role": "system", "content": self.prompts_dict['initial_system'] + self.prompts_dict['initial_tip']},
            {"role": "user", "content": self.prompts_dict['initial_user'] + self.prompts_dict['initial_action']}]
        return messages_temp

    def action_1_mutation_mechanism(self, node):
        messages_temp = [
            {"role": "system", "content": self.prompts_dict['initial_system'] + self.prompts_dict['initial_tip']},
            {"role": "user", "content": self.prompts_dict['initial_user'] +
                                        self.prompts_dict['action_mutation_mechanism'].format(design_idea=node.parent.design_thought,
                                                                                              reward_function=node.parent.reward_func,
                                                                                              epoch_freq=node.epoch_freq,
                                                                                              trained_results=node.parent.history_exec_state,
                                                                                              trained_result_analysis_tip=self.prompts_dict['trained_result_analysis_tip'])
             + self.prompts_dict['initial_action']}
        ]
        return messages_temp

    def action_2_mutation_param(self, node):
        messages_temp = [
            {"role": "system", "content": self.prompts_dict['initial_system'] + self.prompts_dict['initial_tip']},
            {"role": "user", "content": self.prompts_dict['initial_user'] +
                                        self.prompts_dict['action_mutation_param'].format(design_idea=node.parent.design_thought,
                                                                                          reward_function=node.parent.reward_func,
                                                                                          epoch_freq=node.epoch_freq,
                                                                                          trained_results=node.parent.history_exec_state,
                                                                                          trained_result_analysis_tip=self.prompts_dict['trained_result_analysis_tip'])
             + self.prompts_dict['initial_action']}
        ]
        return messages_temp

    def action_3_crossover_elite(self, nodes):
        # todo: prompt needs to indicate which elite, which parent ?
        content = ''
        for i in list(range(len(nodes))):
            content += self.prompts_dict['base_thought_code'].format(i=i + 1,
                                                                     design_idea=nodes[i].design_thought,
                                                                     reward_function=nodes[i].reward_func,
                                                                     trained_results=nodes[i].history_exec_state)
        messages_temp = [
            {"role": "system", "content": self.prompts_dict['initial_system'] + self.prompts_dict['initial_tip']},
            {"role": "user", "content": self.prompts_dict['initial_user'] +
                                        self.prompts_dict['action_crossover_elite'].format(nums=len(nodes), reward_func_group=content, epoch_freq=nodes[0].epoch_freq, trained_result_analysis_tip=self.prompts_dict['trained_result_analysis_tip'])
             + self.prompts_dict['initial_action']}
        ]
        return messages_temp

    def action_4_tree_reasoning(self, nodes):
        # todo: prompt needs to indicate if tree path is good or bad ?
        content = ''
        for i in list(range(len(nodes))):
            content += self.prompts_dict['base_thought_code'].format(i=i+1,
                                                                     design_idea=nodes[i].design_thought,
                                                                     reward_function=nodes[i].reward_func,
                                                                     trained_results=nodes[i].history_exec_state)
        messages_temp = [
            {"role": "system", "content": self.prompts_dict['initial_system'] + self.prompts_dict['initial_tip']},
            {"role": "user", "content": self.prompts_dict['initial_user'] +
                                        self.prompts_dict['action_tree_reasoning'].format(nums=len(nodes), reward_func_group=content, epoch_freq=nodes[0].epoch_freq, trained_result_analysis_tip=self.prompts_dict['trained_result_analysis_tip'])
             + self.prompts_dict['initial_action']}
        ]
        return messages_temp

    def action_5_different_thought(self, nodes):
        content = ''
        for i in list(range(len(nodes))):
            content += self.prompts_dict['base_thought_code'].format(i=i + 1,
                                                                     design_idea=nodes[i].design_thought,
                                                                     reward_function=nodes[i].reward_func,
                                                                     trained_results=nodes[i].history_exec_state)
        messages_temp = [
            {"role": "system", "content": self.prompts_dict['initial_system'] + self.prompts_dict['initial_tip']},
            {"role": "user", "content": self.prompts_dict['initial_user'] +
                                        self.prompts_dict['action_different_thought'].format(nums=len(nodes), reward_func_group=content, epoch_freq=nodes[0].epoch_freq, trained_result_analysis_tip=self.prompts_dict['trained_result_analysis_tip'])
                                        + self.prompts_dict['initial_action']}
        ]
        return messages_temp


    def self_node_value_verify_single(self, node):
        messages_temp = [
            {"role": "system", "content": self.prompts_dict['initial_system_verify']},
            {"role": "user", "content": self.prompts_dict['initial_user'] +
                                        self.prompts_dict['self_node_value_verify_single'].format(design_idea=node.design_thought,reward_function=node.reward_func)}
        ]
        return messages_temp

    def self_node_value_verify_group(self, nodes):
        content = ''
        for i in list(range(len(nodes))):
            content += self.prompts_dict['base_thought_code_wo_result'].format(i=i+1,
                                                                     design_idea=nodes[i].design_thought,
                                                                     reward_function=nodes[i].reward_func)
        messages_temp = [
            {"role": "system", "content": self.prompts_dict['initial_system_verify']},
            {"role": "user", "content": self.prompts_dict['initial_user'] +
                                        self.prompts_dict['self_node_value_verify_group'].format(nums=len(nodes), reward_func_group=content)}
        ]
        return messages_temp

    def simulation_parallel(self, node, env, stable_event, seed):
        """
        Perform the simulation process.
        Given a node, collect the reward function generated by the LLM in the test node by the shape of the propt, and test the final score of the policy training using the reward function.  Including regeneration of error codes and thinking alignment.
        In test mode, reward values are randomly generated.
        """
        if self.test_mode:
            # In test mode, randomly generate rewards
            reward = random.uniform(0, 1)
            success = 1
            logging.info(f"Test Simulation: Node {node.path} assigned random reward {reward}")
            return success, reward # Return to success
        else:
            max_try_num = self.max_try_num
            retry_count = 0
            same_try_cnt = 0
            success_cnt = 0
            responses = []
            response_cur = None
            total_token = 0
            total_completion_token = 0
            response_id = -1
            temp_stable_times = 1
            messages_temp = node.prompt
            reward_fail_bound = 0 # change to -1 in anymal and quadcopter envs

            while retry_count < max_try_num and success_cnt == 0:
                # If the reward function generated for the first time is still unable to generate after multiple iterations after adding the error message, the reward function generated for the first time is discarded and directly starts with a new reward function.
                if same_try_cnt == self.max_same_try_cnt:
                    messages_temp = node.prompt
                    same_try_cnt = 0
                logging.info(f"Node {node.path}: LLM Action Input:\n " + str(messages_temp) + "\n")
                for attempt in range(1000):
                    try:

                        response_cur = openai.ChatCompletion.create(
                            model=self.cfg.model,
                            messages=messages_temp,
                            temperature=self.cfg.temperature,
                            n=1
                        )
                        break
                    except Exception as e:
                        if attempt >= 5:
                            print("Current Openai Attempt Size", attempt)
                        logging.info(f"Attempt {attempt + 1} failed with error: {e}")
                        time.sleep(1)
                if response_cur is None:
                    logging.info("Code terminated due to too many failed attempts!")
                    return node, success_cnt, reward_fail_bound # Return to fail

                response_id += 1
                responses.extend(response_cur.choices)
                total_completion_token += response_cur.usage.completion_tokens
                total_token += response_cur.usage.total_tokens
                logging.info(f"Node {node.path}: LLM Action Output:\n " + responses[response_id].message.content + "\n") # comment off this line if too long

                code_runs = []
                rl_runs = []

                response_cur = responses[response_id].message.content
                node.success_response = response_cur
                logging.info(f"Node {node.path}: Processing Code Run {response_id}")

                design_thought = re.search(r"\{(.*?)\}", response_cur, re.DOTALL).group(1)
                if len(design_thought) == 0:
                    if 'python' in response_cur:
                        design_thought = re.findall(r'^.*?(?=python)', response_cur, re.DOTALL)
                    elif 'import' in response_cur:
                        design_thought = re.findall(r'^.*?(?=import)', response_cur, re.DOTALL)
                    else:
                        design_thought = re.findall(r'^.*?(?=def)', response_cur, re.DOTALL)

                # Regex patterns to extract python code enclosed in GPT response
                patterns = [
                    r'```python(.*?)```',
                    r'```(.*?)```',
                    r'"""(.*?)"""',
                    r'""(.*?)""',
                    r'"(.*?)"',
                ]
                code_string = None
                for pattern in patterns:
                    code_match = re.search(pattern, response_cur, re.DOTALL)
                    if code_match is not None:
                        code_string = code_match.group(1).strip()
                        break
                code_string = response_cur if code_string is None else code_string

                # Remove unnecessary imports
                lines = code_string.split("\n")
                for i, line in enumerate(lines):
                    if line.strip().startswith("def "):
                        code_string = "\n".join(lines[i:])

                # Add the RF-Agent Reward Signature to the environment code
                try:
                    gpt_reward_signature, input_lst = get_function_signature(code_string)
                except Exception as e:
                    logging.info(f"Node {node.path}: Code Run {response_id} cannot parse function signature! With error: {str(e)}")
                    traceback_msg = "Code Run cannot be executed due to function signature error! Please re-write an entirely new reward function!"
                    messages_temp = [{"role": "system", "content": self.prompts_dict['initial_system'] + self.prompts_dict['initial_tip']},
                                     {"role": "user", "content": self.prompts_dict['initial_user'] + self.prompts_dict['initial_failed_feedback'].format(reward_function=code_string, traceback_msg=traceback_msg)}]

                    node.history_exec_state = traceback_msg
                    node.design_thought = design_thought
                    node.reward_func = code_string
                    retry_count += 1
                    same_try_cnt += 1
                    continue

                code_runs.append(code_string)
                reward_signature = [
                    f"self.rew_buf[:], self.rew_dict = {gpt_reward_signature}",
                    f"self.extras['gpt_reward'] = self.rew_buf.mean()",
                    f"for rew_state in self.rew_dict: self.extras[rew_state] = self.rew_dict[rew_state].mean()",
                ]
                indent = " " * 8
                reward_signature = "\n".join([indent + line for line in reward_signature])
                if "def compute_reward(self)" in self.task_code_string:
                    task_code_string_iter = self.task_code_string.replace("def compute_reward(self):",
                                                                          "def compute_reward(self):\n" + reward_signature)
                elif "def compute_reward(self, actions)" in self.task_code_string:
                    task_code_string_iter = self.task_code_string.replace("def compute_reward(self, actions):",
                                                                          "def compute_reward(self, actions):\n" + reward_signature)
                else:
                    raise NotImplementedError

                # Save the new environment code when the output contains valid code string!
                with open(self.output_file, 'w') as file:
                    file.writelines(task_code_string_iter + '\n')
                    file.writelines("from typing import Tuple, Dict" + '\n')
                    file.writelines("import math" + '\n')
                    file.writelines("import torch" + '\n')
                    file.writelines("from torch import Tensor" + '\n')
                    if "@torch.jit.script" not in code_string:
                        code_string = "@torch.jit.script\n" + code_string
                    file.writelines(code_string + '\n')

                with open(f"env_Node{node.path}_response{response_id}_rewardonly.py", 'w') as file:
                    file.writelines(code_string + '\n')

                # Copy the generated environment code to hydra output directory for bookkeeping
                shutil.copy(self.output_file, f"env_Node{node.path}_response{response_id}.py")
                node.exec_file_path = f"env_Node{node.path}_response{response_id}.py"

                # Find the freest GPU to run GPU-accelerated RL
                set_freest_gpu()
                env = os.environ.copy()

                # Execute the python file with flags
                rl_filepath = f"env_Node{node.path}_response{response_id}.txt"
                with open(rl_filepath, 'w') as f:
                    process = subprocess.Popen(['python', '-u', f'{self.issac_root_dir}/train_with_seed.py',
                                                'hydra/output=subprocess',
                                                f'task={self.cfg.env.task}{self.cfg.suffix}',
                                                f'wandb_activate={self.cfg.use_wandb}',
                                                f'wandb_entity={self.cfg.wandb_username}',
                                                f'wandb_project={self.cfg.wandb_project}',
                                                f'headless={not self.cfg.capture_video}',
                                                f'capture_video={self.cfg.capture_video}',
                                                'force_render=False',
                                                f'max_iterations={self.cfg.max_iterations}',
                                                f'seed={seed}'],
                                               stdout=f, stderr=f, env=env)
                [temp_stable_signal, cuda_err] = block_until_training_parallel(rl_filepath, log_status=True, path=node.path, response_id=response_id)
                if temp_stable_times == 1 and temp_stable_signal:
                    stable_event.set()
                    temp_stable_times += 1
                elif cuda_err: # cuda ood error
                    stable_event.set()
                    return node, success_cnt, reward_fail_bound # Return to fail

                rl_runs.append(process)

                # Gather RL training results and construct reward reflection
                code_paths = []

                # todo : No loop here
                rl_run = rl_runs[0]
                rl_run.communicate()
                rl_filepath = f"env_Node{node.path}_response{response_id}.txt"
                code_paths.append(f"env_Node{node.path}_response{response_id}.py")
                try:
                    with open(rl_filepath, 'r') as f:
                        stdout_str = f.read()
                except:
                    logging.info(f"Node {node.path}: Code Run {response_id} cannot be executed due to function signature error!")
                    traceback_msg="Code Run cannot be executed due to function signature error! Please re-write an entirely new reward function!"
                    messages_temp = [{"role": "system", "content": self.prompts_dict['initial_system'] + self.prompts_dict['initial_tip']},
                                     {"role": "user", "content": self.prompts_dict['initial_user'] + self.prompts_dict['initial_failed_feedback'].format(reward_function=code_string, traceback_msg=traceback_msg)}]

                    node.history_exec_state = traceback_msg
                    node.design_thought = design_thought
                    node.reward_func = code_string
                    retry_count += 1
                    same_try_cnt += 1
                    continue

                traceback_msg = filter_traceback(stdout_str)

                if traceback_msg == '':
                    # If RL execution has no error, provide policy statistics feedback
                    logging.info(f"Node {node.path}: Code Run {response_id} has no error!")
                    messages_temp = [{"role": "system", "content": self.prompts_dict['initial_system']},
                                     {"role": "user", "content": self.prompts_dict['initial_user'] + self.prompts_dict['initial_thought_alignment'].format(reward_function=code_string, design_idea=design_thought)}]
                    logging.info(f"Node {node.path}: LLM Thought Align Input:\n " + str(messages_temp) + "\n")
                    for attempt in range(1000):
                        try:

                            response_cur = openai.ChatCompletion.create(
                                model=self.cfg.model,
                                messages=messages_temp,
                                temperature=self.cfg.temperature,
                                n=1
                            )
                            break
                        except Exception as e:
                            if attempt >= 5:
                                print("Current Openai Attempt Size", attempt)
                            logging.info(f"Attempt {attempt + 1} failed with error: {e}")
                            time.sleep(1)
                    if response_cur is None:
                        logging.info("Code terminated due to too many failed attempts!")
                        return node, success_cnt, reward_fail_bound # Return to fail
                    design_thought = response_cur.choices[0].message.content
                    logging.info(f"Node {node.path}: LLM Thought Align Output:\n " + design_thought + "\n")  # may comment off

                    # self verify start
                    logging.info(f"Node {node.path}: Self Verify Start!" + "\n")
                    messages_temp = self.self_node_value_verify_single(node)
                    for attempt in range(1000):
                        try:
                            response_cur = openai.ChatCompletion.create(
                                model=self.cfg.model,
                                messages=messages_temp,
                                temperature=self.cfg.temperature,
                                n=1
                            )
                            break
                        except Exception as e:
                            if attempt >= 5:
                                print("Current Openai Attempt Size", attempt)
                            logging.info(f"Attempt {attempt + 1} failed with error: {e}")
                            time.sleep(1)
                    if response_cur is None:
                        logging.info("Code terminated due to too many failed attempts!")
                        return node, success_cnt, reward_fail_bound # Return to fail
                    self_verify_score = response_cur.choices[0].message.content
                    self_verify_score = re.findall(r"\[(.*?)\]", self_verify_score, re.DOTALL)
                    try:
                        node.self_verify_score = float(self_verify_score[-1])
                        print(str(self_verify_score[-1]))
                    except:
                        node.self_verify_score = 0
                    logging.info(f"Node {node.path}: Self Verify value:" + str(node.self_verify_score) + "\n")
                    #  self verify end

                    lines = stdout_str.split('\n')
                    for i, line in enumerate(lines):
                        if line.startswith('Tensorboard Directory:'):
                            break
                    tensorboard_logdir = line.split(':')[-1].strip()
                    tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)
                    max_iterations = np.array(tensorboard_logs['gt_reward']).shape[0]
                    epoch_freq = max(int(max_iterations // 10), 1)
                    node.epoch_freq = epoch_freq
                    content = ''

                    # Add reward components log to the feedback
                    metric_cur_max = 0
                    task_score = 0
                    for metric in tensorboard_logs:
                        if "/" not in metric:
                            metric_cur = ['{:.2f}'.format(x) for x in tensorboard_logs[metric][::epoch_freq]]
                            metric_cur_max = max(tensorboard_logs[metric])
                            metric_cur_mean = sum(tensorboard_logs[metric]) / len(tensorboard_logs[metric])
                            if "consecutive_successes" == metric:
                                task_score = metric_cur_max
                            metric_cur_min = min(tensorboard_logs[metric])
                            if metric != "gt_reward" and metric != "gpt_reward":
                                if metric != "consecutive_successes":
                                    metric_name = metric
                                else:
                                    metric_name = "task_score"
                                content += f"{metric_name}: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"
                            else:
                                # Provide ground-truth score when success rate not applicable
                                if "consecutive_successes" not in tensorboard_logs:
                                    content += f"ground-truth score: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"
                    # Update the node's historical execution status and reward functions
                    node.summary_path = tensorboard_logdir
                    node.history_exec_state = content
                    node.design_thought = design_thought
                    node.reward_func = code_string
                    success_cnt = 1
                    stable_event.set()
                    return node, success_cnt, task_score  # Return to success

                else:
                    # Otherwise, provide execution traceback error feedback
                    logging.info(f"Node {node.path}: Code Run {response_id} cannot be executed due to environment or other error!")
                    # Update the node's historical execution status and reward functions
                    node.history_exec_state = traceback_msg
                    node.reward_func = code_string
                    messages_temp = [{"role": "system", "content": self.prompts_dict['initial_system'] + self.prompts_dict['initial_tip']},
                                     {"role": "user", "content": self.prompts_dict['initial_user'] + self.prompts_dict['initial_failed_feedback'].format(reward_function=code_string, traceback_msg=traceback_msg)}]
                    retry_count += 1
                    same_try_cnt += 1

            stable_event.set()
            return node, success_cnt, reward_fail_bound

    def backpropagation(self, node, reward):
        """
        Backpropagation updates the node's statistics.
        """
        with self.lock:
            node.reward_cur = reward
            node.Q_value = reward
            node.visits += 1
            node.total_reward += reward
            self.min_Q = min(self.min_Q, node.Q_value)
            self.max_Q = max(self.max_Q, node.Q_value)

            self.elite_set.append(node)
            self.elite_set.sort(key=lambda x: x.reward_cur, reverse=True)
            if len(self.elite_set) > self.elite_max_length:
                self.elite_set = self.elite_set[:self.elite_max_length]

            node = node.parent
            while node.path != 'root':
                decay = 1 - float(self.sim_times / self.max_times)
                node.update(reward, decay)
                logging.info(f"Backpropogation {node.path} over!")
                node = node.parent
            if node.path == 'root':
                node.visits += 1

    def traverse_tree(self, node, tree_info):
        """
        Traverse the entire tree and collect node information.
        """
        node_info = {
            "path": node.path,
            "action": node.action,
            "action_type": node.action['type'] if node.action else None,
            "action_based_nodes_path": node.action_based_nodes_path,
            "success_response": node.success_response,
            "design_thought": node.design_thought,
            "reward_func": node.reward_func,
            "history_exec_state": node.history_exec_state,
            "prompt": node.prompt,
            "Q_value": node.Q_value,
            "visits": node.visits,
            "total_reward": node.total_reward,
            "reward_cur": node.reward_cur,
            "parent": node.parent.path if node.parent else None,
            "children": [child.path for child in node.children] if node.children else None,
            "depth": node.depth,
            "exec_file_path": node.exec_file_path,
            "summary_path": node.summary_path,
            "sim_time": node.sim_time,
            "self_verify_score": node.self_verify_score
        }
        tree_info.append(node_info)
        for child in node.children:
            self.traverse_tree(child, tree_info)

    def save_tree(self, filename="mcts_tree.json"):
        """
        Save the tree structure to the JSON file.
        """
        tree_info = []
        self.traverse_tree(self.root, tree_info)
        with open(filename, 'w') as f:
            json.dump(tree_info, f, indent=4)
        logging.info(f"MCTS tree saved to {filename}")

    def find_best_nodes(self):
        """
        Find nodes with the highest average reward and the highest current reward.
        """
        best_avg_node = None
        best_cur_node = None
        best_avg_value = -float('inf')
        best_cur_value = -float('inf')

        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.visits > 0 and node.path != "root":
                avg_reward = node.total_reward / node.visits
                logging.info(f"Node {node.path}: Visits={node.visits}, Total Reward={node.total_reward}, Avg Reward={avg_reward}, Current Reward={node.reward_cur}")
                if avg_reward > best_avg_value:
                    best_avg_value = avg_reward
                    best_avg_node = node
                if node.reward_cur > best_cur_value:
                    best_cur_value = node.reward_cur
                    best_cur_node = node
            stack.extend(node.children)

        logging.info(
            f"find_best_nodes: Best_avg_node={best_avg_node.path if best_avg_node else 'None'}, Best_cur_node={best_cur_node.path if best_cur_node else 'None'}")
        return best_avg_node, best_cur_node

    def run(self, env=None, train_seed=0):
        """
        The main method of performing RF-Agent, performing multiple simulations to build a search tree.
        """
        simulations_number = self.max_times
        reward_max = 0
        logging.info(f"Starting MCTS with {simulations_number} simulations")

        while self.sim_times < self.max_times:
            action_waiting_simulation_list = []
            c_param = (self.c_param_init - self.c_param_final) * (1 - float(self.sim_times / self.max_times)) + self.c_param_final
            node = self.selection(self.root, c_param)

            if node.path == "root":
                # First expansion: expand with '0i' actions, `self.initial_size` times
                for i in range(self.initial_size):
                    action_waiting_simulation_list.append({'type': '0i', 'index': str(i)})
            else:
                # Subsequent expansions: use actions from self.action_list with corresponding weight numbers
                for action_type, num_expansions in zip(self.action_list, self.action_weight_num):
                    for i in range(num_expansions):
                        # Construct actions based on the action type and its index
                        action_waiting_simulation_list.append({'type': action_type, 'index': str(i)})

            logging.info(f"Node {node.path} waiting actions to expand:\n " + str(action_waiting_simulation_list) + "\n")

            futures = []
            for action in action_waiting_simulation_list:
                child = self.expansion(node, action)
                future = self.executor.submit(self.simulation_parallel, child, env, self.stable_event, train_seed)  # Submit the task to the thread pool
                futures.append(future)
                self.stable_event.wait()  # wait for stable_event triggered
                self.stable_event.clear()  # Clear the event to prepare for the next parallel node

            for future in as_completed(futures):
                child, success, reward = future.result()
                self.sim_times += 1
                child.sim_time = self.sim_times
                # Backpropagation
                self.backpropagation(child, reward)
                if success:
                    logging.info(f"Simulation {self.sim_times}: Node {child.path} succeeded with reward {reward}")
                    if reward > reward_max:
                        reward_max = reward
                else:
                    logging.info(f"Simulation {self.sim_times}: Node {child.path} failed after retries")
                print(f"Simulation {self.sim_times}: Reward={reward}, Current Max Reward={reward_max}")
                self.save_tree()

                # Find the best node
            best_avg_node, best_cur_node = self.find_best_nodes()

            # Print or record information about the best node
            if best_avg_node and best_avg_node.exec_file_path:
                logging.info(
                    f"Best average reward node path: {best_avg_node.path}, file: {best_avg_node.exec_file_path}, average reward: {best_avg_node.total_reward / best_avg_node.visits}")
            if best_cur_node and best_cur_node.exec_file_path:
                logging.info(
                    f"Best current reward node path: {best_cur_node.path}, file: {best_cur_node.exec_file_path}, current reward: {best_cur_node.reward_cur}")

            if self.elite_set:
                for i in range(len(self.elite_set)):
                    node = self.elite_set[i]
                    logging.info(
                        f"Elite reward node {i} path: {node.path}, file: {node.exec_file_path}, current reward: {node.reward_cur}")

        return best_cur_node, best_avg_node, self.elite_set
