import argparse
import ast
import datetime
import json
import os
import re
import sys
import time
import cv2
import yaml
import prompts
from model import parse_explore_rsp, get_qwen_response, get_openai_response, get_qwenmax_response, get_llama_response, parse_llama_explore_rsp, get_gpt4v_response, get_model_response_qwen, get_InternVL_response
from ..utils import *

def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='data directory', default=r'noisy_data')
    parser.add_argument('--model_name', type=str, help='model name')
    parser.add_argument('--config_path', type=str, help='config path', default=r'config.yaml')
    parser.add_argument('--start_num', type=int, help='task start number', default=0)
    parser.add_argument('--model_type', type=str, help='model type', default=r'OpenAI')
    parser.add_argument('--save_path', type=str, help='save results path')
    return parser.parse_args()

args = get_parse()

with open(args.config_path, "r") as f:
    config = yaml.safe_load(f)

data_dir = args.data_dir
# task_dir = os.path.join(data_dir, args.data_type)
prefix = 'data'
all_data = os.listdir(data_dir)

tasks_data = []
for data in all_data:
    if prefix in data:
        tasks_data.append(data) 
    # if data.startswith(prefix):
    #     tasks_data.append(data) 

if args.model_type == 'Qwen':
    API_url = config[args.model_name]['QWEN_API_BASE']
elif args.model_type == 'OpenAI':
    API_url = config[args.model_name]['OPENAI_API_BASE']
    token = config[args.model_name]['OPENAI_API_KEY']
    temperature = config[args.model_name]['TEMPERATURE']
    max_tokens = config[args.model_name]['MAX_TOKENS']
elif args.model_type == 'LLAMA':
    API_url = config[args.model_name]['LLAMA_API_BASE']

for i, task in enumerate(tasks_data[args.start_num:], start=args.start_num):

    prefix, suffix = task.split('_')[0], task.split('_')[-1].split('.')[0]

    task_name = prefix + '_' + suffix + '-'
    task_data = []
    last_act = "None"

    with open(os.path.join(args.data_dir, task), 'r', encoding='gb18030') as f:
        task_info = json.load(f)
    task_desc = task_info['user_instruction']['query']
    task_gt_path = task_info['trajectories']

    for data in all_data:
        if data.startswith(task_name):
            task_data.append(data.split('_')[1])

    task_number = set(re.search(r'\d+-\d+', name).group() for name in task_data)
    sorted_numbers = sorted(task_number, key=lambda x: int(x.split('-')[1]))

    action_history = []
    for round_count, number in enumerate(sorted_numbers):
        print_with_color(f"Round{round_count+1}", "yellow")
        xml_path = os.path.join(task_dir, number + '.xml')
        screenshot_path = os.path.join(task_dir, number + '.png')
        click_actions, scroll_actions, current_page_actions = noisy_actions_generate(xml_path)

        drawn_screenshot = os.path.join(task_dir, number + f'_labeled.png')
        scroll_action_bounds = get_scroll_bounds(scroll_actions)
        draw_bbox_multi(screenshot_path, drawn_screenshot, click_actions)
        draw_bbox_multi(drawn_screenshot, drawn_screenshot, scroll_action_bounds)

        imgcv = cv2.imread(drawn_screenshot)
        imgcv = cv2.resize(imgcv, (1080, 2400))
        cv2.imwrite(drawn_screenshot, imgcv)

        prompt = re.sub(r"<ui_document>", "", prompts.noisy_task_template)
        prompt = re.sub(r"<task_description>", task_desc, prompt)

        prompt = re.sub(r"<last_act>", last_act, prompt)
        print_with_color("Thinking about what to do in the next step...", "yellow")
        
        if args.model_name == 'qwen272B':
            status, rsp = get_model_response_qwen(prompt, [drawn_screenshot])
        elif args.model_name == 'gpt-4o':
            status, rsp = get_openai_response(prompt, [drawn_screenshot], args.model_name, API_url, token)
        elif args.model_name == 'llama90B':
            status, rsp = get_llama_response(prompt, [drawn_screenshot], args.model_name, API_url)
        elif args.model_name == 'gpt-4v':
            status, rsp = get_gpt4v_response(prompt, [drawn_screenshot], args.model_name, API_url, token)
        elif args.model_name == 'InternVL2':
            status, rsp = get_InternVL_response(prompt, drawn_screenshot)
        elif args.model_name == 'qwenvlmax':
            status, rsp = get_qwenmax_response(prompt, [drawn_screenshot], args.model_name, API_url)
        
        if status:
            if args.model_name == 'llama90B':
                res = parse_llama_explore_rsp(rsp)
            else:
                res = parse_explore_rsp(rsp)

            act_name = res[0]
            
            last_act = res[-1]
            res = res[:-1]
            if act_name == "click":
                _, area = res
                try:
                    area_idx = int(re.findall(r'(\d+)', area)[0]) 
                    action_bounds = click_actions[area_idx - 1]
                    action_history.append(str(action_bounds))
                except:
                    print(len(click_actions), area_idx)
                    print('click_error')
                    action_history.append("click error")
            elif act_name == "scroll":
                _, area, direction = res
                
                area_idx = int(re.findall(r'(\d+)', area)[0])
                if area_idx <= len(scroll_action_bounds):    
                    scroll_action = str(scroll_action_bounds[area_idx - 1])
                    scroll_action = "scroll(" + scroll_action + direction + ")"
                    action_history.append(scroll_action)
                else:
                    print('scroll_error')
                    action_history.append("scroll error")
            elif act_name == "input":
                _, text = res
                action_history.append("input(" + text + ")")
            else:
                action_history.append("no such act")
        else:
            print(rsp)
            action_history.append("error")
            print_with_color('error', "red")

    result_path = os.path.join(data_dir, 'noisy_tasks' + args.model_name)
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    print('task completed')
    with open(os.path.join(result_path, f'{task}.txt'), "w", encoding="utf-8") as f:
        for action in action_history:
            f.write(f"{action}\n")