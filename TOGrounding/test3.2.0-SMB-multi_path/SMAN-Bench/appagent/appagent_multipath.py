import argparse
import json
import os
import re
import sys
import time
import yaml
import tqdm
import cv2

_APPAGENT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_APPAGENT_DIR)
if _APPAGENT_DIR not in sys.path:
    sys.path.insert(0, _APPAGENT_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import prompts
from appagent.model import (
    get_openai_response,
    get_qwen_response,
    parse_multi_explore_rsp,
    get_gpt4v_response,
    get_qwenmax_response,
    get_model_response_qwen,
    get_InternVL_response,
)
from utils import *


def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='data directory')
    parser.add_argument('--task_file', type=str, help='task_file', default=r'simple_tasks_sample.json')
    parser.add_argument('--model_name', type=str, help='model name')
    parser.add_argument('--config_path', type=str, help='config path', default=r'config.yaml')
    parser.add_argument('--task_type', type=str, help='task type', default=r'multi_simple')
    parser.add_argument('--start_num', type=int, help='task start number', default=0)
    parser.add_argument('--model_type', type=str, help='model type', default='OpenAI')
    parser.add_argument('--save_path', type=str, help='save path')
    parser.add_argument('--max_rounds', type=int, help='max rounds', default=20)
    parser.add_argument('--limit', type=int, default=None, help='max number of tasks to run')
    return parser.parse_args()


args = get_parse()
data_dir = args.data_dir
MAX_ROUNDS = args.max_rounds

with open(args.config_path, "r") as f:
    config = yaml.safe_load(f)

if args.model_name not in config:
    raise KeyError(f"Model '{args.model_name}' not found in {args.config_path}")

model_cfg = config[args.model_name]
API_url = None
token = None
api_model = model_cfg.get('OPENAI_API_MODEL', args.model_name)
temperature = model_cfg.get('TEMPERATURE', 0.0)
max_tokens = model_cfg.get('MAX_TOKENS', 300)
request_interval = model_cfg.get('REQUEST_INTERVAL', 0)

if args.model_type == 'Qwen':
    API_url = model_cfg['QWEN_API_BASE']
elif args.model_type == 'OpenAI':
    API_url = model_cfg['OPENAI_API_BASE']
    token = model_cfg['OPENAI_API_KEY']

task_path = os.path.join(args.data_dir, args.task_file)

with open(task_path, "r") as f:
    tasks = json.load(f)

tasks_to_run = tasks[args.start_num:]
if args.limit is not None:
    tasks_to_run = tasks_to_run[: args.limit]

for offset, task in tqdm.tqdm(enumerate(tasks_to_run)):
    i = args.start_num + offset
    final_page_name = task['name']
    task_info = task['task']

    task_desc = task_info.split('\n')[-1]

    multi_task_desc = []
    steps = re.findall(r'^\d+\.\s(.*)', task_info, re.MULTILINE)
    for step in steps:
        multi_task_desc.append(step)

    app_prefix = final_page_name.split('0')[0] + '_'
    init_data_path = find_dir_with_prefix(data_dir, app_prefix)
    if init_data_path is None:
        print_with_color(
            f"Skip task {final_page_name}: no graph dir matching '{app_prefix}' under {data_dir}",
            "yellow",
        )
        continue
    task_dir = os.path.join(data_dir, init_data_path)

    if not os.path.exists(task_dir):
        print_with_color(f"Skip task {final_page_name}: task_dir not found: {task_dir}", "yellow")
        continue

    round_count = 0
    last_act = "None"
    task_complete = False

    with open(os.path.join(task_dir, 'all_action_id.json'), "r", encoding='UTF-8') as fp:
        all_action_ids = json.loads(json.load(fp))

    _, id_to_action = build_action_id_maps(all_action_ids)

    with open(os.path.join(task_dir, 'all_page_actions.json')) as fp:
        all_page_actions = json.load(fp)

    with open(os.path.join(task_dir, 'all_page_id.json')) as fp:
        json.loads(json.load(fp))

    with open(os.path.join(task_dir, 'all_triple.json')) as fp:
        all_page_triples = json.load(fp)['data']
    all_page_convert = {}
    for all_page_triple in all_page_triples:
        all_page_convert[all_page_triple[0] + 'act' + str(all_page_triple[1])] = all_page_triple[2]

    current_page_actions = {}
    for current_page_data in all_page_actions['data']:
        current_page_actions[current_page_data['name']] = current_page_data['action_valid']

    current_page_name = final_page_name.split('_')[0]

    ans_action_id = []
    ans_action_info = []
    ans_history_pages = []

    while round_count < MAX_ROUNDS:
        round_count += 1
        print_with_color(f"Round {round_count}", "yellow")

        screenshot_path = os.path.join(task_dir, current_page_name, current_page_name + '-screen.png')
        html_file = os.path.join(task_dir, current_page_name, current_page_name + '-html.txt')
        xml_file = os.path.join(task_dir, current_page_name, current_page_name + '-xml.txt')

        if not os.path.isfile(screenshot_path):
            print_with_color(f"Missing screenshot: {screenshot_path}", "red")
            break

        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        with open(xml_file, 'r', encoding='utf-8') as f:
            xml_content = f.read()

        current_page_all_action_ids = current_page_actions.get(current_page_name, [])
        current_action_infos = []
        for action_id in current_page_all_action_ids:
            current_action_infos.append(id_to_action[int(action_id)])

        click_actions, input_actions_pre, scroll_actions, current_page_all_actions = actions_generate(
            html_content, xml_content
        )
        click_actions, input_actions, scroll_actions, scroll_action_bounds = current_actions_generate(
            current_action_infos, click_actions, input_actions_pre, scroll_actions, current_page_all_actions
        )

        drawn_screenshot = os.path.join(
            task_dir, current_page_name, current_page_name + '_labeled_multi.png'
        )

        draw_bbox_multi(screenshot_path, drawn_screenshot, click_actions)
        draw_bbox_multi(drawn_screenshot, drawn_screenshot, scroll_action_bounds)

        imgcv = cv2.imread(drawn_screenshot)
        imgcv = cv2.resize(imgcv, (1080, 2400))
        cv2.imwrite(drawn_screenshot, imgcv)

        prompt = re.sub(r"<ui_document>", "", prompts.multipath_task_template)
        prompt = re.sub(r"<task_description>", task_desc, prompt)

        task_count = current_page_name.count('_')
        if task_count < len(multi_task_desc) - 1:
            prompt = re.sub(r"<current_task_desc>", multi_task_desc[task_count], prompt)
        else:
            prompt = re.sub(r"<current_task_desc>", multi_task_desc[len(multi_task_desc) - 1], prompt)
        prompt = re.sub(r"<last_act>", last_act, prompt)
        print_with_color("Thinking about what to do in the next step...", "yellow")

        status = False
        rsp = ""

        if args.model_type == 'OpenAI':
            status, rsp = get_openai_response(
                prompt,
                [drawn_screenshot],
                api_model,
                API_url,
                token,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        elif args.model_name in ('qwen272B', 'qwen272B_1'):
            status, rsp = get_model_response_qwen(prompt, [drawn_screenshot])
        elif args.model_name in ('gpt-4o', 'gpt-4-vision-preview'):
            status, rsp = get_openai_response(
                prompt, [drawn_screenshot], args.model_name, API_url, token
            )
        elif args.model_name in ('gpt-4v', 'gpt-4v-1'):
            status, rsp = get_gpt4v_response(
                prompt, [drawn_screenshot], args.model_name, API_url, token
            )
        elif args.model_name == 'qwen-vl-max':
            status, rsp = get_qwenmax_response(
                prompt, [drawn_screenshot], args.model_name, API_url
            )
        elif args.model_name == 'InternVL2':
            status, rsp = get_InternVL_response(prompt, drawn_screenshot)
        else:
            print_with_color(
                f"Unsupported model_name={args.model_name} model_type={args.model_type}",
                "red",
            )

        if request_interval:
            time.sleep(request_interval)

        if status:
            res = parse_multi_explore_rsp(rsp)
            act_name = res[0]
            last_act = res[-1]
            res = res[:-1]

            if act_name == "click":
                _, area = res
                area_idx = int(re.findall(r'(\d+)', str(area))[0])
                current_page_name, action_info, action_id = action_click(
                    click_actions, area_idx, current_page_all_actions,
                    all_action_ids, current_page_name, all_page_convert,
                )
                if action_info == "ERROR":
                    ans_action_info.append(str(area) + "click error")
                    ans_action_id.append(action_id)
                    print_with_color("ERROR: click execution failed", "red")
                else:
                    ans_action_info.append(action_info)
                    ans_action_id.append(action_id)
            elif act_name == "scroll":
                _, area, direction = res
                area_idx = int(re.findall(r'(\d+)', str(area))[0])
                current_page_name, action_info, action_id = action_scroll(
                    scroll_action_bounds, area_idx, direction, all_action_ids,
                    current_page_name, all_page_convert,
                )
                if action_info == "ERROR":
                    ans_action_info.append(str(area) + str(direction) + "scroll error")
                    ans_action_id.append(action_id)
                    print_with_color("ERROR: scroll execution failed", "red")
                else:
                    ans_action_info.append(action_info)
                    ans_action_id.append(action_id)
            elif act_name == "input":
                _, text = res
                current_page_name, action_info, action_id = action_input(
                    text, input_actions, current_page_all_actions, all_action_ids,
                    current_page_name, all_page_convert,
                )
                if action_info == "ERROR":
                    ans_action_info.append('input' + text + "error")
                    ans_action_id.append(action_id)
                    print_with_color("ERROR: input execution failed", "red")
                else:
                    ans_action_info.append(action_info)
                    ans_action_id.append(action_id)
            elif act_name == "back":
                current_page_name = current_page_name.rsplit('_', 1)[0]
                ans_action_id.append(-1)
                ans_action_info.append('back')
            else:
                ans_action_id.append(-3)
                ans_action_info.append('no such action error')
            ans_history_pages.append(current_page_name)
        else:
            ans_history_pages.append(current_page_name)
            ans_action_info.append(str(rsp))
            ans_action_id.append(-3)
            print_with_color('error', "red")

        if current_page_name == final_page_name:
            task_complete = True
            break

    result_path = os.path.join(args.save_path, args.task_type + args.model_name)
    os.makedirs(result_path, exist_ok=True)
    with open(os.path.join(result_path, f'{final_page_name}_{i}.txt'), "w", encoding="utf-8") as f:
        for action_id, action_info, history_page in zip(ans_action_id, ans_action_info, ans_history_pages):
            f.write(f"{action_id}: {action_info}:{history_page}\n")
        if task_complete:
            f.write("Task completed successfully")
        elif round_count >= MAX_ROUNDS:
            f.write("Task finished due to reaching max rounds")
        else:
            f.write("Task failed")

    if task_complete:
        print_with_color("Task completed successfully", "yellow")
    elif round_count >= MAX_ROUNDS:
        print_with_color("Task finished due to reaching max rounds", "yellow")
    else:
        print_with_color("Task finished unexpectedly", "red")
