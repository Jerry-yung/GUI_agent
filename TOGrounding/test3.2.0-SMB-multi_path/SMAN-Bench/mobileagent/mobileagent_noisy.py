import json
import os
import time
import re
import requests
import copy
import argparse
import concurrent.futures
import yaml

from PIL import Image
from prompt import get_action_prompt, get_reflect_prompt, get_memory_prompt, get_process_prompt
from chat import init_action_chat, add_response, init_reflect_chat, add_response_two_image, init_memory_chat
from api import inference_chat, inference_chat_4v, get_model_response, get_qwenmax_response, inference_openai, get_llama_model_response, get_model_response_qwen
from ..utils import *


add_info = "If you want to output the action, output it in the given numbering scheme."

reflection_switch = False
# Memory Setting: If you want to improve the operating speed, you can disable the memory unit. This may reduce the success rate.
memory_switch = True
error_flag = False
all_time = 0


def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='data directory')
    parser.add_argument('--model_name', type=str, help='model name', default=r'gpt-4v')
    parser.add_argument('--config_path', type=str, help='config path')
    parser.add_argument('--start_num', type=int, help='task start number', default=0)
    parser.add_argument('--model_type', type=str, help='model type', default=r'OpenAI')
    parser.add_argument('--save_path', type=str, help='save results path')
    return parser.parse_args()


def get_perception_infos(screenshot_file, click_actions, scroll_action_bounds):
    width, height = Image.open(screenshot_file).size
    coordinates = []
    texts = []
    perception_infos = []

    # 对出现的文字进行记录
    for click_action in click_actions:
        pattern = r'\[(\d+),(\d+)\]\[(\d+),(\d+)\]'
        match = re.search(pattern, click_action)
        if match:
            min_x, min_y, max_x, max_y = map(int, match.groups())
            coordinates.append([min_x, min_y, max_x, max_y])
        text = click_action.split("(")[1].split(',')[0]
        texts.append(text)

    for scroll_action_bound in scroll_action_bounds:
        pattern = r'\[(\d+),(\d+)\]\[(\d+),(\d+)\]'
        match = re.search(pattern, scroll_action_bound)
        if match:
            min_x, min_y, max_x, max_y = map(int, match.groups())
            coordinates.append([min_x, min_y, max_x, max_y])

    for i in range(len(coordinates)):
        perception_info = {"text": "icon", "coordinates": coordinates[i]}
        perception_infos.append(perception_info)

    for i in range(len(perception_infos)):
        perception_infos[i]['coordinates'] = [
            int((perception_infos[i]['coordinates'][0] + perception_infos[i]['coordinates'][2]) / 2),
            int((perception_infos[i]['coordinates'][1] + perception_infos[i]['coordinates'][3]) / 2)]

    return perception_infos, width, height


args = get_parse()

with open(args.config_path, 'r') as f:
    config = yaml.safe_load(f)

if args.model_type == 'Qwen':
    API_url = config[args.model_name]['QWEN_API_BASE']
elif args.model_type == 'OpenAI':
    API_url = config[args.model_name]['OPENAI_API_BASE']
    token = config[args.model_name]['OPENAI_API_KEY']
elif args.model_type == 'LLAMA':
    API_url = config[args.model_name]['LLAMA_API_BASE']
data_dir = args.data_dir


with open(args.config_path, "r") as f:
    config = yaml.safe_load(f)

task_dir = os.path.join(data_dir, args.data_type)
prefix = 'data'
all_data = os.listdir(task_dir)

tasks_data = []
for data in all_data:
    if data.startswith(prefix):
        tasks_data.append(data) 
        
for i, task in enumerate(tasks_data[args.start_num:], start=args.start_num):

    pattern = r'\d+'
    match = re.findall(pattern, task)
    task_name = match[0] + '-'
    task_data = []
    last_act="None"

    with open(os.path.join(task_dir, task), 'r', encoding='gb18030') as f:
        task_info = json.load(f)
    instruction =  task_info['user_instruction']['query']
    task_gt_path = task_info['trajectories']


    for data in all_data:
        if data.startswith(task_name):
            task_data.append(data)
    number = set(re.search(r'\d+-\d+', name).group() for name in task_data)
    sorted_numbers = sorted(number, key=lambda x: int(x.split('-')[1]))

    action_history = []
    last_act = "None"
    gt_act = "None"
    task_complete = False
    
    model_act_history = []

    thought_history = []
    summary_history = []
    action_history = []
    summary = ""
    action = ""
    completed_requirements = ""
    memory = ""
    insight = ""

    task_time = 0
    before_time = time.time()

    for round_count, number in enumerate(sorted_numbers[:-1]):
        round_count += 1
        print_with_color(f"Round{round_count}", "yellow")
        
        if round_count == 1:
            xml_path = os.path.join(task_dir, number + '.xml')
            screenshot_path = os.path.join(task_dir, number + '.png')

            click_actions, scroll_actions, current_page_actions = noisy_actions_generate(xml_path)

            drawn_screenshot = os.path.join(task_dir, number + f'_labeled.png')
            scroll_action_bounds = get_scroll_bounds(scroll_actions)

            perception_infos, width, height = get_perception_infos(screenshot_path, click_actions, current_page_actions)
            draw_bbox_multi(screenshot_path, drawn_screenshot, click_actions)
            draw_bbox_multi(drawn_screenshot, drawn_screenshot, scroll_action_bounds)

        prompt_action = get_action_prompt(instruction, perception_infos, width, height, summary_history, action_history,
                                          summary, action, add_info, error_flag, completed_requirements, memory,
                                          instruction)
        chat_action = init_action_chat()
        chat_action = add_response("user", prompt_action, chat_action, drawn_screenshot, args.model_name)
        if args.model_name == "qwen272B" or args.model_name == "qwen272B_1":
            output_action = get_model_response_qwen(chat_action)
        elif args.model_name == 'qwen-vl-max':
            output_action = get_qwenmax_response(chat_action, args.model_name, API_url)
        elif args.model_name == "llama90B":
            output_action = get_llama_model_response(chat_action, API_url, args.model_name)
        elif args.model_name == 'gpt-4o':
            output_action = inference_openai(chat_action, args.model_name, API_url, token)
        else:
            output_action = inference_chat(chat_action, args.model_name, API_url, token)
        pattern = r"<\|begin_of_text\|><\|start_header_id\|>user<\|end_header_id\|>(.*?)<\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>"
        output_action = re.sub(pattern, "", output_action, flags=re.DOTALL)
        thought = output_action.split("### Thought ###")[-1].split("### Action ###")[0].replace("\n", " ").replace(":","").replace("  ", " ").strip()
        summary = output_action.split("### Operation ###")[-1].replace("\n", " ").replace("  ", " ").strip()
        action = output_action.split("### Action ###")[-1].split("### Operation ###")[0].replace("\n", " ").replace("  ", " ").strip()
        chat_action = add_response("assistant", output_action, chat_action, model_name=args.model_name)

        status = "#" * 50 + " Decision " + "#" * 50
        print(status)
        print(output_action)
        print('#' * len(status))

        if memory_switch:
            prompt_memory = get_memory_prompt(insight)
            chat_action = add_response("user", prompt_memory, chat_action, model_name=args.model_name)
            if args.model_name == "qwen272B" or args.model_name == "qwen272B_1":
                output_memory = get_model_response_qwen(chat_action)
            elif args.model_name == 'qwen-vl-max':
                output_memory = get_qwenmax_response(chat_action, args.model_name, API_url)
            elif args.model_name == "llama90B" or args.model_name == "llama90B_1":
                output_memory = get_llama_model_response(chat_action, API_url, args.model_name)
            elif args.model_name == 'gpt-4o' or args.model_name == 'gpt-4-vision-preview':
                output_memory = inference_openai(chat_action, args.model_name, API_url, token)
            else:
                output_memory = inference_chat(chat_action, args.model_name, API_url, token)
            pattern = r"<\|begin_of_text\|><\|start_header_id\|>user<\|end_header_id\|>(.*?)<\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>"
            output_memory = re.sub(pattern, "", output_memory, flags=re.DOTALL)
            chat_action = add_response("assistant", output_memory, chat_action, model_name=args.model_name)
            status = "#" * 50 + " Memory " + "#" * 50

            print(status)
            print(output_memory)
            print('#' * len(status))
            output_memory = output_memory.split("### Important content ###")[-1].split("\n\n")[0].strip() + "\n"
            if "None" not in output_memory and output_memory not in memory:
                memory += output_memory

        if "click" in action:

            try:
                area_idx = int(re.findall(r'(\d+)', action)[0])
                action_bounds = click_actions[area_idx - 1]
                model_act_history.append('click('+str(action_bounds)+')')
            except:
                print(len(click_actions), area_idx)
                print('click_error')
                model_act_history.append("click error")
        elif 'scroll' in action:
            try:
                area_idx = int(re.findall(r'(\d+)', action)[0])   
                match = re.search(r'"(.*?)"', action)
                if match:
                    direction = match.group(1)
                scroll_action = str(scroll_action_bounds[area_idx - 1])
                scroll_action = "scroll(" + scroll_action + direction + ")"
                model_act_history.append(scroll_action)
            except:
                print('scroll_error')
                model_act_history.append("scroll error")
        elif 'input' in action:
            text = action.split('(')[1].split(')')[0]
            model_act_history.append("input(" + text + ")")
        else:
            model_act_history.append("no such act")

        xml_path = os.path.join(task_dir, sorted_numbers[round_count] + '.xml')
        screenshot_path = os.path.join(task_dir, sorted_numbers[round_count] + '.png')
        click_actions, scroll_actions, current_page_actions = noisy_actions_generate(xml_path)

        drawn_screenshot = os.path.join(task_dir, sorted_numbers[round_count] + f'_labeled.png')
        scroll_action_bounds = get_scroll_bounds(scroll_actions)

        perception_infos, width, height = get_perception_infos(screenshot_path, click_actions, scroll_action_bounds)

        draw_bbox_multi(screenshot_path, drawn_screenshot, click_actions)
        draw_bbox_multi(drawn_screenshot, drawn_screenshot, scroll_action_bounds)

        
        if reflection_switch:
            prompt_reflect = get_reflect_prompt(instruction, last_perception_infos, perception_infos, width, height,
                                                summary, action, add_info)
            chat_reflect = init_reflect_chat()
            chat_reflect = add_response_two_image("user", prompt_reflect, chat_reflect,
                                                  [last_screenshot_file, drawn_screenshot])
            if args.model_name == "qwen272B":
                output_reflect = get_model_response_qwen(chat_reflect)
            elif args.model_name == 'qwen-vl-max':
                output_reflect = get_qwenmax_response(chat_reflect, args.model_name, API_url)
            elif args.model_name == "llama90B":
                output_memory = get_llama_model_response(chat_reflect, API_url, args.model_name)
            elif args.model_name == 'gpt-4o':
                output_reflect = inference_openai(chat_reflect, args.model_name, API_url, token)
            else:
                output_reflect = inference_chat(chat_reflect, args.model_name, API_url, token)
            pattern = r"<\|begin_of_text\|><\|start_header_id\|>user<\|end_header_id\|>(.*?)<\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>"
            output_reflect = re.sub(pattern, "", output_reflect, flags=re.DOTALL)
            reflect = output_reflect.split("### Answer ###")[-1].replace("\n", " ").strip()
            chat_reflect = add_response("assistant", output_reflect, chat_reflect, args.model_name)
            status = "#" * 50 + " Reflcetion " + "#" * 50
            print(status)
            print(output_reflect)
            print('#' * len(status))

            if 'A' in reflect:
                thought_history.append(thought)
                summary_history.append(summary)
                action_history.append(action)

                prompt_planning = get_process_prompt(instruction, thought_history, summary_history, action_history,
                                                     completed_requirements, add_info)
                chat_planning = init_memory_chat()
                chat_planning = add_response("user", prompt_planning, chat_planning, args.model_name)
                if args.model_name == "qwen272B" or args.model_name == "qwen272B_1":
                    output_planning = get_model_response_qwen(chat_planning)
                elif args.model_name == 'qwen-vl-max':
                    output_planning = get_qwenmax_response(chat_planning, args.model_name, API_url)
                elif args.model_name == "llama90B" or args.model_name == "llama90B_1":
                    output_planning = get_llama_model_response(chat_planning, API_url, args.model_name)
                elif args.model_name == 'gpt-4o' or args.model_name == 'gpt-4-vision-preview':
                    output_planning = inference_openai(chat_action, args.model_name, API_url, token)
                else:
                    output_planning = inference_chat(chat_planning, args.model_name, API_url, token, args.model_name)
                pattern = r"<\|begin_of_text\|><\|start_header_id\|>user<\|end_header_id\|>(.*?)<\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>"
                output_planning = re.sub(pattern, "", output_planning, flags=re.DOTALL)
                chat_planning = add_response("assistant", output_planning, chat_planning)
                status = "#" * 50 + " Planning " + "#" * 50
                print(status)
                print(output_planning)
                print('#' * len(status))
                completed_requirements = output_planning.split("### Completed contents ###")[-1].replace("\n",
                                                                                                         " ").strip()

                error_flag = False
            elif 'B' in reflect:
                error_flag = True
                current_page_name = current_page_name.rsplit('_', 1)[0]

                reflect_action.append('B')
                reflect_page.append(current_page_name)

        else:
            thought_history.append(thought)
            summary_history.append(summary)
            action_history.append(action)

            prompt_planning = get_process_prompt(instruction, thought_history, summary_history, action_history,
                                                 completed_requirements, add_info, instruction)
            chat_planning = init_memory_chat()
            if args.model_name == "qwen272B" or args.model_name == "qwen272B_1":
                chat_planning = add_response("user", prompt_planning, chat_planning, model_name=args.model_name)
                output_planning = get_model_response_qwen(chat_planning)
            elif args.model_name == 'qwen-vl-max':
                chat_planning = add_response("user", prompt_planning, chat_planning, model_name=args.model_name)
                output_planning = get_qwenmax_response(chat_planning, args.model_name, API_url)
            elif args.model_name == 'gpt-4o' or args.model_name == 'gpt-4-vision-preview':
                chat_planning = add_response("user", prompt_planning, chat_planning, model_name=args.model_name)
                output_planning = inference_openai(chat_action, args.model_name, API_url, token)
            elif args.model_name == "llama90B" or args.model_name == "llama90B_1":
                chat_planning = add_response("user", prompt_planning, chat_planning, drawn_screenshot, model_name=args.model_name)
                output_planning = get_llama_model_response(chat_planning, API_url, args.model_name)
            elif args.model_name == 'gpt-4o-2024-08-06' or args.model_name == 'gpt-4v':
                chat_planning = add_response("user", prompt_planning, chat_planning, model_name=args.model_name)
                output_planning = inference_chat(chat_planning, args.model_name, API_url, token)
            pattern = r"<\|begin_of_text\|><\|start_header_id\|>user<\|end_header_id\|>(.*?)<\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>"
            output_planning = re.sub(pattern, "", output_planning, flags=re.DOTALL)
            chat_planning = add_response("assistant", output_planning, chat_planning, model_name=args.model_name)
            
            status = "#" * 50 + " Planning " + "#" * 50
            print(status)
            print(output_planning)
            print('#' * len(status))
            completed_requirements = output_planning.split("### Completed contents ###")[-1].replace("\n", " ").strip()


    result_path = os.path.join(args.save_path, args.data_type + args.model_name)
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    print('task completed')
    with open(os.path.join(result_path, f'{task}.txt'), "w", encoding="utf-8") as f:
        for action in model_act_history:
            f.write(f"{action}\n")