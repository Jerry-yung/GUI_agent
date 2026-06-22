import json
import os
import time
import re
import requests
import copy
import concurrent.futures
import argparse
import yaml

from PIL import Image
from prompt import get_action_prompt, get_reflect_prompt, get_memory_prompt, get_process_prompt
from chat import init_action_chat, add_response, init_reflect_chat, add_response_two_image, init_memory_chat
from api import inference_chat, inference_chat_4v, get_model_response, inference_openai, get_qwenmax_response, get_model_response_qwen, get_InternVL_images_response, get_InternVL_response, get_InternVL_text_response
from ..utils import *



def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='data directory')
    parser.add_argument('--task_file', type=str, help='task_file', default=r'simple_tasks_sample.json')
    parser.add_argument('--model_name', type=str, help='model name', default=r'gpt-4v')
    parser.add_argument('--config_path', type=str, help='config path')
    parser.add_argument('--task_type', type=str, help='task type', default=r'multi_complex')
    parser.add_argument('--start_num', type=int, help='task start number', default=0)
    parser.add_argument('--model_type', type=str, help='model type', default='Qwen')
    parser.add_argument('--max_round', type=int, help='max round', default=20)
    parser.add_argument('--save_path', type=str, help='save path')
    return parser.parse_args()

args = get_parse()
task_path = os.path.join(args.data_dir, args.task_file)

reflection_switch = True
all_time = 0
# Memory Setting: If you want to improve the operating speed, you can disable the memory unit. This may reduce the success rate.
memory_switch = True
error_flag = False
add_info = "If you want to output the action, output it in the given numbering scheme."
MAX_ROUNDS = args.max_round

with open(args.config_path, 'r') as f:
    config = yaml.safe_load(f)

if args.model_type == 'Qwen':
    API_url = config[args.model_name]['QWEN_API_BASE']
elif args.model_type == 'OpenAI':
    API_url = config[args.model_name]['OPENAI_API_BASE']
    token = config[args.model_name]['OPENAI_API_KEY']
    
def get_all_files_in_folder(folder_path):
    file_list = []
    for file_name in os.listdir(folder_path):
        file_list.append(file_name)
    return file_list

def get_perception_infos(screenshot_file, click_actions, scroll_action_bounds, input_actions, current_page_all_actions):
    width, height = Image.open(screenshot_file).size
    coordinates = []
    texts = []
    perception_infos = []

    for click_action in click_actions:
        pattern = r'\[(\d+),(\d+)\]\[(\d+),(\d+)\]'
        match = re.search(pattern, click_action)
        if match:
            min_x, min_y, max_x, max_y = map(int, match.groups())
            coordinates.append([min_x, min_y, max_x, max_y])
        text = click_action.split("(")[1].split(',')[0]
        texts.append(text)

    for input_action in input_actions:
        pattern = r'\[(\d+),(\d+)\]\[(\d+),(\d+)\]'
        match = re.search(pattern, input_action)
        if match:
            min_x, min_y, max_x, max_y = map(int, match.groups())
            coordinates.append([min_x, min_y, max_x, max_y])
        input_action = input_action.split("(")[1].split(')')[0]
        input_action_info = current_page_all_actions[input_action]
        text_match = re.search(r'>\s*([^<>]+)\s*<', input_action_info)
        if text_match:  
            text = text_match.group(1)    
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
        perception_infos[i]['coordinates'] = [int((perception_infos[i]['coordinates'][0]+perception_infos[i]['coordinates'][2])/2), int((perception_infos[i]['coordinates'][1]+perception_infos[i]['coordinates'][3])/2)]
        
    return perception_infos, width, height

with open(task_path, "r") as f:
    tasks = json.load(f)
data_dir = args.data_dir
for i, task in enumerate(tasks[args.start_num:], start=args.start_num):
    final_page_name = task['name']
    task_info = task['task']

    instruction = task_info.split('\n')[-1]

    multi_task_desc = []
    steps = re.findall(r'^\d+\.\s(.*)', task_info, re.MULTILINE)
    for step in steps:
        multi_task_desc.append(step)

    init_data_path = find_dir_with_prefix(data_dir, final_page_name.split('0')[0]+'_')
    task_dir = os.path.join(data_dir, init_data_path)

    if not os.path.exists(task_dir):
        continue
    round_count = 0
    last_act = "None"
    gt_act = "None"
    task_complete = False

    all_action_id_file = os.path.join(task_dir, 'all_action_id.json')
    with open(all_action_id_file, "r", encoding='UTF-8') as fp:
        all_action_ids = json.load(fp)
    all_action_ids = json.loads(all_action_ids)

    all_actions_list = []
    for k, v in all_action_ids.items():
        action = {'action_info': k, 'action_id': v}
        all_actions_list.append(action)

    with open(os.path.join(task_dir, 'all_page_actions.json')) as fp:
        all_page_actions = json.load(fp)
    all_page_actions_data = all_page_actions['data']

    current_page_actions = {}
    with open(os.path.join(task_dir, 'all_page_actions.json')) as fp:
        all_page_actions = json.load(fp)
    for current_page_data in all_page_actions['data']:
        current_page_actions[current_page_data['name']] = current_page_data['action_valid']

    with open(os.path.join(task_dir, 'all_page_id.json')) as fp:
        all_page_ids = json.load(fp)
    all_page_ids = json.loads(all_page_ids)

    with open(os.path.join(task_dir, 'all_triple.json')) as fp:
        all_page_triples = json.load(fp)['data']
    all_page_convert = {}
    for all_page_triple in all_page_triples:
        all_page_convert[all_page_triple[0]+'act'+str(all_page_triple[1])] =all_page_triple[2]

    current_page_name = final_page_name.split('_')[0]
    
    action_history = []
    page_history = []

    ans_action_id = []
    ans_action_info = []
    ans_page_name = []
    reflect_action = []
    reflect_page = []
    #
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
    
    while round_count < MAX_ROUNDS:
        round_count += 1
        print_with_color(f'Round {round_count}', 'yellow')
        
        screenshot_path = os.path.join(task_dir, current_page_name, current_page_name + '-screen.png')
        html_file = os.path.join(task_dir, current_page_name, current_page_name + '-html.txt')
        xml_file = os.path.join(task_dir, current_page_name, current_page_name + '-xml.txt')
        with open(html_file, 'r', encoding='utf-8') as f:  # html
            html_content = f.read()
        with open(xml_file, 'r', encoding='utf-8') as f:  # xml
            xml_content = f.read()
                
        current_page_all_action_ids = current_page_actions[current_page_name]
        current_action_infos = []    
        for id in current_page_all_action_ids:
            current_action_infos.append(all_actions_list[int(id)]['action_info'])

        click_actions, input_actions_pre, scroll_actions, current_page_all_actions = actions_generate(html_content, xml_content)
        click_actions, input_actions, scroll_actions, scroll_action_bounds = current_actions_generate(current_action_infos, click_actions, input_actions_pre, scroll_actions, current_page_all_actions)

        scroll_action_bounds = get_scroll_bounds(scroll_actions)
        drawn_screenshot = os.path.join(task_dir, current_page_name, current_page_name + '-labeled.png')
        draw_bbox_multi(screenshot_path, drawn_screenshot, click_actions)
        draw_bbox_multi(drawn_screenshot, drawn_screenshot, scroll_action_bounds)

        if round_count == 1:
            perception_infos, width, height = get_perception_infos(screenshot_path, click_actions, scroll_action_bounds, input_actions_pre, current_page_all_actions)

        task_count = current_page_name.count('_')
        if task_count < len(multi_task_desc) - 1:
            current_instruction = multi_task_desc[task_count]
        else:
            current_instruction = multi_task_desc[len(multi_task_desc)-1]

        prompt_action = get_action_prompt(instruction, perception_infos, width, height, summary_history, action_history, summary, action, add_info, error_flag, completed_requirements, memory, current_instruction)
        chat_action = init_action_chat()
        chat_action = add_response("user", prompt_action, chat_action, drawn_screenshot)
        if args.model_name == 'qwen272B':
            output_action = get_model_response_qwen(chat_action)
        elif args.model_name == 'gpt-4o' or args.model_name == 'gpt-4-vision-preview':
            output_action = inference_openai(chat_action, args.model_name, API_url, token)
        elif args.model_name == 'qwen-vl-max':
            output_action = get_qwenmax_response(chat_action, args.model_name, API_url)
        elif args.model_name == 'InternVL2':
            output_action = get_InternVL_response(chat_action)
        else:
            output_action = inference_chat_4v(chat_action, args.model_name, API_url, token)
        thought = output_action.split("### Thought ###")[-1].split("### Action ###")[0].replace("\n", " ").replace(":", "").replace("  ", " ").strip()
        summary = output_action.split("### Operation ###")[-1].replace("\n", " ").replace("  ", " ").strip()
        action = output_action.split("### Action ###")[-1].split("### Operation ###")[0].replace("\n", " ").replace("  ", " ").strip()
        chat_action = add_response("assistant", output_action, chat_action)

        status = "#" * 50 + " Decision " + "#" * 50
        print(status)
        print(output_action)
        print('#' * len(status))

        if memory_switch:
            prompt_memory = get_memory_prompt(insight)

            chat_action = add_response("user", prompt_memory, chat_action)
            if args.model_name == 'qwen272B':
                output_memory = get_model_response_qwen(chat_action)
            elif args.model_name == 'gpt-4o' or args.model_name == 'gpt-4-vision-preview':
                output_memory = inference_openai(chat_action, args.model_name, API_url, token)
            elif args.model_name == 'qwen-vl-max':
                output_memory = get_qwenmax_response(chat_action, args.model_name, API_url)
            elif args.model_name == 'InternVL2':
                output_memory = get_InternVL_response(chat_action)    
            else:
                output_memory = inference_chat_4v(chat_action, args.model_name, API_url, token)
            chat_action = add_response("assistant", output_memory, chat_action)
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
                new_page_name, action_info, action_id = action_click(click_actions, area_idx, current_page_all_actions, all_action_ids, current_page_name, all_page_convert)
            except:
                ans_action_info.append("no such action")
                ans_action_id.append(-3)
                ans_page_name.append(current_page_name) 
                print_with_color("ERROR: no such action", "red")
            if action_info == "ERROR":
                ans_action_info.append(action + "click error")
                ans_action_id.append(action_id)
                ans_page_name.append(current_page_name) 
                print_with_color("ERROR: click execution failed", "red")
            else:
                ans_action_info.append(action_info)
                ans_action_id.append(action_id)
                ans_page_name.append(new_page_name)
        
        elif "scroll" in action:
            try:
                area_idx = int(re.findall(r'(\d+)', action)[0])
                direction = str(action.split('\"')[1])
                new_page_name, action_info, action_id = action_scroll(scroll_action_bounds, area_idx, direction, all_action_ids, current_page_name, all_page_convert)
            except:
                ans_action_info.append("no such action")
                ans_action_id.append(-3)
                ans_page_name.append(current_page_name)  
                print_with_color("ERROR: no such action", "red")
            if action_info == "ERROR":
                ans_action_info.append(action + "scroll error")
                ans_action_id.append(action_id)  
                ans_page_name.append(current_page_name) 
                print_with_color("ERROR: scroll execution failed", "red")
            else:
                ans_action_info.append(action_info)
                ans_action_id.append(action_id)
                ans_page_name.append(new_page_name)
            
        elif "input" in action:
            text = action.split('(')[1].split(')')[0]

            new_page_name, action_info, action_id = action_input(text, input_actions, current_page_all_actions, all_action_ids, current_page_name, all_page_convert)

            if action_info == "ERROR":
                ans_action_info.append('input' + text + "error")
                ans_action_id.append(action_id)
                ans_page_name.append(current_page_name) 
                print_with_color("ERROR: input execution failed", "red")
            else:
                ans_action_info.append(action_info)
                ans_action_id.append(action_id)
                ans_page_name.append(new_page_name)
        else:
            action_info = 'no such action'
            action_id = -3
            ans_action_info.append(action_info)
            ans_action_id.append(action_id)  
            ans_page_name.append(current_page_name) 

    
        last_perception_infos = copy.deepcopy(perception_infos)
        last_page_name = current_page_name
        last_screenshot_file = os.path.join(task_dir, last_page_name, last_page_name + '-labeled.png')
        current_page_name = new_page_name
        screenshot_path = os.path.join(task_dir, current_page_name, current_page_name + '-screen.png')
        html_file = os.path.join(task_dir, current_page_name, current_page_name + '-html.txt')
        xml_file = os.path.join(task_dir, current_page_name, current_page_name + '-xml.txt')
        with open(html_file, 'r', encoding='utf-8') as f:  # html
            html_content = f.read()
        with open(xml_file, 'r', encoding='utf-8') as f:  # xml
            xml_content = f.read()
                
        current_page_all_action_ids = current_page_actions[current_page_name]
        current_action_infos = []    
        for id in current_page_all_action_ids:
            current_action_infos.append(all_actions_list[int(id)]['action_info'])

        click_actions, input_actions_pre, scroll_actions, current_page_all_actions = actions_generate(html_content, xml_content)
        click_actions, input_actions, scroll_actions, scroll_action_bounds = current_actions_generate(current_action_infos, click_actions, input_actions_pre, scroll_actions, current_page_all_actions)

        scroll_action_bounds = get_scroll_bounds(scroll_actions)


        drawn_screenshot = os.path.join(task_dir, current_page_name, current_page_name + '-labeled.png')
        perception_infos, width, height = get_perception_infos(screenshot_path, click_actions, scroll_action_bounds, input_actions_pre, current_page_all_actions)
        draw_bbox_multi(screenshot_path, drawn_screenshot, click_actions)
        draw_bbox_multi(drawn_screenshot, drawn_screenshot, scroll_action_bounds)

        task_count = current_page_name.count('_')
        if task_count < len(multi_task_desc) - 1:
            current_instruction = multi_task_desc[task_count]
        else:
            current_instruction = multi_task_desc[len(multi_task_desc)-1]
        if reflection_switch:
            prompt_reflect = get_reflect_prompt(instruction, last_perception_infos, perception_infos, width, height, summary, action, add_info, current_instruction)
            chat_reflect = init_reflect_chat()
            chat_reflect = add_response_two_image("user", prompt_reflect, chat_reflect, [last_screenshot_file, drawn_screenshot])
            if args.model_name == 'qwen272B':
                output_reflect = get_model_response_qwen(chat_reflect)
            elif args.model_name == 'gpt-4o' or args.model_name == 'gpt-4-vision-preview':
                output_reflect = inference_openai(chat_reflect, args.model_name, API_url, token)
            elif args.model_name == 'InternVL2':
                output_reflect = get_InternVL_images_response(chat_reflect)
            elif args.model_name == 'qwen-vl-max':
                output_reflect = get_qwenmax_response(chat_reflect, args.model_name, API_url)
            else:
                output_reflect = inference_chat_4v(chat_reflect, args.model_name, API_url, token)
            reflect = output_reflect.split("### Answer ###")[-1].replace("\n", " ").strip()
            chat_reflect = add_response("assistant", output_reflect, chat_reflect)
            status = "#" * 50 + " Reflcetion " + "#" * 50
            print(status)
            print(output_reflect)
            print('#' * len(status))

            if 'A' in reflect:
                thought_history.append(thought)
                summary_history.append(summary)
                action_history.append(action)
                
                prompt_planning = get_process_prompt(instruction, thought_history, summary_history, action_history, completed_requirements, add_info, current_instruction)
                # with open(os.path.join(data_dir, 'error_info', f'prompt_planning{round_count}'), 'w') as f:
                #     f.write(prompt_planning)
                chat_planning = init_memory_chat()
                chat_planning = add_response("user", prompt_planning, chat_planning)
                if args.model_name == 'qwen272B':
                    output_planning = get_model_response_qwen(chat_planning)
                elif args.model_name == 'gpt-4o' or args.model_name == 'gpt-4-vision-preview':
                    output_planning = inference_openai(chat_planning, args.model_name, API_url, token)
                elif args.model_name == 'qwen-vl-max':
                    output_planning = get_qwenmax_response(chat_planning, args.model_name, API_url)
                elif args.model_name == 'InternVL2':
                    output_planning = get_InternVL_text_response(chat_planning)
                else:
                    output_planning = inference_chat_4v(chat_planning, args.model_name, API_url, token)
                
                chat_planning = add_response("assistant", output_planning, chat_planning)
                status = "#" * 50 + " Planning " + "#" * 50
                
                print(status)
                print(output_planning)
                print('#' * len(status))
                completed_requirements = output_planning.split("### Completed contents ###")[-1].replace("\n", " ").strip()
                
                error_flag = False
                reflect_action.append('A')
                reflect_page.append(current_page_name)
            elif 'B' in reflect:

                error_flag = True
                current_page_name = current_page_name.rsplit('_',1)[0]

                reflect_action.append('B')
                reflect_page.append(current_page_name)
            # elif 'C' in reflect:
            #     error_flag = True
        else:
            if round_count >= len(multi_task_desc):
                current_instruction = multi_task_desc[-1]
            else:
                current_instruction = multi_task_desc[round_count]
            thought_history.append(thought)
            summary_history.append(summary)
            action_history.append(action)
            
            prompt_planning = get_process_prompt(instruction, thought_history, summary_history, action_history, completed_requirements, add_info, current_instruction)
            chat_planning = init_memory_chat()
            chat_planning = add_response("user", prompt_planning, chat_planning)
            print(prompt_planning)
            if args.model_name == 'qwen272B':
                output_planning = get_model_response_qwen(chat_planning)
            elif args.model_name == 'gpt-4o' or args.model_name == 'gpt-4-vision-preview':
                output_planning = inference_openai(chat_planning, args.model_name, API_url, token)
            elif args.model_name == 'qwen-vl-max':
                output_planning = get_qwenmax_response(chat_planning, args.model_name, API_url)
            elif output_planning == 'InternVL2':
                output_planning = get_InternVL_text_response(chat_planning)
            else:
                output_planning = inference_chat_4v(chat_planning, args.model_name, API_url, token)
            chat_planning = add_response("assistant", output_planning, chat_planning)
            status = "#" * 50 + " Planning " + "#" * 50
            print(status)
            print(output_planning)
            print('#' * len(status))
            completed_requirements = output_planning.split("### Completed contents ###")[-1].replace("\n", " ").strip()
            reflect_action.append('null')
            reflect_page.append(current_page_name)
        if current_page_name == final_page_name:
            break

    result_path = os.path.join(args.save_path, args.task_type + args.model_name)
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    
    print('task completed')
    action_and_page = zip(ans_action_id, ans_action_info, ans_page_name)
    with open(os.path.join(result_path, f'{final_page_name}_{i}.txt'), "w", encoding="utf-8") as f:
        for action_id, action_info, page_name in action_and_page:
            f.write(f"{action_id}: {action_info}: {page_name}\n")