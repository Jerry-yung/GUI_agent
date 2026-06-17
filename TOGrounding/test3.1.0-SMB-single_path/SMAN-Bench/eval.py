import os
import json
import re
import numpy as np
import pandas as pd
import math
from utils import *

def calculate_direction(point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    dx = x2 - x1
    dy = y2 - y1

    directions = {
        'up': (0, 1),
        'down': (0, -1),
        'left': (-1, 0),
        'right': (1, 0)
    }

    def angle_between_vectors(v1, v2):
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        magnitude_v1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        magnitude_v2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
        cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
        cos_theta = max(-1, min(1, cos_theta))
        return math.acos(cos_theta)

    # 计算与四个基本方向的夹角，并找出最小夹角对应的方向
    min_angle = float('inf')
    best_direction = None

    for direction, unit_vector in directions.items():
        angle = angle_between_vectors((dx, dy), unit_vector)
        if angle < min_angle:
            min_angle = angle
            best_direction = direction

    return best_direction

def noise_data_test(all_ans_path, gt_path):
    # answer file name
    ans_names = os.listdir(all_ans_path)
    gt_action_sum = 0
    all_true_action_sum = 0
    success_num = 0

    all_true_type_sum = 0
    for ans_name in ans_names:
        ans_path = os.path.join(all_ans_path, ans_name)
        ans_number = ans_name.split('.')[0]
        gt_action_path = os.path.join(gt_path, ans_number + '.json')
        with open(ans_path, 'r') as f:
            pred_file = f.read()
        pred_path = pred_file.split('\n')[:-2]
        with open(gt_action_path, 'r', encoding='ISO-8859-1') as f:
            gt_file = json.load(f)
        gt_acts = gt_file['trajectories']
        gt_action_sum += len(pred_path)
        true_action_sum = 0
        for i, action_info in enumerate(pred_path):
            # print(i)
            gt_act_name = gt_acts[i]['action']['action'].lower()
            if gt_act_name == 'back':
                continue
            try:
                gt_act_info = gt_acts[i]['action']['info']
            except:
                print(ans_name, gt_acts[i]['action'])
            if action_info == 'error' or action_info == 'no such act':
                continue
            else:
                pred_act_name = action_info.split('(')[0]
                if pred_act_name == 'click':
                    if gt_act_name == 'click':
                        all_true_type_sum += 0
                        pred_click_area = re.findall(r'\[(\d+),(\d+)\]', action_info)
                        click_area = [[int(x), int(y)] for x, y in pred_click_area]
                        gt_click_coord = gt_act_info['coordinate']
                        if click_area[0][0] < gt_click_coord[0] < click_area[1][0] and click_area[1][0] < \
                                gt_click_coord[1] < click_area[1][1]:
                            true_action_sum += 1
                    else:
                        continue
                elif pred_act_name == 'scroll':
                    if gt_act_name == 'swpie':
                        pred_scroll_area = re.findall(r'\[(\d+),(\d+)\]', action_info)
                        all_true_type_sum += 0
                        scroll_area = [[int(x), int(y)] for x, y in pred_scroll_area]
                        gt_scroll_coord = gt_act_info['coordinate']
                        if scroll_area[0][0] < gt_scroll_coord[0][0] < scroll_area[1][0] and scroll_area[1][0] < \
                                gt_scroll_coord[0][1] < scroll_area[1][1] and scroll_area[0][0] < gt_scroll_coord[1][
                            0] < scroll_area[1][0] and scroll_area[1][0] < gt_scroll_coord[1][1] < scroll_area[1][1]:
                            pred_scroll_direction = action_info.split(']')[-1].split(')')[0]
                            gt_scroll_direction = calculate_direction(gt_scroll_coord[0], gt_scroll_coord[1])
                            if pred_scroll_direction == gt_scroll_direction:
                                true_action_sum += 1
                    else:
                        continue
                else:
                    if gt_act_name == 'text':
                        all_true_type_sum += 0
                        pred_text = action_info.split('(')[1].split(')')[0]
                        f1 = comput_f1(pred_text, gt_act_info['text'])
                        if f1 > 0.5:
                            true_action_sum += 1
                    else:
                        continue
        if true_action_sum == len(pred_path):
            success_num += 1
        all_true_action_sum += true_action_sum
    
    print(len(ans_names))
    print('success_rate', success_num / len(ans_names))
    print('action_acc', all_true_action_sum / gt_action_sum)
    print('type_acc', all_true_type_sum / gt_action_sum)
    
def multi_path_test(file_path, task_type):
    files = os.listdir(file_path)
    print('task_num', len(files))
    success_sum = 0
    gt_action_sum = 0
    all_true_action_sum = 0
    pred_action_sum = 0
    page_convert_num = 0
    for file in files:
        file_name = file.split('.')[0]
        real_path = file_name.split('_')[1:-1]

        if task_type == 'complex':
            pred_paths = []
            with open(os.path.join(file_path, file), 'r') as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                pred_paths.append(line)
                if i == 24 or line.split(':')[-1].strip() == file_name.rsplit('_', 1)[0].strip():
                    break

            pred_path = pred_paths[-1].split(':')[-1].strip().split('_')[1:]
            pred_action_sum += len(pred_paths)

        else:
            pred_paths = []
            with open(os.path.join(file_path, file), 'r') as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                pred_paths.append(line)
                
                if i == 19 or line.split(':')[-1].strip() == file_name.rsplit('_', 1)[0].strip():
                    # if line.split(':')[-1].strip() == file_name.rsplit('_', 1)[0].strip():
                        # print(line.split(':')[-1].strip(), file_name.rsplit('_', 1)[0].strip())
                    break
            pred_path = pred_paths[-1].split(':')[-1].strip().split('_')[1:]
            pred_action_sum += len(pred_paths)
        # print(pred_path)

        gt_action_sum += len(real_path)
        true_action_sum = 0
        page_convert_num += len(pred_path)
        if len(real_path) >= len(pred_path):

            for i, path in enumerate(pred_path):
                if path == real_path[i]:
                    true_action_sum += 1
            if true_action_sum == len(real_path):
                success_sum += 1
        else:
            for i, path in enumerate(real_path):
                if path == pred_path[i]:
                    true_action_sum += 1
        all_true_action_sum += true_action_sum
    print('task_num', len(files))
    print('success_rate', success_sum / len(files))
    print('average_action_num', pred_action_sum / len(files))
    print(gt_action_sum, page_convert_num)
    print('action_acc', all_true_action_sum / gt_action_sum, all_true_action_sum / page_convert_num)


def _action_type(action_str):
    return action_str.split('(')[0] if action_str else ''


def single_path_test(file_path, source_data_dir):
    files = os.listdir(file_path)

    success_sum = 0
    gt_action_sum = 0
    all_true_action_sum = 0
    no_finish_sum = 0
    type_true_action_sum = 0

    no_finish_list = []
    for file in files:
        if not file.endswith('.txt'):
            continue

        file_name = file.split('.')[0]
        task_page_name = file_name.rsplit('_', 1)[0]

        data_directory = find_graph_dir_for_task(source_data_dir, task_page_name)
        if not data_directory:
            print(f"Skip {file}: no graph dir for {task_page_name}")
            continue

        graph_dir = os.path.join(source_data_dir, data_directory)
        try:
            all_action_ids = load_all_action_ids(graph_dir)
            action_str_to_id, id_to_action = build_action_id_maps(all_action_ids)
            gt_ids = gt_action_ids_from_task_page(task_page_name, graph_dir, action_str_to_id)
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            print(f"Skip {file}: GT load failed ({e})")
            continue

        with open(os.path.join(file_path, file), 'r', encoding='utf-8') as f:
            pred_lines = [ln.strip() for ln in f.read().splitlines() if ln.strip()]

        gt_action_sum += len(gt_ids)
        if len(pred_lines) < len(gt_ids):
            no_finish_list.append(file_name)
            no_finish_sum += 1
            continue
        if len(pred_lines) > len(gt_ids):
            pred_lines = pred_lines[:len(gt_ids)]

        true_action_sum = 0
        for i, gt_id in enumerate(gt_ids):
            pred_action_num = pred_lines[i].split(':')[0].strip()
            try:
                pred_id = int(pred_action_num)
            except ValueError:
                continue

            if pred_id >= 0 and pred_id in id_to_action and gt_id in id_to_action:
                if _action_type(id_to_action[pred_id]) == _action_type(id_to_action[gt_id]):
                    type_true_action_sum += 1
            elif pred_id < 0:
                pred_action_name = pred_lines[i].split(':')[1].split(' ')[1] if ':' in pred_lines[i] else ''
                real_action = id_to_action.get(gt_id, '')
                if 'scroll' in pred_action_name and _action_type(real_action) == 'scroll':
                    type_true_action_sum += 1
                if 'click' in pred_action_name and _action_type(real_action) == 'click':
                    type_true_action_sum += 1
                if 'input' in pred_action_name and _action_type(real_action) == 'input':
                    type_true_action_sum += 1

            if pred_action_num == str(gt_id):
                true_action_sum += 1

        all_true_action_sum += true_action_sum

        if true_action_sum == len(gt_ids):
            success_sum += 1
    print('task_num', success_sum, len(files))
    print('success_rate', success_sum / len(files))
    print('action_acc', all_true_action_sum / gt_action_sum)
    print('type_acc', type_true_action_sum / gt_action_sum)
    print('gt_action_num', all_true_action_sum, type_true_action_sum, gt_action_sum)
    print('no_finish', no_finish_list)
