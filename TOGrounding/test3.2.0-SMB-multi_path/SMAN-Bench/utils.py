import base64
import json
import cv2
import numpy as np
import os
import shutil
import subprocess
import time
import re
from anytree import Node
import xml.etree.ElementTree as ET

from colorama import Fore, Style

from put_btext import put_btext
# configs = load_config()

width = 720
height = 1280


def find_dir_with_prefix(data_dir, prefix):
    """Find first subdirectory under data_dir whose name starts with prefix."""
    for name in os.listdir(data_dir):
        if name.startswith(prefix):
            path = os.path.join(data_dir, name)
            if os.path.isdir(path):
                return name
    return None


def parse_xml_to_anytree(xml_code):
    root = ET.fromstring(xml_code)

    def build_anytree(node, element, child_index, seen_elements, counter):
        element_type = element.tag
        # print(element_type)
        # Generate a unique key for the element based on its attributes
        element_key = (
            element_type,
            element.get('resource-id', ''),
            element.get('content-desc', ''),
            element.get('text', ''),
            element.get('clickable', ''),
            element.get('scrollable', ''),
            element.get('package', ''),  ##
            element.get('class', ''),
            element.get('displayed', ''),
            element.get('bounds', ''),
        )
        seen_elements.add(element_key)

        is_leaf = not bool(list(element))

        has_text = bool(element.get('text'))
        has_content_desc = bool(element.get('content-desc'))

        visible = has_text or has_content_desc or 'button' in element_type.lower() or 'edittext' in element.tag.lower()

        leaf_id = counter[0]  
        counter[0] += 1  

        anytree_node = Node(element_type, parent=node, type=element_type, visible=visible, leaf_id=leaf_id,
                            resource_id=element.get('resource-id'), content_desc=element.get('content-desc'),
                            text=element.get('text'), clickable=element.get('clickable'), is_leaf=is_leaf,
                            scrollable=element.get('scrollable'), package=element.get('package'),
                            class_label=element.get('class'), displayed=element.get('displayed'),
                            bounds=element.get('bounds'))

        for idx, child in enumerate(element):
            # print(idx)
            build_anytree(anytree_node, child, idx, seen_elements, counter)

    is_root_leaf = not bool(list(root))

    anytree_root = Node(root.tag, type=root.tag, visible=True, leaf_id=0,  
                        resource_id=root.get('resource-id'), content_desc=root.get('content-desc'),
                        text=root.get('text'), clickable=root.get('clickable'),
                        is_leaf=is_root_leaf, scrollable=root.get('scrollable'), package=root.get('package'),
                        class_label=root.get('class'), displayed=root.get('displayed'), bounds=root.get('bounds'))

    seen_elements = set()
    counter = [1]  

    for idx, child in enumerate(root):
        # print("out",idx)
        build_anytree(anytree_root, child, idx, seen_elements, counter)

    return anytree_root

def any_tree_to_html(node, layer, clickable_label):
    """Turns an AnyTree representation of view hierarchy into HTML.
    Args:
    node: an AnyTree node.
    layer: which layer is the node in.

    Returns:
    results: output HTML.
    """
    results = ''
    if 'ImageView' in node.type:
        node_type = 'img'
    elif 'IconView' in node.type:
        node_type = 'img'
    elif 'Button' in node.type:
        node_type = 'button'
    elif 'Image' in node.type:
        node_type = 'img'
    elif 'MenuItemView' in node.type:
        node_type = 'button'
    elif 'EditText' in node.type:
        node_type = 'input'
    elif 'TextView' in node.type:
        node_type = 'p'
    else:
        node_type = 'div'

    if node.clickable == "true":
        clickable_label = "true"
    elif clickable_label == "true":
        node.clickable = "true"
    if node.text:
        node.text = node.text.replace('\n', '')
    if node.content_desc:
        node.content_desc = node.content_desc.replace('\n', '')

    #  or node.class_label == 'android.widget.EditText'
    if node.is_leaf and node.visible:
        html_close_tag = node_type
        if node.scrollable == "true":
            html_close_tag = node_type
            results = '<{}{}{}{}{}{}{}{}> {} </{}>\n'.format(
                node_type,
                ' id="{}"'.format(node.resource_id)
                if node.resource_id
                else '',
                ' package="{}"'.format(node.package)
                if node.package
                else '',

                ' class="{}"'.format(''.join(node.class_label))
                if node.class_label
                else '',
                ' description="{}"'.format(node.content_desc) if node.content_desc else '',
                ' clickable="{}"'.format(node.clickable) if node.clickable else '',
                ' scrollable="{}"'.format(node.scrollable) if node.scrollable else '',
                ' bounds="{}"'.format(node.bounds) if node.bounds else '',
                '{}'.format(node.text) if node.text else '',
                html_close_tag,
            )
        else:
            results = '<{}{}{}{}{}{}{}> {} </{}>\n'.format(
                node_type,
                ' id="{}"'.format(node.resource_id)
                if node.resource_id
                else '',
                ' package="{}"'.format(node.package)
                if node.package
                else '',

                ' class="{}"'.format(''.join(node.class_label))
                if node.class_label
                else '',

                ' description="{}"'.format(node.content_desc) if node.content_desc else '',
                ' clickable="{}"'.format(node.clickable) if node.clickable else '',
                ' bounds="{}"'.format(node.bounds) if node.bounds else '',
                '{}'.format(node.text) if node.text else '',
                html_close_tag,
            )

    else:
        if node.scrollable == "true":
            html_close_tag = node_type
            results = '<{}{}{}{}{}{}{}> {} </{}>\n'.format(
                node_type,
                ' id="{}"'.format(node.resource_id)
                if node.resource_id
                else '',

                ' class="{}"'.format(''.join(node.class_label))
                if node.class_label
                else '',

                ' descript  ion="{}"'.format(node.content_desc) if node.content_desc else '',
                ' clickable="{}"'.format(node.clickable) if node.clickable else '',
                ' scrollable="{}"'.format(node.scrollable) if node.scrollable else '',
                ' bounds="{}"'.format(node.bounds) if node.bounds else '',

                '{}'.format(node.text) if node.text else '',
                html_close_tag,
            )
        for child in node.children:
            results += any_tree_to_html(child, layer + 1, clickable_label)

    return results



def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def input_generate(input_actions, current_actions, current_page_all_actions):
    new_input_actions = []
    for input_action in input_actions:
        match = re.search(r'input\((.*)\)', input_action)
        if match:
            input_action = match.group(1)
        current_input = 'input(' + current_page_all_actions[input_action] 
        for current_action in current_actions:
            if current_action.startswith(current_input):
                new_input_actions.append(current_action)
    return new_input_actions

def current_actions_generate(current_actions, click_actions, input_actions, scroll_actions, current_page_all_actions):
    new_click_actions = []
    for click_action in click_actions:
        click_action_info = 'click('+current_page_all_actions[click_action]+')'
        if click_action_info in current_actions:
            new_click_actions.append(click_action)
    
    new_scroll_actions = []
    scroll_bounds = []
    
    for scroll_action in scroll_actions:
        pattern = r'scroll\((\[\d+,\d+\]\[\d+,\d+\]),\s*(\w+)\)'
        match = re.search(pattern, scroll_action)
        if match:
            bounds = match.group(1) 
            direction = match.group(2)  
        scroll_action_info = dict_scroll_parameters(bounds, direction)
        scroll_action_info = 'scroll(' + str(scroll_action_info) + ')'
        
        if scroll_action_info in current_actions:
            new_scroll_actions.append(scroll_action)
            scroll_bounds.append(bounds)
    
    new_input_actions = input_generate(input_actions, current_actions, current_page_all_actions)
    return new_click_actions, new_input_actions, new_scroll_actions, scroll_bounds


def noisy_actions_generate(xml):
    
    icons = []
    bounds = []
    with open(xml, 'r', encoding='utf-8') as f:
        xml_content = f.read()
    anytree_root = parse_xml_to_anytree(xml_content)

    new_html = any_tree_to_html(anytree_root, 0, None)
    
    lines = new_html.strip().split('\n')
    html2action = {}
    for line in lines:
        if 'clickable="true"' in line:
            text_match = re.search(r'>\s*([^<>]+)\s*<', line)
            
            bounds_match = re.search(r'bounds="(\[[0-9]+,[0-9]+\]\[[0-9]+,[0-9]+\])"', line)
            if bounds_match:  
                bounds_text = bounds_match.group(1).strip()
            if text_match:
                text = text_match.group(1).strip()
                if text and text != ' ':
                    icons.append(f"click({text}, {bounds_text})")
                    html2action[f"click({text}, {bounds_text})"] = line
                    continue

        if 'scrollable="true"' in line:
            bounds_match = re.search(r'bounds="(\[[0-9]+,[0-9]+\]\[[0-9]+,[0-9]+\])"', line)
            if bounds_match:  
                bounds_text = bounds_match.group(1).strip()
                scroll_actions = [
                    f"scroll({bounds_text},up)",
                    f"scroll({bounds_text},down)",
                    f"scroll({bounds_text},left)",
                    f"scroll({bounds_text},right)"
                ]
                bounds.extend(scroll_actions)
                for action in scroll_actions:
                    html2action[action] = bounds_text
        
    return icons, bounds, html2action

def actions_generate(html, xml):
    inputs = []
    icons = []
    bounds = []
    anytree_root = parse_xml_to_anytree(xml)
    new_html = any_tree_to_html(anytree_root, 0, None)

    lines1 = html.strip().split('\n')
    lines2 = new_html.strip().split('\n')
    html2action = {}

    for line, line2 in zip(lines1, lines2):
        
        count = 1
        if '<input' in line:
            text_match = re.search(r'>\s*([^<>]+)\s*<', line)
            bounds_match = re.search(r'bounds="(\[[0-9]+,[0-9]+\]\[[0-9]+,[0-9]+\])"', line2)
            if bounds_match:  
                bounds_text = bounds_match.group(1).strip()
            if text_match:
                text = text_match.group(1)
            else:
                text = None
            if text:
                if text != ' ':
                    inputs.append(f"input(带有文本{text}的输入框, {bounds_text})")
                    html2action[f"带有文本{text}的输入框, {bounds_text}"] = line
                    continue
                else:
                    inputs.append(f"input(第{count}个空白输入框, {bounds_text})")
                    html2action[f"第{count}个空白输入框, {bounds_text}"] = line
                    count += 1
                    continue

        
        if 'clickable="true"' in line:
            text_match = re.search(r'>\s*([^<>]+)\s*<', line)
            bounds_match = re.search(r'bounds="(\[[0-9]+,[0-9]+\]\[[0-9]+,[0-9]+\])"', line2)
            if bounds_match: 
                bounds_text = bounds_match.group(1).strip()
            if text_match:
                text = text_match.group(1).strip()
                if text and text != ' ':
                    icons.append(f"click({text}, {bounds_text})")
                    html2action[f"click({text}, {bounds_text})"] = line
                    continue

            alt_match = re.search(r'description="([^"]+)"', line)
            if alt_match:
                alt = alt_match.group(1).strip()
                if alt and alt != ' ':
                    icons.append(f"click({alt}, {bounds_text})")
                    html2action[f"click({alt}, {bounds_text})"] = line
                    continue

        if 'scrollable="true"' in line:
            bounds_match = re.search(r'bounds="(\[[0-9]+,[0-9]+\]\[[0-9]+,[0-9]+\])"', line)
            if bounds_match:  
                bounds_text = bounds_match.group(1).strip()
                scroll_actions = [
                    f"scroll({bounds_text},up)",
                    f"scroll({bounds_text},down)",
                    f"scroll({bounds_text},left)",
                    f"scroll({bounds_text},right)"
                ]
                bounds.extend(scroll_actions)
                for action in scroll_actions:
                    html2action[action] = bounds_text

    return icons, inputs, bounds, html2action


def print_with_color(text: str, color=""):
    if color == "red":
        print(Fore.RED + text)
    elif color == "green":
        print(Fore.GREEN + text)
    elif color == "yellow":
        print(Fore.YELLOW + text)
    elif color == "blue":
        print(Fore.BLUE + text)
    elif color == "magenta":
        print(Fore.MAGENTA + text)
    elif color == "cyan":
        print(Fore.CYAN + text)
    elif color == "white":
        print(Fore.WHITE + text)
    elif color == "black":
        print(Fore.BLACK + text)
    else:
        print(text)
    print(Style.RESET_ALL)


def draw_bbox_multi(image_path, output_path, action_list, record_mode=False, dark_mode=False):
    imgcv = cv2.imread(image_path)
    count = 1
    for action in action_list:
        pattern = r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]"
        match = re.search(pattern, action)
        if match:
            # 提取匹配到的坐标
            left, bottom, right, top = map(int, match.groups())
        try:
            if action.split('(')[0] == 'click':
                label = "c" + str(count)
            else:
                label = "s" + str(count)

            text_color = (10, 10, 10) if dark_mode else (255, 250, 250)
            bg_color = (255, 250, 250) if dark_mode else (10, 10, 10)
            imgcv = put_btext(imgcv, label, text_offset_x=(left + right) // 2 + 10,
                                text_offset_y=(top + bottom) // 2 + 10,
                                vspace=10, hspace=10, font_scale=1, thickness=2, background_RGB=bg_color,
                                text_RGB=text_color, alpha=0.5)
        except Exception as e:
            print_with_color(f"ERROR: An exception occurs while labeling the image\n{e}", "red")
        count += 1
        # imgcv = cv2.resize(imgcv, (1080, 2400))
    # imgcv = cv2.resize(imgcv, (1080, 2400))
    cv2.imwrite(output_path, imgcv)
        #  cv2.imencode('.png', imgcv)[1].tofile(output_path)
    return imgcv


def dict_scroll_parameters(bounds, direction):
    match = re.search(r'\[([0-9]+,[0-9]+)\]\[([0-9]+,[0-9]+)\]', bounds)
    bounds = [match.group(1), match.group(2)]
    x1, y1 = map(int, bounds[0].split(','))
    x2, y2 = map(int, bounds[1].split(','))

    
    mid_x = (x1 + x2) // 2
    mid_y = (y1 + y2) // 2

    
    offset_x = (x2 - x1) // 4
    offset_y = (y2 - y1) // 4

    
    scroll_directions = {
        'up': ([mid_x,mid_y + offset_y], [mid_x,mid_y - offset_y]),
        'down': ([mid_x,mid_y - offset_y], [mid_x,mid_y + offset_y]),
        'left': ([mid_x + offset_x,mid_y], [mid_x - offset_x,mid_y]),
        'right': ([mid_x - offset_x,mid_y], [mid_x + offset_x,mid_y])
    }

    return scroll_directions[direction]

def comput_f1(predicted_answer, true_answer):
    if "无人应答" in true_answer:
        return 1
    
    predicted_chars = list(predicted_answer)
    true_chars = list(true_answer)

    
    common = set(predicted_chars) & set(true_chars)
    num_common = sum(min(predicted_chars.count(c), true_chars.count(c)) for c in common)

    
    precision = num_common / len(predicted_chars) if len(predicted_chars) > 0 else 0
    recall = num_common / len(true_chars) if len(true_chars) > 0 else 0

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return f1



def action_click(click_action, area_idx, actions_list, all_action_ids, current_page_name, all_page_convert):
    
    try:
        action_info = actions_list[click_action[area_idx - 1]]
        click_action = 'click('+action_info+')'
        action_id = all_action_ids[click_action]
        
        new_page_name = all_page_convert[current_page_name+'act'+str(action_id)]
        return new_page_name, click_action, action_id
    except Exception as e:
        return current_page_name, "ERROR", -2


def action_scroll(scroll_action_bounds, area_idx, direction, all_action_ids, current_page_name, all_page_convert):
    try:
        scroll_action = dict_scroll_parameters(scroll_action_bounds[area_idx - 1], direction)
        scroll_action = "scroll(" + str((scroll_action)) +")"
        action_id = all_action_ids[scroll_action]
        new_page_name = all_page_convert[current_page_name + 'act' + str(action_id)]
        return new_page_name, scroll_action, action_id
    except Exception:
        return current_page_name, "ERROR", -2


def action_input(text, inputs, current_page_all_actions, all_action_ids, current_page_name, all_page_convert):
    input_action = None
    for input in inputs:
        pattern = r",\s*'([^']+)'"
        match = re.search(pattern, input)
        
        if match:
            result = match.group(1)
            if comput_f1(text, result) < 0.5:
                continue
            else:
                input_action = input
        else:
                continue
    if input_action is None:
        return current_page_name, "ERROR", -2
    else:
        action_id = all_action_ids[input_action]
        new_page_name = all_page_convert[current_page_name + 'act' + str(action_id)]
        return new_page_name, input_action, action_id


def get_scroll_bounds(scroll_actions):
    pattern = re.compile(r'scroll\(\[(\d+),(\d+)\]\[(\d+),(\d+)\],(up|down|left|right)\)')

    unique_bounds = set()

    for scroll_action in scroll_actions:
        match = pattern.match(scroll_action)
        if match:
            
            x1, y1, x2, y2 = map(int, match.groups()[:4])
            bounds_string = f"[{x1},{y1}][{x2},{y2}]"
            unique_bounds.add(bounds_string)

    
    unique_bounds = sorted(unique_bounds)
    return unique_bounds


def load_all_action_ids(graph_dir):
    """Load all_action_id.json (double-encoded JSON string) from a graph directory."""
    path = os.path.join(graph_dir, 'all_action_id.json')
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    return json.loads(raw)


def build_action_id_maps(all_action_ids):
    """Return (action_str_to_id, id_to_action_str) with int ids."""
    action_str_to_id = {k: int(v) for k, v in all_action_ids.items()}
    id_to_action = {}
    for action_str, action_id in action_str_to_id.items():
        id_to_action[action_id] = action_str
    return action_str_to_id, id_to_action


def page_chain_from_task_name(task_page_name):
    """Rebuild page names along a task path, e.g. QQmusic0 -> QQmusic0_34 -> ..."""
    parts = task_page_name.split('_')
    if not parts:
        return []
    chain = [parts[0]]
    for part in parts[1:]:
        chain.append(chain[-1] + '_' + part)
    return chain


def gt_action_ids_from_task_page(task_page_name, graph_dir, action_str_to_id):
    """
    Ground-truth action ids from each step's history_actions[-1] on the task path.
    Skips the root page (no incoming action).
    """
    gt_ids = []
    for page in page_chain_from_task_name(task_page_name)[1:]:
        node_path = os.path.join(graph_dir, page, page + '.json')
        if not os.path.isfile(node_path):
            raise FileNotFoundError(f'Missing node json for GT: {node_path}')
        with open(node_path, 'r', encoding='utf-8') as f:
            node = json.load(f)
        history = node.get('history_actions') or []
        if not history:
            continue
        action_str = history[-1]
        if action_str not in action_str_to_id:
            raise KeyError(f'Action not in all_action_id.json on page {page}')
        gt_ids.append(action_str_to_id[action_str])
    return gt_ids


def find_graph_dir_for_task(source_data_dir, task_page_name):
    """Find {App}_graph_* directory matching task prefix (e.g. QQmusic0 -> QQmusic_)."""
    app_prefix = task_page_name.split('0')[0] + '_' if '0' in task_page_name else task_page_name.split('_')[0] + '_'
    for name in os.listdir(source_data_dir):
        if name.startswith(app_prefix):
            path = os.path.join(source_data_dir, name)
            if os.path.isdir(path):
                return name
    return None

