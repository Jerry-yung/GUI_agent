import re
import json
from abc import abstractmethod
from typing import List
from http import HTTPStatus
import time
import requests
from http import HTTPStatus
import tiktoken
from openai import OpenAI
from transformers import AutoTokenizer

try:
    import dashscope
    from dashscope import get_tokenizer
except ImportError:
    dashscope = None
    get_tokenizer = None

from utils import print_with_color, encode_image
import torch
# from flask import Flask, request, jsonify
#from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, LlamaTokenizer
# from qwen_vl_utils import process_vision_info
from PIL import Image
import base64
import io
import os
# CUDA_VISIBLE_DEVICES=1,2,3,4,5
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#      "qwen2-vl-72B",
#      # torch_dtype="auto",
#      device_map="balanced",
#      torch_dtype="bfloat16",
# )
# processor = AutoProcessor.from_pretrained("qwen2-vl-72B")

import os
import json
import glob
import tqdm
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer

import math
import numpy as np
import torch
import torchvision.transforms as T
try:
    from decord import VideoReader, cpu
except ImportError:
    VideoReader = None
    cpu = None
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
        'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map


_INTERNVL_PATH = 'InternVL2-40B'
_internvl_model = None
_internvl_tokenizer = None


def _get_internvl():
    """Lazy-load InternVL only when needed (requires CUDA + local weights)."""
    global _internvl_model, _internvl_tokenizer
    if _internvl_model is not None and _internvl_tokenizer is not None:
        return _internvl_model, _internvl_tokenizer
    if torch.cuda.device_count() == 0:
        raise RuntimeError(
            "InternVL local inference requires CUDA GPUs. "
            "Use --model_type OpenAI with an API model instead."
        )
    device_map = split_model(_INTERNVL_PATH)
    _internvl_model = AutoModel.from_pretrained(
        _INTERNVL_PATH,
        torch_dtype=torch.bfloat16,
        load_in_8bit=True,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map,
    ).eval()
    _internvl_tokenizer = AutoTokenizer.from_pretrained(
        _INTERNVL_PATH, trust_remote_code=True, use_fast=False
    )
    return _internvl_model, _internvl_tokenizer


def get_InternVL_response(prompt, img):
    model, tokenizer = _get_internvl()
    pixel_values = load_image(img, max_num=12).to(torch.bfloat16).cuda()
    question = '<image>\n' + prompt
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    response = model.chat(tokenizer, pixel_values, question, generation_config)
    return True, response


# from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
# from qwen_vl_utils import process_vision_info
# plan_path = 'UI-TARS-7B-SFT'
# model = Qwen2VLForConditionalGeneration.from_pretrained(plan_path, torch_dtype="auto", device_map="auto").eval()
# processor = AutoProcessor.from_pretrained(plan_path)
def get_uitars_response(image, instruction, model, processor):
    input_text = uitars_prompt.replace('<instruction>', instruction)
    messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image,
            },
            {"type": "text", "text": f"{input_text}"},
        ],
    }
    ]  
    text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
    )
    # print(len(messages))
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,    
        truncation=True,
        padding_side="left",
        max_length=32640,
        return_tensors="pt",
    )
    # num_tokens = inputs.input_ids.shape[1]  # 每条输入的 token 数（假设 batch_size=1）
    # total_tokens = inputs.input_ids.numel()  # 总 token 数（所有输入的总 token 数量）
    print(f"Total tokens: {inputs.input_ids.numel()}")  # 所有 token 的总数量
    # print(f"Number of tokens in input: {num_tokens}")
    inputs = inputs.to("cuda")
    
    # import pdb
    # pdb.set_trace()
    # Inference: Generation of the output
    # repetition_penalty=1.3 # top_p=0.9
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    del inputs
    torch.cuda.empty_cache()
    return output_text


# model_path = 'CogAgent'
# # device = 'cuda:0'
# model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True).eval()
# tokenizer = LlamaTokenizer.from_pretrained(model_path)

def get_cogagent_response(prompt, img):
    image = Image.open(img).convert("RGB")
    query = prompt
    # print(query)
    # print(image)
    input_by_model = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image])
    inputs = {
        'input_ids': input_by_model['input_ids'].unsqueeze(0).to(model.device),
        'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(model.device),
        'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(model.device),
        'images': [[input_by_model['images'][0].to(model.device).to(torch.bfloat16)]],
    }
    if 'cross_images' in input_by_model and input_by_model['cross_images']:
        inputs['cross_images'] = [[input_by_model['cross_images'][0].to(model.device).to(torch.bfloat16)]]
    gen_kwargs = {"max_length": 4096, "do_sample": False}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(outputs[0])
        print(response)
        response = response.split("</s>")[0]
    return True, response

def get_model_response_qwen(input_text, image_path):
    prompt = input_text
    content = [{
    "type": "text",
        "text": prompt
    }]
    for image in image_path:
        content.append({
        "type": "image",
        "image": image
    })
    messages = [
    {
        "role": "user",
        "content": content
    }
    ]
    model.eval()
    # Process the inputs
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    print(f'Total tokens:{inputs.input_ids.numel()}')
    # Inference
    generated_ids = model.generate(**inputs, max_new_tokens=1000)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    del inputs
    torch.cuda.empty_cache()
    return True, output_text[0]

def get_qwenmax_response(prompt, images, model, api):
    if dashscope is None:
        raise ImportError("dashscope is required for qwen-vl-max. Install with: pip install dashscope")
    dashscope.api_key = api
    content = [{
        "text": prompt
    }]
    for img in images:
        # img_path = f"file://{img}"
        img_path = img
        content.append({
            "image": img_path
        })
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]

    while True:
        response = dashscope.MultiModalConversation.call(model=model, messages=messages)  # 输出"Observation:"
        if response.status_code == HTTPStatus.OK:
            break
        else:
            print(f"请求失败，状态码: {response.status_code}")
            print(f'请求失败{response}')
            time.sleep(30)     
    return True, response.output.choices[0].message.content[0]["text"]
    


def get_qwen_response(prompt, images, url):
    # 定义请求的 payload 和文件
    # payload = {"input_text": input_text}
    # files = {"image_path": image_path}
    # 定义 JSON 格式的 payload
    payload = {
        'input_text':"",
        'image_path':[]
    }
    # 发送 POST 请求到模型服务

    payload['input_text'] = prompt
    for img in images:
        payload['image_path'].append(img)

    headers = {"Content-Type": "application/json"}

    # print(payload)
    # 检查响应是否成功
    while True:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            # print(payload)
            output = response.json().get("output_text", [])
            break
        else:
            
            # print(payload)
            print(f"请求失败，状态码: {response.status_code}")
            print(f'请求失败{response}')
            # time.sleep(30)
    return True, output[0]

def get_gpt4v_response(prompt, images, model, api_url, token):
    content = [
        {
            "type": "text",
            "text": prompt
        }
    ]
    for img in images:
        base64_img = encode_image(img)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_img}"
            }
        })
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "temperature": 0.0,
        "max_tokens": 300
    }
    
    res = requests.post(api_url, headers=headers, json=payload)
    print(res)
    # print('messages len:', len(data["messages"]), data["messages"][len(data["messages"])-1]['role'])
    while True:
        try:
            res = requests.post(api_url, headers=headers, json=payload)
            print(res)
            res_json = res.json()
            res_content = res_json['response']
            if 'ERROR' in res_json or 'ERROR' in res_content:
                print('Rate_limit\n')
                print(res_content)
                time.sleep(30)
                continue
            else:
                break
        except:
            print("Network Error:")
            time.sleep(30)
            try:
                print(res.json())
            except:
                print("Request Failed")
        
    
    return True, res_content
def get_openai_response(
    prompt,
    images,
    model,
    api_url,
    token,
    max_tokens=300,
    temperature=0.0,
):
    content = [
        {
            "type": "text",
            "text": prompt,
        }
    ]
    client = OpenAI(api_key=token, base_url=api_url)
    for img in images:
        base64_img = encode_image(img)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_img}",
            },
        })
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return True, response.choices[0].message.content

def _extract_field(rsp, field_name):
    """Extract Observation / Thought / Action / Summary (multiline-safe)."""
    pattern = rf"{field_name}:\s*(.*?)(?=\n(?:Observation|Thought|Action|Summary):|\Z)"
    matches = re.findall(pattern, rsp, re.DOTALL | re.IGNORECASE)
    return matches[0].strip() if matches else None


def parse_explore_rsp(rsp):

    try:
        observation = _extract_field(rsp, "Observation")
        think = _extract_field(rsp, "Thought")
        act = _extract_field(rsp, "Action")
        last_act = _extract_field(rsp, "Summary") or ""

        if not act:
            raise ValueError("missing Action field in model response")

        if observation:
            print_with_color("Observation:", "yellow")
            print_with_color(observation, "magenta")
        if think:
            print_with_color("Thought:", "yellow")
            print_with_color(think, "magenta")
        print_with_color("Action:", "yellow")
        print_with_color(act, "magenta")
        if last_act:
            print_with_color("Summary:", "yellow")
            print_with_color(last_act, "magenta")

        act = act.strip()
        act_name = act.split("(")[0].strip()
        if act_name == "click":
            click_match = re.findall(r"click\((.*?)\)", act, re.DOTALL)
            if not click_match:
                raise ValueError(f"cannot parse click action: {act}")
            area = str(click_match[0]).strip()
            return [act_name, area, last_act]
        elif act_name == "input":
            input_match = re.findall(r"input\((.*?)\)", act, re.DOTALL)
            if not input_match:
                raise ValueError(f"cannot parse input action: {act}")
            input_str = input_match[0].strip().strip('"').strip("'")
            return [act_name, input_str, last_act]
        elif act_name == "scroll":
            scroll_match = re.findall(r"scroll\((.*?)\)", act, re.DOTALL)
            if not scroll_match:
                raise ValueError(f"cannot parse scroll action: {act}")
            params = scroll_match[0]
            if "," not in params:
                raise ValueError(f"cannot parse scroll params: {act}")
            area, scroll_direction = params.split(",", 1)
            area = str(area.strip())
            scroll_direction = scroll_direction.strip().strip('"').strip("'")
            return [act_name, area, scroll_direction, last_act]
        else:
            print_with_color(f"ERROR: Undefined act {act_name}!", "red")
            return ["ERROR"]

    except Exception as e:
        print_with_color(f"ERROR: an exception occurs while parsing the model response: {e}", "red")
        print_with_color(rsp, "red")
        return ["ERROR"]

def parse_llama_explore_rsp(rsp):
    pattern = r"<\|begin_of_text\|><\|start_header_id\|>user<\|end_header_id\|>(.*?)<\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>"
    # 使用 re.search 提取内
    rsp = re.sub(pattern, "", rsp, flags=re.DOTALL)
    try:
        observation = re.findall(r"Observation:\s*(.*?)(?=\n\n|Thought:|$)", rsp, re.DOTALL)[0].strip()
        think = re.findall(r"Thought:\s*(.*?)(?=\n\n|Action:|$)", rsp, re.DOTALL)[0].strip()
        act = re.findall(r"Action:\s*(.*?)(?=\n\n|Summary:|$)", rsp, re.DOTALL)[0].strip()
        last_act = re.findall(r"Summary:\s*(.*?)(?=$)", rsp, re.DOTALL)[0].strip()
        print_with_color("Observation:", "yellow")
        print_with_color(observation, "magenta")
        print_with_color("Thought:", "yellow")
        print_with_color(think, "magenta")
        print_with_color("Action:", "yellow")
        print_with_color(act, "magenta")
        print_with_color("Summary:", "yellow")
        print_with_color(last_act, "magenta")
        # if "FINISH" in act:
        #     return ["FINISH"]
        # print(act)
        act_name = act.split("(")[0]
        if act_name == "click":
            area = str(re.findall(r"click\((.*?)\)", act)[0])
            return [act_name, area, last_act]
        elif act_name == "input":
            input_str = re.findall(r"input\((.*?)\)", act)[0][1:-1]
            return [act_name, input_str, last_act]
        elif act_name == "scroll":
            params = re.findall(r"scroll\((.*?)\)", act)[0]
            area, scroll_direction = params.split(",")
            area = str(area)
            scroll_direction = scroll_direction.strip()[1:-1]
            return [act_name, area, scroll_direction, last_act]
        elif act_name == "back":
            input_str = re.findall(r"back\((.*?)\)", act)[0][1:-1]
            return [act_name, last_act]
        else:
            print_with_color(f"ERROR: Undefined act {act_name}!", "red")
            return ["ERROR"]

    except Exception as e:
        print_with_color(f"ERROR: an exception occurs while parsing the model response: {e}", "red")
        print_with_color(rsp, "red")
        return ["ERROR"]
        
def parse_multi_explore_rsp(rsp):

    try:
        observation = re.findall(r"Observation: (.*?)$", rsp, re.MULTILINE)[0]
        think = re.findall(r"Thought: (.*?)$", rsp, re.MULTILINE)[0]
        act = re.findall(r"Action: (.*?)$", rsp, re.MULTILINE)[0]
        last_act = re.findall(r"Summary: (.*?)$", rsp, re.MULTILINE)[0]
        print_with_color("Observation:", "yellow")
        print_with_color(observation, "magenta")
        print_with_color("Thought:", "yellow")
        print_with_color(think, "magenta")
        print_with_color("Action:", "yellow")
        print_with_color(act, "magenta")
        print_with_color("Summary:", "yellow")
        print_with_color(last_act, "magenta")
        # if "FINISH" in act:
        #     return ["FINISH"]
        # print(act)
        act_name = act.split("(")[0]
        if act_name == "click":
            area = str(re.findall(r"click\((.*?)\)", act)[0])
            return [act_name, area, last_act]
        elif act_name == "input":
            input_str = re.findall(r"input\((.*?)\)", act)[0][1:-1]
            return [act_name, input_str, last_act]
        elif act_name == "scroll":
            params = re.findall(r"scroll\((.*?)\)", act)[0]
            area, scroll_direction = params.split(",")
            area = str(area)
            scroll_direction = scroll_direction.strip()[1:-1]
            return [act_name, area, scroll_direction, last_act]
        elif act_name == "back":
            input_str = re.findall(r"back\((.*?)\)", act)[0][1:-1]
            return [act_name, last_act]
        else:
            print_with_color(f"ERROR: Undefined act {act_name}!", "red")
            return ["ERROR"]

    except Exception as e:
        print_with_color(f"ERROR: an exception occurs while parsing the model response: {e}", "red")
        print_with_color(rsp, "red")
        return ["ERROR"]

def get_llama_response(prompt, images, model, API_url):
    payload = {
        'input_text':"",
        'image_path':[]
    }
    # 发送 POST 请求到模型服务

    payload['input_text'] = prompt
    for img in images:
        payload['image_path'].append(img)
    headers = {"Content-Type": "application/json"}
    # response = requests.post(url, json=payload, headers=headers)
    # 检查响应是否成功
    while True:
        response = requests.post(API_url, json=payload, headers=headers)
        if response.status_code == 200:
            output = response.text
            # print(output)
            break
        else:
            print(f"请求失败，状态码: {response.status_code}")
            print(f'请求失败{response}')
            time.sleep(30)
    # print(output)
    return True, output
