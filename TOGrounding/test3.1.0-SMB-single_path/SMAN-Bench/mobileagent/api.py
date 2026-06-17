import base64
import requests
import time
from http import HTTPStatus
import dashscope
import requests
import openai
import json
import threading
from openai import OpenAI
from utils import *
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import base64
import io
import os
# CUDA_VISIBLE_DEVICES=1,2,3,4,5
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5"
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#      "qwen2-vl-72B",
#      # torch_dtype="auto",
#      device_map="balanced",
#      torch_dtype="bfloat16",
# )
# processor = AutoProcessor.from_pretrained("qwen2-vl-72B")

def get_model_response_qwen(chat):
    payload = {
        'input_text':"",
        'image_path':[]
    }
    # 发送 POST 请求到模型服务
    for role, content in chat:
        payload['input_text'] += (content[0]['text']) 
        try:
            payload['image_path'].append(content[1]['image_url']['url'])
        except:
            pass

    prompt = payload['input_text']
    content = [{
    "type": "text",
        "text": prompt
    }]
    for image in payload['image_path']:
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
    return output_text[0]

# import torch
# from PIL import Image
# from transformers import MllamaForConditionalGeneration, AutoProcessor
# import base64
# model_id = "llama-3-90B-Vision-Instruct"

# model = MllamaForConditionalGeneration.from_pretrained(
#     model_id,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
#     trust_remote_code=True
# )
# processor = AutoProcessor.from_pretrained(model_id)
def get_model_response_llama(chat):
    # model.eval()
    payload = {
        'input_text':"",
        'image_path':[]
    }
    # 发送 POST 请求到模型服务
    for role, content in chat:
        if '<|image|>' in content[0]['text']:
            content[0]['text'] = content[0]['text'].replace('<|image|>', '')
        payload['input_text'] += (content[0]['text']) 
        try:
            payload['image_path'].append(content[1]['image_url']['url'])
        except:
            pass
     #修改后部分
    prompt = payload['input_text']

    image = Image.open(payload['image_path'][0])
    messages = [
        {
            "role": "user",
            "content": [
                {
                    'type': 'image',
                },
                {
                    'type': 'text',
                    'text': prompt
                }
            ],
        }
    ]
    # print(json.dumps(messages))
    # print(len(image))
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    # print(input_text)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(model.device)
    output = model.generate(**inputs, max_new_tokens=500)
    del inputs
    torch.cuda.empty_cache()
    return processor.decode(output[0])
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_model_response(chat, url, model):
    # 定义请求的 payload 和文件
    # payload = {"input_text": input_text}
    # files = {"image_path": image_path}
    # 定义 JSON 格式的 payload

    payload = {
        'input_text':"",
        'image_path':[]
    }
    # 发送 POST 请求到模型服务
    for role, content in chat:
        payload['input_text'] += (content[0]['text']) 
        try:
            payload['image_path'].append(content[1]['image_url']['url'])
        except:
            pass
    headers = {"Content-Type": "application/json"}
    # print(payload)
    # 检查响应是否成功
    while True:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            output = response.json().get("output_text", [])
            break
        else:
            print(f"请求失败，状态码: {response.status_code}")
            print(f'请求失败{response}')
            # time.sleep(30)
    return output[0]
from utils import print_with_color, encode_image

def get_qwenmax_response(chat, model, api):
    dashscope.api_key = api
    messages = []
    for role, content in chat:
        messages.append({
            "role": role,
            "content": content
        })

    while True:
        response = dashscope.MultiModalConversation.call(model=model, messages=messages)  # 输出"Observation:"
        if response.status_code == HTTPStatus.OK:
            break
        else:
            print(f"请求失败，状态码: {response.status_code}")
            print(f'请求失败{response}')
            time.sleep(30)     
    return response.output.choices[0].message.content[0]["text"]

def inference_chat(chat, model, api_url, token):    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

    data = {
        "model": model,
        "messages": [],
        "max_tokens": 2048,
        'temperature': 0.0,
        "seed": 1234
    }

    for role, content in chat:
        data["messages"].append({"role": role, "content": content})
    # print(data["messages"])
    while True:
        try:
            res = requests.post(api_url, headers=headers, json=data)
            res_json = res.json()
            res_content = res_json['response']
            # print_with_color(f'total tokens:' + res_json['usage']['total_tokens'], 'green')
            if 'ERROR' in res_json or 'ERROR' in res_content:
                # print('error', res_json)
                # print_with_color(f'total tokens:' + res_json['usage']['total_tokens'], 'green')
                print('Rate_limit\n')
                print(res_content)
                time.sleep(30)
                continue
        except:
            # print('tokens', res_content)
            time.sleep(30)
            print("Network Error:")
            # print(res)
            try:
                print(res.json())
            except:
                print("Request Failed")
        else:
            break
    
    return res_content

def inference_openai(chat, model, api_url, token):
    messages = []
    for role, content in chat:
        messages.append({"role": role, "content": content})
    base_url = api_url
    # 换成自己的key
    api_key = token
    model_name = model

    client = OpenAI(
        # This is the default and can be omitted
        api_key=api_key,
        base_url=base_url
    )

    response = client.chat.completions.create(
        model = model_name,
        messages = messages,
        max_tokens = 2048,
    )

    return response.choices[0].message.content

def inference_chat_4v(chat, model, api_url, token):    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

    data = {
        "model": model,
        "messages": [],
        "max_tokens": 2048,
        'temperature': 0.0,
        "seed": 1234
    }

    for role, content in chat:
        data["messages"].append({"role": role, "content": content})
    # print('messages len:', len(data["messages"]), data["messages"][len(data["messages"])-1]['role'])
    while True:
        try:
            res = requests.post(api_url, headers=headers, json=data)
            print(res)
            res_json = res.json()
            res_content = res_json['response']
            if 'ERROR' in res_json or 'ERROR' in res_content:
                print('Rate_limit\n')
                print(res_content)
                time.sleep(30)
                continue
        except:
            print("Network Error:")
            time.sleep(30)
            try:
                print(res.json())
            except:
                print("Request Failed")
        else:
            break
    
    return res_content

def get_llama_model_response(chat, url, model):
    # 定义请求的 payload 和文件
    # payload = {"input_text": input_text}
    # files = {"image_path": image_path}
    # 定义 JSON 格式的 payload

    payload = {
        'input_text':"",
        'image_path':[]
    }

    # 发送 POST 请求到模型服务
    for role, content in chat:
        if '<|image|>' in content[0]['text']:
            content[0]['text'] = content[0]['text'].replace('<|image|>', '')
        payload['input_text'] += (content[0]['text']) 
        try:
            payload['image_path'].append(content[1]['image_url']['url'])
        except:
            pass
    # print_with_color(f'chat planning', 'red')
    # print_with_color(f"{payload}: ", 'green')
    headers = {"Content-Type": "application/json"}
    # response = requests.post(url, json=payload, headers=headers)
    # 检查响应是否成功
    # print(payload)
    while True:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            output = response.text
            # print(output)
            break
        else:
            print(f"请求失败，状态码: {response.status_code}")
            print(f'请求失败{response}')
            time.sleep(30)
    return output



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


def get_InternVL_response(chat):
    payload = {
        'input_text':"",
        'image_path':[]
    }

    for role, content in chat:
        payload['input_text'] += (content[0]['text']) 
        try:
            payload['image_path'].append(content[1]['image_url']['url'])
        except:
            pass
    prompt = payload['input_text']
    img = payload['image_path'][0]
    model, tokenizer = _get_internvl()
    pixel_values = load_image(img, max_num=12).to(torch.bfloat16).cuda()
    question = '<image>\n' + prompt
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    response = model.chat(tokenizer, pixel_values, question, generation_config)
    return response

# InternVL2的纯文本回复
def get_InternVL_text_response(chat):
    payload = {
        'input_text':"",
        'image_path':[]
    }
    # 发送 POST 请求到模型服务
    for role, content in chat:
        payload['input_text'] += (content[0]['text']) 
        try:
            payload['image_path'].append(content[1]['image_url']['url'])
        except:
            pass
    prompt = payload['input_text']
    model, tokenizer = _get_internvl()
    question = '<image>\n' + prompt
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    response = model.chat(tokenizer, None, question, generation_config)
    return response

def get_InternVL_images_response(chat):
    payload = {
        'input_text':"",
        'image_path':[]
    }
    # 发送 POST 请求到模型服务
    for role, content in chat:
        payload['input_text'] += (content[0]['text']) 
        try:
            payload['image_path'].append(content[1]['image_url']['url'])
            payload['image_path'].append(content[2]['image_url']['url'])
        except:
            pass
    prompt = payload['input_text']
    # print(payload['image_path'])
    img_1 = payload['image_path'][0]
    img_2 = payload['image_path'][1]
    pixel_values_1 = load_image(img_1, max_num=12).to(torch.bfloat16).cuda()
    pixel_values_2 = load_image(img_2, max_num=12).to(torch.bfloat16).cuda()
    pixel_values = torch.cat((pixel_values_1, pixel_values_2), dim=0)
    model, tokenizer = _get_internvl()
    question = '<image>\n' + prompt
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    response = model.chat(tokenizer, pixel_values, question, generation_config)
    return response