# SMAN-Bench: A Cross-System Benchmark for Mobile Agents under Single- and Multi-path, Ambiguous, and Noisy Tasks
<div align="center">
  
**[ICLR 2026]**

[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/xwk123/MobileBench-v2)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
</div>

## 📢 News
* **[2026.01]** 🎉 SMAN-Bench has been accepted to **ICLR 2026**!

## 📖 Introduction
This repository is the official implementation of **SMAN-Bench**.

SMAN-Bench is a cross-system benchmark designed to evaluate mobile agents under complex scenarios, including:
* **S**ingle-path tasks
* **M**ulti-path tasks
* **A**mbiguous tasks
* **N**oisy tasks

```text
mobilebenchv2/
├── appagent/
│   ├── appagent_ambiguous.py
│   ├── appagent_multipath.py
│   ├── appagent_noisy.py
│   ├── appagent_singlepath.py
│   ├── model.py
│   ├── prompts.py
├── mobileagent/
│   ├── api.py
│   ├── chat.py
│   ├── mobileagent_multipath.py
│   ├── mobileagent_singlepath.py
│   ├── mobileagent_noisy.py
│   ├── mobileagent_ambiguous.py
│   ├── prompt.py
├── config/
│   └── config.yaml
├── eval.py
├── readme.md
├── utils.py
```

## Quick start 

### Step 1.  Configure the model

If you are using a closed-source model, you need to first configure the API URL in `config.yaml`, and then modify the response functions in `appagent/model.py` and `mobileagent/api.py`.  If you are using an open-source model, you can either run inference directly or deploy the model, and then modify the response functions in `appagent/model.py` and `mobileagent/api.py` accordingly.

### Step 2. Start the task

You can try using the following example command to run the task: 

```bash
python appagent/appagent_singlepath.py --data_dir datasets --model_name gpt-4o --config_path config/config.yaml --task_file single_simple --task_file simple_tasks_sample.json --model_type OpenAI --save_path results
```

### Step 3. evaluate the results

You can find the corresponding task type in `eval.py` and fill in the result in the appropriate location for evaluation.
