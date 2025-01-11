# Tool-Factory

> A universal plug-and-play tool usage multimodal framework

![License: Apache-2.0](https://img.shields.io/badge/license-Apache%202.0-green)


🏠 [PRMBench Homepage](#) | 📑 [Paper](#) | 📚 [Documentation](docs/README.md)


## Features
- **Single-Machine Multi-GPU Batch Inference** (Added on 2025-01-11): 
  Batch Infernce and 
  - Rename inferencer to `tf_eval`
  - split tasks and models 
  - `tool_sever.tf_eval.inferencer` and `tool_sever.model_workers` will be deprecated in the next version.



## Installation


```bash
git clone git@github.com:zhaochen0110/Tool-Factory.git
cd Tool-Factory
# optional: create a conda environment
conda create -n tool-server python=3.10
# Install PyTorch and other dependencies
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install -r requirment.txt
pip install -e .

```
If faced with any issues, please refer to our [documentation](docs/README.md## Installation Instructions).

## Example Usages

We released some example scripts/configs to demonstrate how to use our toolkit. You can find them in the `tool_server/tf_eval/scripts` directory.

#### 1. Start all services first

Before you run our inferencer module, it's **necessary** to start all services first. The services include all tool implementation.
```bash
## First, modify the config to adapt to your own environment
## tool_server/tool_workers/scripts/launch_scripts/config/all_service.yaml

## Start all services
cd tool_server/tool_workers/scripts/launch_scripts
python start_server_config.py --config ./config/all_service.yaml

## Press control + C to shutdown all services automatically.
## 关闭服务时 按Ctrl + C 全部服务会自动停止，无需每个手动scancel, 按一下就行，别按多了
```

#### 2. Test our factory through inferencer module


**Evaluation of Qwen2VL on CharXiv Directly**
A simple way to run Tool-Factory.

```bash
accelerate launch  --config_file  ${accelerate_config} \
-m tool_server.tf_eval \
--model qwen2vl \
--model_args pretrained=Qwen/Qwen2-VL-7B-Instruct \
--task_name charxiv \
--verbosity INFO \
--output_path ./scripts/logs/prmtest_classified/reasoneval_7b.jsonl \
--batch_size 2 \
--max_rounds 3 \
--stop_token <stop> \
--controller_addr http://localhost:20001
```

**Evaluation of ReaonEval-7B on PRMBench Using a Config File**
We strongly recommend that using a config file to evaluate tool planning models.


```bash
accelerate launch  --config_file  ${accelerate_config} \
-m tool_server.tf_eval \
--config ${config_file}
```

Config file example:

```yaml
- model_args:
    model: qwen2vl
    model_args: pretrained=Qwen/Qwen2-VL-7B-Instruct
    batch_size: 2
    max_rounds: 3
    stop_token: <stop>
  task_args:
    task_name: charxiv
    resume_from_ckpt:
      charxiv: ./tool_server/tf_eval/scripts/logs/ckpt/charxiv/qwen2vl.jsonl
    save_to_ckpt:
      charxiv: ./tool_server/tf_eval/scripts/logs/ckpt/charxiv/qwen2vl.jsonl
  script_args:
    verbosity: INFO
    output_path: ./tool_server/tf_eval/scripts/logs/results/charxiv/qwen2vl.jsonl
    controller_addr: http://localhost:20001 # Should be the same with your controller address
```

For detailed information and config setting please refer to our [documentation](docs/README.md).
