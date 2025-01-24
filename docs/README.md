# Documentations for Tool Factory

## TF-EVAL Module

The `tf_eval` is the main logic for tool planning model inference. It is based on the `accelerate` framework. The main scripts are under `tool_server/tf_eval`. Before start `tf_eval`, you should make sure tool services are running.

### 1. Config File Formatting

**We released some example scripts/configs to demonstrate how to use our toolkit. You can find them in the `tool_server/tf_eval/scripts` directory.**

You can organize your config as a list of dict or a single dict. It's recommend to use a yaml file. 

```yaml
- model_args:
    # The model series you want to test, must be the same with the file name under tf_eval/models
    model: qwen2vl
    # The arguments that you want to pass to the models, split by a comma.
    model_args: pretrained=/mnt/petrelfs/share_data/mmtool/weights/qwen-cogcom-base
    # The batch size if you want to use batch inference. Caution for OOM.
    batch_size: 2
    # The max rounds you want tool planning model to inference.
    max_rounds: 3
    # When to stop one single round inference, the model will stop when the output contains the stop token.
    stop_token: <stop>
  task_args:
    # The task names you want to evaluate, split by a comma.
    task_name: charxiv
    # checkpoint settings, organize them as a dict
    # taskname: ckpt_file path
    resume_from_ckpt:
        charxiv: ./tool_server/tf_eval/scripts/logs/ckpt/charxiv/qwen2vl.jsonl
  save_to_ckpt:
        charxiv: ./tool_server/tf_eval/scripts/logs/ckpt/charxiv/qwen2vl.jsonl
    script_args:
    verbosity: INFO
    # final result output path
    output_path: ./tool_server/tf_eval/scripts/logs/results/charxiv/qwen2vl.jsonl
```
After setting down the config, please run TF EVAL as:

```bash
accelerate launch  --config_file  ${accelerate_config} \
-m tool_server.tf_eval \
--config ${config_file}
```

Our batch inference and multi-gpu parallel is inferenced based on huggingface accelerate, so please prepare a accelerate config and run based on it.
An example accelerate config is:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

But notice that when testing api modes (e.g. gemini and openai series models), the batch size must be set at 1 and do not use multi-process parallel.


### 2. How to run a tool planning model through VLLM?
We offer a VLLM model implementation in `tool_server/tf_eval/models/vllm_models.py`. This allows you to seamlessly run the model using VLLM by simply replacing the model entry in the configuration file with vllm_models and specifying your model checkpoint using the `pretrained=xxx` parameter in model_args. To enable tensor parallelism, you can configure the tensor parallel size by adding `tensor_parallel=x` to model_args.

A possible example is:


```yaml
model_args:
  model: vllm_models
  model_args: pretrained=/mnt/petrelfs/share_data/quxiaoye/models/Qwen2-VL-72B-Instruct,tensor_parallel=4
  batch_size: 2
  max_rounds: 3
  stop_token: <stop>
task_args:
  task_name: charxiv
  resume_from_ckpt:
    charxiv: ./tool_server/tf_eval/scripts/logs/ckpt/charxiv/qwen2vl72b.jsonl
  save_to_ckpt:
    charxiv: ./tool_server/tf_eval/scripts/logs/ckpt/charxiv/qwen2vl72b.jsonl
script_args:
  verbosity: INFO
  output_path: ./tool_server/tf_eval/scripts/logs/results/charxiv/qwen2vl72b.jsonl
  
```


### 3.Introduction to basic framework of TF Eval

Our `TFEval` framework is consisted with two important concepts: `task` and `model`. You can add custom tasks or models to customize your own evaluation framework.
The tasks and models are connected throuth a pytorch dataset, whose basic implementation can be found at `tool_server/tf_eval/tasks/base_dataset`. The **data loading logic**(`load_data_function()`) and **evaluation logic**(`evaluate_function()`) is implemeted by `task` and the **get data instance logic**(`getitem_fn(self,meta_data,index)`) is implemented by `model`.

The results of the evaluation will be staged in `base_dataset`, so you can call `dataset.store_results(res)` to stage the results temporarily. After the whole evaluation process, the evaluation process will call `evaluate_function()` to get the final results.

The batch inference logic is implemented under `tool_server/tf_eval/tool_inferencer/base_inferencer.py`, which identifies the model batch inferencing and sequential tool calling logics. A data structure called `dynamic_batch` is used across model and inferncer to store the metadata and temporary results.


### 4.How to add a new model?

1. Implement your model inference script under `tool_server/tf_eval/models`

2. Wrap your model inference code with base class `tp_model` (`tool_server.tf_eval.models.abstract_model.tp_model`).

3. Implement the `getitem_function`, `generate`, `generate_conversation_fn`, and `append_conversation_fn`  function in your model class

4. register your model in `AVAILABLE_MODELS` in `tool_server/tf_eval/models/__init__.py`, the key should be the same with the `model` in your config file and your implement python file name. The value should be the class name of your model.

**Notes:**

- When implementing the `getitem_function`, you should return a dict like:
```python
{
    "text": "your text",  # meta_data['idx']
    "image": "your PIL image", # Image.open(meta_data['image_path'])
    "idx": "keep this the same with meta_data['idx']" # meta_data['idx']
}
```

- An example template is:

```python
from .abstract_model import tp_model
class GeminiModels(tp_model):
    def __init__(
            self,
            pretrained = "your-model-name",
        ) -> None:
        super().__init__()
        # your initialize scripts

    def generate_conversation_fn(
        self,
        text,
        image, 
        role = "user",
    ):
        raise NotImplementedError
    
    def append_conversation_fn(
        self, 
        conversation, 
        text, 
        image, 
        role
    ):
        raise  NotImplementedError
    
    def generate(
        self,
        batch: List[DynamicBatchItem],
    ):
        raise NotImplementedError
    
    def getitem_fn(
        self,
        meta_data: List,
        idx: int,
    ):
        raise NotImplementedError

    
```


### 5.How to add a new task?

1. Implement your task under `tool_server/tf_eval/tasks/your-task-name`
2. Implement your `tool_server/tf_eval/tasks/your-task-name/config.yaml` and `tool_server/tf_eval/tasks/your-task-name/task.py`
3. In `task.py`, implement the `load_data_function()` and `evaluate_function(results,meta_data)`
4. No need to register your task, but make sure the `task_name` in your config file is the same with the folder name of your task, that is `your-task-name` in this demo.

**Notes:**

- When implementing the `load_data_function`, you should return a list of dicts, the dict should contain:

```python
dict(
    idx=item_id,
    image_path=image_path, 
    text=text, 
    **item ## Other information that you need to evaluate on the results.
)
```

- In the function `evaluate_function(results,meta_data)`:

    a. `Temprarily, the `results` are a list of objects, where the object is formatted the same as dynamic batch item:

```python
@dataclass
class DynamicBatchItem:
    max_rounds: int
    current_round : int
    status: str = "pending" # pending, processing, finished
    meta_data: Dict = field(default = None)
    conversation: object = field(default = None)
    model_response: List[str] = field(default_factory=list) # A list of model responses across all rounds
    tool_cfg :  List[str] = field(default_factory=list)  # A list of tool_cfgs across all rounds
    tool_response :  List[str] = field(default_factory=list) # A list of tool_responses across all rounds
    new_round_input :  List[str] = field(default_factory=list)
```

    b. `meta_data` is a list of objects, where the object is the same as the one you returned in `load_data_function()`

    c. Remember to remove redundant items using idx.

- When finish one round of batch inference, `dataset.store_results(res)` is called to store the results in the dataset sequentially.


## Installation Instructions

### 1. How to install GroundingDINO?

##### a. install from our code base
install from `tool-agent/LLaVA-Plus-Codebase/dependencies/Grounded-Segment-Anything/GroundingDINO` 
```bash 
cd tool-agent/LLaVA-Plus-Codebase/dependencies/Grounded-Segment-Anything/GroundingDINO
srun -p ${YOUR_PARTITION} pip install -e . # make sure in a GPU environment
```

##### b. If faced with any problems, Install from source


```bash
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git
cd Grounded-Segment-Anything/GroundingDINO

# 如果发现有pyproject.toml,请删除
rm pyproject.toml
# 尝试安装
srun -p MoE pip install -e .
# 若提示缺少MPCXXX，请自行安装MPC并在PATH和LD_LIBRARY_PATH里面指定MPC路径

# 安装成功后再次import
python -c "import groundingdino._C"
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ImportError: libc10.so: cannot open shared object file: No such file or directory

# 查找libc位置
find $(python -c "import torch; print(torch.__path__[0])") -name "libc10.so"
/mnt/petrelfs/haoyunzhuo/anaconda3/envs/tool-factory/lib/python3.10/site-packages/torch/lib/libc10.so

# 把前面这一串加给LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/mnt/petrelfs/haoyunzhuo/anaconda3/envs/tool-factory/lib/python3.10/site-packages/torch/lib

# 再次import
python -c "import groundingdino._C"
```

## Tool Hub

### 1. Tool Manager
From 2025-01-14, we use a single tool manager to manage all tools, including offline tools (which means the tool is computational cheap and no need to use gpus) and online tools (which means the tool is computational expensive and need to use gpus). The tool manager is implemented in `tool_workers/tool_manager/base_manager.py`, which is responsible for tool registration and tool calling. 

Moreover, We surpport users to add their own tools to the tool hub. You can choose to add your own tools to the tool hub by following the instructions below.

### 2. How To Add a New Tool

1. If your tool is simple and no need to use gpus, you can add your tool to the offline tools, which means the tool will be called by the tool manager as a function. 
    - Implement your tool as `tool_server/tool_workers/offline_workers/your_tool_worker.py`. It's recommended to reference other tool implementation to implement your own `generate` function in your script.
    - register your tool in `tool_server/tool_workers/offline_workers/__init__.py`, the key should be the same with the `tool_name` that you want the tool planning model call, and the value should be the file name of your tool implementation. 

2. If your tool is computational expensive and need to use gpus, you can add your tool to the online tools.
    - Implement your tool as `tool_server/tool_workers/online_workers/your_tool_worker.py`. It's recommended to reference other tool implementation to implement your own `generate` function in your script.
    - Wrap the base class `BaseToolWorker` in `tool_server/tool_workers/online_workers/base_tool_worker.py`. The template provided some basic functions that you can use to implement your own tool worker.
