# Tool-Factory

A universal plug-and-play tool usage multimodal framework


### Setup
#### 0. How to install GroundingDINO?

##### 1). install from our code base
install from `tool-agent/LLaVA-Plus-Codebase/dependencies/Grounded-Segment-Anything/GroundingDINO` 
```bash 
cd tool-agent/LLaVA-Plus-Codebase/dependencies/Grounded-Segment-Anything/GroundingDINO
srun -p ${YOUR_PARTITION} pip install -e . ## make sure in a GPU environment
```

##### 2). If faced with any problems

```bash
cd ./tool-agent/LLaVA-Plus-Codebase/dependencies/Grounded-Segment-Anything/GroundingDINO
## 如果发现有pyproject.toml,请删除
rm pyproject.toml
# 尝试安装
srun -p MoE pip install -e .
# 若提示缺少MPCXXX，请自行安装MPC并在PATH和LD_LIBRARY_PATH里面指定MPC路径

## 安装成功后再次import
python -c "import groundingdino._C"
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ImportError: libc10.so: cannot open shared object file: No such file or directory

## 查找libc位置
find $(python -c "import torch; print(torch.__path__[0])") -name "libc10.so"
/mnt/petrelfs/haoyunzhuo/anaconda3/envs/tool-factory/lib/python3.10/site-packages/torch/lib/libc10.so

## 把前面这一串加给LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/mnt/petrelfs/haoyunzhuo/anaconda3/envs/tool-factory/lib/python3.10/site-packages/torch/lib

## 再次import
python -c "import groundingdino._C"
```
#### 1. Install our dependencies

```bash
git clone git@github.com:zhaochen0110/Tool-Factory.git
cd Tool-Factory

conda create -n tool-server python=3.10
# pytorch
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install -r requirment.txt
pip install -e .

```

### Usage

#### 1. Start all services first

```bash
## First, modify the config to adapt to your own environment
## tool_server/tool_workers/scripts/launch_scripts/config/all_service.yaml

## Start all services
cd tool_server/tool_workers/scripts/launch_scripts
python start_server_config.py --config ./config/all_service.yaml

## Press control + C to shutdown all services automatically.
## 关闭服务时 按Ctrl + C 全部服务会自动停止，无需每个手动scancel
```

#### 2. Test our factory through inferencer module

The inferencer module is the main logic of our tool factory, Please reference `tool_server/inferencer/scripts/test_qwen2vlinferencer.py` for
single step inference.

Please reference `tool_server/inferencer/scripts/qwenvl_chartqa_inference.sh` for single loop sequential inference.

### Architechture of This Project

【金山文档 | WPS云文档】 Tool-inplement
https://365.kdocs.cn/l/cu4UPM9NvWLa
