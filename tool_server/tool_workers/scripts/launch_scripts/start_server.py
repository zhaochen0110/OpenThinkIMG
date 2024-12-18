import os
import time
import subprocess
import logging
from pathlib import Path
import requests
from dataclasses import dataclass
from typing import Optional, List


# 配置类，集中管理所有配置参数
@dataclass
class ServerConfig:
    """服务器配置类"""
    # 基础路径配置
    base_dir: str = "/mnt/petrelfs/songmingyang/code/reasoning/tool-agent/tool_server/tool_workers"
    llava_plus_model: str = "/mnt/petrelfs/songmingyang/songmingyang/model/tool-augment/llava_plus_v0_7b"
    dino_config: str = "/mnt/petrelfs/songmingyang/code/reasoning/tool-agent/LLaVA-Plus-Codebase/dependencies/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    dino_model: str = "/mnt/petrelfs/songmingyang/songmingyang/model/tool-augment/groundingdino/groundingdino_swint_ogc.pth"
    sam_model: str = "/mnt/petrelfs/songmingyang/songmingyang/model/tool-augment/groundingdino/sam_vit_h_4b8939.pth"
    
    qwen2vl_model: str = "/mnt/petrelfs/share_data/mmtool/weights/qwen-cogcom-filter"
    
    # 端口配置
    controller_port: int = 20001
    dino_port: int = 20003
    sam_plus_dino_port: int = 20005
    sam_port: int = 20007
    ocr_port: int = 20009
    drawline_port: int = 20011
    crop_port: int = 20013
    
    model_port: int = 40000
    qwen2vl_port: int = 40001
    
    # SLURM配置
    partition: str = "MoE"
    default_calculate_gpus: int = 1
    default_calculate_cpus: int = 16
    default_control_cpus: int = 2
    default_control_gpus: int = 0
    
    
    # 其他配置
    retry_interval: int = 1
    request_timeout: int = 10

class ServerManager:
    """服务器管理类"""
    def __init__(self, config: Optional[ServerConfig] = None):
        # 初始化配置
        self.config = config or ServerConfig()
        self.logger = self._setup_logger()
        
        # 设置路径
        self.log_folder = Path(self.config.base_dir) / "logs/server_log"
        self.log_folder.mkdir(parents=True, exist_ok=True)
        
        # 初始化状态
        self.controller_addr = None
        self._clean_environment()
        os.chdir(self.config.base_dir)
        
        self.slurm_job_ids=[]

    def _setup_logger(self) -> logging.Logger:
        """设置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def _clean_environment(self) -> None:
        """清理环境变量"""
        os.environ.pop('SLURM_JOB_ID', None)
        os.environ["OMP_NUM_THREADS"] = "1"

    def run_srun_command(self, job_name: str, gpus: int, cpus: int, 
                        command: List[str], log_file: str) -> subprocess.Popen:
        """运行SLURM命令"""
        srun_cmd = [
            "srun",
            f"--partition={self.config.partition}",
            f"--job-name={job_name}",
            "--mpi=pmi2",
            f"--gres=gpu:{gpus}",
            "-n1",
            "--ntasks-per-node=1",
            f"-c {cpus}",
            "--kill-on-bad-exit=1",
            "--quotatype=auto",
            f"--output={log_file}",
        ] + command

        self.logger.info(f"Starting job: {job_name}")
        return subprocess.Popen(" ".join(srun_cmd), shell=True, env=os.environ.copy())

    def wait_for_job(self, job_name: str) -> str:
        """等待作业分配并获取节点信息"""
        self.logger.info(f"Waiting for job to start: {job_name}")
        while True:
            # 获取作业ID
            job_id = subprocess.getoutput(
                f"squeue --user=$USER --name={job_name} --noheader --format='%A' | head -n 1"
            ).strip()
            
            if job_id:
                # 获取节点列表
                node_list = subprocess.getoutput(
                    f"squeue --job={job_id} --noheader --format='%N'"
                ).strip()
                if node_list:
                    self.logger.info(f"Job {job_name} is running on node: {node_list}")
                    return {"job_id": job_id, "node_list": node_list}
            time.sleep(self.config.retry_interval)

    def wait_for_worker_addr(self, worker_name: str) -> str:
        """等待worker地址可用"""
        self.logger.info(f"Waiting for {worker_name} worker...")
        attempt = 0
        
        while True:
            try:
                attempt += 1
                response = requests.post(
                    f"{self.controller_addr}/get_worker_address",
                    json={"model": worker_name},
                    timeout=self.config.request_timeout
                )
                response.raise_for_status()
                
                address = response.json().get("address", "")
                if address.strip():
                    self.logger.info(f"Worker {worker_name} is ready at: {address}")
                    return address
                
                self.logger.warning(f"Attempt {attempt}: worker not ready")
                
            except Exception as e:
                self.logger.error(f"Attempt {attempt} failed: {e}")
            
            time.sleep(self.config.retry_interval)

    def start_controller(self) -> str:
        """启动控制器"""
        log_file = self.log_folder / "controller.log"
        command = ["python", "controller.py", "--host", "0.0.0.0", 
                  f"--port", str(self.config.controller_port)]
        
        self.run_srun_command("zc_controller", self.config.default_control_gpus, self.config.default_control_cpus, command, str(log_file))
        
        wait_dict = self.wait_for_job("zc_controller")
        
        node_list = wait_dict["node_list"]
        job_id = wait_dict["job_id"]
        
        self.slurm_job_ids.append(job_id)
        self.controller_addr = f"http://{node_list}:{self.config.controller_port}"
        self.logger.info(f"Controller is running at: {self.controller_addr}")
        return self.controller_addr

    def start_all_workers(self) -> None:
        """启动所有worker服务"""
        # 启动DINO worker
        self.start_dino_worker()
        self.start_crop_worker()
        self.start_drawline_worker()
        self.start_ocr_worker()
        self.start_qwen2vl_worker()

    
    def start_qwen2vl_worker(self) -> None:
        """启动Qwen2VL worker"""
        log_file = self.log_folder / "qwen2vl_worker.log"
        command = [
            "python", "../model_workers/qwen2vl_worker.py",
            "--host", "0.0.0.0",
            "--port", str(self.config.qwen2vl_port),
            "--controller-address", self.controller_addr,
            "--model-path", self.config.qwen2vl_model,
            "--model-name", "Qwen2-VL-7B-Instruct",
        ]
        self.run_srun_command("zc_qwen2vl", self.config.default_calculate_gpus, 
                            self.config.default_calculate_cpus, command, str(log_file))
        wait_dict = self.wait_for_job("zc_qwen2vl")
        job_id = wait_dict["job_id"]
        self.slurm_job_ids.append(job_id)
        
    def start_drawline_worker(self) -> None:
        """启动DINO worker"""
        log_file = self.log_folder / "drawline_worker.log"
        command = [
            "python", "./restructure_worker/drawline_worker.py",
            "--host", "0.0.0.0",
            "--port", str(self.config.drawline_port),
            "--controller-address", self.controller_addr,
        ]
        
        self.run_srun_command("zc_drawline", self.config.default_control_gpus, 
                            self.config.default_control_cpus, command, str(log_file))
        wait_dict = self.wait_for_job("zc_drawline")
        job_id = wait_dict["job_id"]
        self.slurm_job_ids.append(job_id)
        
    def start_ocr_worker(self) -> None:
        """启动OCR worker"""
        log_file = self.log_folder / "ocr_worker.log"
        command = [
            "python", "./ocr_worker.py",
            "--host", "0.0.0.0",
            "--port", str(self.config.ocr_port),
            "--controller-address", self.controller_addr,
        ]
        
        self.run_srun_command("zc_ocr", self.config.default_control_gpus, 
                            self.config.default_control_cpus, command, str(log_file))
        wait_dict = self.wait_for_job("zc_ocr")
        job_id = wait_dict["job_id"]
        self.slurm_job_ids.append(job_id)
        
    def start_crop_worker(self) -> None:
        """启动DINO worker"""
        log_file = self.log_folder / "crop_worker.log"
        command = [
            "python", "./restructure_worker/crop_worker.py",
            "--host", "0.0.0.0",
            "--port", str(self.config.crop_port),
            "--controller-address", self.controller_addr,
        ]
        
        self.run_srun_command("zc_crop", self.config.default_control_gpus, 
                            self.config.default_control_cpus, command, str(log_file))
        wait_dict = self.wait_for_job("zc_crop")
        job_id = wait_dict["job_id"]
        self.slurm_job_ids.append(job_id)
    
    def start_dino_worker(self) -> None:
        """启动DINO worker"""
        log_file = self.log_folder / "dino_worker.log"
        command = [
            "python", "./grounding_dino_worker.py",
            "--host", "0.0.0.0",
            "--port", str(self.config.dino_port),
            "--controller-address", self.controller_addr,
            "--model-config", self.config.dino_config,
            "--model-path", self.config.dino_model
        ]
        
        self.run_srun_command("zc_dino", self.config.default_calculate_gpus, 
                            self.config.default_calculate_cpus, command, str(log_file))
        wait_dict = self.wait_for_job("zc_dino")
        job_id = wait_dict["job_id"]
        self.slurm_job_ids.append(job_id)
    
    def start_sam_worker(self) -> None:
        """启动SAM(Segment Anything Model) worker"""
        log_file = self.log_folder / "sam.log"
        command = [
            "python", "./sam_worker.py",
            "--host", "0.0.0.0",
            "--controller-address", self.controller_addr,
            "--port", str(self.config.sam_port)
        ]
        self.logger.info("Starting SAM worker...")
        self.run_srun_command(
            "zc_sam", 
            self.config.default_calculate_gpus,
            self.config.default_calculate_cpus, 
            command, 
            str(log_file)
        )
        # 等待worker就绪
        wait_dict = self.wait_for_job("zc_sam")
        job_id = wait_dict["job_id"]
        self.slurm_job_ids.append(job_id)

    def start_model_worker(self) -> None:
        """启动LLaVA-Plus Model worker"""
        log_file = self.log_folder / "llava_plus_worker.log"
        command = [
            "python", "-m", "llava_plus.serve.model_worker",
            "--host", "0.0.0.0",
            "--controller-address", self.controller_addr,
            "--port", str(self.config.model_port),
            "--worker-address", "auto",
            "--model-path", self.config.llava_plus_model
        ]
        
        self.logger.info("Starting LLaVA-Plus model worker...")
        self.run_srun_command(
            "zc_llava_plus_worker", 
            self.config.default_calculate_gpus,
            self.config.default_calculate_cpus, 
            command, 
            str(log_file)
        )
        # 等待worker就绪
        wait_dict = self.wait_for_job("zc_llava_plus_worker")
        job_id = wait_dict["job_id"]
        self.slurm_job_ids.append(job_id)
        
    def start_ground_plus_sam_worker(self) -> None:
        """启动SAM(Segment Anything Model) worker"""
        self.wait_for_worker_addr("grounding_dino")
        self.wait_for_worker_addr("sam")
        log_file = self.log_folder / "sam_plus_dino.log"
        command = [
            "python", "./grounded_sam_worker.py",
            "--host", "0.0.0.0",
            "--controller-address", self.controller_addr,
            "--port", str(self.config.sam_plus_dino_port)
        ]
        
        self.logger.info("Starting SAM PLUS DINO worker...")
        self.run_srun_command(
            "zc_sam_plus_dino", 
            self.config.default_control_gpus,
            self.config.default_control_cpus, 
            command, 
            str(log_file)
        )
        # 等待worker就绪
        wait_dict = self.wait_for_job("zc_sam_plus_dino")
        job_id = wait_dict["job_id"]
        self.slurm_job_ids.append(job_id)

    def shutdown_services(self) -> None:
        """关闭所有SLURM服务
        
        使用记录的job_ids列表逐个关闭服务，并进行错误处理和日志记录
        """
        if not hasattr(self, 'slurm_job_ids') or not self.slurm_job_ids:
            self.logger.warning("No SLURM job IDs found to shutdown")
            return
            
        try:
            for job_id in self.slurm_job_ids:
                try:
                    # 检查作业是否仍在运行
                    check_cmd = f"squeue --job={job_id} --noheader"
                    if subprocess.getoutput(check_cmd).strip():
                        # 取消作业
                        subprocess.run(["scancel", str(job_id)], check=True)
                        self.logger.info(f"Successfully cancelled job ID: {job_id}")
                    else:
                        self.logger.info(f"Job ID: {job_id} was already finished")
                        
                except subprocess.CalledProcessError as e:
                    self.logger.error(f"Error cancelling job ID {job_id}: {e}")
                except Exception as e:
                    self.logger.error(f"Unexpected error while cancelling job ID {job_id}: {e}")
                    
            # 清空job_ids列表
            self.slurm_job_ids.clear()
            self.logger.info("All services have been shutdown")
            
        except Exception as e:
            self.logger.error(f"Critical error during shutdown: {e}")
            raise
        finally:
            # 确保清理环境变量
            if 'SLURM_JOB_ID' in os.environ:
                del os.environ['SLURM_JOB_ID']
        
    
        


def main():
    """主函数"""
    try:
        # 创建服务器管理器
        manager = ServerManager()
        
        # 切换到工作目录
        os.chdir(manager.config.base_dir)
        
        # 启动控制器
        manager.start_controller()
        
        # 启动所有worker
        manager.start_all_workers()
        try:
            # 保持运行
            while True:
                time.sleep(1)
            
        except KeyboardInterrupt:
            logger = logging.getLogger(__name__)
            logger.info("正在关闭服务...")
            manager.shutdown_services()
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"发生错误: {e}")
        raise

if __name__ == "__main__":
    main()