import os
import time
import subprocess
import logging
from pathlib import Path
import requests
from dataclasses import dataclass
from typing import Optional, List, Dict
from box import Box
import argparse
import yaml

from tool_server.utils.utils import load_json_file, write_json_file

class ServerManager:
    """Server Manager Class"""
    def __init__(self, config: Optional[Dict] = None):
        # Initialize configuration
        self.config = Box(config)
        self.logger = self._setup_logger()
        self.log_folder = Path(self.config.log_folder)
        self.log_folder.mkdir(parents=True, exist_ok=True)
        
        # Initialize status
        self.controller_addr = None
        self._clean_environment()
        os.chdir(self.config.base_dir)
        
        self.controller_config = self.config.controller_config
        self.model_worker_config = self.config.model_worker_config if "model_worker_config" in self.config else []
        self.tool_worker_config = self.config.tool_worker_config if "tool_worker_config" in self.config else []
        self.slurm_job_ids = []

    def _setup_logger(self) -> logging.Logger:
        """Set up logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def _clean_environment(self) -> None:
        """Clean environment variables"""
        os.environ.pop('SLURM_JOB_ID', None)
        os.environ["OMP_NUM_THREADS"] = "1"

    def run_srun_command(self, job_name: str, gpus: int, cpus: int, 
                        command: List[str], log_file: str, srun_kwargs: Dict = {}, 
                        conda_env: str = None, cuda_visible_devices: str = None) -> subprocess.Popen:
        """Run SLURM command"""
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
            "--quotatype=reserved",
            f"--output={log_file}",
        ]
        
        if conda_env:
            srun_cmd.insert(0, f"source ~/anaconda3/bin/activate {conda_env} &&")
            
        if cuda_visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        
        for k,v in srun_kwargs.items():
            srun_cmd.extend([f"-{k}", str(v)])
        srun_cmd.extend(command)
        
        self.logger.info(f"Starting job: {job_name} with conda environment {conda_env if conda_env else 'original env'}")
        return subprocess.Popen(" ".join(srun_cmd), shell=True, env=os.environ.copy())

    def wait_for_job(self, job_name: str) -> str:
        """Wait for job allocation and get node information"""
        self.logger.info(f"Waiting for job to start: {job_name}")
        while True:
            # Get job ID
            job_id = subprocess.getoutput(
                f"squeue --user=$USER --name={job_name} --noheader --format='%A' | head -n 1"
            ).strip()
            
            if job_id:
                # Get node list
                node_list = subprocess.getoutput(
                    f"squeue --job={job_id} --noheader --format='%N'"
                ).strip()
                if node_list:
                    self.logger.info(f"Job {job_name} is running on node: {node_list}")
                    return {"job_id": job_id, "node_list": node_list}
            time.sleep(self.config.retry_interval)

    def wait_for_worker_addr(self, worker_name: str) -> str:
        """Wait for worker address to be available"""
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
        """Start controller"""
        log_file = self.log_folder / f"{self.controller_config.worker_name}.log"
        script_addr = self.controller_config.cmd.pop("script-addr")
        job_name = self.controller_config.job_name
        command = ["python", script_addr]
        for k, v in self.controller_config.cmd.items():
            command.extend([f"--{k}", str(v)])
        
        
        self.run_srun_command(job_name, self.config.default_control_gpus, self.config.default_control_cpus, command, str(log_file), srun_kwargs=self.controller_config.get("srun_kwargs", {}), conda_env=self.controller_config.get("conda_env", None), cuda_visible_devices=self.controller_config.get("cuda_visible_devices", None))
        
        wait_dict = self.wait_for_job(job_name)
        
        node_list = wait_dict["node_list"]
        job_id = wait_dict["job_id"]
        
        self.slurm_job_ids.append(job_id)
        self.controller_addr = f"http://{node_list}:{self.controller_config.cmd.port}"
        self.logger.info(f"Controller is running at: {self.controller_addr}")
        controller_addr_dict = {"controller_addr": self.controller_addr}
        if "controller_addr_location" in self.controller_config:
            self.controller_addr_location = self.controller_config.controller_addr_location
            write_json_file(controller_addr_dict, self.controller_config.controller_addr_location)
            self.logger.info(f"Controller address saved to: {self.controller_addr_location}")
        else:
            current_file_path = os.path.dirname(os.path.abspath(__file__))
            self.controller_addr_location = f"{current_file_path}/../../online_workers/controller_addr/controller_addr.json"
            write_json_file(controller_addr_dict, self.controller_addr_location)
            self.logger.info(f"Controller address saved to: {self.controller_addr_location}")
        return self.controller_addr

    def start_all_workers(self) -> None:
        """Start all worker services"""
        self.start_model_worker()
        self.start_tool_worker()

    def start_model_worker(self) -> None:
        for config in self.model_worker_config:
            config = list(config.values())[0]
            self.start_worker_by_config(config)
    
    def start_tool_worker(self) -> None:
        for config in self.tool_worker_config:
            config = list(config.values())[0]
            self.start_worker_by_config(config)
    
    def start_worker_by_config(self, config) -> None:
        """Start specific worker"""
        
        if "dependency_worker_name" in config:
            self.wait_for_job(config.dependency_worker_name)
            
        log_file = self.log_folder / f"{config.worker_name}_worker.log"
        script_addr = config.cmd.pop("script-addr")
        job_name = config.job_name
        command = [
            "python", script_addr,
            "--controller-address", self.controller_addr,
        ]
        for k, v in config.cmd.items():
            command.extend([f"--{k}", str(v)])
        
        if config.calculate_type == "control":
            gpus = self.config.default_control_gpus
            cpus = self.config.default_control_cpus
        elif config.calculate_type == "calculate":
            gpus = self.config.default_calculate_gpus
            cpus = self.config.default_calculate_cpus
        else:
            raise ValueError("calculate_type must be 'control' or 'calculate'")
        
        self.run_srun_command(job_name, gpus, cpus, command, str(log_file), srun_kwargs=config.get("srun_kwargs", {}), conda_env=config.get("conda_env", None), cuda_visible_devices=config.get("cuda_visible_devices", None))
        wait_dict = self.wait_for_job(job_name)
        job_id = wait_dict["job_id"]
        self.slurm_job_ids.append(job_id)
        
        if "wait_for_self" in config and config["wait_for_self"]:
            self.wait_for_worker_addr(config.worker_name)

    def shutdown_services(self) -> None:
        """Shut down all SLURM services

        Use the recorded job_ids list to shut down services one by one, handling errors and logging
        """
        if not hasattr(self, 'slurm_job_ids') or not self.slurm_job_ids:
            self.logger.warning("No SLURM job IDs found to shutdown")
            return
        try:
            os.remove(self.controller_addr_location)
            self.logger.info("Controller address file removed")
        except:
            self.logger.warning("Controller address file not found, skipping removal")
            pass
            
        try:
            for job_id in self.slurm_job_ids:
                try:
                    # Check if the job is still running
                    check_cmd = f"squeue --job={job_id} --noheader"
                    if subprocess.getoutput(check_cmd).strip():
                        # Cancel the job
                        subprocess.run(["scancel", str(job_id)], check=True)
                        self.logger.info(f"Successfully cancelled job ID: {job_id}")
                    else:
                        self.logger.info(f"Job ID: {job_id} was already finished")
                        
                except subprocess.CalledProcessError as e:
                    self.logger.error(f"Error cancelling job ID {job_id}: {e}")
                except Exception as e:
                    self.logger.error(f"Unexpected error while cancelling job ID {job_id}: {e}")
                    
            # Clear the job_ids list
            self.slurm_job_ids.clear()
            self.logger.info("All services have been shutdown")
            
        except Exception as e:
            self.logger.error(f"Critical error during shutdown: {e}")
            raise
        finally:
            # Ensure environment variables are cleaned up
            if 'SLURM_JOB_ID' in os.environ:
                del os.environ['SLURM_JOB_ID']


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", type=str, default="/mnt/petrelfs/songmingyang/code/reasoning/tool-agent/tool_server/tool_workers/scripts/launch_scripts/config/all_service_smy.yaml", help="Path to configuration file")
    
    args = argparser.parse_args()
    config_path = Path(args.config)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    try:
        # Create server manager
        manager = ServerManager(config)
        os.chdir(manager.config.base_dir)
        manager.start_controller()
        manager.start_all_workers()
        try:
            # Keep running
            while True:
                time.sleep(1)
            
        except KeyboardInterrupt:
            logger = logging.getLogger(__name__)
            logger.info("Shutting down services...")
            manager.shutdown_services()
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()
