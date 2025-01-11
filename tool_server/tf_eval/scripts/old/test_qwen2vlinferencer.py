from dataclasses import dataclass, field
from transformers import HfArgumentParser

from tool_server.tf_eval.inferencer import QwenInferencer
from tool_server.tf_eval.prompts.qwen72b_assistant import assistant_prompt



if __name__ == "__main__":
    
    @dataclass
    class DataArguments:

        input_path: str = field(default=None)
        output_path: str = field(default=None)
        image_dir_path: str = field(default=None)
        dataset_name: str = field(default="chartqa")
        batch_size: int = field(default=32)
        num_workers: int = field(default=4)
              
    @dataclass
    class InferenceArguments:
        model_name: str = field(default="Qwen2-VL-72B-Instruct")
        controller_addr: str = field(default="http://SH-IDCA1404-10-140-54-119:20001")
        inferencer_name: str = field(default="qwen2vl")
        max_rounds: int = field(default=10)

            
    parser = HfArgumentParser(
    (InferenceArguments, DataArguments))
    
    
    inference_args, data_args = parser.parse_args_into_dataclasses()
    qwen_inferencer = QwenInferencer(inference_args=inference_args, data_args=data_args)
    
    ## Case 1 
    image = "/mnt/petrelfs/haoyunzhuo/mmtool/Tool-Factory/tool_server/tool_workers/restructure_worker/test_cases/two_col_102588.png"
    prompt = assistant_prompt + "What is the difference between the highest and lowest risk index score?"
     
    instance=dict(
        image=image,
        prompt=prompt,
        gen_kwargs={"max_new_tokens":2048},
    )
    generation_logs, conversation_logs = qwen_inferencer.inference_on_one_instance(
        model_name=inference_args.model_name,
        instance=instance,
        max_rounds=inference_args.max_rounds
    )
    print("generation_logs: ", generation_logs)
    print("conversation_logs: ", conversation_logs)
    