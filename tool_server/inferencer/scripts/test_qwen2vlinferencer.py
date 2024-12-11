from tool_server.inferencer.qwen2vl_inferencer import QwenInferencer



if __name__ == "__main__":
    qwen_inferencer = QwenInferencer(controller_addr="http://SH-IDCA1404-10-140-54-5:20001")
    
    model_name = "Qwen2-VL-7B-Instruct"
    ## Case 1 
    image = "/mnt/petrelfs/songmingyang/code/tools/test_imgs/roxy.jpeg"
    prompt = "Please describe this image in detail."
    
    ## Case 2
    image = "/mnt/petrelfs/songmingyang/code/tools/test_imgs/plot_chart1.png"
    prompt = "This image show the variation of trade yearly sales data, please tell me in which year the sales is the highest."
    
    ## Case 3
    image = "/mnt/petrelfs/songmingyang/code/tools/test_imgs/plot_chart1.png"
    prompt = "This image show the variation of trade yearly sales data, please tell me in which year the sales is the lowest, and the value of the lowest sales."
    
    instance=dict(
        image=image,
        prompt=prompt,
        gen_kwargs={"max_new_tokens":512},
    )
    generation_logs, conversation_logs = qwen_inferencer.inference_on_one_instance(
        model_name=model_name,
        instance=instance,
    )
    print("generation_logs: ", generation_logs)
    print("conversation_logs: ", conversation_logs)
    