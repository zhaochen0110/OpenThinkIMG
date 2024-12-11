import argparse
import json

import requests

from io import BytesIO
import base64
from PIL import Image

def b64_encode(img):
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_b64_str = base64.b64encode(buffered.getvalue()).decode()
        return img_b64_str

def main():
    if args.worker_address:
        worker_addr = args.worker_address
    else:
        controller_addr = args.controller_address
        ret = requests.post(controller_addr + "/refresh_all_workers")
        ret = requests.post(controller_addr + "/list_models")
        models = ret.json()["models"]
        models.sort()
        print(f"Models: {models}")

        ret = requests.post(controller_addr + "/get_worker_address",
            json={"model": args.model_name})
        worker_addr = ret.json()["address"]
        print(f"worker_addr: {worker_addr}")

    if worker_addr == "":
        return

    
    headers = {"User-Agent": "LLaVA Client"}
    url = "/mnt/petrelfs/songmingyang/code/tools/test_imgs/roxy.jpeg"
    image = Image.open(url)
    img_b64_str = b64_encode(image)
    img_b64_str = "data:image/jpeg;base64," + img_b64_str
    prompt = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": img_b64_str
                        },
                        {"type": "text", "text": "Describe this image."},
                    ],
                }
            ]
    pload = {
        "model": args.model_name,
        "conversation": prompt,
        "max_new_tokens": 1024,
        "temperature": 0.7,
    }
    response = requests.post(worker_addr + "/worker_generate_stream", headers=headers,
            json=pload, stream=True)
    
    # print(prompt)
    for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            output = data["text"]
            output_str = ""
            for item in output:
                if isinstance(item, str):
                    output_str += item
    print(output_str, end="\r", flush=True)
    print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-address", type=str, default="http://localhost:20001")
    parser.add_argument("--worker-address", type=str)
    parser.add_argument("--model-name", type=str, default="Qwen2-VL-7B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--message", type=str, default=
        "Tell me a story with more than 1000 words.")
    args = parser.parse_args()

    main()
