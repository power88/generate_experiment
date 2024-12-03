from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.dynamic_module_utils import get_imports
import torch
import os
from PIL import Image
from unittest.mock import patch

app = Flask(__name__)

def model_loader():
    repo_name = "./phi-3.5-vision-instruct"
    model = AutoModelForCausalLM.from_pretrained(repo_name,
                                                 trust_remote_code=True,
                                                 torch_dtype="auto",
                                                 _attn_implementation="flash_attention_2").cuda().eval()
    
    processor = AutoProcessor.from_pretrained(repo_name, trust_remote_code=True)

    
    return model, processor


def perform_caption(text, image_path):
    prompt = f"{user_prompt}<|image_1|>\n{text}{prompt_suffix}{assistant_prompt}"
    image = Image.open(image_path).convert("RGB")

    inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")
    generate_ids = model.generate(**inputs,
                                  max_new_tokens=1000,
                                  eos_token_id=processor.tokenizer.eos_token_id,
                                  )
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids,
                                      skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)[0]

    
    return response
    
@app.route('/caption', methods=['POST'])
def api():
    data = request.json
    
    prompt = data.get("prompt")
    image = data.get("image")
    
    caption = perform_caption(prompt, image)

    return jsonify({"caption": caption})

if __name__ == "__main__":
    kwargs = {}
    kwargs['torch_dtype'] = torch.bfloat16

    user_prompt = '<|user|>\n'
    assistant_prompt = '<|assistant|>\n'
    prompt_suffix = "<|end|>\n"

    model, processor = model_loader()

    app.run(port=5000)
    