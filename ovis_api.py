from flask import Flask, request, jsonify
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    GenerationConfig,
)
from transformers.dynamic_module_utils import get_imports
import torch
import os
from PIL import Image
from unittest.mock import patch

app = Flask(__name__)

def remove_flash_attn(filename: str | os.PathLike) -> list[str]:
    if not str(filename).endswith("modeling_ovis.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    try:
        imports.remove("flash_attn")
    except:
        print(f"No flash_attn import to remove")
        pass
    return imports


def model_loader():
    repo_name = "./Ovis1.6-Llama3.2-3B"
    with patch("transformers.dynamic_module_utils.get_imports", remove_flash_attn):
        model = AutoModelForCausalLM.from_pretrained(repo_name, torch_dtype=torch.bfloat16, attn_implementation='eager', multimodal_max_length=8192, trust_remote_code=True).cuda()
    
    text_tokenizer = model.get_text_tokenizer()
    visual_tokenizer = model.get_visual_tokenizer()
    
    return text_tokenizer, visual_tokenizer, model


def perform_caption(text, image_path):

    query = f'<image>\n{text}'
    image = Image.open(image_path)

    prompt, input_ids, pixel_values = model.preprocess_inputs(query, [image])
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
    input_ids = input_ids.unsqueeze(0).to(device=model.device)
    attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
    pixel_values = [pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)]


    with torch.inference_mode():
        gen_kwargs = dict(
            max_new_tokens=1024,
            do_sample=False,
            top_p=None,
            top_k=None,
            temperature=None,
            repetition_penalty=None,
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=text_tokenizer.pad_token_id,
            use_cache=True
        )
        output_ids = model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
        output = text_tokenizer.decode(output_ids, skip_special_tokens=True)
    
    return output
    
@app.route('/caption', methods=['POST'])
def api(image):
    data = request.json
    
    prompt = data.get("prompt")
    image = data.get("image")
    
    perform_caption(prompt, image)

if __name__ == "__main__":
    text_tokenizer, visual_tokenizer, model = model_loader()
    app.run(port=5000)
    