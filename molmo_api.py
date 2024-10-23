from flask import Flask, request, jsonify
import torch.nn.functional as F
from sageattention import sageattn
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    GenerationConfig,
)
from PIL import Image

app = Flask(__name__)

def model_loader():
    repo_name = "./molmo-7B-D-bnb-4bit"
    arguments = {"device_map": "auto", "torch_dtype": "auto", "trust_remote_code": True}
    
    processor = AutoProcessor.from_pretrained(repo_name, **arguments)
    model = AutoModelForCausalLM.from_pretrained(repo_name, **arguments)
    
    return processor, model


def perform_caption(text, image):
    inputs = processor.process(images=[Image.open(image)], text=text)
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
    output = model.generate_from_batch(inputs, GenerationConfig(max_new_tokens=512, stop_strings="<|endoftext|>"), tokenizer=processor.tokenizer)
    
    generated_tokens = output[0, inputs["input_ids"].size(1) :]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return generated_text
    
@app.route('/caption', methods=['POST'])
def api(image):
    data = request.json
    
    prompt = data.get("prompt")
    image = data.get("image")
    
    perform_caption(prompt, image)

if __name__ == "__main__":
    processor, model = model_loader()
    app.run(port=5000)
    