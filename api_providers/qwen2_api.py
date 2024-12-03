from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import base64
import io

app = Flask(__name__)

def model_loader():
    repo_name = "./Qwen2-VL-7B-Instruct-AWQ"
    arguments = {"device_map": "auto", "torch_dtype": torch.float16, "trust_remote_code": True}
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(repo_name, **arguments)
    
    processor = AutoProcessor.from_pretrained(repo_name, **arguments)

    return processor, model

def resize_and_encode_image(image_path, max_size=1024):
    with Image.open(image_path) as img:
        # 获取原始尺寸
        original_width, original_height = img.size

        # 计算调整后的尺寸，保持宽高比
        if max(original_width, original_height) > max_size:
            if original_width > original_height:
                new_width = max_size
                new_height = int((new_width / original_width) * original_height)
            else:
                new_height = max_size
                new_width = int((new_height / original_height) * original_width)
            
            # 调整图像大小
            img = img.resize((new_width, new_height), Image.BICUBIC)
            img = img.convert('RGB')

        # 将图像转为字节流
        img_byte_array = io.BytesIO()
        img.save(img_byte_array, format='PNG')  
        img_byte_array = img_byte_array.getvalue()

    # 将字节流编码为base64
    image_base64 = base64.b64encode(img_byte_array).decode('utf-8')
    return image_base64

def perform_caption(prompt:str, image:Image.Image) -> str:

    image_base64 = resize_and_encode_image(image)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "data:image/png;base64," + image_base64,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text],images=image_inputs,videos=video_inputs,padding=True,return_tensors="pt")
    inputs = inputs.to("cuda")
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text
    
@app.route('/caption', methods=['POST'])
def api():
    data = request.json
    
    prompt = data.get("prompt")
    image = data.get("image")
    
    caption = perform_caption(prompt, image)
    return jsonify({"caption": caption[0]})

if __name__ == "__main__":
    processor, model = model_loader()
    app.run(port=5090)