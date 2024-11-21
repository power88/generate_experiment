from flask import Flask, request, jsonify
from PIL import Image
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import requests
import base64
import io

app = Flask(__name__)

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

def perform_caption(model:str, prompt:str, image:Image.Image) -> str:

    image_base64 = resize_and_encode_image(image)
    data = {
        "model": model, 
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image_base64}},
                    {"type": "text", "text": prompt}
                ]
            }
        ],
        "temperature": 0.3,
        "max_tokens": 4096
    }

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {gpt_api_key if 'gpt' in model else mistral_api_key}"}
    
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["HEAD", "GET", "OPTIONS", "POST"])

    with requests.Session() as s:
        s.mount('https://', HTTPAdapter(max_retries=retries))
        try:
            response = s.post(api_url, headers=headers, json=data, timeout=180)
            response.raise_for_status()
            response_data = response.json()
            if 'error' in response_data:
                return False, f"API error: {response_data['error']['message']}"
            return True, response_data["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            return False, f"API error: {e}"

    
    return output_text
    
@app.route('/caption', methods=['POST'])
def api(prompt, image):
    data = request.json
    
    prompt = data.get("prompt")
    image = data.get("image")
    model = 'gpt-4o' # gpt-4o-mini, pixtral-12b (free) or llama-3.2-vision-11b (via openrouter) will be more suitable for this task
    
    status, caption = perform_caption(model, prompt, image)
    if status:
        return jsonify({"caption": caption})
    else:
        return jsonify({"error": caption}), 500

if __name__ == "__main__":
    api_url = 'http://api.openai.com/v1/chat/completions' # 'https://openrouter.ai/api/v1/chat/completions' or 'https://api.mistral.ai/v1/chat/completions'
    gpt_api_key = 'sk-1145141919810' # If you use openrouter, replace this to your api key.
    mistral_api_key = 'sk-1145141919810' 
    app.run(port=5000)