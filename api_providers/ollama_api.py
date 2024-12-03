from flask import Flask, request, jsonify
import base64
from PIL import Image
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

def perform_caption(prompt:str, image:str, model='llama3.2-vision:11b-instruct-q8_0'):
    image_base64 = resize_and_encode_image(image)
    try:
        response = ollama.chat(
            model=model,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [f'{image_base64}']
            }]
        )
        return response['message']['content']
    except Exception as e:
        print(f'Error: {e}')
        return None

@app.route('/caption', methods=['POST'])
def api():
    try:
        import ollama
    except ImportError as e:
        print('Please install Ollama library first.')
    data = request.json
    
    prompt = data.get("prompt")
    image = data.get("image")
    
    caption = perform_caption(prompt, image)
    return jsonify({"caption": caption[0]})