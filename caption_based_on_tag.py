import os
import json
import requests
from tqdm import tqdm
import base64
from PIL import Image
import io


'''
TODO: 
1. Combine character tag and ORIGINAL SUIT together. (which need a database to convert)
    A list needs to be created, including the original costume and facial type of the character.
    Let AI determine if the character's features match. If they do, it should indicate that the character is wearing their original costume. 
    If not, it should output what the character is actually wearing.
2. Instead of loading model instantly, use API to make captions to support more VLMs.

Notice: The dataset has switched to danbooru because the access to the Pixiv-2.6M dataset has been disabled.
'''


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

def load_tags_from_json(tag_file):
    with open(tag_file, 'r') as f:
        tags_dict = json.load(f)

    if tags_dict['rating'] == 'explicit':
        tags_dict['general'] += ', NSFW Image'

    if ',' in tags_dict['character']:
        # str -> list for multiple characters
        tags_dict['character'] = tags_dict['character'].split(',')
    else:
        tags_dict['character'] = [tags_dict['character']]

    result = {
        'pid': id,
        'general_tags': tags_dict['general'] + tags_dict['meta'],
        'character_tags': tags_dict['character'],
        'copyright_tags': tags_dict['copyright'],
        'artist_tags': tags_dict['artist']
    }
    return result

def generate_prompt(image):
    '''
    tags_dict = {
        'pid': '12345',
        'general_tags': 'sunset, landscape, scenic, NSFW image',
        'character_tags': ['Eiffel Tower', 'Parisian girl'], # ['Parisian girl']
        'copyright_tags': 'Paris, France',
        'artist_tags': 'John Doe'
    }
    '''
    tags_dict = load_tags_from_json(image)
    prompt = f"Here's some accurate tags for this image: {tags_dict['general_tags']}. "

    if 'character_tags' in tags_dict.keys():
        character = tags_dict['character_tags']

        if len(character) == 1:
            if '(' in character and ')' in character:
                character_tag = character.split('(')[0]
                series_tag = tags_dict['copyright_tags']
                prompt += f"The character in this image is '{character_tag}'. The series is '{series_tag}'. If you know about the series, response this series is game or anime."
            else:
                character_tag = character[0]
                prompt += f"The character in this image is '{character_tag}'."
        elif len(character) > 1:
            characters_tag = []
            series = set()
            for value in character:
                if '(' in value and ')' in value:
                    character_tag = value.split('(')[0]
                    series_tag = tags_dict['copyright_tags']
                    characters_tag.append(character_tag)
                    series.add(series_tag)
                else:
                    characters_tag.append(value)
            characters = ", ".join(characters_tag)
            prompt += f"These characters in this image is '{characters}'."
            # tqdm.write(f"Notice: There's many characters in image {tags_dict['pid']}. {prompt}")
            multiple_characters_dict[tags_dict['pid']] = characters
            if len(series) != 0:
                series_prompt = 'The series is'
                for serial in series:
                    series_prompt += f"'{serial}'."
                series_prompt += ' If you know about the series, response this series is game or anime.'
                prompt += series_prompt
            else:
                prompt += f"The image shows a original character."

    if 'artist_tags' in tags_dict.keys():
        # Has artist tags
        artist_prompt = f"This image is created by artist {tags_dict['artist_tags']}"
        prompt += artist_prompt
    else:
        artist_prompt = f"This image is created by an anonymous artist."
        prompt += artist_prompt

    base_prompt = f"Please describe this image based on these tags. The response must include these tags. And should in a word range from 50 to 255. Do not describe anything else. Response the description as sentences (with the following: 'An image of ...'). No need to response as markdown format. If there's tags with 'NSFW images', please make sure to indicate that it is an NSFW image."
    generated_prompt = prompt + '. ' + base_prompt

    return generated_prompt

def image_re_caption(image_path, tags_path):
    generated_prompt = generate_prompt(tags_path)
    # tqdm.write(f'\n\nAsk: {generated_prompt}')
    '''
    # Via OpenAI API
    # Notice: My Flask API only accept image path and prompt. (Due to the fact that transferring base64-based image costs time.)
    # The model I used is Ovis1.6-Llama3.2-3B, which is fast and smart enough in low-parameter VLM. 
    # (It's an amateur scientific statement. If you have any opinion, you're right.)
    img = Image.open(image)
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    '''
    api_url = "http://127.0.0.1:5090/caption"
    # image_base64 = resize_and_encode_image(image_path)

    '''
    {
        "prompt": 'Describe this image.',
        "image": image path
    }
    '''

    # response = requests.post(api_url, json={"prompt": generated_prompt, "image": "data:image/png;base64," + image_base64})
    response = requests.post(api_url,
                             json={"prompt": generated_prompt, "image": image_path})
    
    if response.status_code == 200:
        caption = response.json().get('caption')
        # tqdm.write(f"Response: {caption}")
    else:
        # tqdm.write(f"Failed to get caption: {response.status_code}")
        return None
    
    return caption

def process_image(image_path:str, tag_path:str, output_file:str):

    result = image_re_caption(image_path, tag_path)

    if result is None:
        return
    
    if os.path.exists(output_file):
        os.remove(output_file)
    
    # tqdm.write(f"\n\nVLM: {result}")
    
    with open(output_file, 'w', encoding='UTF-8') as f:
        f.write(result)

def main(tags_dir: str, image_dir: str, output_base_dir: str):
    global multiple_characters_dict
    multiple_characters_dict = {}

    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir, exist_ok=True)

    try:
        for filename in tqdm(os.listdir(image_dir)):
            if filename.endswith('.webp'):
                image_path = os.path.join(image_dir, filename)
                tag_filename = os.path.splitext(filename)[0] + '.json'
                tag_path = os.path.join(tags_dir, tag_filename)

                if not os.path.exists(tag_path):
                    print(f"Tag file {tag_path} not found. Skipping {image_path}.")
                    continue

                filename, extension = os.path.splitext(tag_filename)
                txt_filename = filename + '.txt'
                output_path = os.path.join(output_base_dir, txt_filename)
                if os.path.exists(output_path):
                    # print(f"Tag file {image_path} has been captioned. Skipping...")
                    continue
                process_image(image_path, tag_path, output_path)

    except Exception as e:
        print('Error when processing:', e)

if __name__ == "__main__": 
    captions_output_dir = './NL-captions'
    images_path = './image'
    tags_dir = './tags'

    main(tags_dir, images_path, captions_output_dir)
