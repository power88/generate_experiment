import os
import json
import tarfile
import shutil
import requests
from tqdm import tqdm

'''
TODO: 
1. Combine character tag and ORIGINAL SUIT together. (which need a database to convert)
    A list needs to be created, including the original costume and facial type of the character.
    Let AI determine if the character's features match. If they do, it should indicate that the character is wearing their original costume. 
    If not, it should output what the character is actually wearing.
2. Instead of loading model instantly, use API to make captions to support more VLMs.

Notice: The dataset has switched to danbooru because there's lot's of NSFW images in Pixiv-2.6M dataset.
'''

def load_tags_from_json(image):
    result = {}
    tag_file = os.path.splitext(image)[0] + '.json'
    pid = str(os.path.splitext(os.path.basename(tag_file))[0]).split('_')[0]
    result['pid'] = str(os.path.splitext(os.path.basename(tag_file))[0])
    
    with open(tag_file, 'r') as json_file:
        json_dict = json.load(json_file)
    
    gen_tags = json_dict.get("general tags", [])
    chara_tags = json_dict.get("character tags", [])
    nsfw_tags = json_dict.get('rating class', [])
    
    try:
        artist_tags = pid_uid_dict[pid][1]
        no_artist = False
    except Exception as e:
        no_artist = True
        tqdm.write(f"There's no artist in pid: {pid}")
    
    gen_result = ", ".join(gen_tags)
    result['general_tags'] = gen_result.replace('_', ' ') + ', ' + str(nsfw_tags[0] if nsfw_tags[0] != 'explicit' else 'NSFW' + ' image')
    
    if len(chara_tags) != 0:
        characters_tags = []
        for value in chara_tags:
            characters_tags.append(value.replace('_', ' '))
        result['character_tags'] = characters_tags
    
    if not no_artist:
        result['artist_tags'] = artist_tags
    
    return result

def generate_prompt(image):
    tags_dict = load_tags_from_json(image)
    prompt = f"Here's some accurate tags for this image: {tags_dict['general_tags']}. "
    
    if 'character_tags' in tags_dict.keys():
        character = tags_dict['character_tags']
        
        if len(character) == 1:
            character = character[0]
            if '(' in character and ')' in character:
                character_tag = character.split()[0]
                series_tag = character[character.find('(')+1:character.find(')')]
                prompt += f"The character in this image is '{character_tag}'. The series is '{series_tag}'. If you know about the series, response this series is game or anime."
            else:
                character_tag = character
                prompt += f"The character in this image is '{character_tag}'."
        else:
            characters_tag = []
            series = set()
            for value in character:
                if '(' in value and ')' in value:
                    character_tag = value.split()[0]
                    series_tag = value[value.find('(')+1:value.find(')')]
                    characters_tag.append(character_tag)
                    series.add(series_tag)
                else:
                    characters_tag.append(value)
            characters = ", ".join(characters_tag)
            prompt += f"These characters in this image is '{characters}'."
            tqdm.write(f"Notice: There's many characters in image {tags_dict['pid']}. {prompt}")
            multiple_characters_dict[tags_dict['pid']] = characters
            if len(series) != 0:
                series_prompt = 'The series is'
                for serial in series:
                    series_prompt += f"'{serial}'."
                series_prompt += ' If you know about the series, response this series is game or anime.'
                prompt += series_prompt
    
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

def image_re_caption(image):
    generated_prompt = generate_prompt(image)
    tqdm.write(f'\n\nAsk: {generated_prompt}')
    '''
    # Via OpenAI API
    # Notice: My Flask API only accept image path and prompt. (Due to the fact that transferring base64-based image costs time.)
    # The model I used is Qwen2-VL-3B-AWQ, which is fast and accurate enough in low-parameter VLM. (It's an amateur scientific statement. If you have any opinion, make an issue.)
    img = Image.open(image)
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    '''
    api_url = "http://127.0.0.1:5000/caption"

    '''
    {
        "prompt": 'Describe this image.',
        "image": image path
    }
    '''

    response = requests.post(api_url, json={"prompt": generated_prompt, "image": image})
    
    if response.status_code == 200:
        caption = response.json().get('caption')
        tqdm.write(f"Response: {caption}")
    else:
        tqdm.write(f"Failed to get caption: {response.status_code}")
    
    return caption

def process_image(image_path:str, tag_path:str, output_file:str):
    # output_file = os.path.join(captions_output_dir, os.path.basename(str(os.path.splitext(image_path)[0] + '.txt')))
    result = image_re_caption(image_path)
    
    if os.path.exists(output_file):
        os.remove(output_file)
    
    # tqdm.write(f"\n\nVLM: {result}")
    
    with open(output_file, 'w', encoding='UTF-8') as f:
        f.write(result)
    
    tag_file = os.path.splitext(image_path)[0] + '.json'
    os.remove(image_path)
    os.remove(tag_file)

def extract_tar_data(tar_path:str, extract_path:str) -> dict:
    with tarfile.open(tar_path, 'r') as tar:
        members = tar.getmembers()
        tar.extractall(path=extract_path)
        return [member.name for member in members]

def main(repo_dir:str):
    global pid_uid_dict, multiple_characters_dict
    multiple_characters_dict = {}
    with open('pid-uid.json') as metadata:
        pid_uid_dict = json.load(metadata)
    
    image_dir = os.path.join(repo_dir, 'webp')
    tags_dir = os.path.join(repo_dir, 'tags')
    temp_dir = os.path.join(repo_dir, 'temp')
    
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
    
    try:
        
        for i in range(0, 999):
            images = extract_tar_data(os.path.join(image_dir, f'data-{i:04d}.tar'), temp_dir)
            tags = extract_tar_data(os.path.join(tags_dir, f'data-{i:04d}.tar'), temp_dir)
            image_paths = [os.path.join(temp_dir, img) for img in images]
            
            for img in tqdm(image_paths):
                process_image(img)
        
        shutil.rmtree(temp_dir)
        
        with open(os.path.join(captions_output_dir, 'multiple_characters.json'), 'r') as multiple_characters:
            json.dump(multiple_characters_dict)
            
    except Exception as e:
        print('Error when processing:', e)
        shutil.rmtree(temp_dir)

if __name__ == "__main__": 
    captions_output_dir = './NL-captions'
    dataset_path = './Pixiv-2.6M'
    
    if not os.path.exists(captions_output_dir):
        os.makedirs(captions_output_dir, exist_ok=True)

    main(dataset_path)
