import os
import json
import tarfile
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    GenerationConfig,
)
from PIL import Image
from tqdm import tqdm


def load_tags_from_json(image):
    result = {}
    tag_file = os.path.splitext(image)[0] + '.json'
    with open(tag_file, 'r') as json_file:
        json_dict = json.load(json_file)
    gen_tags = json_dict.get("general tags", [])
    chara_tags = json_dict.get("character tags", [])
    nsfw_tags = json_dict.get('rating class', [])
    gen_result = ", ".join(gen_tags)
    result['general_tags'] = gen_result.replace('_', ' ') + ', ' + str(nsfw_tags[0] if nsfw_tags[0] != 'explicit' else 'NSFW' + ' image')
    if len(chara_tags) != 0:
        chara_tags = json_dict.get("character tags", [])
        characters_tags = []
        for value in chara_tags:
            characters_tags.append(value.replace('_', ' '))
        # chara_result = ", ".join(chara_tags)
        result['character_tags'] = characters_tags
    return result

def model_loader():
    repo_name = "./molmo-7B-D-bnb-4bit"
    arguments = {"device_map": "auto", "torch_dtype": "auto", "trust_remote_code": True}
    
    processor = AutoProcessor.from_pretrained(repo_name, **arguments)

    model = AutoModelForCausalLM.from_pretrained(repo_name, **arguments)
    return processor, model

def image_re_caption(image, processor, model):
    tags_dict = load_tags_from_json(image)
    prompt = f"Here's some accurate tags for this image: {tags_dict['general_tags']}. "
    if 'character_tags' in tags_dict.keys(): # Has character tags
        character = tags_dict['character_tags']
        if len(character) == 1: # ['ganyu (genshin impact)']
            if '(' and ')' in character: # ganyu (genshin impact)
                character_tag = character.split()[0]
                
                series_tag = character[character.find('(')+1:character.find(')')]
                prompt += f"The character in this image is '{character_tag}'. The series is '{series_tag}'. If you know about the series, response this series is game or anime."
            else: # ganyu
                character_tag = character[0]
                prompt += f"The character in this image is '{character_tag}'"
        else: # ['ganyu (genshin impact)', 'slime (genshin impoact)', 'keqing (genshin impact)', 'etc...'] ganyu & keqing vs slime (doge)
            characters_tag = []
            series = set()
            i = 0
            for value in character:
                if '(' and ')' in value: # ['ganyu (genshin impact)', 'slime (genshin impoact)', 'keqing (genshin impact)', 'etc...']
                    character_tag = value.split()[0]
                    series_tag = character[character.find('(')+1:character.find(')')]
                    characters_tag.append(character_tag)
                    series.add(series_tag)
                else: # ['ganyu', 'slime', 'keqing', 'etc...']
                    characters_tag.append(character[i])
                    i += 1
            characters = ", ".join(characters_tag)
            prompt += f"These characters in this image is '{characters}'"
            if len(series) != 0: # Has series tags.
                series_prompt = 'The series is'
                for serial in series: # The set cannot be indexed.
                    series_prompt += f"'{serial}'"
                series_prompt += '. If you know about the series, response this series is game or anime.'
                prompt += series_prompt

    base_prompt = f"Please describe this image based on these tags. The response must include these tags. And should in a word range from 50 to 255. Do not describe anything else. Response the description as sentences (with the following: 'An image of ...'). No need to response as markdown format."
    generated_prompt = prompt + '. ' + base_prompt
    print(f'\nAsk: {generated_prompt}')
    inputs = processor.process(images=[Image.open(image)], text=generated_prompt)
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
    output = model.generate_from_batch(inputs,GenerationConfig(max_new_tokens=512, stop_strings="<|endoftext|>"),tokenizer=processor.tokenizer)
    generated_tokens = output[0, inputs["input_ids"].size(1) :]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return generated_text

def process_image(image_path, processor, model):
    output_file = os.path.join(captions_output_dir, str(os.path.splitext(image_path)[0] + '.txt'))
    result = image_re_caption(image_path, processor, model)
    if os.path.exists(output_file):
        os.remove(output_file)
    with open(output_file, 'w', encoding='UTF-8') as f:
        f.write(result)
    print(f"LLM: {result}") 
    tag_file = os.path.splitext(image_path)[0] + '.json'
    # Remove image and tags to save storage. Commit out it if you want to save.
    os.remove(image_path)
    os.remove(tag_file)

def extract_tar_data(tar_path, extract_path): # Return a list with filenames.
    with tarfile.open(tar_path, 'r') as tar:
        members = tar.getmembers() # Get filenames and return
        tar.extractall(path=extract_path) # Extract
        return [member.name for member in members] # Return filenames in a list

def main(repo_dir):
    image_dir = os.path.join(repo_dir, 'webp')
    tags_dir = os.path.join(repo_dir, 'tags')
    temp_dir = os.path.join(repo_dir, 'temp') # Define a dir to extract (will be removed after task done)
    if not os.path.exists(temp_dir): # mkdir if the folder not exist
        os.makedirs(temp_dir, exist_ok=True)
    try:
        processor, model = model_loader() # Load models

        for i in range(0,999):
            # Extract image tar file to the temp folder
            images = extract_tar_data(os.path.join(image_dir, f'data-{i:04d}.tar'), temp_dir)
            tags = extract_tar_data(os.path.join(tags_dir, f'data-{i:04d}.tar'), temp_dir) # Same as tags, but returned lists cannot be used.

            for img in tqdm(images):
                process_image(img, processor, model)
        
        os.rmdir(temp_dir)
    except Exception as e:
        print('Error when processing:', e)
        os.rmdir(temp_dir)

if __name__ == "__main__":
    captions_output_dir = './NL-captions'
    dataset_path = './Pixiv-2.6M'
    main(dataset_path)
