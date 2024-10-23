import json
import os
import tarfile
import random
from tqdm import tqdm

def extract_tar_data(tar_path, extract_path):
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(path=extract_path)
        return [member.name for member in tar.getmembers()]

def process_json_file(json_file_path, characters_data):
    # Get image id from file name (e.g., '114514_p5.json' -> '114514_p5')
    image_id = os.path.splitext(os.path.basename(json_file_path))[0]
    
    # Open JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    general_tags = json_data.get('general tags', [])
    character_tags = json_data.get('character tags', [])
    
    # Ignore images with zero or multiple characters
    if len(character_tags) != 1:
        return
    
    character_name = character_tags[0]
    
    # Initialize character data if not present
    if character_name not in characters_data:
        characters_data[character_name] = {
            'all_images_count': {'count': 0, 'image_names': []}
        }
    
    # Update total images count
    characters_data[character_name]['all_images_count']['count'] += 1
    characters_data[character_name]['all_images_count']['image_names'].append(image_id)
    
    # Update tags
    for tag in general_tags:
        if tag not in characters_data[character_name]:
            characters_data[character_name][tag] = {'count': 0, 'image_names': []}
        
        characters_data[character_name][tag]['count'] += 1
        characters_data[character_name][tag]['image_names'].append(image_id)

def process_character_data(character_name, character_data, output_dir, max_images=None):
    # Get total image count for the character
    total_images = character_data['all_images_count']['count']
    
    # Get list of tags with counts and associated image names
    tags = [
        {
            'tag': tag,
            'count': data['count'],
            'image_names': data['image_names']
        }
        for tag, data in character_data.items()
        if tag != 'all_images_count'
    ]
    
    # Sort tags by count in descending order
    tags_sorted = sorted(tags, key=lambda x: x['count'], reverse=True)
    
    if not tags_sorted:
        return  # No tags to process
    
    highest_count = tags_sorted[0]['count']
    
    # Select tags whose counts are between 0% and 10% of highest count
    selected_tags = []
    for tag_data in tags_sorted:
        percentage = (tag_data['count'] / highest_count) * 100
        if percentage <= 10 or len(selected_tags) < 7:
            selected_tags.append(tag_data)
    
    # Ensure at least 7 tags are selected
    if len(selected_tags) < 7:
        selected_tags = tags_sorted[:7]
    
    # Prepare the final data structure
    output_data = {}
    for tag_data in selected_tags:
        tag = tag_data['tag']
        image_names = tag_data['image_names']
        count = tag_data['count']
        if max_images and len(image_names) > max_images:
            image_names = random.sample(image_names, max_images)
        output_data[tag] = {'count': count, 'image_names': image_names}
    
    # Save to JSON file
    character_output = {
        character_name: output_data
    }
    # Sanitize file name
    sanitized_name = character_name.replace('/', '_').replace(' ', '_').replace('\\', '_')
    output_file = os.path.join(output_dir, f"{sanitized_name}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(character_output, f, ensure_ascii=False, indent=4)

def main():
    characters_data = {}
    max_images = 20  # Set your max_images value here
    output_dir = './output_dir'  # Set your output directory here
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = '/media/playdustindb/HDD-01/A-DiffusionDatasets/pixiv_booru/temp_dir'
    os.makedirs(temp_dir, exist_ok=True)
    
    tags_dir = '/media/playdustindb/HDD-01/A-DiffusionDatasets/pixiv_booru/tags'  # Update this to your tags directory
    tar_files = sorted([
        os.path.join(tags_dir, f) for f in os.listdir(tags_dir)
        if f.endswith('.tar')
    ])
    
    for tar_file in tqdm(tar_files, desc="Processing tar files"):
        # Extract JSON files from the tar archive
        extract_tar_data(tar_file, temp_dir)
        json_files = [
            os.path.join(temp_dir, f) for f in os.listdir(temp_dir)
            if f.endswith('.json')
        ]
        
        # Process each JSON file
        for json_file in json_files:
            process_json_file(json_file, characters_data)
        
        # Clean up temp directory
        for f in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, f))
    
    # Process data for each character
    for character_name, character_data in tqdm(characters_data.items(), desc="Processing characters"):
        process_character_data(character_name, character_data, output_dir, max_images)
    
    # Clean up temp directory
    os.rmdir(temp_dir)
    print("Processing complete.")

if __name__ == "__main__":
    main()