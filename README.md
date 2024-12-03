# Q: What's this?
A: This is my experiment about using VLMs (via api) to re-caption images based on danbooru tags.

# Q: How to run?
A: Here's some steps to run the experiment:

1. **Download the Model**: 
   - Install ollama
   - Download the model by running `ollama pull (the model you want to download)`
      For example: `ollama pull llama3.2-vision`

2. **Prepare Images and Tags**:
    - Run the script `download_images_and_tags.py` to download images and tags from the danbooru dataset uploaded by DeepGHS.
    - Run the script `convert_tags_to_json.py` to convert the tags to a json format.`

3. **Run the Script**:
     - `python caption_based_on_tag.py`

## Note that the script is still under development. It may not work perfectly yet.