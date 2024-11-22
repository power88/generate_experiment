# Q: What's this?
A: This is my experiment about using VLMs (via api) to recaption images based on tags.

# Q: How to run?
A: Here's some steps to run the experiment:

1. **Download the Model**: 
   - Install ollama
   - Download the model by running `ollama pull minicpm-v`


3. **Prepare Images and Tags**:
   - Download dataset [url](https://huggingface.co/datasets/deepghs/danbooru2024-webp-4Mpixel)
   - Set `dataset_path` in the end of the python file `caption_based_on_tag.py`

4. **Run the Script**:
   - Open Windows Terminal or Linux Terminal (tested and works great on Linux Terminal).
   - Activate the virtual environment:
     - On Windows Terminal: `.\venv\Scripts\activate`
     - On Linux Terminal: `./venv/Scripts/activate`
   - Run the script:
     - `python caption_based_on_tag.py`

# Note that the script is still under development. It may not work perfectly yet.