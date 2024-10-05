# Q: What's this?
A: This is my experiment about using VLM (molmo-7b) to recapture images based on tags.

# Q: How to run?
A: Follow these steps to run the experiment:

1. **Download the Model**: 
   - Download the model from [here](https://huggingface.co/cyan2k/molmo-7B-D-bnb-4bit) (This is the model I used) and place it in this repository.

2. **Install Dependencies**:
   - Follow the installation instructions in the [README](https://github.com/cyan2k/molmo-7b-bnb-4bit) of this repository.

3. **Prepare Images and Tags**:
   - Download dataset [here](https://huggingface.co/datasets/Amber-River/Pixiv-2.6M)
   - Set `dataset_path` in the end of the python file `caption_based_on_tag.py`

4. **Run the Script**:
   - Open Windows Terminal or Linux Terminal (tested and works great on Linux Terminal).
   - Activate the virtual environment:
     - On Windows Terminal: `.\venv\Scripts\activate`
     - On Linux Terminal: `./venv/Scripts/activate`
   - Run the script:
     - `python caption_based_on_tag.py`