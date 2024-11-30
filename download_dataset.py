import os 
import pandas as pd


# If you want to set these environment variables, you have do it here. Else the environment variables will not be used.
os.environ['HF_ENDPOINT'] = 'https://huggingface.co' # Set this to your local endpoint if you want.
os.environ['HF_TOKEN'] = 'hf_xxxxxxxxx' # Token. Set this if you want to download private dataset.

try:
    from cheesechaser.datapool import Danbooru2024SfwDataPool
except ImportError:
    print("cheesechaser not installed, please install it first")



pool = Danbooru2024SfwDataPool() # sfw pool do not require token


if not os.path.exists('metadata.parquet'):
    from huggingface_hub import hf_hub_download
    hf_hub_download(repo_id='deepghs/danbooru2024-sfw', filename='metadata.parquet', repo_type="dataset", local_dir='.', endpoint=os.environ['HF_ENDPOINT'])

df = pd.read_parquet('metadata.parquet', columns=['id'])


ids = set(df.index.tolist())
print(type(ids))
print(f"Total IDs: {len(ids)}")

if not os.path.exists('./images'):
    os.makedirs('./images')


remove_ids = set(int(id.split('.')[0]) for id in os.listdir('./images'))
print(f"Total IDs already downloaded: {len(remove_ids)}")

ids.difference_update(remove_ids)

print(f"Total IDs to download: {len(ids)}")

pool.batch_download_to_directory(resource_ids=ids, dst_dir='./images', max_workers=64, silent=True)
