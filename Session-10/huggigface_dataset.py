
# echo 'export HF_HOME=/hf_home; export HF_HUB_CACHE=/lambda/nfs/ERAv4S09/hf_home/hub; export HF_DATASETS_CACHE=/lambda/nfs/ERAv4S09/hf_home/datasets; export TMPDIR=/lambda/nfs/ERAv4S09/imagenet/tmp' >> ~/.bashrc && source ~/.bashrc


import os

# Set HF caches early (before imports, redirects all blob/temp to /data)
os.environ['HF_HOME'] = '/lambda/nfs/ERAv4S09/hf_home'
os.environ['HF_HUB_CACHE'] = '/lambda/nfs/ERAv4S09/hf_home/hub'
os.environ['HF_DATASETS_CACHE'] = '/lambda/nfs/ERAv4S09/hf_home/datasets'
os.environ['TRANSFORMERS_CACHE'] = '/lambda/nfs/ERAv4S09/hf_home/transformers'
os.environ['TMPDIR'] = '/lambda/nfs/ERAv4S09/imagenet/tmp'
# Create dirs if missing
os.makedirs('/lambda/nfs/ERAv4S09/hf_home/hub', exist_ok=True)
os.makedirs('/lambda/nfs/ERAv4S09/hf_home/datasets', exist_ok=True)
os.makedirs('/lambda/nfs/ERAv4S09/hf_home/transformers', exist_ok=True)
os.makedirs('/lambda/nfs/ERAv4S09/imagenet/tmp', exist_ok=True)

from datasets import load_dataset

# Load with token and custom cache_dir (resumes ~140 GB from shard 13/294, blobs to hub cache on /data)
ds = load_dataset(
    "ILSVRC/imagenet-1k",
    token=os.getenv("HF_TOKEN"),
    cache_dir='/lambda/nfs/ERAv4S09/hf_home/datasets',
    download_mode='reuse_dataset_if_exists'  # Resumes partials
)

print(f"Train: {len(ds['train'])} images")
print(f"Validation: {len(ds['validation'])} images")
print(f"Test: {len(ds['test'])} images")

# Save processed to /data
ds.save_to_disk("/lambda/nfs/ERAv4S09/imagenet/full_dataset")
print("Dataset saved to /lambda/nfs/ERAv4S09/imagenet/full_dataset")


