import torch
from model import DeepSeekConfig, DeepSeekForCausalLM

MODEL_PATH = "deepseek_final.pt"
OUTPUT_PATH = "DeepSeek-MoE-Story-Gen/deepseek_final.pt" # Overwriting the one in the app folder

print(f"Loading {MODEL_PATH}...")
# Load to CPU to avoid VRAM issues
state_dict = torch.load(MODEL_PATH, map_location="cpu")

print("Converting to Float16...")
# Convert every tensor in the state dict to float16
new_state_dict = {}
for k, v in state_dict.items():
    if torch.is_floating_point(v):
        new_state_dict[k] = v.half()
    else:
        new_state_dict[k] = v

print(f"Saving to {OUTPUT_PATH}...")
torch.save(new_state_dict, OUTPUT_PATH)

print("Done! File size should be ~50% of original.")
