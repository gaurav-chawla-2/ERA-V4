import torch
import os

# Paths
input_model_path = "../final_smollm2_135m.pt" 
output_model_path = "final_smollm2_135m_half.pt"

print(f"Loading {input_model_path}...")
state_dict = torch.load(input_model_path, map_location="cpu")

print("Converting to float16...")
new_state_dict = {}
for k, v in state_dict.items():
    if isinstance(v, torch.Tensor):
        if v.is_floating_point():
            new_state_dict[k] = v.half()
        else:
            new_state_dict[k] = v
    else:
        new_state_dict[k] = v

print(f"Saving to {output_model_path}...")
torch.save(new_state_dict, output_model_path)

input_size = os.path.getsize(input_model_path) / (1024 * 1024)
output_size = os.path.getsize(output_model_path) / (1024 * 1024)

print(f"Done.")
print(f"Original size: {input_size:.2f} MB")
print(f"Compressed size: {output_size:.2f} MB")
