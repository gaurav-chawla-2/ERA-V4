import torch
from transformers import AutoModelForCausalLM
from model import SmolLM2ForCausalLM, SmolLM2Config

def validate_weights():
    model_id = "HuggingFaceTB/SmolLM2-135M"
    print(f"Downloading weights for {model_id}...")
    
    # Load official model
    try:
        hf_model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        print("Official model downloaded successfully.")
    except Exception as e:
        print(f"Failed to download official model: {e}")
        return

    # Initialize custom model
    config = SmolLM2Config()
    custom_model = SmolLM2ForCausalLM(config)
    
    print("Loading weights into custom model...")
    
    # Map HF keys to custom model keys if necessary
    # The implementation was designed to match, but let's check for common mismatches
    hf_state_dict = hf_model.state_dict()
    custom_state_dict = custom_model.state_dict()
    
    # Create a new state dict for the custom model
    new_state_dict = {}
    
    # Mapping rules (if any needed). 
    # Our implementation uses:
    # model.embed_tokens -> model.embed_tokens
    # model.layers -> model.layers
    # model.norm -> model.norm
    # lm_head -> lm_head
    
    # Check for mismatches
    keys_missing = []
    keys_unexpected = []
    
    for key in custom_state_dict.keys():
        if key not in hf_state_dict:
            # Check if it's a naming difference
            # Example: HF might use 'model.layers.0.self_attn.q_proj.weight'
            # We use the same.
            keys_missing.append(key)
            
    for key in hf_state_dict.keys():
        if key not in custom_state_dict:
            # Ignore masked bias or other buffer/non-param keys if they exist
            if "inv_freq" in key: # Rotary embedding buffers
                continue
            keys_unexpected.append(key)

    if keys_missing:
        print(f"Missing keys in HF state dict: {keys_missing[:5]} ...")
    if keys_unexpected:
        print(f"Unexpected keys in HF state dict: {keys_unexpected[:5]} ...")

    # Attempt to load
    try:
        # We use strict=False to ignore rotary buffers if they are named differently or not persistent
        custom_model.load_state_dict(hf_state_dict, strict=False)
        print("Weights loaded into custom model (strict=False).")
    except Exception as e:
        print(f"Error loading state dict: {e}")
        return

    # Verify output
    print("Verifying output...")
    custom_model.eval()
    hf_model.eval()
    
    print(f"HF Config: {hf_model.config}")
    
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    
    # Hook to capture intermediate outputs
    hf_outputs = {}
    custom_outputs = {}
    
    def get_hf_hook(name):
        def hook(module, input, output):
            hf_outputs[name] = output
        return hook
        
    def get_custom_hook(name):
        def hook(module, input, output):
            custom_outputs[name] = output
        return hook
    
    # Hook embeddings
    hf_model.model.embed_tokens.register_forward_hook(get_hf_hook("embed_tokens"))
    custom_model.model.embed_tokens.register_forward_hook(get_custom_hook("embed_tokens"))
    
    # Hook first layer
    hf_model.model.layers[0].register_forward_hook(get_hf_hook("layer_0"))
    custom_model.model.layers[0].register_forward_hook(get_custom_hook("layer_0"))
    
    with torch.no_grad():
        hf_output = hf_model(input_ids).logits
        custom_output = custom_model(input_ids)
        
    print(f"HF Output shape: {hf_output.shape}")
    print(f"Custom Output shape: {custom_output.shape}")
    
    # Compare embeddings
    if "embed_tokens" in hf_outputs and "embed_tokens" in custom_outputs:
        diff = torch.abs(hf_outputs["embed_tokens"] - custom_outputs["embed_tokens"]).mean()
        print(f"Embeddings mean diff: {diff.item()}")
        
    # Compare layer 0
    if "layer_0" in hf_outputs and "layer_0" in custom_outputs:
        # HF layer output might be a tuple (hidden_states, ...)
        hf_l0 = hf_outputs["layer_0"]
        if isinstance(hf_l0, tuple):
            hf_l0 = hf_l0[0]
        
        diff = torch.abs(hf_l0 - custom_outputs["layer_0"]).mean()
        print(f"Layer 0 mean diff: {diff.item()}")
    
    # Check difference
    diff = torch.abs(hf_output - custom_output).mean()
    print(f"Mean difference in logits: {diff.item()}")
    
    if diff < 1e-4:
        print("SUCCESS: Models match closely!")
    else:
        print("WARNING: Models outputs differ. Check architecture details.")

if __name__ == "__main__":
    validate_weights()
