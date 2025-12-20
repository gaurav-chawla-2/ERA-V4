import torch
from model import SmolLM2Config, SmolLM2ForCausalLM

def test_model():
    config = SmolLM2Config()
    model = SmolLM2ForCausalLM(config)
    
    print("Model Configuration:")
    print(config)
    
    # Calculate parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal Parameters: {total_params:,}")
    
    # Expected approx 135M
    assert 130_000_000 < total_params < 140_000_000, f"Parameter count {total_params} not in expected range for 135M model"
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"\nRunning forward pass with input shape: {input_ids.shape}")
    logits = model(input_ids)
    
    print(f"Output logits shape: {logits.shape}")
    
    assert logits.shape == (batch_size, seq_len, config.vocab_size), "Output shape mismatch"
    
    print("\nTest Passed!")

if __name__ == "__main__":
    test_model()
