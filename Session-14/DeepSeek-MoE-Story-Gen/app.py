import torch
import gradio as gr
from transformers import AutoTokenizer
from model import DeepSeekConfig, DeepSeekForCausalLM

# Constants
MODEL_PATH = "deepseek_final.pt"
TOKENIZER_ID = "HuggingFaceTB/cosmo2-tokenizer"
MAX_LENGTH = 512

# Check for GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load Model
def load_model():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print("Loading model config and weights...")
    config = DeepSeekConfig()
    config.vocab_size = tokenizer.vocab_size
    
    model = DeepSeekForCausalLM(config)
    
    # Load weights
    try:
        # Load directly
        state_dict = torch.load(MODEL_PATH, map_location=device)
        
        # Fix keys if they were saved from a compiled model
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("_orig_mod.", "")
            new_state_dict[new_key] = v
        state_dict = new_state_dict
        
        model.load_state_dict(state_dict)
        
        # If on CPU and weights are half, cast to float for compatibility/speed
        if device == "cpu" and any(p.dtype == torch.float16 for p in model.parameters()):
            print("Converting model to float32 for CPU execution...")
            model.float()
            
    except FileNotFoundError:
        print(f"Error: {MODEL_PATH} not found. Please ensure the model file is in the same directory.")
        return None, None, None
        
    model.to(device)
    model.eval()
    return model, tokenizer, config

model, tokenizer, config = load_model()

def generate(prompt, max_new_tokens=100, temperature=0.7, top_k=50):
    if model is None:
        return "Error: Model not loaded."
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Generation loop
    idx = input_ids
    for _ in range(max_new_tokens):
        # Crop context if needed
        idx_cond = idx[:, -config.max_position_embeddings:]
        
        with torch.no_grad():
            logits = model(idx_cond)
        
        # Focus on last token
        logits = logits[:, -1, :] 
        
        # Apply Temperature
        logits = logits / temperature
        
        # Apply Top-K
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        # Sample
        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # Append
        idx = torch.cat((idx, idx_next), dim=1)
        
        # Stop if EOS (optional, depending on training)
        if idx_next.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(idx[0], skip_special_tokens=True)

# Gradio Interface
with gr.Blocks(title="DeepSeek MoE Story Generator") as demo:
    gr.Markdown("# DeepSeek MoE Story Generator")
    gr.Markdown("A custom Mixture-of-Experts language model trained on Smollm2 dataset.")
    
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(label="Prompt", value="Once upon a time", lines=3)
            with gr.Row():
                max_tokens = gr.Slider(minimum=10, maximum=500, value=100, step=10, label="Max New Tokens")
                temperature = gr.Slider(minimum=0.1, maximum=2.0, value=0.7, label="Temperature")
                top_k = gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Top-K")
            generate_btn = gr.Button("Generate", variant="primary")
            
        with gr.Column():
            output_text = gr.Textbox(label="Generated Text", lines=10)
            
    generate_btn.click(
        fn=generate,
        inputs=[prompt_input, max_tokens, temperature, top_k],
        outputs=output_text
    )

if __name__ == "__main__":
    demo.launch()
