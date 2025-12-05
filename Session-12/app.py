import torch
import gradio as gr
from model import GPT, GPTConfig
import tiktoken
import os

# Configuration
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"

print(f"Using device: {device}")

# Load Model
model_path = "final_model.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

# Initialize model with same config as training
config = GPTConfig()
model = GPT(config)

# Load state dict
checkpoint = torch.load(model_path, map_location=device)
# Handle if checkpoint is a full checkpoint dict or just state_dict
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)
model.to(device)
model.eval()

# Tokenizer
enc = tiktoken.get_encoding('gpt2')

def generate_text(start_text, max_new_tokens=100, temperature=0.8, top_k=50):
    if not start_text:
        return "Please enter some text to start."
    
    # Encode input
    start_ids = enc.encode(start_text)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
    
    # Generate
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = x if x.size(1) <= config.block_size else x[:, -config.block_size:]
            
            # Forward pass
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            x = torch.cat((x, idx_next), dim=1)
            
    # Decode output
    output_tokens = x[0].tolist()
    decoded_text = enc.decode(output_tokens)
    return decoded_text

# Gradio Interface
with gr.Blocks(title="GPT Text Generation") as demo:
    gr.Markdown("# GPT Text Generation")
    gr.Markdown("Enter some text and the model will continue it.")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Input Text", placeholder="Once upon a time...", lines=5)
            with gr.Row():
                max_tokens = gr.Slider(minimum=10, maximum=500, value=100, step=10, label="Max New Tokens")
                temperature = gr.Slider(minimum=0.1, maximum=2.0, value=0.8, step=0.1, label="Temperature")
                top_k = gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Top-K")
            generate_btn = gr.Button("Generate", variant="primary")
        
        with gr.Column():
            output_text = gr.Textbox(label="Generated Text", lines=10)
    
    generate_btn.click(
        fn=generate_text,
        inputs=[input_text, max_tokens, temperature, top_k],
        outputs=output_text
    )

if __name__ == "__main__":
    demo.launch()
