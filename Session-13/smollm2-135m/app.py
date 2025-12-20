import os
import zipfile
import torch
from transformers import AutoTokenizer
import gradio as gr
from model import SmolLM2Config, SmolLM2ForCausalLM

# Constants
MODEL_PATH = "final_smollm2_135m_half.pt"
TOKENIZER_ID = "HuggingFaceTB/cosmo2-tokenizer"

def load_model():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
         device = "mps"
    
    print(f"Using device: {device}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    except Exception as e:
        print(f"Failed to load tokenizer {TOKENIZER_ID}: {e}")
        print("Falling back to gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    config = SmolLM2Config()
    try:
        config.vocab_size = tokenizer.vocab_size
    except:
        pass # use default
        
    print(f"Initializing model with vocab_size={config.vocab_size}")
    model = SmolLM2ForCausalLM(config)
    
    # Load state dict
    if os.path.exists(MODEL_PATH):
        print(f"Loading state dict from {MODEL_PATH}")
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"Model file {MODEL_PATH} not found. Please ensure it is uploaded (use Git LFS for large files).")

    model.to(device)
    model.eval()
    
    return model, tokenizer, device

# Load global model
print("Loading model...")
model, tokenizer, device = load_model()
print("Model loaded.")

def generate_text(prompt, max_new_tokens):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=int(max_new_tokens))
    
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# SmolLM2 Text Generator")
    gr.Markdown("Generate text using a trained SmolLM2-135M model.")
    
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(label="Prompt", value="Once upon a time", lines=3)
            max_tokens_input = gr.Slider(minimum=10, maximum=500, value=100, step=10, label="Max New Tokens")
            submit_btn = gr.Button("Generate")
        
        with gr.Column():
            output_text = gr.Textbox(label="Generated Text", lines=10)
            
    submit_btn.click(
        fn=generate_text,
        inputs=[prompt_input, max_tokens_input],
        outputs=output_text
    )

if __name__ == "__main__":
    demo.launch()
