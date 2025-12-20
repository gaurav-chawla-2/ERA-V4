import os
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer
from tqdm import tqdm
from model import DeepSeekConfig, DeepSeekForCausalLM

# Configuration
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 8 
SEQ_LEN = 128
LEARNING_RATE = 6e-4 
MIN_LR = 6e-5 # 10%
WARMUP_STEPS = 500
MAX_STEPS = 10000 # User requested 10k steps
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
MODEL_ID = "HuggingFaceTB/cosmo2-tokenizer"

# Data Paths (Relative to Session-14)
DATA_DIR = "../Session-13" 
FILES = {
    'input.txt': 1.0,
    'code.txt': 1.0,
    'math.txt': 1.0
}

# Precision
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=DEVICE.type, dtype=ptdtype) if DEVICE.type != 'cpu' else torch.nullcontext()

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, seq_len):
        # Resolve path
        if not os.path.exists(file_path):
             # Try relative
             file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)
             
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Creating empty dataset.")
            self.encodings = tokenizer("", return_tensors='pt')
            self.seq_len = seq_len
            self.total_len = 0
            return

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        self.encodings = tokenizer(text, return_tensors='pt', add_special_tokens=True)
        self.seq_len = seq_len
        self.total_len = self.encodings.input_ids.size(1)

    def __len__(self):
        if self.total_len <= self.seq_len:
            return 0
        return (self.total_len - 1) // self.seq_len

    def __getitem__(self, idx):
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len
        
        if end_idx + 1 > self.total_len:
            end_idx = self.total_len - 1
            
        input_ids = self.encodings.input_ids[0, start_idx:end_idx]
        target_ids = self.encodings.input_ids[0, start_idx+1:end_idx+1]
        
        if input_ids.size(0) < self.seq_len:
            pad_len = self.seq_len - input_ids.size(0)
            input_ids = torch.cat([input_ids, torch.zeros(pad_len, dtype=torch.long)])
            target_ids = torch.cat([target_ids, torch.zeros(pad_len, dtype=torch.long)])
            
        return input_ids, target_ids

def get_mixed_dataloader(tokenizer, seq_len, batch_size, weights):
    datasets = []
    sample_weights = []
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    for filename, weight in weights.items():
        # Construct path to ../Session-13/filename
        # We assume DATA_DIR is explicitly outside
        filepath = os.path.join(script_dir, DATA_DIR, filename)
        
        ds = TextDataset(filepath, tokenizer, seq_len)
        if len(ds) > 0:
            datasets.append(ds)
            sample_weights.extend([weight] * len(ds))
    
    if not datasets:
        print("No valid datasets found. Creating dummy dataset.")
        # Create a tiny dummy dataset to prevent crash if files missing
        dummy_text = "Dummy text " * 1000
        ds = TextDataset("dummy", tokenizer, seq_len) 
        ds.encodings = tokenizer(dummy_text, return_tensors='pt')
        ds.total_len = ds.encodings.input_ids.size(1)
        datasets.append(ds)
        sample_weights.extend([1.0] * len(ds))

    concat_dataset = torch.utils.data.ConcatDataset(datasets)
    sampler = WeightedRandomSampler(sample_weights, num_samples=MAX_STEPS * batch_size * GRAD_ACCUM_STEPS, replacement=True)
    
    return DataLoader(concat_dataset, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=True) # Workers 0 for simple debugging

def get_lr(it):
    if it < WARMUP_STEPS:
        return LEARNING_RATE * (it + 1) / (WARMUP_STEPS + 1)
    if it > MAX_STEPS:
        return MIN_LR
    decay_ratio = (it - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)

def generate_text(model, tokenizer, device, prompt="Once upon a time", max_new_tokens=50):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    model.train()
    return text

def main():
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)
        
    print(f"Using device: {DEVICE}")
    print(f"Precision: {dtype}")

    # Check for resume
    start_step = 0
    checkpoint_path = "checkpoint_latest.pt"

    
    # Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    except:
        print("Falling back to gpt2 tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model
    config = DeepSeekConfig()
    config.vocab_size = tokenizer.vocab_size
    # Adjust config as needed, typically keeping defaults we defined in model.py
    model = DeepSeekForCausalLM(config).to(DEVICE)
    
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Compilation
    if DEVICE.type != 'mps':
        print("Compiling model...")
        try:
             model = torch.compile(model)
             print("Model compiled.")
        except Exception as e:
             print(f"Compilation failed: {e}")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=0.1)
    criterion = nn.CrossEntropyLoss()
    
    # Data
    dataloader = get_mixed_dataloader(tokenizer, SEQ_LEN, BATCH_SIZE, FILES)
    
    # Resume if checkpoint definition exists
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step']
        print(f"Resuming from step {start_step}")
    
    # Training Loop
    model.train()
    iterator = iter(dataloader)
    
    print(f"\nStarting Training for {MAX_STEPS} steps (from {start_step})...")
    
    losses = []
    
    progress_bar = tqdm(range(start_step, MAX_STEPS), desc="Training")
    
    for i in progress_bar:
        # LR Schedule
        lr = get_lr(i)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        accum_loss = 0
        for _ in range(GRAD_ACCUM_STEPS):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                batch = next(iterator)
            
            input_ids, target_ids = batch
            input_ids, target_ids = input_ids.to(DEVICE), target_ids.to(DEVICE)
            
            with ctx:
                logits = model(input_ids)
                loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                loss = loss / GRAD_ACCUM_STEPS
            
            loss.backward()
            accum_loss += loss.item()
            
        optimizer.step()
        optimizer.zero_grad()
        
        losses.append(accum_loss)
        progress_bar.set_postfix({'loss': accum_loss, 'lr': lr})
        
        # Save Checkpoint every 500 steps
        if (i+1) % 500 == 0:
            torch.save({
                'step': i + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': accum_loss,
            }, checkpoint_path)
            # print(f"Saved checkpoint to {checkpoint_path}") # Optional to reduce clutter
        
        if (i+1) % 1000 == 0:
            print(f"\n[Step {i+1}] Avg Loss: {sum(losses[-100:])/100:.4f}")
            print(f"Generation: {generate_text(model, tokenizer, DEVICE)}")

    # Save
    torch.save(model.state_dict(), "deepseek_final.pt")
    print("Model saved to deepseek_final.pt")
    
    # Generate 5 outputs
    print("\n=== Generating 5 Outputs ===")
    prompts = [
        "The future of AI is",
        "Deep learning allows us to",
        "Once upon a time in a digital world,",
        "The quick brown fox",
        "Python is a programming language that"
    ]
    
    for idx, p in enumerate(prompts):
        output = generate_text(model, tokenizer, DEVICE, prompt=p, max_new_tokens=50)
        print(f"Output {idx+1}:\n{output}\n{'-'*20}")

if __name__ == "__main__":
    main()
