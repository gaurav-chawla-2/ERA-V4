import os
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer
from tqdm import tqdm
from model import SmolLM2Config, SmolLM2ForCausalLM

# Configuration
# Karpathy suggests ~0.5M tokens per batch. 
# 128 seq_len * 4 batch_size = 512 tokens. 
# To get ~0.5M, we need grad_accum ~1000. That might be too slow for this demo.
# Let's aim for a reasonable "large" batch size, e.g., 4096 tokens (32 * 128).
# Batch size 4 * 8 grad_accum = 32 effective batch size.
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 8 
SEQ_LEN = 128
LEARNING_RATE = 6e-4 # Slightly higher max LR for cosine schedule
MIN_LR = 3e-4 # 10% of max LR
WARMUP_STEPS = 100
MAX_STEPS = 5050
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
MODEL_ID = "HuggingFaceTB/cosmo2-tokenizer"

# Precision settings
# MPS supports float16, but bfloat16 is better if available (Ampere+ or M-series chips)
# We'll try to use bfloat16 if on CUDA/Ampere, otherwise float16 or float32.
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=DEVICE.type, dtype=ptdtype) if DEVICE.type != 'cpu' else torch.nullcontext()

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, seq_len):
        if not os.path.exists(file_path):
             script_dir = os.path.dirname(os.path.abspath(__file__))
             file_path = os.path.join(script_dir, file_path)
             
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
    
    for filename, weight in weights.items():
        ds = TextDataset(filename, tokenizer, seq_len)
        if len(ds) > 0:
            datasets.append(ds)
            sample_weights.extend([weight] * len(ds))
    
    if not datasets:
        raise ValueError("No valid datasets found.")
        
    concat_dataset = torch.utils.data.ConcatDataset(datasets)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(concat_dataset), replacement=True)
    
    # Optimization: num_workers and pin_memory
    return DataLoader(concat_dataset, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=True)

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < WARMUP_STEPS:
        return LEARNING_RATE * (it + 1) / (WARMUP_STEPS + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > MAX_STEPS:
        return MIN_LR
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)

def generate_text(model, tokenizer, device, prompt="First Citizen:", max_new_tokens=20):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    model.train()
    return text

def train_stage(stage_name, model, dataloader, optimizer, criterion, device, steps, start_step_offset=0, tokenizer=None, generate_every=500, save_every=5, resume_from_step=0):
    print(f"\n=== {stage_name} ===")
    
    # Adjust for resumption
    start_range = resume_from_step
    if start_range >= steps:
        print(f"Skipping {stage_name} (completed)")
        return

    model.train()
    iterator = iter(dataloader)
    losses = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    progress_bar = tqdm(range(start_range, steps), desc=f"{stage_name}")
    
    for i in progress_bar:
        global_step = start_step_offset + i
        
        # Determine LR for this step
        lr = get_lr(global_step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        # Gradient Accumulation
        accum_loss = 0
        for micro_step in range(GRAD_ACCUM_STEPS):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                batch = next(iterator)
                
            input_ids, target_ids = batch
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            
            # Mixed Precision Forward Pass
            with ctx:
                logits = model(input_ids)
                loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                loss = loss / GRAD_ACCUM_STEPS # Scale loss
            
            # Backward Pass
            loss.backward()
            accum_loss += loss.item()
        
        # Optimizer Step
        optimizer.step()
        optimizer.zero_grad()
        
        losses.append(accum_loss)
        progress_bar.set_postfix({'loss': accum_loss, 'lr': lr})
        
        # Generate
        if tokenizer and (i + 1) % generate_every == 0:
            print(f"\n[{stage_name} | Step {i+1}] Generated: {generate_text(model, tokenizer, device)}")
            
        # Save Checkpoint
        # Save Checkpoint
        if (i + 1) % save_every == 0:
            checkpoint_path = os.path.join(script_dir, "checkpoint_latest.pt")
            save_dict = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'step': global_step + 1,
            }
            torch.save(save_dict, checkpoint_path)

    avg_loss = sum(losses) / len(losses)
    print(f"{stage_name} Average Loss: {avg_loss:.4f}")

def main():
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)
        
    print(f"Using device: {DEVICE}")
    print(f"Precision: {dtype}")
    
    # Initialize Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    except:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize Model
    config = SmolLM2Config()
    config.vocab_size = tokenizer.vocab_size
    model = SmolLM2ForCausalLM(config).to(DEVICE)
    
    # Optimization: torch.compile
    # MPS support for compile is experimental/limited. We'll try it if not on MPS, or skip.
    if DEVICE.type != 'mps': 
        print("Compiling model...")
        try:
            model = torch.compile(model)
            print("Model compiled.")
        except Exception as e:
            print(f"Could not compile model: {e}")
    else:
        print("Skipping torch.compile on MPS (experimental).")

    # Optimization: Fused AdamW
    # Check if fused is supported
    use_fused = 'fused' in optim.AdamW.__init__.__code__.co_varnames
    extra_args = dict(fused=True) if use_fused and DEVICE.type == 'cuda' else dict()
    print(f"Using fused AdamW: {use_fused and DEVICE.type == 'cuda'}")
    
    # Weight Decay: Apply only to 2D parameters (weights), not biases/layernorms
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': 0.1},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    optimizer = optim.AdamW(optim_groups, lr=LEARNING_RATE, betas=(0.9, 0.95), eps=1e-8, **extra_args)
    
    criterion = nn.CrossEntropyLoss()

    GENERATE_EVERY = 500
    SAVE_EVERY = 50

    # Check for checkpoint
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(script_dir, "checkpoint_latest.pt")
    start_global_step = 0
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        
        # Handle legacy checkpoints (state_dict only) vs new dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_global_step = checkpoint['step']
            print(f"Resuming from step {start_global_step}")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded legacy checkpoint (model weights only). Starting from step 0.")

    def get_stage_resume_step(global_resume_step, stage_start_offset, stage_steps):
        # If we are past this stage, return stage_steps (so it skips)
        if global_resume_step >= stage_start_offset + stage_steps:
            return stage_steps
        # If we are before this stage, return 0
        if global_resume_step <= stage_start_offset:
            return 0
        # Check alignment
        local_step = global_resume_step - stage_start_offset
        return local_step

    # --- Stage 1: Foundational Learning ---
    weights_s1 = {'input.txt': 1.0}
    loader_s1 = get_mixed_dataloader(tokenizer, SEQ_LEN, BATCH_SIZE, weights_s1)
    resume_s1 = get_stage_resume_step(start_global_step, 0, 1000)
    train_stage("Stage 1: Foundation", model, loader_s1, optimizer, criterion, DEVICE, 
                steps=1000, start_step_offset=0, tokenizer=tokenizer, generate_every=GENERATE_EVERY, save_every=SAVE_EVERY, resume_from_step=resume_s1)
    
    # --- Stage 2: Intermediate ---
    weights_s2 = {'input.txt': 0.8, 'code.txt': 0.2}
    loader_s2 = get_mixed_dataloader(tokenizer, SEQ_LEN, BATCH_SIZE, weights_s2)
    resume_s2 = get_stage_resume_step(start_global_step, 1000, 1000)
    train_stage("Stage 2: General + Code", model, loader_s2, optimizer, criterion, DEVICE, 
                steps=1000, start_step_offset=1000, tokenizer=tokenizer, generate_every=GENERATE_EVERY, save_every=SAVE_EVERY, resume_from_step=resume_s2)
    
    # --- Stage 3: Specialized ---
    weights_s3 = {'input.txt': 0.6, 'code.txt': 0.2, 'math.txt': 0.2}
    loader_s3 = get_mixed_dataloader(tokenizer, SEQ_LEN, BATCH_SIZE, weights_s3)
    resume_s3 = get_stage_resume_step(start_global_step, 2000, 1000)
    train_stage("Stage 3: Specialized", model, loader_s3, optimizer, criterion, DEVICE, 
                steps=1000, start_step_offset=2000, tokenizer=tokenizer, generate_every=GENERATE_EVERY, save_every=SAVE_EVERY, resume_from_step=resume_s3)
    
    # --- Stage 4: Refinement ---
    weights_s4 = {'input.txt': 0.4, 'code.txt': 0.3, 'math.txt': 0.3}
    loader_s4 = get_mixed_dataloader(tokenizer, SEQ_LEN, BATCH_SIZE, weights_s4)
    resume_s4 = get_stage_resume_step(start_global_step, 3000, 2000)
    train_stage("Stage 4: Refinement", model, loader_s4, optimizer, criterion, DEVICE, 
                steps=2000, start_step_offset=3000, tokenizer=tokenizer, generate_every=GENERATE_EVERY, save_every=SAVE_EVERY, resume_from_step=resume_s4)

    # --- Stage 5: Final Polish ---
    weights_s5 = {'input.txt': 0.4, 'code.txt': 0.3, 'math.txt': 0.3}
    loader_s5 = get_mixed_dataloader(tokenizer, SEQ_LEN, BATCH_SIZE, weights_s5)
    resume_s5 = get_stage_resume_step(start_global_step, 5000, 50)
    train_stage("Stage 5: Final Polish", model, loader_s5, optimizer, criterion, DEVICE, 
                steps=50, start_step_offset=5000, tokenizer=tokenizer, generate_every=GENERATE_EVERY, save_every=SAVE_EVERY, resume_from_step=resume_s5)

    # Save Final Model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(script_dir, "final_smollm2_135m.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nFinal model saved to {checkpoint_path}")

if __name__ == "__main__":
    main()
