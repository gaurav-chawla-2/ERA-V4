import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from model import SmolLM2Config, SmolLM2ForCausalLM

# Configuration
BATCH_SIZE = 4
SEQ_LEN = 128
LEARNING_RATE = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
MODEL_ID = "HuggingFaceTB/cosmo2-tokenizer"

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, seq_len):
        # Handle path relative to script location if needed, or use provided path
        if not os.path.exists(file_path):
             # Try relative to this script
             script_dir = os.path.dirname(os.path.abspath(__file__))
             file_path = os.path.join(script_dir, "input.txt")
             
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

def generate_text(model, tokenizer, device, prompt="First Citizen:", max_new_tokens=20):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    model.train()
    return text

def train_phase(model, dataloader, optimizer, criterion, device, steps, tokenizer=None, generate_every=None):
    model.train()
    iterator = iter(dataloader)
    losses = []
    
    progress_bar = tqdm(range(steps), desc="Training Steps")
    for i in progress_bar:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)
            
        input_ids, target_ids = batch
        input_ids, target_ids = input_ids.to(device), target_ids.to(device)
        
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        progress_bar.set_postfix({'loss': loss.item()})
        
        if generate_every and (i + 1) % generate_every == 0 and tokenizer:
            print(f"\n[Step {i+1}] Generated: {generate_text(model, tokenizer, device)}")

    return sum(losses) / len(losses)

def train_epochs(model, dataloader, optimizer, criterion, device, epochs, tokenizer=None):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            input_ids, target_ids = batch
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
        if tokenizer:
            print(f"\n[Epoch {epoch+1}] Generated: {generate_text(model, tokenizer, device)}")

def main():
    print(f"Using device: {DEVICE}")
    
    # Initialize Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    except:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    dataset = TextDataset("Session-13/input.txt", tokenizer, SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # --- Phase 1: Train 5 steps, predict every step ---
    print("\n=== Phase 1: Initial Training (5 steps) ===")
    config = SmolLM2Config()
    config.vocab_size = tokenizer.vocab_size
    model = SmolLM2ForCausalLM(config).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    train_phase(model, dataloader, optimizer, criterion, DEVICE, steps=5, tokenizer=tokenizer, generate_every=1)
    
    # Save Checkpoint
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(script_dir, "checkpoint_step_5.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nCheckpoint saved to {checkpoint_path}")
    
    # Stop model (delete)
    del model
    del optimizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("Model stopped and memory cleared.")
    
    # --- Phase 2: Load and Train 50 steps ---
    print("\n=== Phase 2: Resume Training (50 steps) ===")
    
    # Re-init
    model = SmolLM2ForCausalLM(config).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    print(f"Loaded checkpoint from {checkpoint_path}")
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    train_phase(model, dataloader, optimizer, criterion, DEVICE, steps=50, tokenizer=tokenizer, generate_every=10)
    
    # --- Phase 3: Train 5 Epochs ---
    print("\n=== Phase 3: Train 5 Epochs ===")
    train_epochs(model, dataloader, optimizer, criterion, DEVICE, epochs=5, tokenizer=tokenizer)
    
    print("\nTraining Complete.")

if __name__ == "__main__":
    main()
