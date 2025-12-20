# Session 13: SmolLM2-135M Training

This project implements and trains a 135M parameter language model based on the SmolLM2 architecture (a Llama-style transformer). The model is trained using a 5-stage curriculum learning strategy on a mixed dataset of general text, code, and math examples.

## Model Architecture
- **Type**: Llama-style Decoder-only Transformer
- **Parameters**: ~135M
- **Hidden Size**: 576
- **Intermediate Size**: 1536
- **Heads**: 9 Attention Heads, 3 Key-Value Heads (GQA)
- **Layers**: 30
- **Context Length**: 2048 (RoPE)
- **Vocab Size**: 49152 (Cosmo2 Tokenizer)

## Training Configuration
- **Optimizer**: AdamW (Fused if available)
- **Learning Rate**: Max 6e-4, Min 3e-4 (Cosine Decay with Warmup)
- **Batch Size**: 32 (effective) = 4 per gpu * 8 grad accum steps
- **Sequence Length**: 128 (for training speed)
- **Total Steps**: 5050

## Dataset Strategy (Curriculum Learning)
The model was trained in 5 distinct stages, gradually shifting the data distribution from general text to more specialized domains (code, math).

| Stage | Name | Steps | Data Mix (Weights) |
|-------|------|-------|--------------------|
| 1 | Foundation | 0-1000 | `input.txt` (1.0) |
| 2 | General + Code | 1000-2000 | `input.txt` (0.8), `code.txt` (0.2) |
| 3 | Specialized | 2000-3000 | `input.txt` (0.6), `code.txt` (0.2), `math.txt` (0.2) |
| 4 | Refinement | 3000-5000 | `input.txt` (0.4), `code.txt` (0.3), `math.txt` (0.3) |
| 5 | Final Polish | 5000-5050 | `input.txt` (0.4), `code.txt` (0.3), `math.txt` (0.3) |

## Training Results

### Training Log Summary
Below is a summary of the training log captured during the session.

- **Stage 1 (Foundation)**: Loss started high and converged significantly as the model learned basic language structure.
- **Stage 2 (General + Code)**: Introduction of code data. Initial spike in loss followed by adaptation.
- **Stage 3 (Specialized)**: Math data added. Further adaptation observed.
- **Stage 4 (Refinement)**: Longest training phase to consolidate knowledge across all domains.
- **Final Loss**: The model achieved a low stable loss, indicating successful convergence on the mixed dataset.

*(Note: Detailed loss curves and generation samples can be found in `training.log`)*

### Sample Generations
During training, the model generated samples every 500 steps. 
*(See `training.log` for specific text outputs, e.g., "First Citizen:", code snippets, and math problem attempts)*

## Hugging Face App
The trained model has been deployed to a Hugging Face Space using Gradio.

- **App Folder**: `smollm2-135m/`
- **Model File**: `final_smollm2_135m_half.pt` (FP16 Compressed)
- **Deployment**: The app uses `gradio` to provide a text generation interface.
- **Features**:
    - Automatic FP16 model loading.
    - Adjustable generation parameters (max tokens).
    - Queue management for stable inference.

## Usage
To run the training locally:
```bash
pip install -r requirements.txt
python train.py
```

To run the app locally:
```bash
cd smollm2-135m
pip install -r requirements.txt
python app.py
```
