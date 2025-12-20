---
title: SmolLM2 135M
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

# SmolLM2 Gradio App

This is a Hugging Face Gradio application for the SmolLM2 model trained in Session 13.

## Deployment

1. Create a new Space on Hugging Face (Select SDK: Gradio).
2. Upload the contents of this folder to the Space.
   - You can use `git` to push these files.
   - Or upload manually via the storage interface.
3. The `app.py` script loads `final_smollm2_135m_half.pt` (FP16 precision).

## Model

The model is a SmolLM2-135M dense transformer model.
The Checkpoint is stored in FP16 precision to reduce size (~310MB).

## Local Run

```bash
pip install -r requirements.txt
python app.py
```
