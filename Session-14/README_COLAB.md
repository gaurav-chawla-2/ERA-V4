# Running SmolLM2 Training on Google Colab

Follow these steps to train your model using Google Colab's free GPU resources.

## 1. Prepare Files
Ensure your `Session-13` folder contains:
- `train.py`
- `model.py`
- `input.txt`
- `code.txt`
- `math.txt`
- `requirements.txt`

## 2. Upload to Google Drive
1.  Go to your Google Drive.
2.  Upload the entire `Session-13` folder. 
    *   *Recommendation*: Keep the path simple, e.g., `MyDrive/ERA-V4/Session-13`.

## 3. Open Colab and Setup
1.  Open [Google Colab](https://colab.research.google.com/).
2.  Create a **New Notebook**.
3.  **Enable GPU**: Go to `Runtime` > `Change runtime type` > Select **T4 GPU** (or better).

## 4. Run the Following Cells

### Step 1: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 2: Navigate to Folder
Adjust the path if you uploaded it somewhere else.
```python
import os
os.chdir('/content/drive/MyDrive/ERA-V4/Session-13')
print("Current Working Directory:", os.getcwd())
```

### Step 3: Install Dependencies
```python
!pip install -r requirements.txt
```

### Step 4: Run Training
```python
!python train.py
```

## Notes
- **Checkpoints**: The script is configured to save checkpoints in the same directory as `train.py`. Since you mounted Drive, your checkpoints (`checkpoint_latest.pt`, `final_smollm2_135m.pt`) will be saved directly to your Google Drive and persist after the Colab session ends.
- **Speed**: On a T4 GPU, training should be significantly faster than on CPU/MPS.
- **Timeouts**: Colab has idle timeouts. Keep the tab open.
