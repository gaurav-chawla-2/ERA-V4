import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

try:
    from app import generate_text
    print("Successfully imported generate_text from app.py")
    
    text = "The quick brown fox"
    print(f"Generating text for input: '{text}'")
    
    generated = generate_text(text, max_new_tokens=20)
    print(f"Generated text: {generated}")
    
    if len(generated) > len(text):
        print("Verification SUCCESS: Text generation works.")
    else:
        print("Verification FAILED: No text generated.")
        sys.exit(1)
        
except ImportError as e:
    print(f"ImportError: {e}")
    print("Please ensure requirements are installed.")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)
