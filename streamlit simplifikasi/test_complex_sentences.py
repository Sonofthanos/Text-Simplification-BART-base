
import os
import sys
import torch
import warnings
from contextlib import contextmanager

# Suppress warnings
warnings.filterwarnings("ignore")

# Add current directory to path so we can import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Mock streamlit to prevent errors/warnings when importing core
import streamlit
from unittest.mock import MagicMock

# Create a mock for st.cache_resource that just returns the function unchanged
def mock_cache_resource(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

# Apply mocks before importing core
if not hasattr(streamlit, "cache_resource"):
    streamlit.cache_resource = mock_cache_resource
else:
    # If it exists (e.g. if we are running via streamlit), we can leave it or mock it to avoid caching issues in script
    # For a simple script run, let's just mock it to be safe
    streamlit.cache_resource = mock_cache_resource

# Mock other streamlit functions used in core to avoid errors if they are called
streamlit.error = MagicMock()
streamlit.warning = MagicMock()
streamlit.info = MagicMock()

try:
    from core import load_model, simplify_text
except ImportError:
    print("Error: Could not import 'core'. Make sure 'core.py' is in the same directory.")
    sys.exit(1)

def main():
    # 1. Setup Model Path
    model_path = "SimpleBART-base model"
    if not os.path.exists(model_path):
        print(f"Error: Model path '{model_path}' not found.")
        return

    print(f"Loading model from: {model_path}...")
    
    # 2. Load Model
    # Note: load_model returns (model, tokenizer, device, training_config)
    try:
        model, tokenizer, device, training_config = load_model(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    if model is None:
        print("Model failed to load.")
        return

    print("Model loaded successfully!")
    print("-" * 50)

    # 3. Define 20 Complex Sentences
    sentences = [
        "The implementation of the new policy caused significant confusion among employees.",
        "The athlete ran a distance of 100 meters in just 10 seconds.",
        "Furthermore, the results indicate a strong correlation between stress and lack of sleep.",
        "It is imperative that we conserve water during the dry season.",
        "The medication alleviates symptoms associated with the common cold."
        ]

    # 4. Process Each Sentence
    print(f"Starting simplification of {len(sentences)} sentences...\n")

    for i, original_text in enumerate(sentences, 1):
        print(f"Sentence {i}:")
        print(f"Original  : {original_text}")
        
        try:
            # simplify_text returns (simplified_text, inference_time, normalized_input, info)
            simplified_text, inference_time, normalized_input, info = simplify_text(
                original_text, model, tokenizer, device, adaptive=True
            )
            print(f"Simplified: {simplified_text}")
            # Optional: Print metrics if needed
            # print(f"Time: {inference_time:.4f}s | FKGL In: {info.get('input_fkgl',0):.2f} -> Out: {info.get('output_fkgl',0):.2f}")
        except Exception as e:
            print(f"Error simplifying sentence: {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    main()
