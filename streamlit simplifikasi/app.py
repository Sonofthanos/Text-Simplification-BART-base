import streamlit as st
import torch
import os

from core import load_model
from ui import show_simplification_page

# Config wajib di paling awal
st.set_page_config(
    page_title="BART Text Simplification",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    # Sidebar: model path & device
    with st.sidebar:
        st.header("Settings")
        
        # Default path check
        default_path = "SimpleBART-base model"
        if not os.path.exists(default_path):
             st.warning(f"Default model path '{default_path}' not found.")
        
        model_path = st.text_input(
            "Model Path",
            value=default_path,
            help="Path to folder containing config.json, model.safetensors, etc.",
        )

        device_name = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        st.caption(f"Running on: {device_name}")
        st.markdown("---")
        st.markdown("### About")
        st.info("Fine-tuned BART model trained on WikiLarge for sentence simplification.")

    # Load model (Cached)
    model, tokenizer, device, training_config = load_model(model_path)

    if model is None:
        st.error(f"Failed to load model from `{model_path}`.")
        st.info("Please check the path and ensure it contains the necessary files.")
        return

    # Call the UI Function
    show_simplification_page(model, tokenizer, device)


if __name__ == "__main__":
    main()