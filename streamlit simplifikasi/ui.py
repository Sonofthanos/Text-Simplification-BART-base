import streamlit as st
from typing import Optional, Dict, Any

from core import simplify_text, compute_metrics_single


def get_custom_css():
    """Returns custom CSS for a cleaner UI."""
    return """
    <style>
        /* Global font and padding adjustments */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        
        /* Typography */
        h1, h2, h3 {
            font-family: 'Inter', sans-serif;
            font-weight: 600;
        }
        
        /* Text Area styling */
        .stTextArea textarea {
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            padding: 10px;
            font-size: 16px;
            line-height: 1.6;
            box-shadow: none;
        }
        .stTextArea textarea:focus {
            border-color: #ff4b4b; /* Streamlit primary color */
            box-shadow: 0 0 0 1px #ff4b4b;
        }

        /* Metrics styling */
        div[data-testid="stMetric"] {
            background-color: #f0f2f6; /* Fallback */
            background-color: var(--secondary-background-color);
            padding: 15px;
            border-radius: 10px;
            border: 1px solid rgba(128, 128, 128, 0.1);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            text-align: center;
        }
        div[data-testid="stMetricValue"] {
            font-size: 24px;
            font-weight: 700;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 14px;
            color: #666;
        }

        /* Button styling */
        .stButton button {
            border-radius: 8px;
            font-weight: 600;
            padding: 0.5rem 1rem;
            width: 100%;
        }

        /* Expander styling */
        .streamlit-expanderHeader {
            font-weight: 500;
            border-radius: 8px;
            background-color: #f0f2f6;
        }
        
        /* Hide default header/footer */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
    """


def show_simplification_page(model, tokenizer, device):
    """Main page with a cleaner, two-column layout."""
    
    # Inject Custom CSS
    st.markdown(get_custom_css(), unsafe_allow_html=True)

    st.title("Text Simplification")
    st.markdown("---")

    # Main Layout: 2 Columns
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("Input")
        input_text = st.text_area(
            "Complete Text",
            key="input_text",
            height=300,
            placeholder="Paste your complex text here...",
            label_visibility="collapsed",
        )
        
        # Advanced options in an expander to keep UI clean
        with st.expander("Advanced Options & Reference", expanded=False):
            reference_text = st.text_area(
                "Reference Text (Optional)",
                key="reference_text",
                height=100,
                placeholder="Paste simplified reference text for evaluation...",
                help="Used to calculate SARI and BLEU scores."
            )

        simplify_btn = st.button("Simplify Text", type="primary", use_container_width=True)

    with col_right:
        st.subheader("Output")
        
        # Placeholder for output or persistent state
        if 'simplified_text' not in st.session_state:
            st.info("generated text will appear here.")
            output_container = st.empty()
        else:
            # We will fill this later
            pass

    # Processing Logic
    if simplify_btn:
        if not input_text.strip():
            with col_left:
                st.warning("⚠️ Please provide input text.")
            return

        with col_right:
            with st.spinner("Processing..."):
                simplified_text, inference_time, normalized_input, info = simplify_text(
                    input_text, model, tokenizer, device, adaptive=True
                )
                
                # Update session state
                st.session_state['simplified_text'] = simplified_text
                st.session_state['inference_info'] = info
                st.session_state['inference_time'] = inference_time
                st.session_state['normalized_input'] = normalized_input

    # Display Results if available
    if 'simplified_text' in st.session_state:
        simplified_val = st.session_state['simplified_text']
        info_val = st.session_state['inference_info']
        time_val = st.session_state['inference_time']
        norm_input_val = st.session_state['normalized_input']

        # Show Output Text in Right Column
        with col_right:
            st.text_area(
                "Simplified Output",
                value=simplified_val,
                height=300,
                label_visibility="collapsed",
                disabled=False # Allow copying
            )
            
            # Action buttons for output? Copy button is built-in to code blocks but text_area has it too now?
            # Actually st.code allows easy copy, but text_area is editable. Keep text_area.

        # Metrics Section below columns
        st.markdown("### Analysis")
        
        # Metrics setup
        input_fkgl = info_val.get('input_fkgl', 0.0)
        output_fkgl = info_val.get('output_fkgl', 0.0)
        input_len = info_val.get('input_len', 0)
        output_len = info_val.get('output_len', 0)
        
        # Layout: 4 Columns
        # Check for Reference
        ref_input = st.session_state.get('reference_text', "").strip()
        
        # Determine columns based on mode
        if ref_input:
            cols = st.columns(5)
            c1, c2, c3, c4, c5 = cols
        else:
            cols = st.columns(4)
            c1, c2, c3, c4 = cols

        with c1:
            st.metric(
                label="Readability (FKGL)",
                value=f"{output_fkgl:.1f}",
                delta=f"{output_fkgl - input_fkgl:.1f}",
                delta_color="inverse",
                help="Flesch-Kincaid Grade Level. Lower is easier to read."
            )
            
        with c2:
            st.metric(
                label="Word Count",
                value=f"{output_len}",
                delta=f"{output_len - input_len}",
                delta_color="inverse",
                help="Number of words."
            )

        with c3:
             st.metric(
                label="Inference Time",
                value=f"{time_val:.3f}s",
                help="Time taken to simplify text."
            )

        if ref_input:
            metrics = compute_metrics_single(
                source=norm_input_val,
                prediction=simplified_val,
                reference=ref_input,
            )
            with c4:
                st.metric("SARI Score", f"{metrics.get('sari', 0.0):.1f}")
            with c5:
                st.metric("BLEU Score", f"{metrics.get('bleu', 0.0):.1f}")
        else:
            with c4:
                st.metric(
                    "Compression", 
                    f"{info_val.get('compression_ratio', 0.0):.2f}x",
                    help="Ratio of Output length / Input length"
                )
