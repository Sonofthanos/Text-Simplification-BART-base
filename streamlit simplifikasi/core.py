# core.py
import re
import time
import json
from pathlib import Path

import streamlit as st
import torch
import textstat  # <--- NEW IMPORT
from transformers import BartTokenizer, AutoModelForSeq2SeqLM
from sacrebleu import corpus_bleu
from easse.sari import corpus_sari


class TextPreprocessor:
    """Text preprocessing supaya konsisten dengan pipeline training."""

    PTB_REPLACEMENTS = [
        (r'\b-?lrb-?\b', '('),
        (r'\b-?rrb-?\b', ')'),
        (r'\b-?lsb-?\b', '['),
        (r'\b-?rsb-?\b', ']'),
        (r'\b-?lcb-?\b', '{'),
        (r'\b-?rcb-?\b', '}'),
        (r'\s+-lrb-\s+', ' ('),
        (r'\s+-rrb-\s+', ') '),
        (r'\s+lrb\s+', ' ('),
        (r'\s+rrb\s+', ') '),
        (r'^-?lrb-?\s*', ''),
        (r'\s*-?rrb-?$', ''),
    ]

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalisasi teks (PTB tokens, whitespace, tanda baca akhir)."""
        if not text:
            return ""

        text = text.strip()

        # Normalisasi PTB tokens (case-insensitive)
        for pattern, replacement in TextPreprocessor.PTB_REPLACEMENTS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Rapikan whitespace
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s([,.!?;:])", r"\1", text)

        # Pastikan ada tanda baca di akhir
        if text and text[-1] not in ".!?":
            text += "."

        return text


@st.cache_resource
def load_model(model_path: str):
    """
    Load model dan tokenizer dari folder model_path.
    Dik-cache oleh Streamlit supaya tidak load ulang tiap interaksi.
    """
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = BartTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        model.to(device)
        model.eval()

        # Baca training_config.json jika ada
        config_path = Path(model_path) / "training_config.json"
        training_config = None
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                training_config = json.load(f)

        return model, tokenizer, device, training_config

    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, None, None


def _select_adaptive_params(input_len: int):
    """
    Menentukan parameter generasi secara adaptif berdasarkan panjang input.
    Logikanya mengikuti simplify_adaptive di notebook.
    """
    # Perkiraan kasar: 1 kata ≈ 1.2 token BPE
    approx_tokens = int(input_len * 1.2)

    if input_len < 12:
        # KALIMAT PENDEK: jangan terlalu dikompresi
        max_len = max(25, int(approx_tokens * 1.2))   # boleh sedikit lebih panjang
        min_len = max(6, int(approx_tokens * 0.9))    # minimal ~90% panjang awal
        length_pen = 1.0                               # netral
        num_beams = 6
        category = "SHORT"
        emoji = "🔹"

    elif input_len < 25:
        # KALIMAT SEDANG: kompresi moderat (target 70–90%)
        max_len = max(40, int(approx_tokens * 1.3))
        min_len = max(12, int(approx_tokens * 0.75))  # jangan kurang dari ~75%
        length_pen = 0.9                               # hindari terlalu pendek
        num_beams = 6
        category = "MEDIUM"
        emoji = "🔸"

    else:
        # KALIMAT PANJANG: jaga informasi (kompresi ringan 70–100%)
        max_len = max(60, int(approx_tokens * 1.4))
        min_len = max(18, int(approx_tokens * 0.8))   # minimal ~80%
        length_pen = 0.9
        num_beams = 6
        category = "LONG"
        emoji = "🔶"

    params = {
        "max_length": max_len,
        "min_length": min_len,
        "length_penalty": length_pen,
        "num_beams": num_beams,
        "no_repeat_ngram_size": 2,
        "category": category,
        "emoji": emoji,
        "input_len": input_len,
        "approx_tokens": approx_tokens,
    }
    return params


def simplify_text(
    text: str,
    model,
    tokenizer,
    device,
    adaptive: bool = True,
    max_source_length: int = 128,
    num_beams: int = 5,
    max_gen_length: int = 80,
    min_gen_length: int = 10,
    no_repeat_ngram_size: int = 3,
    length_penalty: float = 1.0,
):
    """
    Simplifikasi satu kalimat + Hitung FKGL.
    """
    # Preprocessing
    normalized_text = TextPreprocessor.normalize_text(text)
    input_len = len(normalized_text.split())

    if adaptive:
        adapt = _select_adaptive_params(input_len)
        num_beams = adapt["num_beams"]
        max_gen_length = adapt["max_length"]
        min_gen_length = adapt["min_length"]
        no_repeat_ngram_size = adapt["no_repeat_ngram_size"]
        length_penalty = adapt["length_penalty"]
    else:
        adapt = {
            "category": "FIXED",
            "emoji": "⚙️",
            "input_len": input_len,
            "approx_tokens": int(input_len * 1.2),
            "max_length": max_gen_length,
            "min_length": min_gen_length,
            "length_penalty": length_penalty,
            "num_beams": num_beams,
            "no_repeat_ngram_size": no_repeat_ngram_size,
        }

    # Tokenisasi source
    inputs = tokenizer(
        normalized_text,
        max_length=max_source_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    # Generate
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_gen_length,
            min_length=min_gen_length,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty,
            early_stopping=True,
        )
    inference_time = time.time() - start_time

    # Decode
    simplified_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # --- HITUNG METRIK OUTPUT ---
    out_len = len(simplified_text.split())
    adapt["output_len"] = out_len
    adapt["compression_ratio"] = (out_len / input_len) if input_len > 0 else 0.0

    # --- HITUNG FKGL (Input vs Output) ---
    # Flesch-Kincaid Grade Level: Menunjukkan tingkat kelas pendidikan (misal 8.0 = kelas 8 SMP)
    try:
        input_fkgl = textstat.flesch_kincaid_grade(normalized_text)
        output_fkgl = textstat.flesch_kincaid_grade(simplified_text)
    except Exception:
        input_fkgl, output_fkgl = 0.0, 0.0
    
    adapt["input_fkgl"] = input_fkgl
    adapt["output_fkgl"] = output_fkgl

    return simplified_text, inference_time,normalized_text, adapt


def compute_metrics_single(source: str, prediction: str, reference: str):
    """Hitung SARI dan SacreBLEU untuk satu pasangan kalimat."""
    try:
        sari_score = corpus_sari(
            [source],
            [prediction],
            [[reference]],
        )
    except Exception as e:
        st.warning(f"⚠️ SARI calculation error: {str(e)}")
        sari_score = 0.0

    try:
        bleu_score = corpus_bleu(
            [prediction],
            [[reference]],
        ).score
    except Exception as e:
        st.warning(f"⚠️ BLEU calculation error: {str(e)}")
        bleu_score = 0.0

    return {
        "sari": round(float(sari_score), 2),
        "bleu": round(float(bleu_score), 2),
    }