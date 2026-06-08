"""
Script untuk evaluasi manual model BART Text Simplification
Jalankan: python evaluate_model.py
"""

import torch
import json
from pathlib import Path
from transformers import BartTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from sacrebleu import corpus_bleu
from easse.sari import corpus_sari
from tqdm import tqdm
import re

model_path = "SimpleBART-base model"

class TextPreprocessor:
    """Text preprocessing sesuai dengan training"""
    
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
        """Normalisasi teks sesuai preprocessing saat training"""
        if not text:
            return ""
        
        text = text.strip()
        
        # PTB tokens (case-insensitive)
        for pattern, replacement in TextPreprocessor.PTB_REPLACEMENTS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s([,.!?;:])', r'\1', text)
        
        # Ensure ending punctuation
        if text and text[-1] not in '.!?':
            text += '.'
        
        return text

def evaluate_model(model_path, num_samples=None):
    """
    Evaluasi model pada test set
    
    Args:
        model_path: Path ke folder model
        num_samples: Jumlah sample untuk evaluasi (None = semua)
    """
    
    print("="*70)
    print("🚀 BART Text Simplification - Model Evaluation")
    print("="*70)
    
    # Load model
    print(f"\n📦 Loading model from: {model_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")
    
    tokenizer = BartTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully!")
    
    # Load test dataset
    print("\n📥 Loading WikiLarge test dataset...")
    dataset = load_dataset("bogdancazan/wikilarge-text-simplification")
    test_data = dataset['test']
    
    if num_samples:
        test_data = test_data.select(range(min(num_samples, len(test_data))))
    
    print(f"📊 Evaluating on {len(test_data)} samples")
    
    # Prepare data
    sources = []
    references = []
    predictions = []
    
    print("\n🔄 Generating predictions...")
    for example in tqdm(test_data):
        # Get source and target
        source = example.get('Normal', example.get('source', ''))
        target = example.get('Simple', example.get('target', ''))
        
        # Preprocess
        source = TextPreprocessor.normalize_text(source)
        target = TextPreprocessor.normalize_text(target)
        
        # Skip invalid
        if not source or not target:
            continue
        
        # Tokenize
        inputs = tokenizer(
            source,
            max_length=128,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=80,
                min_length=10,
                num_beams=5,
                no_repeat_ngram_size=3,
                length_penalty=1.0,
                early_stopping=True
            )
        
        # Decode
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        sources.append(source)
        references.append(target)
        predictions.append(prediction)
    
    # Calculate metrics
    print("\n📊 Calculating metrics...")
    
    try:
        bleu_score = corpus_bleu(predictions, [references]).score
    except Exception as e:
        print(f"⚠️  BLEU calculation error: {e}")
        bleu_score = 0.0
    
    try:
        sari_score = corpus_sari(sources, predictions, [references])
    except Exception as e:
        print(f"⚠️  SARI calculation error: {e}")
        sari_score = 0.0
    
    # Display results
    print("\n" + "="*70)
    print("📈 EVALUATION RESULTS")
    print("="*70)
    print(f"✅ SARI Score:  {sari_score:.2f}")
    print(f"✅ BLEU Score:  {bleu_score:.2f}")
    print(f"📊 Samples:     {len(predictions)}")
    print("="*70)
    
    # Show examples
    print("\n📝 Sample Predictions:")
    print("-"*70)
    for i in range(min(3, len(predictions))):
        print(f"\n[Example {i+1}]")
        print(f"Source:     {sources[i]}")
        print(f"Prediction: {predictions[i]}")
        print(f"Reference:  {references[i]}")
        print("-"*70)
    
    # Save results
    results = {
        "model_path": model_path,
        "test_samples": len(predictions),
        "test_sari": round(sari_score, 2),
        "test_bleu": round(bleu_score, 2),
        "device": device,
        "examples": [
            {
                "source": sources[i],
                "prediction": predictions[i],
                "reference": references[i]
            }
            for i in range(min(5, len(predictions)))
        ]
    }
    
    output_file = Path(model_path) / "evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Update training_config.json
    config_file = Path(model_path) / "training_config.json"
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
        config['test_sari'] = round(sari_score, 2)
        config['test_bleu'] = round(bleu_score, 2)
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✅ Updated: {config_file}")
    
    return results

if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "SimpleBART-base model"  # Sesuaikan path Anda
    NUM_SAMPLES = None  # None = evaluasi semua, atau set angka (e.g., 1000)
    
    # Run evaluation
    results = evaluate_model(MODEL_PATH, NUM_SAMPLES)
    
    print("\n✅ Evaluation complete!")
    print("\n💡 Tips:")
    print("   - Hasil sudah disimpan di evaluation_results.json")
    print("   - training_config.json sudah diupdate dengan metrics terbaru")
    print("   - Sekarang Anda bisa menjalankan Streamlit app!")