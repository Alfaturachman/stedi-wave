"""
Standalone test script untuk debug prediction - FIXED VERSION
Letakkan file ini di folder yang sama dengan manage.py
Jalankan: python test_debug.py
"""

import os
import sys
import django

# Setup Django
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stetoskop.settings')
django.setup()

# Import setelah Django setup
import numpy as np
import librosa
import torch
import joblib
from audio.views import (
    load_pytorch_model, 
    load_gb_model, 
    extract_features,
    MODEL_GB_PATH,
    CLASS_THRESHOLDS
)

# Global variables
gb_model = None
classFolders = None

def load_models():
    """Load models dan return gb_model, classFolders"""
    global gb_model, classFolders
    
    if gb_model is None:
        print("Loading models...")
        
        # Load GB model langsung dari file
        gb_model, classFolders = joblib.load(MODEL_GB_PATH)
        print(f"✓ GB model loaded from {MODEL_GB_PATH}")
        print(f"✓ Classes: {classFolders}")
        
        # Load PyTorch model juga
        load_pytorch_model()
        print(f"✓ PyTorch model loaded")
    
    return gb_model, classFolders


def test_single_audio(audio_path):
    """Test prediction untuk 1 file audio"""
    print("\n" + "="*70)
    print(f"TESTING: {os.path.basename(audio_path)}")
    print("="*70)
    
    try:
        # Load models
        gb_model_local, classes = load_models()
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=22050)
        duration = len(y) / sr
        print(f"✓ Audio loaded: {duration:.2f}s, {len(y)} samples")
        
        # Extract features
        feat = extract_features(y, sr)
        print(f"✓ Features extracted: shape={feat.shape}")
        
        # GB prediction
        probs = gb_model_local.predict_proba(feat)[0]
        pred_idx = int(np.argmax(probs))
        
        print(f"\n{'='*70}")
        print("GB MODEL PREDICTIONS:")
        print(f"{'='*70}")
        
        # Sort by probability
        sorted_indices = np.argsort(probs)[::-1]
        
        for i, idx in enumerate(sorted_indices, 1):
            cls = classes[idx]
            prob = probs[idx]
            threshold = CLASS_THRESHOLDS.get(cls, 0.3)
            
            status = "✓ PASS" if prob >= threshold else "✗ FAIL"
            marker = " ← TOP PREDICTION" if idx == pred_idx else ""
            
            print(f"  {i}. {cls:12s} : {prob:.4f} (threshold: {threshold:.2f}) [{status}]{marker}")
        
        print(f"\n{'='*70}")
        top_class = classes[pred_idx]
        top_prob = probs[pred_idx]
        top_threshold = CLASS_THRESHOLDS.get(top_class, 0.3)
        
        if top_prob >= top_threshold:
            print(f"RESULT: {top_class.upper()} (confidence: {top_prob:.4f})")
            print(f"STATUS: ✓ ACCEPTED")
        else:
            print(f"RESULT: Unknown (best was {top_class} with {top_prob:.4f})")
            print(f"STATUS: ✗ REJECTED (below threshold {top_threshold:.2f})")
        
        print(f"{'='*70}\n")
        
        return {
            'predicted_class': top_class,
            'confidence': float(top_prob),
            'passed_threshold': top_prob >= top_threshold,
            'all_probs': {classes[i]: float(probs[i]) for i in range(len(classes))}
        }
        
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_multiple_samples():
    """Test beberapa sample dari setiap class"""
    
    dataset_dir = r"C:\Users\Lenovo\Downloads\DATASET_LUNG TYPE SOUND-20250929T090952Z-1-001\DATASET_LUNG TYPE SOUND"
    
    if not os.path.exists(dataset_dir):
        print(f"ERROR: Dataset directory not found: {dataset_dir}")
        return
    
    results = {}
    
    for class_name in ['bron', 'crackles', 'crep', 'mengi', 'normal']:
        class_dir = os.path.join(dataset_dir, class_name)
        
        if not os.path.exists(class_dir):
            print(f"WARNING: Class directory not found: {class_name}")
            continue
        
        wav_files = [f for f in os.listdir(class_dir) if f.lower().endswith('.wav')]
        
        if not wav_files:
            print(f"WARNING: No wav files found in {class_name}")
            continue
        
        # Test first 3 files dari setiap class
        test_files = wav_files[:3]
        results[class_name] = []
        
        for wav_file in test_files:
            audio_path = os.path.join(class_dir, wav_file)
            result = test_single_audio(audio_path)
            
            if result:
                results[class_name].append(result)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    total_correct = 0
    total_samples = 0
    
    for class_name, class_results in results.items():
        if not class_results:
            continue
        
        correct = sum(1 for r in class_results if r['predicted_class'] == class_name and r['passed_threshold'])
        total = len(class_results)
        avg_conf = np.mean([r['confidence'] for r in class_results])
        
        total_correct += correct
        total_samples += total
        
        print(f"\n{class_name.upper()}:")
        print(f"  Correct: {correct}/{total} ({correct/total*100:.1f}%)")
        print(f"  Avg confidence: {avg_conf:.4f}")
        print(f"  Threshold: {CLASS_THRESHOLDS.get(class_name, 0.3):.2f}")
    
    if total_samples > 0:
        print(f"\n{'='*70}")
        print(f"OVERALL ACCURACY: {total_correct}/{total_samples} ({total_correct/total_samples*100:.1f}%)")
        print(f"{'='*70}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("LUNG SOUND PREDICTION DEBUG TOOL")
    print("="*70)
    
    # Pilih mode
    print("\nMode:")
    print("1. Test single file")
    print("2. Test multiple samples (3 per class)")
    
    choice = input("\nPilih mode (1/2): ").strip()
    
    if choice == "1":
        audio_path = input("Masukkan path audio file: ").strip()
        if os.path.exists(audio_path):
            test_single_audio(audio_path)
        else:
            print(f"File not found: {audio_path}")
    
    elif choice == "2":
        test_multiple_samples()
    
    else:
        print("Invalid choice!")