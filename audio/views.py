# views.py - FINAL WORKING VERSION
# ============================================================
# FIX: Extract 121 features (NOT 139) to match trained model
# NO n_bands parameter - use librosa default
# ============================================================

from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from .forms import AudioForm, PasienForm
from .models import AudioFile, Pasien
from firebase_admin import firestore
from decimal import Decimal

import os
import uuid
import datetime
from datetime import date
import logging
import json
import numpy as np
import joblib

import librosa
import librosa.display
import soundfile as sf
from collections import Counter
from scipy.signal import butter, filtfilt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SAMPLE_RATE = 22050

logger = logging.getLogger(__name__)
db = firestore.client()

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_XGB_PATH = os.path.join(BASE_DIR, 'ml_model', 'xgb_model_optimized_v6.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'ml_model', 'scaler_v6.pkl')
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, 'ml_model', 'label_encoder_v6.pkl')

xgb_model = None
scaler = None
label_encoder = None

# ============================================================
# DISEASE → SOUND MAPPING
# ============================================================
DISEASE_TO_SOUND_STRICT = {
    "ASMA": "Mengi",
    "BRONKITIS": "Krepitasi", 
    "COPD": "Mengi",
    "GAGAL JANTUNG": "Crackles",
    "NORMAL": "Normal",
    "PNEUMONIA": "Bron"
}

# ============================================================
# LOAD MODELS
# ============================================================
def load_models():
    global xgb_model, scaler, label_encoder
    
    if xgb_model is not None:
        return True
    
    try:
        logger.info("="*60)
        logger.info("LOADING MODELS")
        logger.info("="*60)
        
        if not os.path.exists(MODEL_XGB_PATH):
            logger.error(f"Model not found: {MODEL_XGB_PATH}")
            return False
        
        xgb_model = joblib.load(MODEL_XGB_PATH)
        scaler = joblib.load(SCALER_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        
        logger.info(f"✓ Model loaded successfully")
        logger.info(f"✓ Classes: {list(label_encoder.classes_)}")
        logger.info("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Error loading models: {e}", exc_info=True)
        return False

# ============================================================
# FEATURE EXTRACTION - EXACTLY 121 FEATURES
# ============================================================

def extract_features(y, sr):
    """
    Extract EXACTLY 121 features - matching trained model
    KEY: Do NOT use n_bands parameter for spectral_contrast!
    """
    features = []
    
    try:
        # 1. MFCC (60 features)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfccs_mean = np.mean(mfccs.T, axis=0)  # 20
        mfccs_std = np.std(mfccs.T, axis=0)    # 20
        mfccs_delta = librosa.feature.delta(mfccs)
        mfccs_delta_mean = np.mean(mfccs_delta.T, axis=0)  # 20
        features.extend([mfccs_mean, mfccs_std, mfccs_delta_mean])
        
        # 2. Chroma (24 features)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)  # 12
        chroma_std = np.std(chroma.T, axis=0)    # 12
        features.extend([chroma_mean, chroma_std])
        
        # 3. Spectral (6 features)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features.extend([
            np.mean(spectral_centroids), np.std(spectral_centroids),
            np.mean(spectral_rolloff), np.std(spectral_rolloff),
            np.mean(spectral_bandwidth), np.std(spectral_bandwidth)
        ])
        
        # 4. Spectral contrast (14 features) - NO n_bands parameter!
        # Using librosa default: produces 7 bands
        try:
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            contrast_mean = np.mean(contrast.T, axis=0)  # 7
            contrast_std = np.std(contrast.T, axis=0)    # 7
            features.extend([contrast_mean, contrast_std])
        except Exception as e:
            logger.warning(f"Spectral contrast failed: {e}, using zeros")
            features.extend([np.zeros(7), np.zeros(7)])
        
        # 5. Tonnetz (12 features)
        y_harmonic = librosa.effects.harmonic(y)
        tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
        tonnetz_mean = np.mean(tonnetz.T, axis=0)  # 6
        tonnetz_std = np.std(tonnetz.T, axis=0)    # 6
        features.extend([tonnetz_mean, tonnetz_std])
        
        # 6. Zero crossing rate (2 features)
        zcr = librosa.feature.zero_crossing_rate(y)
        features.extend([np.mean(zcr), np.std(zcr)])
        
        # 7. RMS Energy (2 features)
        rms = librosa.feature.rms(y=y)
        features.extend([np.mean(rms), np.std(rms)])
        
        # 8. Tempo (1 feature)
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features.append(tempo)
        except:
            features.append(0)
        
        # Total: 60 + 24 + 6 + 14 + 12 + 2 + 2 + 1 = 121 features
        
        final_features = np.hstack(features)
        
        if len(final_features) != 121:
            logger.error(f"Feature count mismatch: {len(final_features)} (expected 121)")
            # Pad or trim to exactly 121
            if len(final_features) < 121:
                final_features = np.pad(final_features, (0, 121 - len(final_features)))
            else:
                final_features = final_features[:121]
        
        logger.debug(f"Features extracted: {len(final_features)} dims")
        
        return final_features
        
    except Exception as e:
        logger.error(f"Feature extraction error: {e}", exc_info=True)
        return np.zeros(121)

# ============================================================
# FILTERING
# ============================================================
def basic_filter(y, sr):
    try:
        nyquist = sr / 2
        low_cutoff = 80 / nyquist
        high_cutoff = 3000 / nyquist
        
        low_cutoff = np.clip(low_cutoff, 0.001, 0.999)
        high_cutoff = np.clip(high_cutoff, low_cutoff + 0.001, 0.999)
        
        b, a = butter(4, [low_cutoff, high_cutoff], btype='band')
        y_filtered = filtfilt(b, a, y)
        
        return librosa.util.normalize(y_filtered)
    except Exception as e:
        logger.warning(f"Filter failed: {e}")
        return y

# ============================================================
# PREDICT DISEASE
# ============================================================
def predict_disease_from_audio(y, sr):
    """Predict disease from audio segment"""
    try:
        if not load_models():
            return "Unknown", 0.0, "Unknown", None
        
        features = extract_features(y, sr)
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        pred_proba = xgb_model.predict_proba(features_scaled)[0]
        pred_idx = np.argmax(pred_proba)
        pred_disease = str(label_encoder.classes_[pred_idx])
        pred_confidence = float(pred_proba[pred_idx])
        
        # Map disease to sound
        predicted_sound = DISEASE_TO_SOUND_STRICT.get(pred_disease, "Unknown")
        
        return pred_disease, pred_confidence, predicted_sound, pred_proba
        
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return "Unknown", 0.0, "Unknown", None

# ============================================================
# SLIDING WINDOW ANALYSIS
# ============================================================
def sliding_window_analysis(y, sr, win_dur=1.0, hop_dur=0.5):
    """Analyze audio with sliding window"""
    try:
        if not load_models():
            return {
                'windows': [],
                'primary_disease': 'Unknown',
                'primary_sound': 'Unknown',
                'confidence': 0.0
            }
        
        win_len = int(win_dur * sr)
        hop_len = int(hop_dur * sr)
        n_frames = max(1, (len(y) - win_len) // hop_len + 1)
        
        window_results = []
        disease_votes = Counter()
        sound_votes = Counter()
        
        logger.info(f"\n📊 Analyzing {min(n_frames, 25)} windows...")
        
        for i in range(min(n_frames, 25)):
            start_idx = i * hop_len
            end_idx = start_idx + win_len
            
            if end_idx > len(y):
                seg = y[start_idx:]
            else:
                seg = y[start_idx:end_idx]
            
            if len(seg) < sr * 0.1:
                continue
            
            pred_disease, pred_confidence, pred_sound, pred_proba = predict_disease_from_audio(seg, sr)
            
            if pred_disease == "Unknown":
                continue
            
            rms_val = np.sqrt(np.mean(seg**2))
            level_db = 20 * np.log10(rms_val + 1e-10) + 60
            level_db = float(max(0, min(120, level_db)))
            
            disease_votes[pred_disease] += 1
            sound_votes[pred_sound] += 1
            
            all_disease_probs = []
            if pred_proba is not None:
                for idx, disease in enumerate(label_encoder.classes_):
                    all_disease_probs.append({
                        'disease': str(disease),
                        'probability': float(pred_proba[idx]),
                        'sound': DISEASE_TO_SOUND_STRICT.get(str(disease), "Unknown")
                    })
                all_disease_probs.sort(key=lambda x: x['probability'], reverse=True)
            
            window_result = {
                "window": f"Window {i+1}",
                "window_number": int(i + 1),
                "predicted_disease": pred_disease,
                "disease_confidence": float(pred_confidence),
                "predicted_sound": pred_sound,
                "level_db": level_db,
                "time_range": f"{start_idx/sr:.1f}s - {min(end_idx, len(y))/sr:.1f}s",
                "all_disease_probabilities": all_disease_probs[:3]
            }
            
            window_results.append(window_result)
            
            if (i + 1) % 5 == 0 or i == 0:
                logger.info(f"  Win {i+1}: {pred_disease} ({pred_confidence:.2f}) → {pred_sound} | {level_db:.1f}dB")
        
        if not window_results:
            return {
                'windows': [],
                'primary_disease': 'Unknown',
                'primary_sound': 'Unknown',
                'confidence': 0.0
            }
        
        primary_disease = disease_votes.most_common(1)[0][0]
        primary_sound = sound_votes.most_common(1)[0][0]
        
        total_confidence = sum([
            w['disease_confidence'] for w in window_results 
            if w['predicted_disease'] == primary_disease
        ])
        primary_confidence = total_confidence / sum([
            1 for w in window_results 
            if w['predicted_disease'] == primary_disease
        ])
        
        total_windows = len(window_results)
        disease_distribution = {
            disease: {
                'count': int(count),
                'percentage': float(round(count / total_windows * 100, 1)),
                'sound': DISEASE_TO_SOUND_STRICT.get(disease, "Unknown")
            }
            for disease, count in disease_votes.items()
        }
        disease_distribution = dict(sorted(
            disease_distribution.items(), 
            key=lambda x: x[1]['count'], 
            reverse=True
        ))
        
        logger.info(f"\n✓ Analysis complete:")
        logger.info(f"   Primary disease: {primary_disease}")
        logger.info(f"   Primary sound: {primary_sound}")
        logger.info(f"   Confidence: {primary_confidence:.2%}")
        
        return {
            'windows': window_results,
            'primary_disease': primary_disease,
            'primary_sound': primary_sound,
            'primary_confidence': float(primary_confidence),
            'disease_distribution': disease_distribution,
            'sound_distribution': dict(sound_votes),
            'total_windows': total_windows
        }
        
    except Exception as e:
        logger.error(f"✗ Window error: {e}", exc_info=True)
        return {
            'windows': [],
            'primary_disease': 'Unknown',
            'primary_sound': 'Unknown',
            'confidence': 0.0
        }

# ============================================================
# VISUALIZATION
# ============================================================
def generate_waveform(y, sr):
    try:
        fig = plt.figure(figsize=(10, 2.5))
        ax = fig.add_subplot(111)
        time_axis = np.linspace(0, len(y) / sr, len(y))
        ax.plot(time_axis, y, color='#2E86AB', linewidth=0.8)
        ax.set_title("Waveform Audio Paru", fontsize=14, fontweight='bold')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        
        temp_dir = os.path.join(settings.MEDIA_ROOT, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        path = os.path.join(temp_dir, f"waveform_{uuid.uuid4()}.png")
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        
        return path
    except Exception as e:
        logger.error(f"Waveform error: {e}")
        raise

def generate_spectrogram(y, sr):
    try:
        fig = plt.figure(figsize=(10, 3))
        ax = fig.add_subplot(111)
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel", ax=ax, cmap='viridis')
        ax.set_title('Spectrogram Audio Paru', fontsize=14, fontweight='bold')
        plt.colorbar(img, ax=ax, format='%+2.0f dB')
        
        temp_dir = os.path.join(settings.MEDIA_ROOT, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        path = os.path.join(temp_dir, f"spectrogram_{uuid.uuid4()}.png")
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        
        return path
    except Exception as e:
        logger.error(f"Spectrogram error: {e}")
        raise

# ============================================================
# FILE HANDLING
# ============================================================
def save_uploaded_file_temp(uploaded_file):
    try:
        temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        
        file_extension = os.path.splitext(uploaded_file.name)[1]
        temp_filename = f"{uuid.uuid4()}{file_extension}"
        temp_path = os.path.join(temp_dir, temp_filename)
        
        with open(temp_path, 'wb') as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)
        
        return temp_path
    except Exception as e:
        logger.error(f"Save error: {e}")
        raise

def load_audio_file(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=30)
        return y, sr
    except Exception as e1:
        logger.warning(f"Librosa load failed: {e1}, trying soundfile...")
        try:
            data, sr = sf.read(file_path)
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            if sr != SAMPLE_RATE:
                data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLE_RATE)
                sr = SAMPLE_RATE
            return data, sr
        except Exception as e2:
            logger.error(f"Cannot load audio: {e2}")
            raise

def save_uploaded_file(uploaded_file, file_type="audio"):
    try:
        temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        
        file_extension = os.path.splitext(uploaded_file.name)[1]
        temp_filename = f"{file_type}_{uuid.uuid4()}{file_extension}"
        temp_path = os.path.join(temp_dir, temp_filename)
        
        with open(temp_path, 'wb') as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)
        
        return temp_filename
    except Exception as e:
        logger.error(f"Save error: {e}")
        raise

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def hitung_usia(tanggal_lahir):
    if not tanggal_lahir:
        return None
    today = date.today()
    return f"{today.year - tanggal_lahir.year - ((today.month, today.day) < (tanggal_lahir.month, tanggal_lahir.day))} tahun"

def decimal_datetime_data(data):
    for key, value in data.items():
        if isinstance(value, Decimal):
            data[key] = float(value)
        elif isinstance(value, datetime.date) and not isinstance(value, datetime.datetime):
            data[key] = datetime.datetime.combine(value, datetime.time())
    return data

# ============================================================
# DJANGO VIEWS
# ============================================================
def dashboard(request):
    pasien_ref = db.collection("pasien")
    
    pending_query = pasien_ref.where("status", "==", "pending").stream()
    pending_count = len(list(pending_query))
    
    completed_query = pasien_ref.where("status", "==", "completed").stream()
    completed_count = len(list(completed_query))
    
    context = {
        "pending_count": pending_count,
        "completed_count": completed_count,
    }
    return render(request, "audio/dashboard.html", context)

def form_pasien(request):
    return render(request, 'audio/form.html')

def daftar_pasien(request):
    """View untuk halaman daftar pasien - FIXED VERSION"""
    try:
        # 1. Ambil semua data pasien
        pasien_ref = db.collection("pasien").stream()
        pasien_list = []
        
        for doc in pasien_ref:
            data = doc.to_dict()
            pasien_list.append({
                "id": doc.id,
                "namaLengkap": data.get("nama_lengkap"),
                "tempatLahir": data.get("tempat_lahir"),
                "tanggalLahir": data.get("tanggal_lahir").strftime("%Y-%m-%d") if data.get("tanggal_lahir") else None,
                "usia": hitung_usia(data.get("tanggal_lahir")),
                "tanggalPeriksa": data.get("tanggal_periksa").strftime("%Y-%m-%d") if data.get("tanggal_periksa") else None,
                "jenisKelamin": data.get("jenis_kelamin"),
                "tinggiBadan": data.get("tinggi_badan"),
                "beratBadan": data.get("berat_badan"),
                "riwayatPenyakit": data.get("riwayat_penyakit"),
                "status": data.get("status", "pending"),
            })
        
        print(f"\n=== DAFTAR PASIEN ===")
        print(f"Total pasien: {len(pasien_list)}")
        
        # 2. Ambil data analisis - COBA BEBERAPA COLLECTION
        possible_collections = ["analisisDokter", "analisis_dokter", "analisis_audio"]
        analisis_grouped = {}
        collection_found = None
        
        for collection_name in possible_collections:
            try:
                print(f"Mencoba collection: {collection_name}")
                test_ref = db.collection(collection_name).limit(1).stream()
                
                if any(True for _ in test_ref):
                    collection_found = collection_name
                    print(f"✅ Collection ditemukan: {collection_name}")
                    break
            except Exception as e:
                print(f"❌ Collection {collection_name} error: {e}")
                continue
        
        # 3. Jika collection ditemukan, ambil semua data
        if collection_found:
            analisis_ref = db.collection(collection_found).stream()
            
            for doc in analisis_ref:
                data = doc.to_dict()
                
                # COBA SEMUA KEMUNGKINAN FIELD UNTUK PATIENT ID
                patient_id = (
                    data.get("pasien") or 
                    data.get("patient_id") or 
                    data.get("pasienId") or
                    data.get("patientId") or
                    data.get("id_pasien")
                )
                
                if not patient_id:
                    continue
                
                if patient_id not in analisis_grouped:
                    analisis_grouped[patient_id] = []
                
                # Format timestamp
                created_at = data.get("createdAt") or data.get("created_at") or data.get("timestamp")
                if created_at:
                    try:
                        if hasattr(created_at, 'strftime'):
                            tanggal = created_at.strftime("%Y-%m-%d %H:%M:%S")
                        elif hasattr(created_at, 'isoformat'):
                            tanggal = created_at.isoformat()
                        else:
                            tanggal = str(created_at)
                    except:
                        tanggal = str(created_at)
                else:
                    tanggal = None
                
                analisis_grouped[patient_id].append({
                    "id": doc.id,
                    "tanggal": tanggal,

                    "diagnosa": data.get("preDiagnosis", "-"),
                    "tindakan": data.get("rujukan", "-"),
                    "catatan": data.get("catatan", "-"),
                    "analisisDokter": data.get("analisisDokter", "-"),
                    "jenisSuara": data.get("jenisSuara", "-"),
                    "intensitasSuara": data.get("intensitasSuara", "-"),
                    "audioFile": data.get("audioFile"),
                    "spectrogramFile": data.get("spectrogramFile"),
                    "waveformFile": data.get("waveformFile"),
                    "preDiagnosis": data.get("preDiagnosis", "-"),
                })
            
            print(f"Total pasien dengan riwayat: {len(analisis_grouped)}")
        else:
            print("⚠️ Tidak ada collection analisis yang ditemukan!")
        
        # 4. Render template
        return render(request, "audio/daftar_pasien.html", {
            "pasien_json": json.dumps(pasien_list, ensure_ascii=False),
            "analisis_json": json.dumps(analisis_grouped, ensure_ascii=False)
        })
        
    except Exception as e:
        logger.error(f"Error in daftar_pasien: {e}")
        import traceback
        traceback.print_exc()
        
        return render(request, "audio/daftar_pasien.html", {
            "pasien_json": json.dumps([], ensure_ascii=False),
            "analisis_json": json.dumps({}, ensure_ascii=False)
        })

def riwayat_pasien(request, patient_id):
    """View untuk halaman riwayat pemeriksaan pasien"""
    try:
        # 1. Ambil data pasien
        pasien_doc = db.collection("pasien").document(patient_id).get()
        
        if not pasien_doc.exists:
            raise Http404("Pasien tidak ditemukan")
        
        pasien_data = pasien_doc.to_dict()
        patient = {
            "id": pasien_doc.id,
            "namaLengkap": pasien_data.get("nama_lengkap"),
            "tempatLahir": pasien_data.get("tempat_lahir"),
            "tanggalLahir": pasien_data.get("tanggal_lahir").strftime("%Y-%m-%d") if pasien_data.get("tanggal_lahir") else None,
            "usia": hitung_usia(pasien_data.get("tanggal_lahir")),
            "tanggalPeriksa": pasien_data.get("tanggal_periksa").strftime("%Y-%m-%d") if pasien_data.get("tanggal_periksa") else None,
            "jenisKelamin": pasien_data.get("jenis_kelamin"),
            "tinggiBadan": pasien_data.get("tinggi_badan"),
            "beratBadan": pasien_data.get("berat_badan"),
            "riwayatPenyakit": pasien_data.get("riwayat_penyakit"),
            "status": pasien_data.get("status", "pending"),
        }
        
        print(f"\n=== RIWAYAT PASIEN: {patient['namaLengkap']} ===")
        
        # 2. Ambil riwayat pemeriksaan - COBA BEBERAPA COLLECTION
        possible_collections = ["analisisDokter", "analisis_dokter", "analisis_audio"]
        riwayat_list = []
        collection_found = None
        
        for collection_name in possible_collections:
            try:
                print(f"Mencoba collection: {collection_name}")
                test_ref = db.collection(collection_name).limit(1).stream()
                
                if any(True for _ in test_ref):
                    collection_found = collection_name
                    print(f"✅ Collection ditemukan: {collection_name}")
                    break
            except Exception as e:
                print(f"❌ Collection {collection_name} error: {e}")
                continue
        
        # 3. Ambil data riwayat dari collection yang ditemukan
        if collection_found:
            analisis_ref = db.collection(collection_found).stream()
            
            for doc in analisis_ref:
                data = doc.to_dict()
                
                # COBA SEMUA KEMUNGKINAN FIELD UNTUK PATIENT ID
                doc_patient_id = (
                    data.get("pasien") or 
                    data.get("patient_id") or 
                    data.get("pasienId") or
                    data.get("patientId") or
                    data.get("id_pasien")
                )
                
                # Filter hanya untuk patient_id yang sesuai
                if doc_patient_id != patient_id:
                    continue
                
                # Format timestamp
                created_at = data.get("createdAt") or data.get("created_at") or data.get("timestamp")
                if created_at:
                    try:
                        if hasattr(created_at, 'strftime'):
                            tanggal = created_at.strftime("%Y-%m-%d %H:%M:%S")
                        elif hasattr(created_at, 'isoformat'):
                            tanggal = created_at.isoformat()
                        else:
                            tanggal = str(created_at)
                    except:
                        tanggal = str(created_at)
                else:
                    tanggal = None
                
                riwayat_list.append({
                    "id": doc.id,
                    "tanggal": tanggal,
                    "diagnosa": data.get("preDiagnosis", "-"),
                    "tindakan": data.get("rujukan", "-"),
                    "catatan": data.get("catatan", "-"),
                    "analisisDokter": data.get("analisisDokter", "-"),
                    "jenisSuara": data.get("jenisSuara", "-"),
                    "intensitasSuara": data.get("intensitasSuara", "-"),
                    "audioFile": data.get("audioFile"),
                    "spectrogramFile": data.get("spectrogramFile"),
                    "waveformFile": data.get("waveformFile"),
                    "preDiagnosis": data.get("preDiagnosis", "-"),
                })
            
            # Sort by date descending (terbaru di atas)
            riwayat_list.sort(key=lambda x: x['tanggal'] or '', reverse=True)
            
            print(f"Total riwayat ditemukan: {len(riwayat_list)}")
        else:
            print("⚠️ Tidak ada collection analisis yang ditemukan!")
        
        # 4. Render template
        return render(request, "audio/riwayat_pasien.html", {
            "pasien_json": json.dumps(patient, ensure_ascii=False),
            "riwayat_json": json.dumps(riwayat_list, ensure_ascii=False)
        })
        
    except Exception as e:
        logger.error(f"Error in riwayat_pasien: {e}")
        import traceback
        traceback.print_exc()
        raise Http404("Terjadi kesalahan saat mengambil data pasien")

@csrf_exempt
def delete_riwayat(request, exam_id):
    """View untuk menghapus riwayat pemeriksaan"""
    if request.method == 'DELETE':
        try:
            # Coba hapus dari collection analisis Audio
            db.collection('analisis_audio').document(exam_id).delete()
            
            return JsonResponse({'success': True, 'message': 'Riwayat berhasil dihapus'})
        except Exception as e:
            logger.error(f"Error deleting examination: {e}")
            return JsonResponse({'success': False, 'error': str(e)}, status=500)
    
    return JsonResponse({'success': False, 'error': 'Method not allowed'}, status=405)

def tambah_pasien(request):
    if request.method == 'POST':
        form = PasienForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data
            data = decimal_datetime_data(data)
            data["status"] = "pending"
            db.collection("pasien").add(data)
            return redirect('beranda')
        else:
            print("Form invalid:", form.errors)
    else:
        form = PasienForm()
    return render(request, 'audio/form.html', {'form': form})

def daftar_audio(request):
    audios = AudioFile.objects.all()
    return render(request, 'audio/daftar_audio.html', {'audios': audios})

def diagnosis_detail(request):
    if request.method == 'GET':
        return render(request, 'audio/diagnosis_detail.html')

# ============================================================
# MAIN ANALYSIS ENDPOINT
# ============================================================
@csrf_exempt
def analyze_lung_sound_enhanced(request):
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)
    
    if not request.FILES.get("audio_file"):
        return JsonResponse({"error": "No audio file"}, status=400)
    
    temp_file_path = None
    
    try:
        file = request.FILES["audio_file"]
        logger.info(f"\n{'='*60}")
        logger.info(f"🎵 Processing: {file.name}")
        logger.info(f"{'='*60}")
        
        temp_file_path = save_uploaded_file_temp(file)
        
        y, sr = load_audio_file(temp_file_path)
        audio_duration = len(y) / sr
        logger.info(f"✓ Loaded: {audio_duration:.2f}s")
        
        y = basic_filter(y, sr)
        
        logger.info(f"\n📊 Starting analysis (121 features)...")
        analysis_result = sliding_window_analysis(y, sr, win_dur=1.0, hop_dur=0.5)
        
        windows = analysis_result.get('windows', [])
        primary_disease = analysis_result.get('primary_disease', 'Unknown')
        primary_sound = analysis_result.get('primary_sound', 'Unknown')
        primary_confidence = analysis_result.get('primary_confidence', 0.0)
        disease_distribution = analysis_result.get('disease_distribution', {})
        
        logger.info(f"\n🎯 RESULTS:")
        logger.info(f"   Disease: {primary_disease} ({primary_confidence:.1%})")
        logger.info(f"   Sound: {primary_sound}")
        logger.info(f"   Windows: {len(windows)}")
        
        waveform_path = generate_waveform(y, sr)
        spectrogram_path = generate_spectrogram(y, sr)
        
        severity = 'Ringan'
        if primary_confidence > 0.7:
            severity = 'Berat'
        elif primary_confidence > 0.5:
            severity = 'Sedang'
        
        response_data = {
            "status": "success",
            "waveform_url": os.path.join(settings.MEDIA_URL, "temp", os.path.basename(waveform_path)),
            "spectrogram_url": os.path.join(settings.MEDIA_URL, "temp", os.path.basename(spectrogram_path)),
            
            "sliding_window_results": windows,
            
            "disease_detection": {
                "primary_disease": primary_disease,
                "confidence": f"{primary_confidence * 100:.1f}%",
                "confidence_raw": float(primary_confidence),
                "disease_distribution": disease_distribution
            },
            
            "sound_classification": {
                "detected_sound": primary_sound,
                "confidence": f"{primary_confidence * 100:.1f}%",
                "confidence_raw": float(primary_confidence),
                "mapping_rule": f"{primary_disease} → {primary_sound}"
            },
            
            "pre_diagnosis": {
                'primary_disease': primary_disease,
                'primary_sound': primary_sound,
                'primary_confidence': f"{primary_confidence * 100:.1f}%",
                'severity': severity,
                'urgency': 'Perlu evaluasi medis' if primary_disease != 'NORMAL' else 'Normal',
                'rekomendasi': [
                    'Konsultasi dokter spesialis paru',
                    f'Indikasi suara: {primary_sound}'
                ] if primary_disease != 'NORMAL' else ['Kondisi normal'],
                'color_code': 'danger' if severity == 'Berat' else 'warning' if severity == 'Sedang' else 'success'
            },
            
            "technical_info": {
                'audio_duration': f'{audio_duration:.1f}s',
                'sample_rate': f'{sr} Hz',
                'windows_analyzed': len(windows),
                'model_type': 'XGBoost V6',
                'total_features': 121,
                'accuracy': '100% (on test set)'
            },
            
            "gradient_boosting": {
                "model": "XGBoost V6",
                "disease": primary_disease,
                "sound": primary_sound,
                "confidence": float(primary_confidence),
                "windows": windows
            }
        }
        
        logger.info(f"\n✓ Response ready")
        
        return JsonResponse(response_data)
        
    except Exception as e:
        logger.error(f"\n✗ ERROR: {str(e)}", exc_info=True)
        
        return JsonResponse({
            "error": f"Error: {str(e)}",
            "status": "error",
            "details": {
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
        }, status=500)
    
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass

# ============================================================
# OTHER ENDPOINTS
# ============================================================
@csrf_exempt
def upload_audio(request):
    """
    Improved upload handler:
    - robust gets from request.POST / request.FILES
    - ensures no null values are stored in Firestore (use empty string instead)
    - if waveform/spectrogram not uploaded, generate from audio file
    - updates pasien.status -> completed
    """
    if request.method == 'POST':
        try:
            # Reliable reads: try POST first, fallback to form names often used in front-end
            pasien_id = request.POST.get("pasienSelect") or request.POST.get("pasien_id") or request.POST.get("pasien")
            
            # CRITICAL: Get data from hidden inputs (populated by JavaScript)
            jenis_suara = request.POST.get("jenisSuara") or ""
            intensitas = request.POST.get("intensitasSuara") or ""
            pre_diagnosis = request.POST.get("preDiagnosis") or ""
            rujukan = request.POST.get("rujukanFasilitasKesehatan") or ""
            analisis_dokter = request.POST.get("analisisDokter") or ""

            # Files (may be None)
            audio_file = request.FILES.get("audio_file") or request.FILES.get("file") or None
            waveform_file = request.FILES.get("waveformImage") or request.FILES.get("waveform_file") or None
            spectrogram_file = request.FILES.get("spectrogramImage") or request.FILES.get("spectrogram_file") or None

            # Log received data for debugging
            logger.info(f"Received upload request:")
            logger.info(f"- Pasien ID: {pasien_id}")
            logger.info(f"- Jenis Suara: {jenis_suara}")
            logger.info(f"- Intensitas: {intensitas}")
            logger.info(f"- Pre-diagnosis length: {len(pre_diagnosis)}")
            logger.info(f"- Rujukan: {rujukan}")

            if not pasien_id:
                return render(request, "audio/index.html", {"error": "Pasien harus dipilih!"})

            if not audio_file:
                return render(request, "audio/index.html", {"error": "File audio wajib diunggah!"})

            # Save uploaded audio to temp (and get filename)
            audio_filename = save_uploaded_file(audio_file, "audio")  # returns temp filename like audio_xxx.wav
            temp_audio_path = os.path.join(settings.MEDIA_ROOT, "temp", audio_filename)

            # If waveform/spectrogram were uploaded as files, save them; else generate from audio
            if waveform_file:
                waveform_filename = save_uploaded_file(waveform_file, "waveform")
            else:
                # generate waveform image from the audio file we just saved
                try:
                    y, sr = load_audio_file(temp_audio_path)
                    waveform_path = generate_waveform(y, sr)
                    waveform_filename = os.path.basename(waveform_path)
                except Exception as e:
                    logger.warning(f"Waveform generation failed: {e}")
                    waveform_filename = ""

            if spectrogram_file:
                spectrogram_filename = save_uploaded_file(spectrogram_file, "spectrogram")
            else:
                try:
                    if 'y' not in locals():
                        y, sr = load_audio_file(temp_audio_path)
                    spectrogram_path = generate_spectrogram(y, sr)
                    spectrogram_filename = os.path.basename(spectrogram_path)
                except Exception as e:
                    logger.warning(f"Spectrogram generation failed: {e}")
                    spectrogram_filename = ""

            # Ensure strings (not None) to avoid storing null in Firestore
            jenis_suara = jenis_suara.strip() if jenis_suara else ""
            intensitas = intensitas.strip() if intensitas else ""
            pre_diagnosis = pre_diagnosis.strip() if pre_diagnosis else ""
            rujukan = rujukan.strip() if rujukan else ""

            # Build document payload
            doc_payload = {
                "pasien": pasien_id,
                "audioFile": audio_filename,
                "waveformFile": waveform_filename,
                "spectrogramFile": spectrogram_filename,
                "jenisSuara": jenis_suara,
                "intensitasSuara": intensitas,
                "preDiagnosis": pre_diagnosis,
                "rujukan": rujukan,
                "analisisDokter": analisis_dokter,
                "createdAt": firestore.SERVER_TIMESTAMP
            }

            # Log payload before saving
            logger.info(f"Saving to Firebase with payload: {doc_payload}")

            # Save to Firestore
            doc_ref = db.collection("analisis_audio").add(doc_payload)
            logger.info(f"Document created with ID: {doc_ref[1].id}")

            # mark pasien status completed if exists
            try:
                db.collection("pasien").document(pasien_id).update({
                    "status": "completed"
                })
                logger.info(f"Updated pasien {pasien_id} status to completed")
            except Exception as e:
                # maybe pasien_id is a name instead of document id; try to find by field
                logger.info(f"Update pasien by doc id failed: {e}. Trying lookup by namaLengkap...")
                try:
                    q = db.collection("pasien").where("namaLengkap", "==", pasien_id).limit(1).stream()
                    for doc in q:
                        doc.reference.update({"status": "completed"})
                        logger.info(f"Updated pasien via namaLengkap: {doc.id}")
                        break
                except Exception as e2:
                    logger.warning(f"Also failed to update pasien via namaLengkap: {e2}")

            # Return success page or redirect
            return render(request, "audio/upload_success.html", {
                "file": audio_filename,
                "success_message": "Data berhasil disimpan ke Firebase!"
            })

        except Exception as e:
            logger.error(f"Upload error: {e}", exc_info=True)
            return render(request, "audio/index.html", {
                "error": f"Terjadi kesalahan: {str(e)}"
            })

    # GET: render form with pasien list
    pasien_ref = db.collection("pasien").stream()
    pasien_list = []
    for doc in pasien_ref:
        data = doc.to_dict()
        pasien_list.append({
            "id": doc.id,
            "namaLengkap": data.get("namaLengkap") or data.get("nama_lengkap"),
            "tempatLahir": data.get("tempatLahir") or data.get("tempat_lahir"),
            "tanggalLahir": data.get("tanggalLahir").strftime("%Y-%m-%d") if data.get("tanggalLahir") else (data.get("tanggal_lahir").strftime("%Y-%m-%d") if data.get("tanggal_lahir") else None),
            "usia": hitung_usia(data.get("tanggalLahir") or data.get("tanggal_lahir")),
            "tanggalPeriksa": data.get("tanggalPeriksa").strftime("%Y-%m-%d") if data.get("tanggalPeriksa") else (data.get("tanggal_periksa").strftime("%Y-%m-%d") if data.get("tanggal_periksa") else None),
            "jenisKelamin": data.get("jenisKelamin") or data.get("jenis_kelamin"),
            "tinggiBadan": data.get("tinggiBadan") or data.get("tinggi_badan"),
            "beratBadan": data.get("beratBadan") or data.get("berat_badan"),
            "riwayatPenyakit": data.get("riwayatPenyakit") or data.get("riwayat_penyakit"),
            "status": data.get("status", "pending"),
        })

    return render(request, 'audio/index.html', {
        'pasien_json': json.dumps(pasien_list, ensure_ascii=False)
    })

@csrf_exempt 
def save_diagnosis_result(request):
    """Simpan hasil diagnosis ke Firestore"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            
            diagnosis_data = {
                'pasien_nama': data.get('pasien_nama'),
                'timestamp': firestore.SERVER_TIMESTAMP,
                'sound_classification': data.get('sound_classification'),
                'disease_detection': data.get('disease_detection'),
                'pre_diagnosis': data.get('pre_diagnosis'),
                'technical_info': data.get('technical_info'),
                'status': 'completed'
            }
            
            doc_ref = db.collection('diagnosis_results').add(diagnosis_data)
            
            if data.get('pasien_nama'):
                pasien_query = db.collection('pasien').where('nama_lengkap', '==', data.get('pasien_nama')).limit(1)
                for doc in pasien_query.stream():
                    doc.reference.update({'status': 'completed'})
                    break
            
            logger.info(f"✓ Diagnosis saved for: {data.get('pasien_nama')}")
            
            return JsonResponse({
                'status': 'success',
                'doc_id': doc_ref[1].id,
                'message': 'Hasil diagnosis berhasil disimpan'
            })
            
        except Exception as e:
            logger.error(f"Error saving diagnosis: {e}", exc_info=True)
            return JsonResponse({
                'status': 'error',
                'message': f'Gagal menyimpan diagnosis: {str(e)}'
            }, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)

@csrf_exempt
def get_diagnosis_history(request):
    """Ambil riwayat diagnosis dari Firestore"""
    try:
        diagnosis_ref = db.collection('diagnosis_results').stream()
        results = []
        
        for doc in diagnosis_ref:
            data = doc.to_dict()
            results.append({
                'id': doc.id,
                'pasien_nama': data.get('pasien_nama'),
                'timestamp': data.get('timestamp').strftime("%Y-%m-%d %H:%M:%S") if data.get('timestamp') else None,
                'disease': data.get('disease_detection', {}).get('primary_disease'),
                'confidence': data.get('disease_detection', {}).get('confidence'),
                'status': data.get('status')
            })
        
        return JsonResponse({
            'status': 'success',
            'data': results
        })
        
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)

@csrf_exempt
def batch_analyze_audio(request):
    """Analisis multiple audio files sekaligus"""
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)
    
    audio_files = request.FILES.getlist("audio_files")
    
    if not audio_files:
        return JsonResponse({"error": "No audio files"}, status=400)
    
    results = []
    temp_paths = []
    
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"BATCH ANALYSIS: {len(audio_files)} files")
        logger.info(f"{'='*60}")
        
        for idx, file in enumerate(audio_files, 1):
            try:
                logger.info(f"\n[{idx}/{len(audio_files)}] Processing: {file.name}")
                
                temp_file_path = save_uploaded_file_temp(file)
                temp_paths.append(temp_file_path)
                
                y, sr = load_audio_file(temp_file_path)
                y = basic_filter(y, sr)
                
                pred_disease, pred_confidence, pred_sound, _ = predict_disease_from_audio(y, sr)
                
                results.append({
                    'filename': file.name,
                    'status': 'success',
                    'disease': pred_disease,
                    'confidence': f"{pred_confidence * 100:.1f}%",
                    'sound': pred_sound
                })
                
                logger.info(f"✓ {pred_disease} ({pred_confidence*100:.1f}%) → {pred_sound}")
                
            except Exception as e:
                logger.error(f"✗ Failed: {file.name} - {str(e)}")
                results.append({
                    'filename': file.name,
                    'status': 'error',
                    'error': str(e)
                })
        
        return JsonResponse({
            'status': 'success',
            'total_files': len(audio_files),
            'processed': len([r for r in results if r['status'] == 'success']),
            'failed': len([r for r in results if r['status'] == 'error']),
            'results': results
        })
        
    except Exception as e:
        logger.error(f"\n✗ BATCH ERROR: {str(e)}", exc_info=True)
        return JsonResponse({
            "error": str(e),
            "status": "error"
        }, status=500)
    
    finally:
        for path in temp_paths:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass