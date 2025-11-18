# ====== TAMBAHKAN FILE BARU: audio/lung_sound_analyzer.py ======
"""
Sistem analisis suara paru yang lebih akurat dengan pre-diagnosis
Letakkan file ini di folder audio/ dengan nama: lung_sound_analyzer.py
"""

import numpy as np
import librosa
import logging
from datetime import datetime
from django.conf import settings
import os

logger = logging.getLogger(__name__)

class LungSoundAnalyzer:
    """
    Analyzer suara paru dengan rule-based system yang lebih akurat
    dan sistem pre-diagnosis berdasarkan intensitas + jenis suara
    """
    
    def __init__(self):
        self.classes = ['normal', 'crackles', 'crep', 'mengi', 'bron']
        self.intensity_thresholds = {
            'sangat_rendah': -40,  # < -40 dB
            'rendah': -30,         # -40 to -30 dB
            'normal': -20,         # -30 to -20 dB
            'tinggi': -10,         # -20 to -10 dB
            'sangat_tinggi': 0     # > -10 dB
        }
        
    def extract_enhanced_features(self, y, sr):
        """Extract fitur yang lebih komprehensif untuk deteksi akurat"""
        try:
            features = {}
            
            # 1. Basic audio properties
            duration = len(y) / sr
            rms_energy = np.sqrt(np.mean(y**2))
            
            # 2. Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            
            # 3. MFCC for texture analysis
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            
            # 4. Spectral contrast (bagus untuk membedakan suara abnormal)
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            
            # 5. Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            
            # 6. Onset detection (untuk crackles)
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)
            
            # 7. Harmonic-percussive separation
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            
            # Compile features
            features = {
                # Basic
                'duration': duration,
                'rms_energy': float(rms_energy),
                'intensity_db': float(20 * np.log10(rms_energy + 1e-10)),
                
                # Spectral
                'spectral_centroid_mean': float(np.mean(spectral_centroids)),
                'spectral_centroid_std': float(np.std(spectral_centroids)),
                'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
                'spectral_rolloff_std': float(np.std(spectral_rolloff)),
                'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
                'spectral_bandwidth_std': float(np.std(spectral_bandwidth)),
                'zero_crossing_rate_mean': float(np.mean(zero_crossing_rate)),
                'zero_crossing_rate_std': float(np.std(zero_crossing_rate)),
                
                # MFCC statistics
                'mfcc_mean': np.mean(mfcc, axis=1),
                'mfcc_std': np.std(mfcc, axis=1),
                
                # Spectral contrast
                'spectral_contrast_mean': np.mean(spectral_contrast, axis=1),
                'spectral_contrast_std': np.std(spectral_contrast, axis=1),
                
                # Chroma
                'chroma_mean': np.mean(chroma, axis=1),
                'chroma_std': np.std(chroma, axis=1),
                
                # Onset information
                'onset_count': len(onset_times),
                'onset_density': len(onset_times) / duration if duration > 0 else 0,
                
                # Harmonic vs Percussive
                'harmonic_ratio': float(np.mean(y_harmonic**2) / (np.mean(y**2) + 1e-10)),
                'percussive_ratio': float(np.mean(y_percussive**2) / (np.mean(y**2) + 1e-10)),
                
                # Frequency domain features
                'dominant_frequency': float(np.mean(spectral_centroids)),
                'frequency_range': float(np.mean(spectral_bandwidth)),
                'high_freq_ratio': float(np.mean(spectral_rolloff) / (sr/2)),
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting enhanced features: {e}")
            raise
    
    def classify_lung_sound_rule_based(self, features):
        """
        Rule-based classification yang lebih akurat berdasarkan karakteristik medis
        """
        scores = {class_name: 0 for class_name in self.classes}
        
        # Ekstrak fitur penting
        intensity_db = features['intensity_db']
        spectral_centroid = features['spectral_centroid_mean']
        onset_density = features['onset_density']
        zero_crossing_rate = features['zero_crossing_rate_mean']
        harmonic_ratio = features['harmonic_ratio']
        percussive_ratio = features['percussive_ratio']
        spectral_bandwidth = features['spectral_bandwidth_mean']
        duration = features['duration']
        
        # 1. NORMAL BREATH SOUNDS
        # - Low to moderate intensity
        # - Smooth spectral characteristics
        # - Low onset density (few sharp transitions)
        if (-35 <= intensity_db <= -15 and 
            onset_density < 5 and 
            zero_crossing_rate < 0.1 and
            harmonic_ratio > 0.3):
            scores['normal'] += 40
            
        if (1000 <= spectral_centroid <= 3000 and
            spectral_bandwidth < 2000):
            scores['normal'] += 30
            
        # 2. CRACKLES (Fine & Coarse)
        # - Sudden sharp sounds = high onset density
        # - High percussive component
        # - Variable intensity
        if onset_density > 8:  # Many sudden sounds
            scores['crackles'] += 50
            
        if percussive_ratio > 0.4:  # Sharp, percussive sounds
            scores['crackles'] += 30
            
        if zero_crossing_rate > 0.15:  # Rapid changes in signal
            scores['crackles'] += 25
            
        # 3. CREPITATION (Fine crackles, paper-like)
        # - Very high onset density
        # - High frequency components
        # - Short duration bursts
        if (onset_density > 12 and 
            spectral_centroid > 2500):
            scores['crep'] += 55
            
        if (zero_crossing_rate > 0.2 and
            percussive_ratio > 0.5):
            scores['crep'] += 35
            
        # 4. WHEEZING (MENGI)
        # - Continuous musical sound
        # - High harmonic content
        # - Narrow frequency band
        # - Higher frequency
        if (harmonic_ratio > 0.6 and 
            spectral_centroid > 2000):
            scores['mengi'] += 50
            
        if (spectral_bandwidth < 1500 and  # Narrow band = musical
            duration > 1.0):  # Continuous
            scores['mengi'] += 40
            
        if onset_density < 3:  # Continuous, not interrupted
            scores['mengi'] += 20
            
        # 5. BRONCHIAL SOUNDS
        # - Higher intensity than normal
        # - Broader frequency range
        # - Hollow, tubular quality
        if (intensity_db > -20 and 
            spectral_bandwidth > 2500):
            scores['bron'] += 45
            
        if (2000 <= spectral_centroid <= 4000 and
            harmonic_ratio > 0.4):
            scores['bron'] += 35
            
        # Normalisasi scores
        max_score = max(scores.values())
        if max_score > 0:
            for class_name in scores:
                scores[class_name] = scores[class_name] / max_score
        
        # Tentukan prediksi
        predicted_class = max(scores, key=scores.get)
        confidence = scores[predicted_class]
        
        return {
            'label': predicted_class,
            'confidence': float(confidence),
            'all_probabilities': scores,
            'features_used': {
                'intensity_db': intensity_db,
                'spectral_centroid': spectral_centroid,
                'onset_density': onset_density,
                'harmonic_ratio': harmonic_ratio,
                'percussive_ratio': percussive_ratio
            }
        }
    
    def get_intensity_category(self, intensity_db):
        """Kategorikan intensitas suara dalam dB"""
        if intensity_db < -40:
            return 'sangat_rendah'
        elif intensity_db < -30:
            return 'rendah'
        elif intensity_db < -20:
            return 'normal'
        elif intensity_db < -10:
            return 'tinggi'
        else:
            return 'sangat_tinggi'
    
    def generate_pre_diagnosis(self, lung_sound_result, patient_data=None):
        """
        Generate pre-diagnosis berdasarkan jenis suara + intensitas + data pasien
        """
        sound_type = lung_sound_result['label']
        confidence = lung_sound_result['confidence']
        intensity_db = lung_sound_result['features_used']['intensity_db']
        intensity_category = self.get_intensity_category(intensity_db)
        
        # Base diagnosis berdasarkan jenis suara
        diagnosis_map = {
            'normal': {
                'condition': 'Suara napas normal',
                'severity': 'Normal',
                'recommendations': [
                    'Tidak ada kelainan yang terdeteksi',
                    'Lanjutkan gaya hidup sehat',
                    'Kontrol rutin sesuai jadwal'
                ],
                'possible_conditions': []
            },
            
            'crackles': {
                'condition': 'Suara napas crackles terdeteksi',
                'severity': 'Ringan hingga Sedang',
                'recommendations': [
                    'Diperlukan evaluasi lebih lanjut',
                    'Pertimbangkan foto rontgen dada',
                    'Konsultasi dengan dokter spesialis paru'
                ],
                'possible_conditions': [
                    'Pneumonia',
                    'Edema paru',
                    'Fibrosis paru',
                    'Bronkiektasis'
                ]
            },
            
            'crep': {
                'condition': 'Suara crepitation (fine crackles)',
                'severity': 'Ringan hingga Berat',
                'recommendations': [
                    'Perlu pemeriksaan segera',
                    'Foto rontgen dada diperlukan',
                    'Evaluasi fungsi jantung',
                    'Konsultasi dokter spesialis'
                ],
                'possible_conditions': [
                    'Edema paru akut',
                    'Pneumonia interstitial',
                    'Gagal jantung kongestif',
                    'Fibrosis paru awal'
                ]
            },
            
            'mengi': {
                'condition': 'Wheezing (mengi) terdeteksi',
                'severity': 'Ringan hingga Berat',
                'recommendations': [
                    'Evaluasi saluran napas',
                    'Tes fungsi paru',
                    'Pertimbangkan bronkodilator',
                    'Kontrol asma jika ada riwayat'
                ],
                'possible_conditions': [
                    'Asma bronkial',
                    'PPOK (Penyakit Paru Obstruktif Kronik)',
                    'Bronkitis',
                    'Obstruksi saluran napas'
                ]
            },
            
            'bron': {
                'condition': 'Suara napas bronkial',
                'severity': 'Sedang hingga Berat',
                'recommendations': [
                    'Pemeriksaan radiologi segera',
                    'Evaluasi konsolidasi paru',
                    'Konsultasi dokter spesialis paru',
                    'Pertimbangkan terapi antibiotik'
                ],
                'possible_conditions': [
                    'Pneumonia lobar',
                    'Konsolidasi paru',
                    'Atelektasis',
                    'Masa di paru'
                ]
            }
        }
        
        base_diagnosis = diagnosis_map.get(sound_type, diagnosis_map['normal'])
        
        # Modifikasi berdasarkan intensitas
        severity_modifier = {
            'sangat_rendah': {
                'severity_adj': 'Sangat Ringan',
                'urgency': 'Observasi',
                'additional_rec': ['Monitoring berkala', 'Hidrasi yang cukup']
            },
            'rendah': {
                'severity_adj': 'Ringan',
                'urgency': 'Kontrol Rutin',
                'additional_rec': ['Observasi gejala', 'Istirahat cukup']
            },
            'normal': {
                'severity_adj': 'Sedang',
                'urgency': 'Evaluasi Medis',
                'additional_rec': ['Pemeriksaan lanjutan', 'Monitoring ketat']
            },
            'tinggi': {
                'severity_adj': 'Berat',
                'urgency': 'Perlu Perhatian Medis',
                'additional_rec': ['Evaluasi segera', 'Pertimbangkan rawat inap']
            },
            'sangat_tinggi': {
                'severity_adj': 'Sangat Berat',
                'urgency': 'URGENT - Perlu Tindakan Segera',
                'additional_rec': ['Rujukan emergency', 'Monitoring intensif']
            }
        }
        
        intensity_mod = severity_modifier.get(intensity_category, severity_modifier['normal'])
        
        # Generate final diagnosis
        pre_diagnosis = {
            'timestamp': datetime.now().isoformat(),
            'sound_analysis': {
                'detected_sound': sound_type.upper(),
                'confidence_percentage': f"{confidence * 100:.1f}%",
                'intensity_db': f"{intensity_db:.1f} dB",
                'intensity_category': intensity_category,
                'all_probabilities': lung_sound_result['all_probabilities']
            },
            'clinical_assessment': {
                'primary_condition': base_diagnosis['condition'],
                'severity_level': intensity_mod['severity_adj'],
                'urgency_level': intensity_mod['urgency'],
                'confidence_level': 'Tinggi' if confidence > 0.7 else 'Sedang' if confidence > 0.5 else 'Rendah'
            },
            'possible_diagnoses': base_diagnosis['possible_conditions'],
            'recommendations': base_diagnosis['recommendations'] + intensity_mod['additional_rec'],
            'next_steps': self._get_next_steps(sound_type, intensity_category, confidence),
            'disclaimer': 'Hasil ini adalah analisis awal menggunakan AI dan memerlukan konfirmasi dokter untuk diagnosis definitif.'
        }
        
        # Tambahan berdasarkan data pasien jika ada
        if patient_data:
            pre_diagnosis['patient_factors'] = self._analyze_patient_factors(patient_data, sound_type)
        
        return pre_diagnosis
    
    def _get_next_steps(self, sound_type, intensity_category, confidence):
        """Tentukan langkah selanjutnya berdasarkan hasil analisis"""
        if sound_type == 'normal' and intensity_category in ['normal', 'rendah']:
            return [
                'Kontrol kesehatan rutin',
                'Maintian gaya hidup sehat',
                'Observasi gejala respirasi'
            ]
        
        urgent_cases = ['sangat_tinggi', 'tinggi']
        if intensity_category in urgent_cases or confidence > 0.8:
            return [
                'Konsultasi dokter dalam 24 jam',
                'Foto rontgen dada',
                'Tes darah lengkap',
                'Evaluasi fungsi paru'
            ]
        
        return [
            'Konsultasi dokter dalam 1-3 hari',
            'Observasi gejala tambahan',
            'Istirahat dan hidrasi yang cukup',
            'Hindari faktor pencetus'
        ]
    
    def _analyze_patient_factors(self, patient_data, sound_type):
        """Analisis faktor pasien yang mempengaruhi diagnosis"""
        factors = {
            'risk_factors': [],
            'protective_factors': [],
            'additional_considerations': []
        }
        
        # Contoh analisis berdasarkan data pasien
        if patient_data.get('usia'):
            age = patient_data['usia']
            if age > 65:
                factors['risk_factors'].append('Usia lanjut (>65 tahun)')
            elif age < 5:
                factors['risk_factors'].append('Usia balita (<5 tahun)')
        
        if patient_data.get('riwayat_penyakit'):
            history = patient_data['riwayat_penyakit'].lower()
            if 'asma' in history:
                factors['risk_factors'].append('Riwayat asma')
            if 'rokok' in history or 'merokok' in history:
                factors['risk_factors'].append('Riwayat merokok')
            if 'jantung' in history:
                factors['risk_factors'].append('Riwayat penyakit jantung')
        
        return factors
    
    def analyze_lung_sound_complete(self, y, sr, patient_data=None):
        """
        Analisis lengkap: deteksi suara + pre-diagnosis
        """
        try:
            # Extract enhanced features
            features = self.extract_enhanced_features(y, sr)
            
            # Classify using rule-based system
            classification_result = self.classify_lung_sound_rule_based(features)
            
            # Generate pre-diagnosis
            pre_diagnosis = self.generate_pre_diagnosis(classification_result, patient_data)
            
            # Compile complete result
            complete_result = {
                'sound_classification': classification_result,
                'pre_diagnosis': pre_diagnosis,
                'technical_details': {
                    'analysis_timestamp': datetime.now().isoformat(),
                    'audio_duration': features['duration'],
                    'sample_rate': sr,
                    'features_extracted': len(features),
                    'analyzer_version': '2.0_enhanced'
                }
            }
            
            return complete_result
            
        except Exception as e:
            logger.error(f"Error in complete lung sound analysis: {e}")
            raise


# ====== UTILITY FUNCTIONS ======
def format_diagnosis_for_display(pre_diagnosis):
    """Format diagnosis untuk ditampilkan di template"""
    return {
        'jenis_suara': pre_diagnosis['sound_analysis']['detected_sound'],
        'confidence': pre_diagnosis['sound_analysis']['confidence_percentage'],
        'intensitas': pre_diagnosis['sound_analysis']['intensity_db'],
        'tingkat_keparahan': pre_diagnosis['clinical_assessment']['severity_level'],
        'urgency': pre_diagnosis['clinical_assessment']['urgency_level'],
        'kemungkinan_penyakit': pre_diagnosis['possible_diagnoses'],
        'rekomendasi': pre_diagnosis['recommendations'],
        'langkah_selanjutnya': pre_diagnosis['next_steps']
    }

def get_color_code_by_severity(severity_level):
    """Return color code untuk UI berdasarkan tingkat keparahan"""
    color_map = {
        'Normal': 'success',
        'Sangat Ringan': 'info',
        'Ringan': 'warning',
        'Sedang': 'warning',
        'Berat': 'danger',
        'Sangat Berat': 'danger'
    }
    return color_map.get(severity_level, 'secondary')