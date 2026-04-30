import librosa
import numpy as np
import cv2
import webrtcvad

class VoiceProcessor:
    def __init__(self, target_sr=16000):
        # Menggunakan 16000 Hz agar sinkron dengan kebutuhan WebRTC VAD
        self.target_sr = target_sr
        # Mode 3: Tingkat agresivitas tertinggi untuk memfilter noise di lingkungan Surabaya
        self.vad = webrtcvad.Vad(3) 

    def is_speech(self, audio_frame, sample_rate):
        """Mengecek apakah frame audio mengandung komponen vokal manusia."""
        return self.vad.is_speech(audio_frame, sample_rate)

    def preprocess_audio(self, y):
        """Mengubah sinyal waveform menjadi Mel-Spectrogram (dB)."""
        # Parameter n_mels=128 harus konsisten dengan arsitektur Koe no Katachi
        mel_spec = librosa.feature.melspectrogram(y=y, sr=self.target_sr, n_mels=128)
        # Konversi ke Log Scale (decibel) untuk menonjolkan tekstur vokal
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db

    def prepare_for_resnet(self, mel_spec_db):
        """Transformasi final agar input identik dengan data training di Colab."""
        # 1. Resize ke 224x224 (Dimensi standar input ResNet-50)
        img_resized = cv2.resize(mel_spec_db, (224, 224))
        
        # 2. Normalisasi Min-Max (0-1)
        # Epsilon 1e-6 ditambahkan untuk mencegah pembagian dengan nol jika audio hening
        img_min = img_resized.min()
        img_max = img_resized.max()
        img_norm = (img_resized - img_min) / (img_max - img_min + 1e-6)
        
        # 3. Stack menjadi 3 channel (RGB)
        # Model dilatih dengan data yang di-repeat 3x, jadi kita samakan di sini
        img_rgb = np.stack([img_norm] * 3, axis=-1) # Shape: (224, 224, 3)
        
        # CATATAN: Normalisasi Mean/Std ImageNet dihapus sesuai kesepakatan training terbaru.
        
        # 4. Transpose format: (H, W, C) -> (C, H, W)
        # PyTorch membutuhkan dimensi channel (C) di urutan pertama
        img_chw = img_rgb.transpose(2, 0, 1)
        
        # 5. Expand Batch Dimension agar menjadi (1, 3, 224, 224)
        img_batch = np.expand_dims(img_chw, axis=0).astype(np.float32)
        
        return img_batch