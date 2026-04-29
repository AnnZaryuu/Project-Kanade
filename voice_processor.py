import librosa
import numpy as np
import webrtcvad
import struct

class VoiceProcessor:
    def __init__(self, target_sr=22050):
        self.target_sr = target_sr
        # Mode 3 adalah yang paling agresif dalam memfilter noise
        self.vad = webrtcvad.Vad(3) 

    def is_speech(self, audio_frame, sample_rate):
        """Mengecek apakah potongan audio (frame) berisi suara manusia"""
        # webrtcvad butuh data dalam format PCM 16-bit
        # audio_frame harus berdurasi 10, 20, atau 30ms
        return self.vad.is_speech(audio_frame, sample_rate)

    def preprocess_audio(self, y):
        """
        Alur Preprocessing:
        1. Normalization (sudah dilakukan librosa saat load)
        2. Mel-Spectrogram
        3. Log Scaling (dB)
        """
        # Buat Mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=self.target_sr, n_mels=128)
        
        # Ubah ke skala Log (Desibel) - Ini poin Rescaling kamu
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Standarisasi ukuran agar sesuai input ResNet (128x128 atau sesuai training)
        # Kita perlu pastikan dimensi waktu (width) konsisten
        return mel_spec_db

    def prepare_for_resnet(self, mel_spec_db):
        """Menyiapkan matriks agar siap masuk ke ResNet-50"""
        # Tambahkan dimensi Batch dan Channel agar jadi (1, 3, 128, W)
        # Kita repeat 3x agar seolah-olah gambar RGB
        img = np.stack([mel_spec_db] * 3, axis=0)
        img = np.expand_dims(img, axis=0)
        return img