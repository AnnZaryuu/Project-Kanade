import librosa
import numpy as np
import torch
from voice_processor import VoiceProcessor
from model_loader import ModelLoader

# 1. Inisialisasi
TARGET_SR = 16000
processor = VoiceProcessor(target_sr=TARGET_SR)
loader = ModelLoader(
    model_path='model_resnet50_koe_no_katachi_NEW.pth',
    class_names_path='class_names.json'
)

def test_song(file_path):
    print(f"\n--- Mengetes File: {file_path} ---")
    
    # 2. Load audio asli (kita ambil 5 detik di bagian tengah lagu agar ada vokal)
    # Ubah offset=60.0 ke detik di mana vokal penyanyi terdengar jelas
    y, sr = librosa.load(file_path, sr=TARGET_SR, duration=5.0, offset=60.0)
    
    # 3. Proses melalui pipeline yang sama dengan Flask
    spec = processor.preprocess_audio(y)
    input_tensor = processor.prepare_for_resnet(spec)
    
    # 4. Prediksi
    result = loader.predict(input_tensor)
    
    print(f"Hasil Prediksi: {result['singer']}")
    print(f"Confidence: {result['confidence']:.4f}")
    
    if result['confidence'] > 0.80:
        print("✅ Status: Model Sangat Yakin")
    else:
        print("⚠️ Status: Model Ragu (Mungkin ada kemiripan frekuensi)")

# --- EKSEKUSI ---
# Ganti dengan path lagu asli di laptopmu (misal .mp3 atau .wav)
try:
    # Contoh: test_song("lagu_milet.mp3")
    # test_song("Milet - Anytime Anywhere.mp3")
    # test_song("ZUTOMAYO - Inside Joke.mp3")
    test_song("Eve - Record .mp3")
except Exception as e:
    print(f"Gagal memuat file: {e}")