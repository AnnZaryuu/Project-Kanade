import os
import numpy as np
import io
import traceback
from collections import Counter
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from pydub import AudioSegment

# Import modul internal
from voice_processor import VoiceProcessor
from model_loader import ModelLoader

app = Flask(__name__)
CORS(app)

# --- KONFIGURASI PATH & FFmpeg ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ffmpeg_path = os.path.join(BASE_DIR, "ffmpeg.exe")
ffprobe_path = os.path.join(BASE_DIR, "ffprobe.exe")

AudioSegment.converter = ffmpeg_path
AudioSegment.ffprobe = ffprobe_path

# --- INISIALISASI SISTEM ---
TARGET_SR = 16000
# Gunakan BUFFER_SIZE > 1 untuk mendapatkan efek "Majority Voting" yang stabil
BUFFER_SIZE = 5 
voting_buffer = []

try:
    # Inisialisasi Processor dan Model baru (Acc 94.62%)
    processor = VoiceProcessor(target_sr=TARGET_SR)
    model_info = ModelLoader(
        model_path='model_resnet50_koe_no_katachi_NEW.pth',
        class_names_path='class_names.json'
    )
    print("✅ Sistem Siap: Model dan Processor berhasil dimuat.")
except Exception as e:
    print(f"❌ Gagal Inisialisasi: {e}")
    traceback.print_exc()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global voting_buffer
    
    if 'audio' not in request.files:
        return jsonify({"error": "Audio tidak ditemukan"}), 400

    audio_file = request.files['audio']

    try:
        # 1. Load Audio dari Request
        audio_bytes = io.BytesIO(audio_file.read())
        try:
            audio = AudioSegment.from_file(audio_bytes)
        except Exception as e:
            return jsonify({"error": f"Gagal membaca format audio: {str(e)}"}), 500

        # 2. Standarisasi (Mono, 16000Hz)
        audio = audio.set_frame_rate(TARGET_SR).set_channels(1)
        y = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0

        if len(y) < 100:
            return jsonify({"singer": "Suara terlalu pendek", "status": "Filtered"})

        # 3. Voice Activity Detection (VAD) - Frame 30ms
        y_pcm = (y * 32767).astype(np.int16).tobytes()
        samples_per_30ms = int(TARGET_SR * 0.03) # 480 sampel
        frame_len_bytes = samples_per_30ms * 2   # 960 bytes

        if len(y_pcm) < frame_len_bytes:
            return jsonify({"singer": "Mendengarkan...", "status": "VAD Filtered"})

        vad_frame = y_pcm[:frame_len_bytes]
        if not processor.is_speech(vad_frame, TARGET_SR):
            return jsonify({
                "singer": "Hening/Noise...",
                "confidence": 0.0,
                "status": "VAD Filtered"
            })

        # 4. Inference Model (ResNet-50)
        spec = processor.preprocess_audio(y)
        input_tensor = processor.prepare_for_resnet(spec)
        result = model_info.predict(input_tensor)
        
        current_singer = result['singer']
        current_conf = result['confidence']

        # 5. Majority Voting Logic
        voting_buffer.append(current_singer)
        if len(voting_buffer) > BUFFER_SIZE:
            voting_buffer.pop(0)

        # Ambil keputusan berdasarkan suara terbanyak di buffer
        final_decision = Counter(voting_buffer).most_common(1)[0][0]

        print(f"DEBUG: Raw: {current_singer} ({current_conf:.2f}) | Vote: {final_decision}")

        return jsonify({
            "singer": final_decision,
            "confidence": current_conf,
            "status": "Success",
            "raw_prediction": current_singer
        })

    except Exception as e:
        print("⚠️ Error saat prediksi:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Threaded=True agar UI tidak freeze saat memproses audio
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)