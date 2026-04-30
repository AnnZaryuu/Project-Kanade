import torch
import torchvision.models as models
import torch.nn as nn
import json
import os
import numpy as np

class ModelLoader:
    def __init__(self, model_path: str, class_names_path: str, device: str = 'cpu'):
        """
        Inisialisasi Model Loader untuk Project Koe no Katachi.
        """
        self.device = torch.device(device)
        self.model_path = model_path
        
        # 1. Load Mapping Penyanyi dari JSON
        if not os.path.exists(class_names_path):
            raise FileNotFoundError(f"❌ File JSON tidak ditemukan: {class_names_path}")
            
        with open(class_names_path, 'r') as f:
            # Pastikan class_names terisi mapping terbaru dari Colab
            self.class_names = json.load(f)
        
        # 2. Bangun Arsitektur ResNet-50
        # Menggunakan jumlah kelas dinamis sesuai isi JSON
        self.model = self._setup_architecture(num_classes=len(self.class_names))
        
        # 3. Load Bobot (.pth)
        self._load_weights()
        
        # 4. Set ke Mode Evaluasi
        self.model.eval()

    def _setup_architecture(self, num_classes: int):
        """
        Menyiapkan struktur ResNet-50.
        """
        # weights=None karena kita menggunakan bobot custom sendiri
        model = models.resnet50(weights=None)
        
        # Modifikasi Fully Connected layer (fc) untuk output penyanyi
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
        return model.to(self.device)

    def _load_weights(self):
        """
        Memasukkan file .pth ke arsitektur model.
        """
        if os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"✅ Berhasil memuat model: {os.path.basename(self.model_path)}")
        else:
            raise FileNotFoundError(f"❌ File model {self.model_path} tidak ditemukan!")

    def predict(self, input_tensor):
        """
        Prediksi penyanyi dari input spektrogram.
        """
        with torch.no_grad():
            # Konversi ke tensor jika input masih berupa numpy array
            if isinstance(input_tensor, np.ndarray):
                tensor = torch.from_numpy(input_tensor).float().to(self.device)
            else:
                tensor = input_tensor.to(self.device)
            
            # Forward Pass
            outputs = self.model(tensor)
            
            # Ambil probabilitas (Softmax)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Cari skor tertinggi
            conf, predicted = torch.max(probabilities, 1)
            class_id = str(predicted.item())
            
            return {
                "class_id": class_id,
                "singer": self.class_names.get(class_id, "Unknown"),
                "confidence": float(conf.item())
            }