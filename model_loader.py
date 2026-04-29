import torch
import torchvision.models as models
import torch.nn as nn
import json
import os

class ModelLoader:
    def __init__(self, model_path, class_names_path, device='cpu'):
        self.device = torch.device(device)
        self.model_path = model_path
        
        # 1. Load Class Names
        with open(class_names_path, 'r') as f:
            self.class_names = json.load(f)
        
        # 2. Inisialisasi Arsitektur ResNet-50
        self.model = self.load_architecture(num_classes=len(self.class_names))
        
        # 3. Load Weights (.pth)
        self.load_weights()
        self.model.eval() # Set ke mode evaluasi (penting untuk inference!)

    def load_architecture(self, num_classes):
        # Gunakan weights=None karena kita akan pakai weights custom dari .pth
        model = models.resnet50(weights=None)
        
        # Modifikasi layer terakhir agar sesuai dengan 10 penyanyi
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        return model.to(self.device)

    def load_weights(self):
        if os.path.exists(self.model_path):
            # Load state_dict (bobot) ke dalam arsitektur
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            print(f"✅ Berhasil memuat model dari: {self.model_path}")
        else:
            raise FileNotFoundError(f"❌ File model tidak ditemukan di: {self.model_path}")

    def predict(self, input_tensor):
        """Melakukan prediksi dari tensor input"""
        with torch.no_grad():
            input_tensor = torch.from_numpy(input_tensor).float().to(self.device)
            outputs = self.model(input_tensor)
            
            # Ambil probabilitas menggunakan Softmax
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Ambil index kelas dengan nilai tertinggi
            conf, predicted = torch.max(probabilities, 1)
            
            return {
                "class_id": str(predicted.item()),
                "singer": self.class_names[str(predicted.item())],
                "confidence": conf.item()
            }