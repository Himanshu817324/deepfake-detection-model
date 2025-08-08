# python-model-server/app.py
import os
import cv2
import torch
import torch.nn as nn
import timm
import numpy as np
from flask import Flask, request, jsonify
from facenet_pytorch import MTCNN
from torchvision import transforms
from PIL import Image
import io
import tempfile
import warnings
import requests # <-- Add this import

warnings.filterwarnings('ignore')

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Model Download URL ---
# IMPORTANT: Replace this with the direct download link to your .pth file from Hugging Face Hub
MODEL_URL = "https://huggingface.co/dhaal2025/deepfake-detector/resolve/main/best_deepfake_model.pth?download=true" 
MODEL_PATH = 'best_deepfake_model.pth'


# --- Model Architecture ---
class DeepfakeDetector(nn.Module):
    def __init__(self, num_frames=8, hidden_size=128, num_classes=2, dropout=0.5):
        super(DeepfakeDetector, self).__init__()
        self.num_frames = num_frames
        self.hidden_size = hidden_size
        
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        feature_dim = self.backbone.num_features
        
        self.lstm = nn.LSTM(feature_dim, hidden_size, batch_first=True, dropout=dropout)
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        if len(x.shape) == 5:
            num_frames = x.size(1)
            x = x.view(batch_size * num_frames, x.size(2), x.size(3), x.size(4))
        else:
            num_frames = 1

        features = self.backbone(x)
        
        if num_frames > 1:
            features = features.view(batch_size, num_frames, -1)
        else:
            features = features.unsqueeze(1)

        lstm_out, _ = self.lstm(features)
        
        if num_frames == 1:
            pooled_features = lstm_out[:, -1, :]
        else:
            pooled_features = torch.mean(lstm_out, dim=1)

        output = self.classifier(pooled_features)
        return output

# --- Global Variables ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
mtcnn = None
transform = None

def download_file(url, filename):
    """Helper function to download a file from a URL."""
    print(f"Downloading model from {url}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Model downloaded successfully to {filename}")
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False
    return True

def load_model():
    """Download the model if it doesn't exist, then load it."""
    global model, mtcnn, transform
    
    # Download the model file if it's not already present
    if not os.path.exists(MODEL_PATH):
        if not MODEL_URL or "YOUR_HUGGING_FACE" in MODEL_URL:
             print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
             print("!!! ERROR: MODEL_URL is not set in app.py.             !!!")
             print("!!! You must upload best_deepfake_model.pth to a host  !!!")
             print("!!! (like Hugging Face Hub) and set the download URL.  !!!")
             print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
             return
        download_file(MODEL_URL, MODEL_PATH)

    print("Loading model...")
    model = DeepfakeDetector()
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    model.to(DEVICE)
    model.eval()
    
    print("Initializing MTCNN...")
    mtcnn = MTCNN(
        image_size=128, margin=20, keep_all=False, 
        min_face_size=40, device=DEVICE, select_largest=True
    )
    print("MTCNN initialized.")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def predict_on_frames(face_frames):
    """Run prediction on a list of face frames."""
    if not face_frames:
        return {"result": "error", "message": "No faces detected."}

    face_tensors = torch.stack(face_frames).to(DEVICE)
    
    with torch.no_grad():
        if face_tensors.dim() == 3:
            face_tensors = face_tensors.unsqueeze(0)
        if face_tensors.dim() == 4:
             face_tensors = face_tensors.unsqueeze(0)

        outputs = model(face_tensors)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
        
        result_label = 'FAKE' if predicted.item() == 1 else 'AUTHENTIC'
        confidence_score = confidence.item() * 100
        
        return {"result": result_label.lower(), "confidence": round(confidence_score, 2)}

# --- API Endpoints ---
@app.route('/predict/<media_type>', methods=['POST'])
def predict(media_type):
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        if media_type == 'image':
            image_bytes = file.read()
            pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            face = mtcnn(pil_image)
            if face is None:
                return jsonify({"result": "error", "message": "No face detected in the image."})
            face = (face - 127.5) / 128.0
            face_tensor = transform(face.permute(1, 2, 0).cpu().numpy()).to(DEVICE)
            return jsonify(predict_on_frames([face_tensor]))
        
        elif media_type == 'video':
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                file.save(tmp.name)
                video_path = tmp.name

            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices = np.linspace(0, total_frames - 1, 8, dtype=int)
            face_frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    face = mtcnn(pil_frame)
                    if face is not None:
                        face = (face - 127.5) / 128.0
                        face_tensor = transform(face.permute(1, 2, 0).cpu().numpy()).to(DEVICE)
                        face_frames.append(face_tensor)
            cap.release()
            os.unlink(video_path)
            return jsonify(predict_on_frames(face_frames))
        
        else:
            return jsonify({"error": "Invalid media type"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


load_model()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)