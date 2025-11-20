---
title: DeepFake Detection
emoji: 🔍
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
app_port: 8080
---

# DeepGuard: Deepfake Face Detection System - Backend

A comprehensive Python backend for detecting deepfake images and videos using deep learning.

## 🚀 Features

- **XceptionNet-based Model**: Fine-tuned pretrained model for binary classification (Real vs Fake)
- **Face Detection**: Automatic face detection and cropping using MTCNN
- **Video Support**: Process video files by analyzing multiple frames
- **Frequency Analysis**: Optional FFT/DCT analysis for improved detection
- **Grad-CAM Visualization**: Heatmaps showing manipulated regions
- **REST API**: Flask-based API with CORS support for Next.js frontend

## 📋 Requirements

- Python 3.8 or higher
- TensorFlow 2.15+
- OpenCV
- Flask

## 🛠️ Installation

1. **Navigate to backend directory**:
```powershell
cd backend
```

2. **Create a virtual environment** (recommended):
```powershell
python -m venv venv
.\venv\Scripts\activate
```

3. **Install dependencies**:
```powershell
pip install -r requirements.txt
```

## 🎯 Model Training

### Option 1: Train with Your Data

Prepare your dataset in the following structure:
```
data/
  train/
    real/
      image1.jpg
      image2.jpg
      ...
    fake/
      image1.jpg
      image2.jpg
      ...
  val/
    real/
      ...
    fake/
      ...
```

Then run:
```powershell
python train_model.py
```

### Option 2: Create Model Structure for Testing

If you don't have training data yet, create a model structure:
```powershell
python train_model.py
```

This will create an untrained model structure that you can use for testing the API.

## 🚀 Running the Backend

Start the Flask server:
```powershell
python app.py
```

The API will be available at: `http://localhost:5000`

## 📡 API Endpoints

### 1. Health Check
```
GET /health
```

**Response**:
```json
{
  "status": "running",
  "app_name": "DeepGuard: Deepfake Face Detection System",
  "version": "1.0.0",
  "model_loaded": true
}
```

### 2. Predict (Image/Video)
```
POST /predict
```

**Request**:
- Method: POST
- Content-Type: multipart/form-data
- Body: file (image: .jpg, .png | video: .mp4, .avi, .mov)

**Response (Image)**:
```json
{
  "label": "FAKE",
  "confidence": 87.45,
  "probabilities": {
    "REAL": 12.55,
    "FAKE": 87.45
  },
  "frequency_score": 0.678,
  "adjusted_confidence": 88.91,
  "processing_time": 1.23,
  "file_type": "image"
}
```

**Response (Video)**:
```json
{
  "label": "FAKE",
  "confidence": 85.32,
  "probabilities": {
    "REAL": 14.68,
    "FAKE": 85.32
  },
  "frames_analyzed": 10,
  "consistency_score": 92.45,
  "total_frames": 10,
  "faces_detected": 9,
  "processing_time": 5.67,
  "file_type": "video",
  "frame_predictions": [
    {"frame_number": 1, "prob_fake": 82.3},
    {"frame_number": 2, "prob_fake": 88.1},
    ...
  ]
}
```

### 3. Grad-CAM Visualization
```
POST /gradcam
```

**Request**:
- Method: POST
- Content-Type: multipart/form-data
- Body: file (image only: .jpg, .png)

**Response**:
```json
{
  "gradcam_image": "data:image/png;base64,iVBORw0KG...",
  "prediction": {
    "label": "FAKE",
    "confidence": 87.45,
    "probabilities": {
      "REAL": 12.55,
      "FAKE": 87.45
    }
  },
  "layer_name": "block14_sepconv2_act"
}
```

## 🧪 Testing the API

### Using cURL (PowerShell)

**Test Image**:
```powershell
$file = Get-Item "path\to\image.jpg"
curl.exe -X POST -F "file=@$($file.FullName)" http://localhost:5000/predict
```

**Test Video**:
```powershell
$file = Get-Item "path\to\video.mp4"
curl.exe -X POST -F "file=@$($file.FullName)" http://localhost:5000/predict
```

### Using Python

```python
import requests

# Test image
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/predict',
        files={'file': f}
    )
    print(response.json())

# Test video
with open('video.mp4', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/predict',
        files={'file': f}
    )
    print(response.json())
```

## 📁 Project Structure

```
backend/
│
├── app.py                      # Main Flask application
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
├── train_model.py             # Model training script
│
├── model/
│   ├── predictor.py           # Model inference logic
│   └── xception_model.h5      # Trained model (generated)
│
├── utils/
│   ├── face_detector.py       # MTCNN face detection
│   ├── preprocessing.py       # Image/video preprocessing
│   ├── frequency_analysis.py  # FFT/DCT analysis
│   └── gradcam.py            # Grad-CAM visualization
│
├── uploads/                   # Temporary file storage
└── logs/                      # Application logs
```

## ⚙️ Configuration

Edit `config.py` to customize:

- **Port**: Change `PORT` (default: 5000)
- **CORS Origins**: Add your frontend URL to `CORS_ORIGINS`
- **Video Frame Count**: Adjust `VIDEO_FRAME_COUNT` (default: 10)
- **Model Path**: Change `MODEL_PATH` if needed
- **File Size Limit**: Modify `MAX_FILE_SIZE` (default: 50MB)

## 🔧 Environment Variables

You can override settings using environment variables:

```powershell
$env:PORT = "8000"
$env:DEBUG = "True"
$env:LOG_LEVEL = "DEBUG"
python app.py
```

## 📊 Model Details

- **Architecture**: XceptionNet (pretrained on ImageNet)
- **Input Size**: 224×224×3
- **Output**: Binary classification (Real: 0, Fake: 1)
- **Training**: Two-phase fine-tuning
  - Phase 1: Train top layers only
  - Phase 2: Fine-tune with unfrozen layers

## 🎨 Features Explained

### Face Detection (MTCNN)
- Automatically detects faces in images/videos
- Crops face with 20% padding
- Handles multiple faces (uses largest)

### Frequency Analysis (Optional)
- FFT (Fast Fourier Transform) analysis
- DCT (Discrete Cosine Transform) analysis
- Detects artifacts in frequency domain
- Adjusts confidence scores

### Grad-CAM Visualization
- Shows which regions influenced the decision
- Highlights manipulated areas
- Helps understand model predictions

## 🐛 Troubleshooting

### Model Not Found
If you see "Model file not found" error:
```powershell
python train_model.py
```

### No Face Detected
Ensure:
- Image contains a clear, frontal face
- Face is not too small or occluded
- Image quality is good

### CORS Issues
Add your frontend URL to `CORS_ORIGINS` in `config.py`:
```python
CORS_ORIGINS = [
    "http://localhost:3000",
    "your-frontend-url"
]
```

## 📝 Logs

Application logs are stored in:
- `logs/deepguard.log`

Monitor logs in real-time:
```powershell
Get-Content logs\deepguard.log -Wait
```

## 🔐 Security Notes

For production deployment:
- Use environment variables for sensitive config
- Implement authentication/authorization
- Add rate limiting
- Validate file types thoroughly
- Use HTTPS
- Limit file upload sizes

## 📈 Performance Tips

- Use GPU for faster inference (CUDA-enabled TensorFlow)
- Reduce `VIDEO_FRAME_COUNT` for faster video processing
- Implement caching for repeated requests
- Use asynchronous processing for large files

## 🤝 Integration with Frontend

The backend is ready to work with your Next.js frontend. Ensure:

1. Backend is running on `http://localhost:5000`
2. Frontend is configured to use this URL
3. CORS is properly configured

Example frontend integration:
```javascript
const formData = new FormData();
formData.append('file', file);

const response = await fetch('http://localhost:5000/predict', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log(result);
```

## 📚 Additional Resources

- TensorFlow: https://www.tensorflow.org/
- Flask: https://flask.palletsprojects.com/
- MTCNN: https://github.com/ipazc/mtcnn
- Xception Paper: https://arxiv.org/abs/1610.02357

## 📄 License

This project is part of DeepGuard: Deepfake Face Detection System.

## 👨‍💻 Author

Built as a comprehensive deepfake detection solution.

---

**Happy Detection! 🎯**
