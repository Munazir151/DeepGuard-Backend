"""
Configuration file for DeepGuard Backend
"""

import os

# Application Configuration
APP_NAME = "DeepGuard: Deepfake Face Detection System"
VERSION = "1.0.0"
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 5000))

# CORS Configuration
# Add your Vercel deployment URL here after deployment
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3000",
    # Add your production frontend URLs below:
    # "https://your-app.vercel.app",
    # "https://deepguard.vercel.app",
]

# Allow all origins in development (for testing)
if os.getenv("FLASK_ENV") != "production":
    CORS_ORIGINS.append("*")

# File Upload Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Model Configuration
# Using the trained model from Epoch 1 (74.4% validation accuracy)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'xception_model.h5')
INPUT_SIZE = (224, 224)

# Video Processing Configuration
VIDEO_FRAME_COUNT = 10  # Number of frames to extract from video
VIDEO_SAMPLING_METHOD = 'uniform'  # 'uniform' or 'random'

# Face Detection Configuration
FACE_DETECTION_PADDING = 0.2  # 20% padding around detected face
FACE_REQUIRED_SIZE = (224, 224)

# Frequency Analysis Configuration
ENABLE_FREQUENCY_ANALYSIS = True
FREQUENCY_WEIGHT = 0.2  # Weight for frequency analysis in final decision

# Logging Configuration
LOG_FOLDER = os.path.join(os.path.dirname(__file__), 'logs')
LOG_FILE = os.path.join(LOG_FOLDER, 'deepguard.log')
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LOG_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
