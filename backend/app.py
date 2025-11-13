"""
DeepGuard: Deepfake Face Detection System - Backend API
Flask REST API for deepfake detection with image and video support
"""

import os
import sys
import logging
import time
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import base64
from io import BytesIO

# Import local modules
from config import *
from model.predictor import get_predictor
from utils.face_detector import get_face_detector
from utils.preprocessing import ImagePreprocessor, VideoPreprocessor
from utils.frequency_analysis import analyze_image_frequency
from utils.gradcam import create_gradcam_visualization

# Configure logging
os.makedirs(LOG_FOLDER, exist_ok=True)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Enable CORS
CORS(app, origins=CORS_ORIGINS, supports_credentials=True)

# Global instances (lazy loading)
predictor = None
face_detector = None


def init_models():
    """Initialize models on first request"""
    global predictor, face_detector
    
    if predictor is None:
        logger.info("Initializing model predictor...")
        try:
            predictor = get_predictor(MODEL_PATH)
            logger.info("✓ Model predictor initialized")
        except Exception as e:
            logger.error(f"✗ Failed to initialize predictor: {str(e)}")
            raise
    
    if face_detector is None:
        logger.info("Initializing face detector...")
        try:
            face_detector = get_face_detector()
            logger.info("✓ Face detector initialized")
        except Exception as e:
            logger.error(f"✗ Failed to initialize face detector: {str(e)}")
            raise


def allowed_file(filename, file_type='image'):
    """Check if file extension is allowed"""
    if '.' not in filename:
        return False
    
    ext = filename.rsplit('.', 1)[1].lower()
    
    if file_type == 'image':
        return ext in ALLOWED_IMAGE_EXTENSIONS
    elif file_type == 'video':
        return ext in ALLOWED_VIDEO_EXTENSIONS
    else:
        return ext in ALLOWED_IMAGE_EXTENSIONS or ext in ALLOWED_VIDEO_EXTENSIONS


def process_image(file_bytes, filename):
    """
    Process an image file for deepfake detection
    
    Args:
        file_bytes: bytes, image file data
        filename: str, original filename
        
    Returns:
        dict: Prediction results
    """
    try:
        logger.info(f"Processing image: {filename}")
        start_time = time.time()
        
        # Load image
        image = ImagePreprocessor.load_image_from_bytes(file_bytes)
        if image is None:
            return {'error': 'Failed to load image'}, 400
        
        # Detect and extract face
        logger.info("Detecting face in image...")
        face = face_detector.extract_face(image, required_size=FACE_REQUIRED_SIZE)
        
        if face is None:
            logger.warning("No face detected in image")
            return {'error': 'No face detected in image'}, 400
        
        logger.info("✓ Face detected and extracted")
        
        # Preprocess for model
        preprocessed = ImagePreprocessor.preprocess_for_model(face)
        
        # Make prediction
        logger.info("Running model prediction...")
        prediction = predictor.predict_single(preprocessed)
        
        if prediction is None:
            return {'error': 'Prediction failed'}, 500
        
        logger.info(f"✓ Prediction: {prediction['label']} ({prediction['confidence']:.2f}%)")
        
        # Optional: Add frequency analysis
        if ENABLE_FREQUENCY_ANALYSIS:
            logger.info("Running frequency analysis...")
            freq_analysis = analyze_image_frequency(face)
            freq_score = freq_analysis['frequency_score']
            
            # Adjust confidence based on frequency analysis
            adjustment = (freq_score - 0.5) * FREQUENCY_WEIGHT * 100
            adjusted_confidence = prediction['confidence']
            
            if prediction['label'] == 'FAKE':
                adjusted_confidence = min(100, adjusted_confidence + adjustment)
            else:
                adjusted_confidence = max(0, adjusted_confidence - abs(adjustment))
            
            prediction['frequency_score'] = round(freq_score, 3)
            prediction['adjusted_confidence'] = round(adjusted_confidence, 2)
            logger.info(f"✓ Frequency analysis complete (score: {freq_score:.3f})")
        
        # Add processing time
        processing_time = time.time() - start_time
        prediction['processing_time'] = round(processing_time, 2)
        prediction['file_type'] = 'image'
        
        logger.info(f"✓ Image processing completed in {processing_time:.2f}s")
        return prediction, 200
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return {'error': f'Processing failed: {str(e)}'}, 500


def process_video(file_bytes, filename):
    """
    Process a video file for deepfake detection
    
    Args:
        file_bytes: bytes, video file data
        filename: str, original filename
        
    Returns:
        dict: Prediction results
    """
    try:
        logger.info(f"Processing video: {filename}")
        start_time = time.time()
        
        # Save video temporarily
        temp_video_path = os.path.join(UPLOAD_FOLDER, secure_filename(filename))
        with open(temp_video_path, 'wb') as f:
            f.write(file_bytes)
        
        # Extract frames
        logger.info(f"Extracting {VIDEO_FRAME_COUNT} frames from video...")
        frames = VideoPreprocessor.extract_frames(
            temp_video_path,
            num_frames=VIDEO_FRAME_COUNT,
            method=VIDEO_SAMPLING_METHOD
        )
        
        # Clean up temporary file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        
        if not frames:
            logger.warning("Failed to extract frames from video")
            return {'error': 'Failed to extract frames from video'}, 400
        
        logger.info(f"✓ Extracted {len(frames)} frames")
        
        # Process each frame
        preprocessed_frames = []
        faces_detected = 0
        
        for i, frame in enumerate(frames):
            # Detect and extract face
            face = face_detector.extract_face(frame, required_size=FACE_REQUIRED_SIZE)
            
            if face is not None:
                preprocessed = ImagePreprocessor.preprocess_for_model(face)
                preprocessed_frames.append(preprocessed)
                faces_detected += 1
            else:
                logger.debug(f"No face in frame {i+1}")
        
        if not preprocessed_frames:
            logger.warning("No faces detected in any frame")
            return {'error': 'No faces detected in video'}, 400
        
        logger.info(f"✓ Detected faces in {faces_detected}/{len(frames)} frames")
        
        # Make prediction on all frames
        logger.info("Running model prediction on frames...")
        prediction = predictor.predict_video_frames(preprocessed_frames)
        
        if prediction is None:
            return {'error': 'Prediction failed'}, 500
        
        logger.info(f"✓ Video prediction: {prediction['label']} ({prediction['confidence']:.2f}%)")
        
        # Add processing time
        processing_time = time.time() - start_time
        prediction['processing_time'] = round(processing_time, 2)
        prediction['file_type'] = 'video'
        prediction['total_frames'] = len(frames)
        prediction['faces_detected'] = faces_detected
        
        logger.info(f"✓ Video processing completed in {processing_time:.2f}s")
        return prediction, 200
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return {'error': f'Processing failed: {str(e)}'}, 500


# API Endpoints

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        status = {
            'status': 'running',
            'app_name': APP_NAME,
            'version': VERSION,
            'model_loaded': predictor is not None,
            'face_detector_loaded': face_detector is not None
        }
        
        if predictor is not None:
            model_info = predictor.get_model_info()
            if model_info:
                status['model_info'] = {
                    'input_shape': str(model_info['input_shape']),
                    'classes': model_info['class_names']
                }
        
        logger.info("Health check successful")
        return jsonify(status), 200
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    Accepts image or video file and returns deepfake detection result
    """
    try:
        # Initialize models on first request
        init_models()
        
        # Check if file is present
        if 'file' not in request.files:
            logger.warning("No file in request")
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            logger.warning("Empty filename")
            return jsonify({'error': 'No file selected'}), 400
        
        filename = secure_filename(file.filename)
        file_bytes = file.read()
        
        logger.info(f"Received file: {filename} ({len(file_bytes)} bytes)")
        
        # Determine file type and process accordingly
        if allowed_file(filename, 'image'):
            result, status_code = process_image(file_bytes, filename)
        elif allowed_file(filename, 'video'):
            result, status_code = process_video(file_bytes, filename)
        else:
            logger.warning(f"Unsupported file type: {filename}")
            return jsonify({'error': 'Unsupported file type. Allowed: jpg, png, mp4, avi, mov'}), 400
        
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"Prediction endpoint error: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/gradcam', methods=['POST'])
def gradcam():
    """
    Grad-CAM visualization endpoint
    Returns heatmap showing important regions for the prediction
    """
    try:
        # Initialize models
        init_models()
        
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        filename = secure_filename(file.filename)
        file_bytes = file.read()
        
        logger.info(f"Generating Grad-CAM for: {filename}")
        
        # Only process images
        if not allowed_file(filename, 'image'):
            return jsonify({'error': 'Grad-CAM only supports image files'}), 400
        
        # Load and process image
        image = ImagePreprocessor.load_image_from_bytes(file_bytes)
        if image is None:
            return jsonify({'error': 'Failed to load image'}), 400
        
        # Detect and extract face
        face = face_detector.extract_face(image, required_size=FACE_REQUIRED_SIZE)
        if face is None:
            return jsonify({'error': 'No face detected in image'}), 400
        
        # Preprocess for model
        preprocessed = ImagePreprocessor.preprocess_for_model(face)
        
        # Generate Grad-CAM
        logger.info("Generating Grad-CAM visualization...")
        gradcam_result = create_gradcam_visualization(
            predictor.model,
            preprocessed,
            original_image=face
        )
        
        if gradcam_result is None:
            return jsonify({'error': 'Failed to generate Grad-CAM'}), 500
        
        # Convert overlay image to base64
        _, buffer = cv2.imencode('.png', gradcam_result['overlay'])
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Make prediction
        prediction = predictor.predict_single(preprocessed)
        
        result = {
            'gradcam_image': f'data:image/png;base64,{img_base64}',
            'prediction': prediction,
            'layer_name': gradcam_result['layer_name']
        }
        
        logger.info("✓ Grad-CAM generated successfully")
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Grad-CAM endpoint error: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/', methods=['GET'])
def index():
    """Root endpoint"""
    return jsonify({
        'app_name': APP_NAME,
        'version': VERSION,
        'endpoints': {
            '/health': 'GET - Health check',
            '/predict': 'POST - Deepfake detection (image/video)',
            '/gradcam': 'POST - Grad-CAM visualization (image only)'
        },
        'documentation': 'Send POST request to /predict with file in form-data'
    }), 200


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({'error': f'File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB'}), 413


@app.errorhandler(500)
def internal_server_error(error):
    """Handle internal server error"""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info(f"Starting {APP_NAME} v{VERSION}")
    logger.info("=" * 60)
    logger.info(f"Host: {HOST}:{PORT}")
    logger.info(f"Debug mode: {DEBUG}")
    logger.info(f"Model path: {MODEL_PATH}")
    logger.info(f"Upload folder: {UPLOAD_FOLDER}")
    logger.info(f"Log file: {LOG_FILE}")
    logger.info("=" * 60)
    
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        logger.warning(f"⚠ Model file not found: {MODEL_PATH}")
        logger.warning("⚠ Please train the model first using train_model.py")
        logger.warning("⚠ Or run: python train_model.py to create a model structure")
    else:
        logger.info(f"✓ Model file found: {MODEL_PATH}")
    
    # Start Flask app
    try:
        app.run(host=HOST, port=PORT, debug=DEBUG)
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        sys.exit(1)
