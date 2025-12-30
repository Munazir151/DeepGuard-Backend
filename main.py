from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from typing import Optional
import time

app = FastAPI(title="Deepfake Image Detection API")

# Add CORS middleware to allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://localhost:3001", 
        "http://localhost:3002",
        "https://deep-guard.vercel.app",
        "https://deep-guard.vercel.app/"
    ],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
print("Loading model...")
models = {}

model_files = {
    "inception": "deepfake_detection_inception.h5",
}

for model_name, model_file in model_files.items():
    try:
        print(f"Loading {model_name} from {model_file}...")
        models[model_name] = tf.keras.models.load_model(model_file, compile=False)
        print(f"✓ Successfully loaded {model_name}")
    except Exception as e:
        print(f"✗ Failed to load {model_name}: {str(e)[:100]}")

if not models:
    print("WARNING: No models were loaded successfully!")
else:
    print(f"Successfully loaded {len(models)} model(s)!")
    print(f"Available models: {list(models.keys())}")

IMAGE_SIZE = (224, 224)

# Model-specific preprocessing
preprocessing_functions = {
    "inception": inception_preprocess,
}

def get_preprocess_fn(model_name):
    return preprocessing_functions.get(model_name, preprocess_input)

def preprocess_image(image_bytes, model_name="xception"):
    # Convert bytes to OpenCV image
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE)
    
    # Use model-specific preprocessing
    preprocess_fn = get_preprocess_fn(model_name)
    img = preprocess_fn(img)
    img = np.expand_dims(img, axis=0)

    return img


@app.get("/models")
async def get_models():
    """Get list of available models"""
    return {
        "models": list(models.keys()),
        "total": len(models),
        "default": "ensemble" if len(models) > 1 else list(models.keys())[0] if models else None
    }


@app.post("/predict")
async def predict_image(
    file: UploadFile = File(...),
    model_name: Optional[str] = Query(default="ensemble", description="Model to use: ensemble or specific model name")
):
    if not models:
        return {"error": "No models are currently loaded"}
    
    start_time = time.time()
    image_bytes = await file.read()
    
    if model_name == "ensemble":
        # Use all models and average predictions
        predictions = []
        model_results = {}
        
        for name, model in models.items():
            img = preprocess_image(image_bytes, name)
            if img is None:
                return {"error": "Invalid image file"}
            
            pred = model.predict(img, verbose=0)[0][0]
            predictions.append(pred)
            model_results[name] = {
                "prediction": round(float(pred) * 100, 2),
                "label": "FAKE" if pred >= 0.5 else "REAL"
            }
        
        # Average ensemble prediction
        avg_prediction = np.mean(predictions)
        
        label = "FAKE" if avg_prediction >= 0.5 else "REAL"
        confidence = avg_prediction if avg_prediction >= 0.5 else 1 - avg_prediction
        
        processing_time = time.time() - start_time
        
        return {
            "label": label,
            "confidence": round(confidence * 100, 2),
            "probabilities": {
                "REAL": round((1 - avg_prediction) * 100, 2),
                "FAKE": round(avg_prediction * 100, 2)
            },
            "processing_time": round(processing_time, 2),
            "model_used": "ensemble",
            "individual_models": model_results
        }
    
    else:
        # Use specific model
        if model_name not in models:
            return {"error": f"Model '{model_name}' not found. Available models: {list(models.keys())}"}
        
        img = preprocess_image(image_bytes, model_name)
        if img is None:
            return {"error": "Invalid image file"}
        
        model = models[model_name]
        prediction = model.predict(img, verbose=0)[0][0]
        
        label = "FAKE" if prediction >= 0.5 else "REAL"
        confidence = prediction if prediction >= 0.5 else 1 - prediction
        
        processing_time = time.time() - start_time
        
        return {
            "label": label,
            "confidence": round(confidence * 100, 2),
            "probabilities": {
                "REAL": round((1 - prediction) * 100, 2),
                "FAKE": round(prediction * 100, 2)
            },
            "processing_time": round(processing_time, 2),
            "model_used": model_name
        }
