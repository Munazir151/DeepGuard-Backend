#!/bin/sh
# Render startup script with better error handling

echo "========================================="
echo "DeepFake Detection Backend - Starting..."
echo "========================================="
echo "PORT: $PORT"
echo "FLASK_ENV: $FLASK_ENV"
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo ""

# Create required directories
echo "Creating directories..."
mkdir -p uploads logs data/temp model
echo "✓ Directories created"

# List files to verify structure
echo ""
echo "Files in current directory:"
ls -lh

echo ""
echo "Files in model directory:"
ls -lh model/ || echo "Model directory not accessible"

# Check if model file exists
echo ""
if [ -f "model/xception_model.h5" ]; then
    echo "✓ Model file found: model/xception_model.h5"
    ls -lh model/xception_model.h5
else
    echo "⚠ WARNING: Model file not found at model/xception_model.h5"
    echo "  Predictions will fail until model is uploaded"
    echo "  The API will still start for health checks"
fi

# Check Python dependencies
echo ""
echo "Checking critical dependencies..."
python -c "import flask; print('✓ Flask:', flask.__version__)" || echo "✗ Flask not found"
python -c "import tensorflow; print('✓ TensorFlow:', tensorflow.__version__)" || echo "✗ TensorFlow not found"
python -c "import cv2; print('✓ OpenCV:', cv2.__version__)" || echo "✗ OpenCV not found"

# Start Gunicorn
echo ""
echo "========================================="
echo "Starting Gunicorn on port $PORT..."
echo "========================================="

# Use --preload to catch startup errors early
exec gunicorn app:app \
    --bind 0.0.0.0:$PORT \
    --timeout 300 \
    --workers 1 \
    --threads 2 \
    --worker-class gthread \
    --log-level debug \
    --access-logfile - \
    --error-logfile - \
    --capture-output \
    --enable-stdio-inheritance
