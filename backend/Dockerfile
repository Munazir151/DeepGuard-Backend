FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ .

# Copy and make startup script executable
RUN chmod +x start.sh

# Create necessary directories
RUN mkdir -p uploads logs data/temp model

# Expose port
EXPOSE 8080

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production

# Run the application using startup script
CMD ["./start.sh"]
