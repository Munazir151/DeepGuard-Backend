# Quick Docker Start Script for DeepFake Detection Backend
# Run this script from the backend directory

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  DeepFake Detection Backend - Docker Deployment  " -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is running
Write-Host "Checking Docker..." -ForegroundColor Yellow
try {
    docker info | Out-Null
    Write-Host "✓ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "✗ Docker is not running. Please start Docker Desktop." -ForegroundColor Red
    Write-Host "  Download from: https://www.docker.com/products/docker-desktop" -ForegroundColor Yellow
    exit 1
}

# Check if model file exists
Write-Host ""
Write-Host "Checking model file..." -ForegroundColor Yellow
if (Test-Path "model\xception_model.h5") {
    Write-Host "✓ Model file found" -ForegroundColor Green
} else {
    Write-Host "⚠ Warning: Model file not found at model\xception_model.h5" -ForegroundColor Yellow
    Write-Host "  The container will start but predictions may fail." -ForegroundColor Yellow
}

# Stop and remove existing container if running
Write-Host ""
Write-Host "Cleaning up existing containers..." -ForegroundColor Yellow
docker-compose down 2>$null

# Build and start the container
Write-Host ""
Write-Host "Building and starting Docker container..." -ForegroundColor Yellow
Write-Host "This may take a few minutes on first run..." -ForegroundColor Gray
docker-compose up -d --build

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "==================================================" -ForegroundColor Green
    Write-Host "  ✓ Backend is running successfully!             " -ForegroundColor Green
    Write-Host "==================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Backend URL: " -NoNewline
    Write-Host "http://localhost:8080" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Quick Tests:" -ForegroundColor Yellow
    Write-Host "  Health Check:  " -NoNewline
    Write-Host "http://localhost:8080/health" -ForegroundColor Cyan
    Write-Host "  Ping:          " -NoNewline
    Write-Host "http://localhost:8080/ping" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Useful Commands:" -ForegroundColor Yellow
    Write-Host "  View logs:     " -NoNewline -ForegroundColor Gray
    Write-Host "docker-compose logs -f" -ForegroundColor White
    Write-Host "  Stop backend:  " -NoNewline -ForegroundColor Gray
    Write-Host "docker-compose down" -ForegroundColor White
    Write-Host "  Restart:       " -NoNewline -ForegroundColor Gray
    Write-Host "docker-compose restart" -ForegroundColor White
    Write-Host ""
    
    # Wait a moment for container to start
    Start-Sleep -Seconds 3
    
    # Test the health endpoint
    Write-Host "Testing health endpoint..." -ForegroundColor Yellow
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8080/health" -TimeoutSec 5 -UseBasicParsing
        if ($response.StatusCode -eq 200) {
            Write-Host "✓ Health check passed!" -ForegroundColor Green
            $response.Content | ConvertFrom-Json | ConvertTo-Json
        }
    } catch {
        Write-Host "⚠ Health check pending... Container may still be starting." -ForegroundColor Yellow
        Write-Host "  Run: docker-compose logs -f" -ForegroundColor Gray
    }
    
    Write-Host ""
    Write-Host "Opening logs in 5 seconds... (Press Ctrl+C to exit)" -ForegroundColor Gray
    Start-Sleep -Seconds 5
    docker-compose logs -f
    
} else {
    Write-Host ""
    Write-Host "✗ Failed to start container" -ForegroundColor Red
    Write-Host "Check the errors above for details." -ForegroundColor Yellow
    exit 1
}
