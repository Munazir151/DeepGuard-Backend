# 🚀 Quick Start - Docker Deployment

Deploy the DeepFake Detection backend using Docker in 3 simple steps!

## Prerequisites
- Docker Desktop installed ([Download](https://www.docker.com/products/docker-desktop))
- Model file exists at `backend/model/xception_model.h5`

## Option 1: One-Click Start (Easiest) ⚡

Open PowerShell in the `backend` directory and run:

```powershell
.\docker-start.ps1
```

That's it! The script will:
- ✅ Check if Docker is running
- ✅ Build the container image
- ✅ Start the backend on http://localhost:8080
- ✅ Run health checks
- ✅ Show logs

## Option 2: Manual Docker Compose 🐳

```powershell
# Navigate to backend directory
cd backend

# Start the container
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

## Option 3: Docker CLI 🔧

```powershell
# Navigate to backend directory
cd backend

# Build the image
docker build -t deepfake-backend .

# Run the container
docker run -d --name deepfake-backend -p 8080:8080 deepfake-backend

# View logs
docker logs -f deepfake-backend

# Stop and remove
docker stop deepfake-backend && docker rm deepfake-backend
```

## Test Your Deployment ✅

Once running, test these endpoints:

```powershell
# Health check
curl http://localhost:8080/health

# Ping
curl http://localhost:8080/ping

# Or open in browser:
# http://localhost:8080/health
```

## Common Commands 📝

```powershell
# View running containers
docker ps

# View logs
docker-compose logs -f

# Restart backend
docker-compose restart

# Stop backend
docker-compose down

# Rebuild from scratch
docker-compose up -d --build --no-cache
```

## Troubleshooting 🔍

**Port 8080 already in use?**
```powershell
# Check what's using port 8080
netstat -ano | findstr :8080

# Change port in docker-compose.yml:
ports:
  - "9000:8080"  # Use port 9000 instead
```

**Container won't start?**
```powershell
# Check logs for errors
docker-compose logs

# Ensure Docker Desktop has enough memory (4GB+)
# Settings → Resources → Advanced → Memory
```

**Model not found?**
- Ensure `xception_model.h5` exists in `backend/model/`
- The file should be ~80-100MB

## What's Next? 🎯

- Frontend integration: Update `my-app/.env.local` with `NEXT_PUBLIC_API_URL=http://localhost:8080`
- Production deployment: See `DOCKER_DEPLOYMENT.md` for cloud deployment options
- Monitoring: Add logging and metrics

## Full Documentation 📚

For detailed instructions, see: `DOCKER_DEPLOYMENT.md`
