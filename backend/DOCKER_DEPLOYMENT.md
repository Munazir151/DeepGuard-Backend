# Render Deployment Guide - DeepFake Detection Backend

This guide will help you deploy the DeepFake Detection backend to Render using Docker.

## Prerequisites

- Render account ([Sign up free](https://render.com))
- GitHub account (to connect your repository)
- Model file (`xception_model.h5`) in the `backend/model/` directory

## Deployment Steps

### Step 1: Push Code to GitHub

1. **Ensure your code is committed:**
   ```powershell
   cd "C:\Users\Mohammed Munazir\deepfake detection2"
   git add .
   git commit -m "Prepare for Render deployment"
   git push origin main
   ```

### Step 2: Create Web Service on Render

1. **Go to Render Dashboard:**
   - Visit [https://dashboard.render.com](https://dashboard.render.com)
   - Click **"New +"** → **"Web Service"**

2. **Connect Repository:**
   - Select **"Build and deploy from a Git repository"**
   - Click **"Connect GitHub"** and authorize Render
   - Select your `deepfake_detection2` repository

3. **Configure Service:**
   - **Name:** `deepfake-detection-backend`
   - **Region:** Choose closest to your users
   - **Branch:** `main`
   - **Root Directory:** `backend`
   - **Runtime:** `Docker`
   - **Docker Build Context:** `backend`
   - **Dockerfile Path:** `./Dockerfile`

4. **Set Environment Variables:**
   Click **"Advanced"** and add:
   ```
   PORT=8080
   FLASK_ENV=production
   PYTHONUNBUFFERED=1
   HOST=0.0.0.0
   ```

5. **Select Plan:**
   - **Free Plan:** Limited resources, good for testing
   - **Starter Plan ($7/month):** Recommended for production
   - **Standard Plan:** Higher performance

6. **Deploy:**
   - Click **"Create Web Service"**
   - Render will automatically build and deploy your Docker container
   - Wait 5-10 minutes for the first deployment

### Step 3: Verify Deployment

Once deployed, Render provides a URL like: `https://deepfake-detection-backend.onrender.com`

**Test the endpoints:**
```powershell
# Health check
curl https://your-app-name.onrender.com/health

# Ping
curl https://your-app-name.onrender.com/ping
```

## Important Files for Render

Your repository already includes these required files:

- ✅ `backend/Dockerfile` - Container configuration
- ✅ `backend/requirements.txt` - Python dependencies
- ✅ `backend/start.sh` - Startup script
- ✅ `backend/render.yaml` - Render configuration (optional)

## Environment Variables

Configure these in Render Dashboard → Environment:

| Variable | Value | Description |
|----------|-------|-------------|
| `PORT` | `8080` | Port the server listens on (auto-set by Render) |
| `FLASK_ENV` | `production` | Environment mode |
| `PYTHONUNBUFFERED` | `1` | Enable Python logging |
| `HOST` | `0.0.0.0` | Host address |

## Monitoring & Logs

### View Logs
1. Go to your service in Render Dashboard
2. Click **"Logs"** tab
3. View real-time deployment and runtime logs

### Health Checks
Render automatically monitors: `https://your-app.onrender.com/health`

### Metrics
- View CPU, Memory usage in Dashboard
- Set up alerts for downtime

## Updating Your Deployment

Render automatically redeploys when you push to GitHub:

```powershell
# Make changes to your code
git add .
git commit -m "Update backend"
git push origin main
```

Render will detect the push and automatically rebuild and redeploy.

### Manual Deploy
You can also trigger manual deploys:
1. Go to Render Dashboard
2. Click **"Manual Deploy"** → **"Deploy latest commit"**

## Troubleshooting

### Build Fails

**Check logs in Render Dashboard:**
- Look for errors during Docker build
- Ensure `requirements.txt` has all dependencies
- Verify `Dockerfile` syntax is correct

**Common issues:**
- Missing `xception_model.h5` - ensure file is in repository
- Out of memory - upgrade to a larger plan
- Timeout during build - large model file may need optimization

### Model File Too Large

If the model file is too large for GitHub:

**Option 1: Use Git LFS**
```powershell
git lfs install
git lfs track "backend/model/*.h5"
git add .gitattributes
git add backend/model/xception_model.h5
git commit -m "Add model with Git LFS"
git push
```

**Option 2: Download at Runtime**
Modify `start.sh` to download model from cloud storage (S3, Google Drive, etc.)

### Service Won't Start

1. **Check logs** for error messages
2. **Verify PORT** is set to match Render's assigned port
3. **Test locally** with Docker first
4. **Check health endpoint** returns 200 OK

### Slow Performance on Free Plan

Free plan limitations:
- Service spins down after 15 minutes of inactivity
- First request after spin-down takes ~30 seconds
- Limited CPU/RAM

**Solutions:**
- Upgrade to Starter plan for always-on service
- Use cron-job.org to ping every 14 minutes (keeps service alive)

## Connecting Frontend

Update your Next.js frontend environment:

**In `my-app/.env.local`:**
```env
NEXT_PUBLIC_API_URL=https://your-app-name.onrender.com
```

**Update CORS in `backend/config.py`:**
```python
CORS_ORIGINS = [
    "http://localhost:3000",
    "https://your-frontend.vercel.app",  # Add your Vercel URL
]
```

Commit and push changes for automatic redeployment.

## Cost Optimization

### Free Plan
- Good for development/testing
- Service spins down after inactivity
- 750 hours/month free

### Starter Plan ($7/month)
- Always-on service
- Better performance
- Recommended for production

### Tips
- Use free tier during development
- Upgrade to paid when launching
- Monitor usage in Dashboard

## Security Best Practices

1. **Environment Variables:**
   - Never commit secrets to GitHub
   - Use Render's Environment Variables for sensitive data

2. **HTTPS:**
   - Render provides free SSL/TLS certificates
   - All traffic is encrypted by default

3. **API Rate Limiting:**
   - Implement rate limiting in Flask
   - Use Render's built-in DDoS protection

4. **Keep Dependencies Updated:**
   ```powershell
   pip list --outdated
   pip install --upgrade <package>
   ```

## Next Steps

- ✅ Deploy backend to Render
- ✅ Deploy frontend to Vercel
- ✅ Update CORS settings
- ✅ Test end-to-end workflow
- ✅ Set up monitoring/alerts
- ✅ Configure custom domain (optional)

## Support & Resources

- [Render Documentation](https://render.com/docs)
- [Render Community Forum](https://community.render.com)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- Check service logs in Render Dashboard
