# DeepGuard Chrome Extension - Installation Guide

## Quick Install Steps

### 1. Start Your Backend API
```powershell
cd "C:\Users\Mohammed Munazir\deepfake detection2\backend"
python app.py
```
The backend should run on `http://127.0.0.1:5000`

### 2. Load Extension in Chrome

1. **Open Chrome Extensions Page**
   - Open Google Chrome
   - Go to `chrome://extensions/`
   - Or click Menu (⋮) → Extensions → Manage Extensions

2. **Enable Developer Mode**
   - Toggle ON the "Developer mode" switch in the top-right corner

3. **Load Unpacked Extension**
   - Click "Load unpacked" button
   - Navigate to: `C:\Users\Mohammed Munazir\deepfake detection2\chrome-extension`
   - Click "Select Folder"

4. **Verify Installation**
   - You should see "DeepGuard - Deepfake Detector" in your extensions list
   - The extension icon should appear in your Chrome toolbar
   - Status should show "Errors" in red if icons are missing (this is OK for now)

### 3. Configure Extension

1. **Click the DeepGuard extension icon** in Chrome toolbar
2. **Check API Status** - Should show "API Online" (green)
   - If offline, verify backend is running on port 5000
3. **Enable Platforms**
   - ✅ Enable Auto-Scan
   - ✅ Enable Show Warnings  
   - ✅ Enable Instagram
   - ✅ Enable Facebook
4. **Verify API Endpoint**: `http://127.0.0.1:5000`

### 4. Test the Extension

1. **Go to Instagram** (instagram.com)
2. **Play any video** (Reels, Stories, Posts)
3. **Watch for:**
   - Biometric scanning animation (glowing corners, scan line)
   - Detection badge in top-right corner showing REAL/FAKE
   - Console logs (F12 → Console) showing detection details

## What You'll See

### During Scanning:
- Dark overlay with grid pattern
- Multiple scanning boxes with glowing corners
- Vertical scan line moving through the frame
- "Scanning for faces..." text with animated dots

### After Detection:
- **Green badge** = AUTHENTIC (real face)
- **Red badge** = DEEPFAKE (fake/manipulated)
- Badge shows:
  - Prediction (DEEPFAKE/AUTHENTIC)
  - Confidence percentage
  - Frequency analysis score
  - Processing time
  - Timestamp

### If Deepfake Detected:
- Warning notification in top-right
- Detailed information about the detection
- Badge pulses with red glow

## Troubleshooting

### ❌ Extension shows "API Offline"
**Fix:** 
- Make sure backend is running: `python app.py` in backend folder
- Check console for errors: F12 → Console
- Verify endpoint is `http://127.0.0.1:5000` (not localhost:8080)

### ❌ No scanning happens
**Fix:**
- Open extension popup, enable "Auto-Scan"
- Enable the platform you're testing (Instagram/Facebook)
- Refresh the page (F5)

### ❌ "No face detected" badge
**Normal:** The video frame doesn't contain a visible face
- Extension will keep scanning every 3 seconds
- When a face appears, it will be detected

### ❌ Extension icons missing (manifest errors)
**Fix:** Create PNG icons or ignore for now (extension still works)
```powershell
# Icons needed (optional):
# chrome-extension/icons/icon16.png
# chrome-extension/icons/icon48.png
# chrome-extension/icons/icon128.png
```

## Features

✅ **Continuous Scanning** - Rescans every 3 seconds while video plays
✅ **Real-time Detection** - Instant deepfake analysis
✅ **Biometric Animation** - Professional face scanning effect
✅ **Detailed Results** - Confidence, frequency score, processing time
✅ **Smart Notifications** - Warns only on new deepfake detections
✅ **Platform Support** - Instagram Reels, Stories, Facebook videos
✅ **Statistics** - Track videos scanned and deepfakes found

## Console Commands (for debugging)

Press F12 to open Developer Tools, then check Console tab:

```javascript
// You should see logs like:
🛡️ DeepGuard: Content script loaded
🛡️ DeepGuard: Monitoring Instagram
🔍 Scanning Instagram video frame...
🛡️ DeepGuard Detection Result: {
  prediction: "FAKE",
  confidence: "87.32%",
  frequencyScore: "68.21%",
  processingTime: "0.43s",
  platform: "Instagram"
}
```

## Uninstalling

1. Go to `chrome://extensions/`
2. Find "DeepGuard - Deepfake Detector"
3. Click "Remove"
4. Confirm removal

## Need Help?

- Check browser console (F12) for error messages
- Verify backend API is running and accessible
- Make sure you're on Instagram or Facebook
- Try refreshing the page after enabling the extension

---

**Note:** This extension requires a running backend API. Make sure to start `python app.py` before using the extension!
