# 🚀 Quick Setup Guide - DeepGuard Chrome Extension

## Prerequisites

✅ Chrome browser installed
✅ DeepGuard backend API running (see `backend/README.md`)

## Installation Steps

### 1. Create Icon Files (Required)

Chrome needs PNG icons. Quick method:

**Option A: Use Favicon Generator**
1. Visit https://favicon.io/emoji-favicons/shield/
2. Download the generated icons
3. Extract and rename:
   - `android-chrome-512x512.png` → `icon128.png`
   - `android-chrome-192x192.png` → `icon48.png`  
   - `favicon-16x16.png` → `icon16.png`
4. Copy all three to `chrome-extension/icons/`

**Option B: Create Simple PNG Icons**
```powershell
# You can use any 128x128 PNG image as a temporary icon
# Copy it three times with different names:
cp my-icon.png icons/icon16.png
cp my-icon.png icons/icon48.png
cp my-icon.png icons/icon128.png
```

### 2. Load Extension in Chrome

1. Open Chrome browser
2. Navigate to `chrome://extensions/`
3. Enable **Developer mode** (toggle in top-right corner)
4. Click **Load unpacked** button
5. Navigate to and select the `chrome-extension` folder
6. Click **Select Folder**

✅ DeepGuard should now appear in your extensions!

### 3. Configure Backend URL

1. Click the DeepGuard icon in Chrome toolbar
2. In the **API Endpoint** field, enter your backend URL:
   - Local: `http://localhost:8080`
   - Render: `https://your-app.onrender.com`
3. Click **Save Settings**
4. Check that API status shows **"API Online"** (green)

### 4. Enable Auto-Scan

In the extension popup, ensure these are enabled (✓):
- ✅ Auto-Scan Videos
- ✅ Show Warnings
- ✅ Instagram
- ✅ Facebook

### 5. Test It Out!

1. Go to Instagram or Facebook
2. Play any video
3. Watch for the scanning indicator
4. See the result badge appear on the video

## Troubleshooting

### Icons not showing?
- Make sure you have `icon16.png`, `icon48.png`, `icon128.png` in the `icons/` folder
- Reload the extension: `chrome://extensions/` → click reload icon

### API shows offline?
- Check backend is running: `http://localhost:8080/health`
- Verify URL in extension settings
- Check browser console for CORS errors

### Videos not being scanned?
- Refresh the page after installing extension
- Check auto-scan is enabled
- Look for errors in browser console (F12)

### Permission denied errors?
- Some video players may block frame capture
- Try different videos/posts

## Usage Tips

### View Statistics
Click the extension icon to see:
- Total videos scanned
- Deepfakes detected
- Detection accuracy

### Manual Scan
- Right-click any video
- Select "Scan for Deepfakes"

### Reset Stats
- Double-click the statistics section in the popup
- Confirm to reset all counters

## Development Mode

### View Console Logs
1. Open Developer Tools (F12)
2. Go to Console tab
3. Look for messages starting with "🛡️ DeepGuard"

### Debug Content Script
1. Right-click on page → Inspect
2. Console tab will show content script logs
3. Check for video detection and API calls

### Debug Popup
1. Right-click extension icon → Inspect popup
2. Separate DevTools window opens
3. View popup.js logs and errors

### Debug Background Script
1. Go to `chrome://extensions/`
2. Find DeepGuard
3. Click "service worker" link
4. View background.js logs

## Updating the Extension

After making code changes:

1. Go to `chrome://extensions/`
2. Find DeepGuard extension
3. Click the reload icon (🔄)
4. Refresh any open social media pages

## Supported Platforms

Currently tested on:
- ✅ Instagram (Reels, Stories, Feed videos)
- ✅ Facebook (Video posts, Reels)
- ⏳ Twitter (Coming soon)
- ⏳ YouTube (Coming soon)

## Next Steps

1. **Create Better Icons** - Design professional icons for production
2. **Test on Social Media** - Try scanning various videos
3. **Adjust Settings** - Customize scan behavior
4. **Monitor Statistics** - Track detection performance
5. **Report Issues** - Help improve the extension

## Performance Tips

- Extension uses minimal resources
- Videos are scanned only when playing
- Results are cached to avoid re-scanning
- Background processing doesn't affect browsing

## Security & Privacy

- ✅ No data collected or stored remotely
- ✅ All analysis via your own backend
- ✅ No tracking or analytics
- ✅ Open source - review all code

## Publishing (Optional)

To publish to Chrome Web Store:

1. Create a Chrome Web Store developer account
2. Pay $5 one-time fee
3. Prepare store listing materials:
   - Screenshots
   - Promotional images
   - Description
4. Package extension as .zip
5. Upload and submit for review

## Questions?

- Check the main README in the project root
- Review backend API documentation
- Open an issue on GitHub
- Test with sample videos first

---

Happy scanning! 🛡️ Stay safe online.
