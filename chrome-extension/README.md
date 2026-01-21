# DeepGuard Chrome Extension 🛡️

AI-powered deepfake detection with 97.73% accuracy using ResNet50 for Instagram, Facebook, and other social media platforms.

## Features

✅ **Real-time Detection** - Automatically scans videos as you browse with 97.73% accuracy
✅ **ResNet50 AI Model** - Advanced deep learning trained on 240,000+ images
✅ **GPU Accelerated** - Fast processing using PyTorch with CUDA support
✅ **Multi-Platform Support** - Works on Instagram, Facebook, Twitter, YouTube
✅ **Visual Warnings** - Clear indicators for deepfake content
✅ **Privacy Focused** - All processing via your own backend API
✅ **Statistics Tracking** - Monitor scanned videos and detections
✅ **Customizable Settings** - Control when and where to scan

## Installation

### Step 1: Load the Extension

1. Open Chrome and go to `chrome://extensions/`
2. Enable **Developer mode** (toggle in top right)
3. Click **Load unpacked**
4. Select the `chrome-extension` folder

### Step 2: Configure API Endpoint

1. Click the DeepGuard extension icon
2. Enter your backend API URL (default: `http://localhost:8080`)
3. Click **Save Settings**

### Step 3: Start Browsing

Visit Instagram, Facebook, or other supported platforms and the extension will automatically scan videos!

## How It Works

1. **Video Detection** - Monitors social media pages for video content
2. **Frame Capture** - Captures frames from playing videos
3. **AI Analysis** - Sends frames to your backend API for deepfake detection
4. **Visual Feedback** - Shows results directly on the video

## Settings

### Auto-Scan Videos
Automatically analyze videos when they appear on the page

### Show Warnings
Display notifications when deepfakes are detected

### Platform Toggles
Enable/disable scanning for specific platforms:
- Instagram
- Facebook
- Twitter
- YouTube

### API Endpoint
Your backend server URL for deepfake analysis

## Usage

### Automatic Scanning
When enabled, videos are automatically scanned as they play on supported platforms.

### Manual Scanning
Right-click any video → "Scan for Deepfakes"

### View Results
Results appear as badges on videos:
- ✅ Green badge = Authentic
- ⚠️ Red badge = Potential deepfake

### Check Statistics
Click the extension icon to view:
- Total videos scanned
- Deepfakes detected
- Detection accuracy

## API Integration

The extension connects to your DeepGuard backend API:

**Endpoint:** `POST /predict`

**Request:**
```
FormData with 'file' field containing video frame
```

**Response:**
```json
{
  "prediction": "REAL" | "FAKE",
  "confidence": 94.3,
  "probabilities": {
    "REAL": 94.3,
    "FAKE": 5.7
  }
}
```

## Requirements

- Chrome browser (version 88+)
- DeepGuard backend API running (see main README)

## Privacy & Security

- **No Data Collection** - Extension doesn't collect or store personal data
- **Local Processing** - All analysis via your own backend server
- **No Third Parties** - Direct connection to your API only
- **Open Source** - Full code transparency

## Troubleshooting

### Extension not scanning
- Check if API endpoint is correct in settings
- Verify backend is running (`http://localhost:8080/health`)
- Enable auto-scan in extension settings
- Check browser console for errors

### API connection failed
- Ensure backend server is running
- Check CORS settings in backend
- Verify API URL in extension settings
- Try manual API health check

### Videos not detected
- Refresh the page after enabling the extension
- Some platforms may block frame capture
- Check if the platform is enabled in settings

## Development

### File Structure
```
chrome-extension/
├── manifest.json      # Extension configuration
├── popup.html         # Settings popup UI
├── popup.js          # Settings logic
├── content.js        # Page interaction script
├── background.js     # Background service worker
├── styles.css        # UI styles
├── icons/            # Extension icons
└── README.md         # This file
```

### Testing Locally
1. Make changes to extension files
2. Go to `chrome://extensions/`
3. Click reload icon for DeepGuard
4. Test on social media platforms

### Building for Production
The extension is already production-ready. To distribute:
1. Zip the `chrome-extension` folder
2. Upload to Chrome Web Store
3. Or distribute the `.zip` file directly

## Supported Platforms

- ✅ Instagram (Reels, Stories, Posts)
- ✅ Facebook (Videos, Reels)
- ✅ Twitter/X (Video posts)
- ✅ YouTube (Videos)
- More platforms coming soon!

## Keyboard Shortcuts

- **Ctrl+Shift+D** - Toggle auto-scan (coming soon)
- **Right-click video** - Manual scan option

## Performance

- Minimal CPU usage
- Scans on-demand to save resources
- Cached results to avoid re-scanning
- Efficient frame capture

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License - See main project LICENSE file

## Support

- 📧 Email: support@deepguard.ai
- 🐛 Issues: GitHub Issues
- 💬 Discussions: GitHub Discussions

## Changelog

### v1.0.0 (Initial Release)
- Instagram video scanning
- Facebook video scanning
- Real-time detection
- Visual indicators
- Statistics tracking
- Settings panel

## Roadmap

- [ ] Video timeline scanning
- [ ] Batch scanning
- [ ] Export scan reports
- [ ] TikTok support
- [ ] LinkedIn support
- [ ] Offline mode
- [ ] Keyboard shortcuts
- [ ] Custom themes

---

Made with ❤️ for a safer internet
