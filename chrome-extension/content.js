// Content script - runs on Instagram, Facebook, etc.

console.log('🛡️ DeepGuard: Content script loaded');

// Configuration
let settings = {
  autoScan: true,
  showWarnings: true,
  enableInstagram: true,
  enableFacebook: true,
  apiEndpoint: 'http://127.0.0.1:5000'
};

// Load settings
chrome.storage.sync.get(settings, (items) => {
  settings = items;
  console.log('Settings loaded:', settings);
  initializeDetection();
});

// Listen for settings updates
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'settingsUpdated') {
    settings = request.settings;
    console.log('Settings updated:', settings);
  }
});

// Track scanned videos to avoid duplicates
const scannedVideos = new Set();
const videoAnalysis = new Map();
const videoScanIntervals = new Map();
const videoControlButtons = new Map();
const videoScanningPaused = new Map();
const SCAN_INTERVAL = 3000; // Scan every 3 seconds

// Initialize detection based on platform
function initializeDetection() {
  const hostname = window.location.hostname;
  
  console.log(`🛡️ DeepGuard: Monitoring ${hostname}`);
  
  // Work on all websites - detect all videos and images
  watchForAllVideos();
  watchForAllImages();
}

// Watch for videos on all websites
function watchForAllVideos() {
  const observer = new MutationObserver((mutations) => {
    const videos = document.querySelectorAll('video');
    videos.forEach(video => {
      if (!scannedVideos.has(video) && isVideoPlaying(video)) {
        scannedVideos.add(video);
        if (settings.autoScan) {
          scanVideo(video, 'Web');
        }
      }
    });
  });

  observer.observe(document.body, {
    childList: true,
    subtree: true
  });

  // Also check initially
  setTimeout(() => {
    const videos = document.querySelectorAll('video');
    videos.forEach(video => {
      if (!scannedVideos.has(video)) {
        scannedVideos.add(video);
        if (settings.autoScan) {
          scanVideo(video, 'Web');
        }
      }
    });
  }, 2000);
}

// Watch for images on all websites
function watchForAllImages() {
  console.log('📸 Starting universal image monitoring...');
  
  // Process existing images immediately
  processVisibleImages();
  
  // Use MutationObserver to detect new images
  const observer = new MutationObserver((mutations) => {
    processVisibleImages();
  });
  
  observer.observe(document.body, {
    childList: true,
    subtree: true
  });
  
  // Also process on scroll to catch lazy-loaded images
  let scrollTimeout;
  window.addEventListener('scroll', () => {
    clearTimeout(scrollTimeout);
    scrollTimeout = setTimeout(() => {
      processVisibleImages();
    }, 300);
  }, { passive: true });
  
  // Periodic check for dynamically loaded images (especially for Instagram)
  setInterval(() => {
    processVisibleImages();
  }, 2000);
}

// Process all visible images on the page
function processVisibleImages() {
  const images = document.querySelectorAll('img');
  let processedCount = 0;
  
  images.forEach(img => {
    // Skip if already processed
    if (img.dataset.deepguardEnabled) return;
    
    // Skip images inside result badges or scan overlays
    if (img.closest('.image-scan-result') || img.closest('.deepguard-image-scan-overlay')) return;
    
    // Skip our own extension icons
    if (img.classList.contains('scan-icon-img')) return;
    
    // Skip very small images (icons, buttons, etc.)
    if (img.width < 200 || img.height < 200) return;
    
    // Skip images that are likely UI elements (small aspect ratio)
    const aspectRatio = img.width / img.height;
    if (aspectRatio > 5 || aspectRatio < 0.2) return;
    
    // Check if image is in viewport or nearby (for better performance)
    const rect = img.getBoundingClientRect();
    const isNearViewport = rect.top < window.innerHeight + 500 && rect.bottom > -500;
    
    if (!isNearViewport) return;
    
    // Mark as processed
    img.dataset.deepguardEnabled = 'true';
    
    // Add scan button
    addImageScanButton(img);
    processedCount++;
  });
  
  if (processedCount > 0) {
    console.log(`✅ Added scan buttons to ${processedCount} images`);
  }
}

// Watch for Instagram videos
function watchForInstagramVideos() {
  const observer = new MutationObserver((mutations) => {
    // Look for video elements
    const videos = document.querySelectorAll('video');
    videos.forEach(video => {
      if (!scannedVideos.has(video) && isVideoPlaying(video)) {
        scannedVideos.add(video);
        if (settings.autoScan) {
          scanVideo(video, 'Instagram');
        }
      }
    });
  });

  observer.observe(document.body, {
    childList: true,
    subtree: true
  });

  // Also check initially
  setTimeout(() => {
    const videos = document.querySelectorAll('video');
    videos.forEach(video => {
      if (!scannedVideos.has(video)) {
        scannedVideos.add(video);
        if (settings.autoScan) {
          scanVideo(video, 'Instagram');
        }
      }
    });
  }, 2000);
}

// Watch for Instagram images
function watchForInstagramImages() {
  console.log('📸 Starting Instagram image monitoring...');
  
  // Check for images periodically
  setInterval(() => {
    const images = document.querySelectorAll('img');
    let processedCount = 0;
    
    images.forEach(img => {
      // Skip if already processed
      if (img.dataset.deepguardEnabled) return;
      
      // Skip images inside result badges
      if (img.closest('.image-scan-result') || img.closest('.deepguard-image-scan-overlay')) return;
      
      // Skip very small images
      if (img.width < 100 || img.height < 100) return;
      
      // Mark as processed
      img.dataset.deepguardEnabled = 'true';
      
      // Add scan button
      addImageScanButton(img, 'Instagram');
      processedCount++;
    });
    
    if (processedCount > 0) {
      console.log(`✅ Added scan buttons to ${processedCount} Instagram images`);
    }
  }, 2000);
}

// Watch for Facebook videos
function watchForFacebookVideos() {
  const observer = new MutationObserver((mutations) => {
    const videos = document.querySelectorAll('video');
    videos.forEach(video => {
      if (!scannedVideos.has(video) && isVideoPlaying(video)) {
        scannedVideos.add(video);
        if (settings.autoScan) {
          scanVideo(video, 'Facebook');
        }
      }
    });
  });

  observer.observe(document.body, {
    childList: true,
    subtree: true
  });
}

// Watch for Facebook images
function watchForFacebookImages() {
  console.log('📸 Starting Facebook image monitoring...');
  
  // Check for images periodically
  setInterval(() => {
    const images = document.querySelectorAll('img');
    let processedCount = 0;
    
    images.forEach(img => {
      if (img.dataset.deepguardEnabled) return;
      
      // Skip images inside result badges
      if (img.closest('.image-scan-result') || img.closest('.deepguard-image-scan-overlay')) return;
      
      if (img.width < 100 || img.height < 100) return;
      
      img.dataset.deepguardEnabled = 'true';
      addImageScanButton(img, 'Facebook');
      processedCount++;
    });
    
    if (processedCount > 0) {
      console.log(`✅ Added scan buttons to ${processedCount} Facebook images`);
    }
  }, 2000);
}

// Add scan button overlay to images
function addImageScanButton(imgElement, platform) {
  // Skip if image already has a scan button
  if (imgElement.dataset.deepguardScanButton) return;
  
  // Mark as having scan button and ensure scanning flag is clear
  imgElement.dataset.deepguardScanButton = 'true';
  imgElement.dataset.deepguardScanning = 'false';
  
  // Create wrapper if image doesn't have a positioned parent
  let container = imgElement.parentElement;
  
  // Check if we need to wrap the image
  const containerPosition = window.getComputedStyle(container).position;
  if (containerPosition === 'static') {
    // Create a wrapper
    const wrapper = document.createElement('div');
    wrapper.style.position = 'relative';
    wrapper.style.display = 'inline-block';
    wrapper.style.width = imgElement.width + 'px';
    wrapper.style.height = imgElement.height + 'px';
    
    imgElement.parentNode.insertBefore(wrapper, imgElement);
    wrapper.appendChild(imgElement);
    container = wrapper;
  }
  
  // Remove any existing result badges before adding scan button
  const existingResults = container.querySelectorAll('.image-scan-result');
  existingResults.forEach(r => {
    console.log('🗑️ Removing old result badge before adding scan button');
    r.remove();
  });
  
  // Create scan button overlay
  const scanOverlay = document.createElement('div');
  scanOverlay.className = 'deepguard-image-scan-overlay show';
  scanOverlay.innerHTML = `
    <div class="image-scan-button" title="Click to scan for deepfakes">
      <div class="scan-logo">
        <img src="${chrome.runtime.getURL('icons/icon48.png')}" alt="Scan" class="scan-icon-img">
      </div>
    </div>
  `;
  
  container.appendChild(scanOverlay);
  
  // Click to scan
  const button = scanOverlay.querySelector('.image-scan-button');
  button.addEventListener('click', async (e) => {
    e.stopPropagation();
    e.preventDefault();
    console.log('🔍 User clicked scan button on', platform);
    scanOverlay.classList.add('scanning');
    await scanImage(imgElement, platform, scanOverlay);
  });
}

// Scan image for deepfakes
async function scanImage(imgElement, platform, overlay) {
  // Prevent multiple simultaneous scans on the same image
  if (imgElement.dataset.deepguardScanning === 'true') {
    console.log('⏭️ Already scanning this image, skipping...');
    return;
  }
  
  try {
    imgElement.dataset.deepguardScanning = 'true';
    console.log(`🔍 Starting scan for ${platform} image...`);
    console.log('Image details:', {
      src: imgElement.src?.substring(0, 100),
      width: imgElement.width,
      height: imgElement.height
    });
    
    // Update button to show scanning state (simple spinner, no face animation)
    if (overlay) {
      overlay.innerHTML = `
        <div class="image-scan-button scanning">
          <div class="scan-logo scanning-face">
            <img src="${chrome.runtime.getURL('icons/icon48.png')}" alt="Scanning" class="scan-icon-img" style="opacity: 0.7;">
          </div>
        </div>
      `;
      
      // Add face scanning overlay to the image container
      const container = imgElement.parentElement;
      addImageScanningOverlay(container, imgElement);
    }
    
    // Get image data
    console.log('Capturing image data...');
    const imageData = await getImageData(imgElement);
    
    if (!imageData) {
      console.error('❌ Failed to capture image data - CORS issue or image not accessible');
      if (overlay) {
        overlay.classList.remove('scanning');
        overlay.innerHTML = `
          <div class="image-scan-button error">
            <div class="scan-logo">
              <img src="${chrome.runtime.getURL('icons/icon48.png')}" alt="Error" class="scan-icon-img" style="opacity: 0.6;">
            </div>
          </div>
        `;
        
        // Show error message
        const container = imgElement.parentElement;
        const errorBadge = document.createElement('div');
        errorBadge.className = 'image-scan-result error';
        errorBadge.innerHTML = `
          <div class="result-badge-compact">
            <span class="badge-icon">⚠️</span>
            <span class="badge-label">Cannot access image</span>
          </div>
        `;
        container.appendChild(errorBadge);
        setTimeout(() => errorBadge.remove(), 4000);
        
        // Reset button after delay
        setTimeout(() => {
          overlay.innerHTML = `
            <div class="image-scan-button">
              <div class="scan-logo">
                <img src="${chrome.runtime.getURL('icons/icon48.png')}" alt="Scan" class="scan-icon-img">
              </div>
            </div>
          `;
          const button = overlay.querySelector('.image-scan-button');
          button.addEventListener('click', async (e) => {
            e.stopPropagation();
            e.preventDefault();
            const oldResults = container.querySelectorAll('.image-scan-result');
            oldResults.forEach(r => r.remove());
            overlay.classList.add('scanning');
            await scanImage(imgElement, platform, overlay);
          });
        }, 3000);
      }
      return;
    }
    
    console.log('✅ Image captured, sending to API...');
    
    // Send to backend API
    const result = await analyzeFrame(imageData);
    
    console.log('API Response:', result);
    
    // Check for errors
    if (result.error) {
      console.log('⚠️ Backend error:', result.error);
      
      if (overlay) {
        overlay.classList.remove('scanning');
        const errorMsg = result.error.toLowerCase().includes('no face') 
          ? 'No face detected' 
          : result.error.substring(0, 30);
        
        // Show error badge
        const container = imgElement.parentElement;
        const errorBadge = document.createElement('div');
        errorBadge.className = 'image-scan-result error';
        errorBadge.innerHTML = `
          <div class="result-badge-compact">
            <span class="badge-icon">⚠️</span>
            <span class="badge-label">${errorMsg}</span>
          </div>
        `;
        container.appendChild(errorBadge);
        setTimeout(() => errorBadge.remove(), 4000);
        
        // Reset button
        overlay.innerHTML = `
          <div class="image-scan-button">
            <div class="scan-logo">
              <img src="${chrome.runtime.getURL('icons/icon48.png')}" alt="Scan" class="scan-icon-img">
            </div>
          </div>
        `;
        const button = overlay.querySelector('.image-scan-button');
        button.addEventListener('click', async (e) => {
          e.stopPropagation();
          e.preventDefault();
          const oldResults = container.querySelectorAll('.image-scan-result');
          oldResults.forEach(r => r.remove());
          overlay.classList.add('scanning');
          await scanImage(imgElement, platform, overlay);
        });
      }
      return;
    }
    
    // Update stats
    updateStats(result);
    
    console.log('✅ Scan complete, displaying result');
    
    // Show result on image
    displayImageResult(imgElement, result, platform, overlay);

  } catch (error) {
    console.error('❌ Error scanning image:', error);
    if (overlay) {
      overlay.innerHTML = `
        <div class="image-scan-button error">
          <div class="scan-logo">
            <img src="${chrome.runtime.getURL('icons/icon48.png')}" alt="Error" class="scan-icon-img" style="opacity: 0.6;">
          </div>
        </div>
      `;
      setTimeout(() => {
        overlay.innerHTML = `
          <div class="image-scan-button">
            <div class="scan-logo">
              <img src="${chrome.runtime.getURL('icons/icon48.png')}" alt="Scan" class="scan-icon-img">
            </div>
          </div>
        `;
      }, 4000);
    }
  } finally {
    imgElement.dataset.deepguardScanning = 'false';
  }
}

// Get image data as base64
async function getImageData(imgElement) {
  try {
    // First, try to fetch the image through fetch API to bypass CORS
    const imageUrl = imgElement.src;
    
    // Fetch the image
    const response = await fetch(imageUrl);
    const blob = await response.blob();
    
    // Convert blob to data URL
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  } catch (error) {
    console.error('Error fetching image:', error);
    
    // Fallback: try canvas method for same-origin images
    try {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      
      // Set canvas size to image size
      canvas.width = imgElement.naturalWidth || imgElement.width;
      canvas.height = imgElement.naturalHeight || imgElement.height;
      
      // Draw image to canvas
      ctx.drawImage(imgElement, 0, 0);
      
      // Convert to data URL
      return canvas.toDataURL('image/jpeg', 0.9);
    } catch (canvasError) {
      console.error('Canvas fallback also failed:', canvasError);
      return null;
    }
  }
}

// Display scan result on image
function displayImageResult(imgElement, result, platform, overlay) {
  // Validate that we have a real result before displaying
  if (!result || !result.label) {
    console.warn('⚠️ No valid result to display, skipping badge');
    return;
  }
  
  const prediction = result.prediction || result.label || 'UNKNOWN';
  const isFake = prediction.toUpperCase() === 'FAKE';
  
  let confidence = result.confidence || result.adjusted_confidence || 0;
  if (confidence > 1 && confidence <= 100) {
    // Already percentage
  } else if (confidence >= 0 && confidence <= 1) {
    confidence = confidence * 100;
  }
  
  console.log('Displaying image result:', { prediction, confidence, isFake });
  
  // Create wrapper for result badge if needed
  const container = imgElement.parentElement;
  
  // Remove ALL existing result badges thoroughly
  // Check both container and document for any orphaned badges
  const existingInContainer = container.querySelectorAll('.image-scan-result');
  existingInContainer.forEach(r => {
    console.log('🗑️ Removing existing result badge');
    r.remove();
  });
  
  // Also check if there are any badges directly on the image element
  const existingOnImage = imgElement.querySelectorAll('.image-scan-result');
  existingOnImage.forEach(r => r.remove());
  
  // Create result badge element
  const resultBadge = document.createElement('div');
  resultBadge.className = `image-scan-result ${isFake ? 'fake' : 'real'}`;
  resultBadge.innerHTML = `
    <div class="result-badge-compact">
      <span class="badge-icon">${isFake ? '⚠️' : '✓'}</span>
      <span class="badge-label">${isFake ? 'FAKE' : 'REAL'}</span>
    </div>
    <div class="result-details-hover">
      <div class="detail-row">
        <span class="detail-label">Confidence:</span>
        <span class="detail-value">${confidence.toFixed(1)}%</span>
      </div>
      ${result.frequency_score ? `
        <div class="detail-row">
          <span class="detail-label">Frequency:</span>
          <span class="detail-value">${(result.frequency_score * 100).toFixed(1)}%</span>
        </div>
      ` : ''}
      ${result.processing_time ? `
        <div class="detail-row">
          <span class="detail-label">Time:</span>
          <span class="detail-value">${result.processing_time.toFixed(2)}s</span>
        </div>
      ` : ''}
    </div>
  `;
  
  // Ensure unique badge per container - use data attribute
  resultBadge.dataset.imageResultBadge = 'true';
  container.appendChild(resultBadge);
  
  // Reset the scan button overlay to allow re-scanning
  if (overlay) {
    overlay.classList.remove('scanning');
    overlay.classList.add('has-result', 'show');
    overlay.innerHTML = `
      <div class="image-scan-button" title="Click to re-scan">
        <div class="scan-logo">
          <img src="${chrome.runtime.getURL('icons/icon48.png')}" alt="Re-scan" class="scan-icon-img">
        </div>
      </div>
    `;
    
    // Re-attach click handler for re-scanning
    const button = overlay.querySelector('.image-scan-button');
    button.addEventListener('click', async (e) => {
      e.stopPropagation();
      e.preventDefault();
      console.log('🔍 Re-scanning image...');
      // Remove old results
      const oldResults = container.querySelectorAll('.image-scan-result');
      oldResults.forEach(r => r.remove());
      overlay.classList.remove('has-result');
      overlay.classList.add('scanning');
      await scanImage(imgElement, platform, overlay);
    });
  }
  
  console.log('🛡️ DeepGuard Image Result:', {
    prediction,
    confidence: confidence.toFixed(2) + '%',
    platform
  });
}

// Check if video is playing
function isVideoPlaying(video) {
  return !!(video.currentTime > 0 && !video.paused && !video.ended && video.readyState > 2);
}

// Scan video for deepfakes (continuous mode)
async function scanVideo(videoElement, platform) {
  // Clear existing interval if any
  if (videoScanIntervals.has(videoElement)) {
    clearInterval(videoScanIntervals.get(videoElement));
  }
  
  // Initialize pause state
  videoScanningPaused.set(videoElement, false);
  
  // Add control buttons
  addScanControlButtons(videoElement, platform);
  
  // Start continuous scanning
  const scanInterval = setInterval(async () => {
    // Skip if scanning is manually paused or video is paused
    if (videoScanningPaused.get(videoElement) || !isVideoPlaying(videoElement)) {
      return;
    }
    
    await performSingleScan(videoElement, platform);
  }, SCAN_INTERVAL);
  
  videoScanIntervals.set(videoElement, scanInterval);
  
  // Perform initial scan immediately (unless paused)
  if (!videoScanningPaused.get(videoElement)) {
    await performSingleScan(videoElement, platform);
  }
  
  // Listen for video end to clean up
  videoElement.addEventListener('ended', () => stopVideoScan(videoElement));
}

// Perform a single scan iteration
async function performSingleScan(videoElement, platform) {
  try {
    console.log(`🔍 Scanning ${platform} video frame...`);
    
    // Show scanning indicator with biometric face scan animation
    addScanningIndicator(videoElement, 'Scanning for faces...');
    
    // Capture frame from video
    const frameData = captureVideoFrame(videoElement);
    
    if (!frameData) {
      console.log('Could not capture video frame');
      removeScanningIndicator(videoElement);
      return;
    }
    
    // Send frame to backend API
    const result = await analyzeFrame(frameData);
    
    // Check for errors in response
    if (result.error) {
      console.log('Backend error:', result.error);
      removeScanningIndicator(videoElement);
      
      // Check if it's a "no face detected" error
      const errorMsg = result.error.toLowerCase();
      if (errorMsg.includes('no face')) {
        showInfoBadge(videoElement, 'No face detected', 'warning');
        console.log('ℹ️ No face detected in current frame');
      } else if (errorMsg.includes('file') || errorMsg.includes('load')) {
        showInfoBadge(videoElement, 'Frame capture failed', 'error');
        console.log('⚠️ Frame processing error:', result.error);
      } else {
        showInfoBadge(videoElement, result.error, 'error');
        console.log('⚠️ API error:', result.error);
      }
      return;
    }
    
    // Update stats only for successful detections
    updateStats(result);
    
    // Show result (supports multiple faces)
    displayResult(videoElement, result, platform);
    
    // Store analysis
    videoAnalysis.set(videoElement, result);
    
  } catch (error) {
    console.error('Error scanning video:', error);
    removeScanningIndicator(videoElement);
    
    // Show error badge with appropriate message
    const errorMsg = error.message || 'Unknown error';
    if (errorMsg.includes('Network') || errorMsg.includes('reach')) {
      showInfoBadge(videoElement, 'API Offline - Check connection', 'error');
      console.log('❌ Cannot reach API - Is backend running?');
    } else {
      showInfoBadge(videoElement, 'Scan failed', 'error');
      console.log('❌ Scan error:', errorMsg);
    }
  }
}

// Stop continuous video scanning
function stopVideoScan(videoElement) {
  if (videoScanIntervals.has(videoElement)) {
    clearInterval(videoScanIntervals.get(videoElement));
    videoScanIntervals.delete(videoElement);
    console.log('Stopped video scanning');
  }
  
  // Clean up control buttons
  if (videoControlButtons.has(videoElement)) {
    const controls = videoControlButtons.get(videoElement);
    if (controls && controls.parentElement) {
      controls.remove();
    }
    videoControlButtons.delete(videoElement);
  }
  
  // Clean up pause state
  videoScanningPaused.delete(videoElement);
  
  // Remove badges and overlays
  removeScanningIndicator(videoElement);
  const container = videoElement.parentElement;
  if (container) {
    const badges = container.querySelectorAll('.deepguard-badge, .deepguard-info-badge');
    badges.forEach(b => b.remove());
  }
}

// Capture frame from video
function captureVideoFrame(video) {
  try {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    return canvas.toDataURL('image/jpeg', 0.8);
  } catch (error) {
    console.error('Error capturing frame:', error);
    return null;
  }
}

// Analyze frame using backend API
async function analyzeFrame(frameData) {
  const blob = dataURLtoBlob(frameData);
  
  const formData = new FormData();
  formData.append('file', blob, 'frame.jpg');
  
  try {
    const response = await fetch(`${settings.apiEndpoint}/predict`, {
      method: 'POST',
      body: formData
    });
    
    const data = await response.json();
    
    // Return data even if status is not OK (to handle 400 errors properly)
    if (!response.ok) {
      // Add status code to error data
      data.statusCode = response.status;
    }
    
    return data;
  } catch (error) {
    console.error('API request failed:', error);
    throw new Error('Network error - Cannot reach API');
  }
}

// Convert dataURL to Blob
function dataURLtoBlob(dataURL) {
  const arr = dataURL.split(',');
  const mime = arr[0].match(/:(.*?);/)[1];
  const bstr = atob(arr[1]);
  let n = bstr.length;
  const u8arr = new Uint8Array(n);
  while (n--) {
    u8arr[n] = bstr.charCodeAt(n);
  }
  return new Blob([u8arr], { type: mime });
}

// Add scanning indicator overlay
function addScanningIndicator(video, text, isError = false) {
  // Only remove old overlay, keep badges
  const oldOverlay = video.parentElement?.querySelector('.deepguard-scanning-overlay');
  if (oldOverlay) oldOverlay.remove();
  
  const overlay = document.createElement('div');
  overlay.className = 'deepguard-scanning-overlay';
  overlay.innerHTML = `
    <div class="deepguard-scanning-content">
      <div class="deepguard-spinner"></div>
      <div class="deepguard-text">${text}</div>
      <div class="ai-analysis-dots">
        <div class="ai-dot"></div>
        <div class="ai-dot"></div>
        <div class="ai-dot"></div>
        <div class="ai-dot"></div>
      </div>
    </div>
  `;
  
  if (isError) {
    overlay.style.background = 'rgba(248, 113, 113, 0.9)';
  }
  
  const container = video.parentElement;
  container.style.position = 'relative';
  container.appendChild(overlay);
  
  // Add biometric scanning animation
  if (!isError) {
    addBiometricScanAnimation(container, video);
  }
  
  // Remove overlay after scanning (keep badges)
  setTimeout(() => {
    if (overlay.parentElement) {
      overlay.style.opacity = '0';
      setTimeout(() => overlay.remove(), 300);
    }
  }, isError ? 3000 : 2000);
}

// Add scan control buttons
function addScanControlButtons(videoElement, platform) {
  const container = videoElement.parentElement;
  if (!container) return;
  
  // Remove existing controls
  const existingControls = container.querySelector('.deepguard-controls');
  if (existingControls) existingControls.remove();
  
  const controls = document.createElement('div');
  controls.className = 'deepguard-controls';
  controls.innerHTML = `
    <button class="control-btn pause-btn" title="Pause Scanning">
      <span class="btn-icon">⏸️</span>
    </button>
    <button class="control-btn resume-btn" title="Resume Scanning" style="display: none;">
      <span class="btn-icon">▶️</span>
    </button>
    <button class="control-btn scan-now-btn" title="Scan Now">
      <span class="btn-icon">🔍</span>
    </button>
    <button class="control-btn stop-btn" title="Stop Scanning">
      <span class="btn-icon">⏹️</span>
    </button>
  `;
  
  container.style.position = 'relative';
  container.appendChild(controls);
  
  // Button event listeners
  const pauseBtn = controls.querySelector('.pause-btn');
  const resumeBtn = controls.querySelector('.resume-btn');
  const scanNowBtn = controls.querySelector('.scan-now-btn');
  const stopBtn = controls.querySelector('.stop-btn');
  
  pauseBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    videoScanningPaused.set(videoElement, true);
    pauseBtn.style.display = 'none';
    resumeBtn.style.display = 'block';
    console.log('🛡️ Scanning paused');
  });
  
  resumeBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    videoScanningPaused.set(videoElement, false);
    resumeBtn.style.display = 'none';
    pauseBtn.style.display = 'block';
    console.log('🛡️ Scanning resumed');
    // Trigger immediate scan
    performSingleScan(videoElement, platform);
  });
  
  scanNowBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    console.log('🛡️ Manual scan triggered');
    performSingleScan(videoElement, platform);
  });
  
  stopBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    stopVideoScan(videoElement);
    controls.remove();
    console.log('🛡️ Scanning stopped');
  });
  
  videoControlButtons.set(videoElement, controls);
}

// Add biometric face scanning animation
function addBiometricScanAnimation(container, video) {
  const rect = video.getBoundingClientRect();
  
  // Add multiple scanning boxes for face detection effect
  const scanBoxes = [
    { width: 0.4, height: 0.5, delay: 0 },    // Main face area
    { width: 0.3, height: 0.35, delay: 0.3 }, // Secondary scan
    { width: 0.5, height: 0.6, delay: 0.6 }   // Wide scan
  ];
  
  scanBoxes.forEach((box, index) => {
    const boxWidth = rect.width * box.width;
    const boxHeight = rect.height * box.height;
    const left = (rect.width - boxWidth) / 2;
    const top = (rect.height - boxHeight) / 2;
    
    const faceBox = document.createElement('div');
    faceBox.className = 'biometric-scan-box';
    faceBox.style.width = boxWidth + 'px';
    faceBox.style.height = boxHeight + 'px';
    faceBox.style.left = left + 'px';
    faceBox.style.top = top + 'px';
    faceBox.style.animationDelay = `${box.delay}s`;
    
    // Add corner markers
    faceBox.innerHTML = `
      <div class="scan-corner scan-corner-tl"></div>
      <div class="scan-corner scan-corner-tr"></div>
      <div class="scan-corner scan-corner-bl"></div>
      <div class="scan-corner scan-corner-br"></div>
      <div class="scan-line"></div>
    `;
    
    container.appendChild(faceBox);
  });
}

// Add face scanning overlay for images (single scan)
function addImageScanningOverlay(container, imgElement) {
  // Remove any existing scanning overlays
  const existingOverlays = container.querySelectorAll('.image-scanning-overlay');
  existingOverlays.forEach(o => o.remove());
  
  // Create scanning overlay
  const scanOverlay = document.createElement('div');
  scanOverlay.className = 'image-scanning-overlay';
  
  // Calculate center area for face detection box
  const imgWidth = imgElement.width;
  const imgHeight = imgElement.height;
  const boxWidth = imgWidth * 0.5;
  const boxHeight = imgHeight * 0.6;
  const left = (imgWidth - boxWidth) / 2;
  const top = (imgHeight - boxHeight) / 2;
  
  scanOverlay.innerHTML = `
    <div class="full-image-scan-line"></div>
    <div class="image-scan-box" style="width: ${boxWidth}px; height: ${boxHeight}px; left: ${left}px; top: ${top}px;">
      <div class="scan-corner scan-corner-tl"></div>
      <div class="scan-corner scan-corner-tr"></div>
      <div class="scan-corner scan-corner-bl"></div>
      <div class="scan-corner scan-corner-br"></div>
      <div class="scan-line"></div>
    </div>
  `;
  
  container.appendChild(scanOverlay);
  
  // Remove after 2 seconds (single scan animation)
  setTimeout(() => {
    scanOverlay.remove();
  }, 2000);
}

// Remove scanning indicator
function removeScanningIndicator(video) {
  const overlay = video.parentElement?.querySelector('.deepguard-scanning-overlay');
  if (overlay) {
    overlay.remove();
  }
  // Remove all biometric scan boxes
  const scanBoxes = video.parentElement?.querySelectorAll('.biometric-scan-box');
  if (scanBoxes) {
    scanBoxes.forEach(box => box.remove());
  }
  // Remove old face detection boxes if any
  const faceBoxes = video.parentElement?.querySelectorAll('.face-detection-box');
  if (faceBoxes) {
    faceBoxes.forEach(box => box.remove());
  }
}

// Display scan result
function displayResult(video, result, platform) {
  // Validate that we have a real result before displaying
  if (!result || (!result.label && !result.prediction)) {
    console.warn('⚠️ No valid result to display, skipping badge');
    return;
  }
  
  // Handle different result formats from backend
  const prediction = result.prediction || result.label || 'UNKNOWN';
  const isFake = prediction.toUpperCase() === 'FAKE';
  
  // Get confidence - try different possible fields
  let confidence = result.confidence || result.adjusted_confidence || 0;
  
  // Backend returns confidence as 0-100, ensure it's in that range
  if (confidence > 1 && confidence <= 100) {
    // Already in percentage
  } else if (confidence >= 0 && confidence <= 1) {
    // Convert from 0-1 to percentage
    confidence = confidence * 100;
  }
  
  // Get additional metadata
  const frequencyScore = result.frequency_score;
  const processingTime = result.processing_time;
  const faceDetected = !result.error || result.label !== undefined;
  
  // Create compact result badge
  const badge = document.createElement('div');
  badge.className = 'deepguard-badge';
  
  // Build detailed info panel HTML
  let detailsHTML = `
    <div class="badge-details-panel">
      <div class="detail-row">
        <span class="detail-label">Confidence:</span>
        <span class="detail-value">${confidence.toFixed(1)}%</span>
      </div>
  `;
  
  if (frequencyScore !== undefined) {
    detailsHTML += `
      <div class="detail-row">
        <span class="detail-label">Frequency:</span>
        <span class="detail-value">${(frequencyScore * 100).toFixed(1)}%</span>
      </div>
    `;
  }
  
  if (processingTime !== undefined) {
    detailsHTML += `
      <div class="detail-row">
        <span class="detail-label">Scan Time:</span>
        <span class="detail-value">${processingTime.toFixed(2)}s</span>
      </div>
    `;
  }
  
  detailsHTML += `
      <div class="detail-row">
        <span class="detail-label">Time:</span>
        <span class="detail-value">${new Date().toLocaleTimeString()}</span>
      </div>
    </div>
  `;
  
  badge.innerHTML = `
    <div class="deepguard-badge-content ${isFake ? 'fake' : 'real'}">
      <div class="badge-main">
        <span class="badge-icon">${isFake ? '⚠️' : '✓'}</span>
        <span class="badge-label">${isFake ? 'FAKE' : 'REAL'}</span>
      </div>
      ${detailsHTML}
    </div>
  `;
  
  const container = video.parentElement;
  container.style.position = 'relative';
  
  // Remove existing badges
  const existingBadges = container.querySelectorAll('.deepguard-badge');
  existingBadges.forEach(b => b.remove());
  
  container.appendChild(badge);
  
  // Log detailed results to console for debugging
  console.log('🛡️ DeepGuard Detection Result:', {
    prediction,
    confidence: confidence.toFixed(2) + '%',
    frequencyScore: frequencyScore ? (frequencyScore * 100).toFixed(2) + '%' : 'N/A',
    processingTime: processingTime ? processingTime.toFixed(2) + 's' : 'N/A',
    platform,
    timestamp: new Date().toISOString()
  });
  
  // Keep badge visible during continuous scanning
  setTimeout(() => {
    if (badge.parentElement) {
      badge.style.opacity = '0.85';
    }
  }, 5000);
}

// Show warning notification
function showWarningNotification(platform, confidence, result) {
  const notification = document.createElement('div');
  notification.className = 'deepguard-notification';
  
  // Build detailed message
  let detailsMsg = `${confidence.toFixed(1)}% confidence`;
  if (result.frequency_score) {
    detailsMsg += ` | Frequency anomaly: ${(result.frequency_score * 100).toFixed(1)}%`;
  }
  if (result.processing_time) {
    detailsMsg += ` | Analyzed in ${result.processing_time.toFixed(2)}s`;
  }
  
  notification.innerHTML = `
    <div class="notification-content">
      <div class="notification-icon">⚠️</div>
      <div class="notification-body">
        <div class="notification-title">Deepfake Detected!</div>
        <div class="notification-message">This ${platform} video contains manipulated content</div>
        <div class="notification-details">${detailsMsg}</div>
      </div>
      <button class="notification-close">✕</button>
    </div>
  `;
  
  document.body.appendChild(notification);
  
  // Close notification
  notification.querySelector('.notification-close').addEventListener('click', () => {
    notification.remove();
  });
  
  // Auto-remove after 8 seconds
  setTimeout(() => {
    notification.style.opacity = '0';
    setTimeout(() => notification.remove(), 300);
  }, 8000);
}

// Show informational badge
function showInfoBadge(video, message, type = 'info') {
  const container = video.parentElement;
  if (!container) return;
  
  // Remove existing info badges
  const existingInfo = container.querySelectorAll('.deepguard-info-badge');
  existingInfo.forEach(b => b.remove());
  
  // Determine icon and style based on type
  let icon = 'ℹ️';
  let badgeClass = 'info';
  
  if (type === 'warning') {
    icon = '⚠️';
    badgeClass = 'warning';
  } else if (type === 'error') {
    icon = '❌';
    badgeClass = 'error';
  }
  
  const infoBadge = document.createElement('div');
  infoBadge.className = `deepguard-info-badge ${badgeClass}`;
  infoBadge.innerHTML = `
    <div class="info-badge-content">
      <span class="info-icon">${icon}</span>
      <span class="info-text">${message}</span>
    </div>
  `;
  
  container.style.position = 'relative';
  container.appendChild(infoBadge);
  
  // Auto-hide after duration based on type
  const duration = type === 'error' ? 5000 : 3000;
  setTimeout(() => {
    if (infoBadge.parentElement) {
      infoBadge.style.opacity = '0';
      setTimeout(() => infoBadge.remove(), 300);
    }
  }, duration);
}

// Update statistics
function updateStats(result) {
  chrome.storage.local.get({
    videosScanned: 0,
    deepfakesFound: 0
  }, (items) => {
    const isFake = result.prediction === 'FAKE' || result.prediction === 'fake';
    
    chrome.storage.local.set({
      videosScanned: items.videosScanned + 1,
      deepfakesFound: items.deepfakesFound + (isFake ? 1 : 0)
    });
  });
}
