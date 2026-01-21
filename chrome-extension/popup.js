// Popup script for settings and stats

// Load settings and stats on popup open
document.addEventListener('DOMContentLoaded', async () => {
  loadSettings();
  loadStats();
  checkAPIStatus();
  
  // Setup event listeners
  document.getElementById('saveBtn').addEventListener('click', saveSettings);
  document.getElementById('autoScan').addEventListener('change', toggleAutoScan);
});

// Load saved settings from storage
function loadSettings() {
  chrome.storage.sync.get({
    autoScan: true,
    showWarnings: true,
    enableInstagram: true,
    enableFacebook: true,
    apiEndpoint: 'http://127.0.0.1:5000'
  }, (items) => {
    document.getElementById('autoScan').checked = items.autoScan;
    document.getElementById('showWarnings').checked = items.showWarnings;
    document.getElementById('enableInstagram').checked = items.enableInstagram;
    document.getElementById('enableFacebook').checked = items.enableFacebook;
    document.getElementById('apiEndpoint').value = items.apiEndpoint;
  });
}

// Save settings to storage
function saveSettings() {
  const settings = {
    autoScan: document.getElementById('autoScan').checked,
    showWarnings: document.getElementById('showWarnings').checked,
    enableInstagram: document.getElementById('enableInstagram').checked,
    enableFacebook: document.getElementById('enableFacebook').checked,
    apiEndpoint: document.getElementById('apiEndpoint').value
  };

  chrome.storage.sync.set(settings, () => {
    // Show success feedback
    const btn = document.getElementById('saveBtn');
    const originalText = btn.textContent;
    btn.textContent = '✓ Saved!';
    btn.style.background = 'rgba(74, 222, 128, 0.3)';
    
    setTimeout(() => {
      btn.textContent = originalText;
      btn.style.background = '';
    }, 2000);
    
    // Notify content scripts about settings change
    chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
      if (tabs[0]) {
        chrome.tabs.sendMessage(tabs[0].id, {
          action: 'settingsUpdated',
          settings: settings
        });
      }
    });
  });
}

// Load statistics
function loadStats() {
  chrome.storage.local.get({
    videosScanned: 0,
    deepfakesFound: 0
  }, (items) => {
    document.getElementById('videosScanned').textContent = items.videosScanned;
    document.getElementById('deepfakesFound').textContent = items.deepfakesFound;
  });
}

// Toggle auto-scan
function toggleAutoScan(e) {
  chrome.storage.sync.set({ autoScan: e.target.checked });
}

// Check API status
async function checkAPIStatus() {
  const statusDiv = document.getElementById('apiStatus');
  const statusDot = document.getElementById('statusDot');
  const statusText = document.getElementById('statusText');
  
  try {
    const settings = await chrome.storage.sync.get({ apiEndpoint: 'http://127.0.0.1:5000' });
    const response = await fetch(`${settings.apiEndpoint}/health`, {
      method: 'GET',
      signal: AbortSignal.timeout(5000)
    });
    
    if (response.ok) {
      const data = await response.json();
      statusDiv.className = 'api-status online';
      statusDot.className = 'status-dot online';
      statusText.textContent = `API Online - ${data.app_name}`;
    } else {
      throw new Error('API returned error');
    }
  } catch (error) {
    statusDiv.className = 'api-status offline';
    statusDot.className = 'status-dot offline';
    statusText.textContent = 'API Offline - Check settings';
  }
}

// Reset stats (hidden feature - double click on stats)
document.querySelector('.stats')?.addEventListener('dblclick', () => {
  if (confirm('Reset all statistics?')) {
    chrome.storage.local.set({
      videosScanned: 0,
      deepfakesFound: 0
    }, () => {
      loadStats();
    });
  }
});
