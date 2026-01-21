// Background service worker

console.log('🛡️ DeepGuard: Background service worker started');

// Listen for extension installation
chrome.runtime.onInstalled.addListener((details) => {
  if (details.reason === 'install') {
    console.log('DeepGuard installed!');
    
    // Set default settings
    chrome.storage.sync.set({
      autoScan: true,
      showWarnings: true,
      enableInstagram: true,
      enableFacebook: true,
      apiEndpoint: 'http://127.0.0.1:8000'
    });
    
    // Initialize stats
    chrome.storage.local.set({
      videosScanned: 0,
      deepfakesFound: 0
    });
    
    // Open welcome page
    chrome.tabs.create({
      url: 'https://github.com/Munazir151/deepfake_detection2'
    });
  }
});

// Handle messages from content scripts
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'scanVideo') {
    // Handle video scan request
    handleVideoScan(request.data, sender.tab.id);
  }
  
  if (request.action === 'updateBadge') {
    // Update extension badge with detection count
    updateBadge(request.count);
  }
});

// Handle video scan
async function handleVideoScan(data, tabId) {
  try {
    const settings = await chrome.storage.sync.get({ apiEndpoint: 'http://127.0.0.1:5000' });
    
    // Send notification to user
    chrome.notifications.create({
      type: 'basic',
      iconUrl: 'icons/icon128.png',
      title: 'DeepGuard',
      message: 'Scanning video for deepfakes...',
      priority: 1
    });
    
  } catch (error) {
    console.error('Error handling video scan:', error);
  }
}

// Update extension badge
function updateBadge(count) {
  if (count > 0) {
    chrome.action.setBadgeText({ text: count.toString() });
    chrome.action.setBadgeBackgroundColor({ color: '#f87171' });
  } else {
    chrome.action.setBadgeText({ text: '' });
  }
}

// Context menu for manual scanning
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: 'scanVideo',
    title: 'Scan for Deepfakes',
    contexts: ['video']
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === 'scanVideo') {
    chrome.tabs.sendMessage(tab.id, {
      action: 'manualScan'
    });
  }
});
