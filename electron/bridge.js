// bridge.js - Simple connection between Electron and your backend
const { ipcMain } = require('electron');

function setupAvatarBridge(avatarWindow, mainWindow) {
  // This connects the minimal avatar to your FULL app
  
  // When avatar receives voice command, forward to main app
  ipcMain.on('avatar-voice-command', (event, audioData) => {
    console.log('🎤 Forwarding voice command to main app');
    
    // Option 1: Send to your existing full app (running on port 3000)
    fetch('http://localhost:8000/api/voice_query', {
      method: 'POST',
      body: audioData
    })
    .then(response => response.json())
    .then(result => {
      // Send result to both windows
      if (mainWindow) {
        mainWindow.webContents.send('process-voice-result', result);
      }
      if (avatarWindow) {
        avatarWindow.webContents.send('avatar-update', {
          type: 'text',
          text: "Processing your request...",
          typing: true
        });
      }
    });
  });
  
  // When main app has updates, forward to avatar
  ipcMain.on('update-avatar', (event, data) => {
    if (avatarWindow && !avatarWindow.isDestroyed()) {
      avatarWindow.webContents.send('avatar-update', data);
    }
  });
}

module.exports = { setupAvatarBridge };