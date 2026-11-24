const { app, BrowserWindow, globalShortcut, ipcMain, screen } = require('electron');
const express = require('express');
const http = require('http');
const WebSocket = require('ws');

app.commandLine.appendSwitch('autoplay-policy', 'no-user-gesture-required');

let avatarWindow = null;
let isRecording = false;
let visibilityCheckInterval;
let externalHttpServer = null;
let highlightWss = null;
const highlightClients = new Set();

const BACKEND_URL = 'http://127.0.0.1:8002';
let currentVoice = 'jonnydepp';

function broadcastHighlightSync(message) {
  const payload = JSON.stringify(message);

  for (const client of highlightClients) {
    if (client.readyState === WebSocket.OPEN) {
      client.send(payload);
    }
  }
}

ipcMain.on('highlight-sync', (event, message) => {
  broadcastHighlightSync(message);
});

function setLedState(state) {
  if (!avatarWindow || avatarWindow.isDestroyed()) return;

  avatarWindow.webContents.executeJavaScript(`
    try {
      if (window.setLedState) {
        window.setLedState("${state}");
      } else {
        const led = document.getElementById('statusLed');
        if (led) {
          led.classList.remove('blink', 'recording');
          if ("${state}" === 'recording') {
            led.classList.add('recording');
          } else if ("${state}" === 'processing' || "${state}" === 'playing') {
            led.classList.add('blink');
          }
        }
      }
    } catch (e) {}
  `).catch(() => {});
}

function createWindow() {
  const primaryDisplay = screen.getPrimaryDisplay();
  const { width: screenWidth } = primaryDisplay.workAreaSize;

  const avatarWidth = 260;
  const avatarHeight = 330;
  const xPos = screenWidth - avatarWidth;

  avatarWindow = new BrowserWindow({
    width: avatarWidth,
    height: avatarHeight,
    x: xPos,
    y: 0,
    transparent: true,
    frame: false,
    alwaysOnTop: true,
    focusable: true,
    resizable: false,
    skipTaskbar: true,
    backgroundColor: '#00000000',
    type: 'toolbar',
    show: false,
    minWidth: avatarWidth,
    maxWidth: avatarWidth,
    minHeight: avatarHeight,
    maxHeight: avatarHeight,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      webSecurity: false,
      partition: 'persist:imtalker',
      autoplayPolicy: 'no-user-gesture-required'
    }
  });

  avatarWindow.setIgnoreMouseEvents(false);

  console.log(`📐 Avatar window positioned at (${xPos}, 0) with size ${avatarWidth}x${avatarHeight}`);

  const forceAlwaysOnTop = () => {
    if (!avatarWindow || avatarWindow.isDestroyed()) return;

    const levels = ['screen-saver', 'modal-panel', 'main-menu', 'status', 'pop-up-menu', 'critical'];

    for (const level of levels) {
      try {
        avatarWindow.setAlwaysOnTop(true, level);
        break;
      } catch (err) {
        continue;
      }
    }

    avatarWindow.setIgnoreMouseEvents(false);

    if (!avatarWindow.isVisible()) {
      avatarWindow.showInactive();
    }

    if (avatarWindow.isMinimized()) {
      avatarWindow.restore();
    }
  };

  avatarWindow.once('ready-to-show', () => {
    avatarWindow.show();
    avatarWindow.setIgnoreMouseEvents(false);
    forceAlwaysOnTop();
    console.log('👁️ Avatar window shown and pinned on top');
  });

  visibilityCheckInterval = setInterval(() => {
    if (avatarWindow && !avatarWindow.isDestroyed()) {
      forceAlwaysOnTop();
      avatarWindow.setIgnoreMouseEvents(false);
    }
  }, 2000);

  avatarWindow.on('blur', () => {
    console.log('🔍 Avatar window blurred - maintaining visibility');
    avatarWindow.setOpacity(0.95);
    avatarWindow.setAlwaysOnTop(true, 'floating');
    avatarWindow.setIgnoreMouseEvents(false);
  });

  avatarWindow.on('focus', () => {
    console.log('🔍 Avatar window focused');
    avatarWindow.setOpacity(1.0);
    avatarWindow.setAlwaysOnTop(true, 'normal');
    avatarWindow.setIgnoreMouseEvents(false);
  });

  avatarWindow.on('minimize', (event) => {
    event.preventDefault();
    console.log('🚫 Minimize blocked');
    forceAlwaysOnTop();
  });

  avatarWindow.webContents.session.clearCache().then(() => {
    const timestamp = Date.now();
    const avatarHtmlPath = `file://${__dirname}/avatar.html?t=${timestamp}`;

    console.log('🚀 Loading avatar window:', avatarHtmlPath);
    avatarWindow.loadURL(avatarHtmlPath);
  }).catch(err => {
    console.error('Cache clear failed:', err);
    avatarWindow.loadURL(`file://${__dirname}/avatar.html`);
  });

  avatarWindow.webContents.on('dom-ready', () => {
    console.log('✅ DOM ready - injecting click fixes');

    avatarWindow.webContents.insertCSS(`
      * {
        pointer-events: auto !important;
      }

      html, body {
        pointer-events: auto !important;
        background: transparent !important;
      }

      button, input, #ai-text, #connectBtn, #sendBtn, #userInput, .wakeword-status-bar {
        pointer-events: auto !important;
        -webkit-app-region: no-drag !important;
        position: relative;
        z-index: 9999;
      }

      button, #connectBtn, #sendBtn {
        cursor: pointer !important;
      }

      input, #ai-text, #userInput {
        cursor: text !important;
      }

      .drag-bar {
        -webkit-app-region: drag !important;
        pointer-events: auto !important;
        z-index: 99999;
      }

      .drag-bar * {
        pointer-events: none !important;
        -webkit-app-region: drag !important;
      }
    `);

    avatarWindow.webContents.executeJavaScript(`
      console.log('=== INJECTED CLICK FIX ===');

      setTimeout(() => {
        const buttons = document.querySelectorAll('button');
        console.log('Found buttons:', buttons.length);

        buttons.forEach(btn => {
          btn.disabled = false;
          btn.style.pointerEvents = 'auto';
          btn.style.cursor = 'pointer';
          btn.style.zIndex = '9999';
          console.log('Enabled button:', btn.id || btn.textContent);
        });

        const inputs = document.querySelectorAll('input');
        inputs.forEach(input => {
          input.disabled = false;
          input.style.pointerEvents = 'auto';
          input.style.cursor = 'text';
          input.style.zIndex = '9999';
          console.log('Enabled input:', input.id);
        });

        const userInput = document.getElementById('userInput');
        if (userInput) {
          userInput.focus();
        }
      }, 100);

      document.addEventListener('click', (e) => {
        console.log('Click detected on:', e.target.tagName, e.target.id || e.target.className);
      }, true);

      console.log('Click fix injected successfully');
    `);
  });

  avatarWindow.webContents.on('did-finish-load', () => {
    console.log('✅ Avatar window loaded');
    avatarWindow.setIgnoreMouseEvents(false);
    forceAlwaysOnTop();
    registerGlobalShortcuts();
  });

  avatarWindow.webContents.on('did-fail-load', (event, errorCode, errorDescription) => {
    console.error('❌ Load failed:', errorCode, errorDescription);

    setTimeout(() => {
      avatarWindow.loadURL(`file://${__dirname}/avatar.html?retry=${Date.now()}`);
    }, 2000);
  });

  avatarWindow.on('closed', () => {
    if (visibilityCheckInterval) {
      clearInterval(visibilityCheckInterval);
    }

    avatarWindow = null;
  });
}

async function playGeneratedSessionInAvatarWindow(sessionId, totalChunks) {
  if (!avatarWindow || avatarWindow.isDestroyed()) {
    throw new Error('Avatar window not ready');
  }

  const js = `
    (async () => {
      const BACKEND_URL = ${JSON.stringify(BACKEND_URL)};
      const sessionId = ${JSON.stringify(sessionId)};
      const totalChunks = ${JSON.stringify(totalChunks)};

      window.__externalReadAloudActive = true;

      if (window.setLedState) {
        window.setLedState('playing');
      } else {
        const led = document.getElementById('statusLed');
        if (led) {
          led.classList.add('blink');
          led.classList.remove('recording');
        }
      }

      function sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
      }

      function keepGreenDotOnTop() {
        const statusLed = document.getElementById('statusLed');

        if (statusLed) {
          statusLed.style.display = 'block';
          statusLed.style.visibility = 'visible';
          statusLed.style.opacity = '0.95';
          statusLed.style.zIndex = '100000';
          statusLed.style.pointerEvents = 'none';
        }
      }

      function forceBaseAvatarVisible() {
        document.body.classList.remove('playing-video');

        const externalVideo = document.getElementById('externalReadAloudVideo');
        if (externalVideo) {
          try {
            externalVideo.pause();
            externalVideo.loop = false;
            externalVideo.removeAttribute('src');
            externalVideo.load();
            externalVideo.remove();
          } catch (err) {
            console.warn('Could not remove external video:', err);
          }
        }

        document.querySelectorAll('video').forEach(video => {
          if (video.id === 'v_idle') {
            video.style.opacity = '1';
            video.style.zIndex = '5';
            video.style.display = 'block';
            video.style.visibility = 'visible';
            video.loop = true;
            video.muted = true;
            video.defaultMuted = true;
            video.volume = 0;

            try {
              video.play().catch(() => {});
            } catch (err) {
              console.warn('Could not play idle video:', err);
            }

            return;
          }

          if (video.id === 'v1' || video.id === 'v2') {
            video.pause();
            video.loop = false;
            video.removeAttribute('src');
            video.load();
            video.classList.remove('active');
            video.style.opacity = '0';
            video.style.zIndex = '2';
            video.style.display = 'block';
            video.style.visibility = 'visible';
          }
        });

        document.querySelectorAll('img, canvas, .avatar-image, #avatarImage, #avatar, .base-avatar').forEach(el => {
          try {
            if (el.dataset.externalReadAloudPrevVisibility !== undefined) {
              el.style.visibility = el.dataset.externalReadAloudPrevVisibility;
              delete el.dataset.externalReadAloudPrevVisibility;
            } else {
              el.style.visibility = 'visible';
            }

            if (el.dataset.externalReadAloudPrevOpacity !== undefined) {
              el.style.opacity = el.dataset.externalReadAloudPrevOpacity;
              delete el.dataset.externalReadAloudPrevOpacity;
            } else {
              el.style.opacity = '1';
            }
          } catch (err) {
            console.warn('Could not restore base avatar element:', err);
          }
        });

        keepGreenDotOnTop();
      }

      function hideBaseAvatarImage() {
        document.body.classList.add('playing-video');

        const elements = document.querySelectorAll('img, canvas, .avatar-image, #avatarImage, #avatar, .base-avatar');

        elements.forEach(el => {
          if (el.id === 'externalReadAloudVideo' || el.id === 'statusLed') {
            return;
          }

          const rect = el.getBoundingClientRect();

          if (rect.width > 50 && rect.height > 50) {
            if (el.dataset.externalReadAloudPrevVisibility === undefined) {
              el.dataset.externalReadAloudPrevVisibility = el.style.visibility || '';
            }

            if (el.dataset.externalReadAloudPrevOpacity === undefined) {
              el.dataset.externalReadAloudPrevOpacity = el.style.opacity || '';
            }

            el.style.visibility = 'hidden';
            el.style.opacity = '0';
          }
        });

        keepGreenDotOnTop();
      }

      function stopExistingVideos() {
        const videos = document.querySelectorAll('video');

        videos.forEach(video => {
          if (video.id === 'externalReadAloudVideo') {
            return;
          }

          try {
            if (video.id === 'v_idle') {
              video.pause();
              video.loop = true;
              video.muted = true;
              video.defaultMuted = true;
              video.volume = 0;
              return;
            }

            video.pause();
            video.loop = false;
            video.muted = false;
            video.volume = 1.0;
          } catch (err) {
            console.warn('Could not stop existing video:', err);
          }
        });
      }

      function getVideoElement() {
        const container = document.querySelector('.video-container');

        if (!container) {
          throw new Error('No .video-container found');
        }

        let video = document.getElementById('externalReadAloudVideo');

        if (!video) {
          video = document.createElement('video');
          video.id = 'externalReadAloudVideo';

          const statusLed = document.getElementById('statusLed');

          if (statusLed && statusLed.parentElement === container) {
            container.insertBefore(video, statusLed);
          } else {
            container.appendChild(video);
          }
        }

        if (video.parentElement !== container) {
          const statusLed = document.getElementById('statusLed');

          if (statusLed && statusLed.parentElement === container) {
            container.insertBefore(video, statusLed);
          } else {
            container.appendChild(video);
          }
        }

        video.autoplay = false;
        video.loop = false;
        video.muted = false;
        video.defaultMuted = false;
        video.volume = 1.0;
        video.controls = false;
        video.playsInline = true;
        video.setAttribute('playsinline', 'true');
        video.removeAttribute('muted');

        video.style.position = 'absolute';
        video.style.inset = '0';
        video.style.left = '0';
        video.style.top = '0';
        video.style.width = '100%';
        video.style.height = '100%';
        video.style.objectFit = 'cover';
        video.style.zIndex = '50';
        video.style.display = 'block';
        video.style.visibility = 'visible';
        video.style.opacity = '1';
        video.style.pointerEvents = 'none';
        video.style.background = 'transparent';
        video.style.borderRadius = '5px';

        const idleVideo = document.getElementById('v_idle');
        if (idleVideo) {
          idleVideo.style.opacity = '0';
          idleVideo.style.zIndex = '1';
        }

        document.body.classList.add('playing-video');
        keepGreenDotOnTop();

        return video;
      }

      async function waitForChunk(chunkIndex) {
        const url = BACKEND_URL + '/wait_for_chunk/' + sessionId + '/' + chunkIndex;

        while (true) {
          const response = await fetch(url);
          const data = await response.json();

          if (data.ready && data.url) {
            return BACKEND_URL + data.url;
          }

          if (data.error && data.error !== 'Timeout') {
            throw new Error(data.error);
          }

          await sleep(500);
        }
      }

      function playVideo(url) {
        return new Promise((resolve, reject) => {
          stopExistingVideos();
          hideBaseAvatarImage();

          const video = getVideoElement();

          video.pause();
          video.loop = false;
          video.muted = false;
          video.defaultMuted = false;
          video.volume = 1.0;
          video.autoplay = false;
          video.controls = false;
          video.playsInline = true;
          video.removeAttribute('muted');

          video.onended = () => {
            console.log('✅ Read-aloud chunk ended:', url);
            resolve();
          };

          video.onerror = () => {
            reject(new Error('Video playback failed: ' + url));
          };

          video.onloadedmetadata = () => {
            console.log(
              '🎬 Read-aloud video metadata:',
              'duration=', video.duration,
              'muted=', video.muted,
              'volume=', video.volume,
              'videoWidth=', video.videoWidth,
              'videoHeight=', video.videoHeight,
              'src=', url
            );
          };

          video.oncanplay = async () => {
            try {
              video.muted = false;
              video.defaultMuted = false;
              video.volume = 1.0;
              video.loop = false;
              video.removeAttribute('muted');

              keepGreenDotOnTop();

              await video.play();

              console.log(
                '🔊 Playing read-aloud video:',
                'muted=', video.muted,
                'volume=', video.volume,
                'paused=', video.paused
              );
            } catch (err) {
              reject(err);
            }
          };

          video.src = url;
          video.load();
        });
      }

      try {
        console.log('▶️ External read-aloud playback starting:', sessionId, totalChunks);

        for (let i = 0; i < totalChunks; i++) {
          const videoUrl = await waitForChunk(i);
          console.log('▶️ Playing lipsync chunk:', i, videoUrl);
          await playVideo(videoUrl);
        }

        console.log('✅ External read-aloud playback complete:', sessionId);
      } catch (err) {
        console.error('❌ External read-aloud playback failed:', err);
      } finally {
        window.__externalReadAloudActive = false;
        forceBaseAvatarVisible();

        if (window.setLedState) {
          window.setLedState('listening');
        } else {
          const led = document.getElementById('statusLed');
          if (led) {
            led.classList.remove('blink', 'recording');
          }
        }
      }
    })();
  `;

  await avatarWindow.webContents.executeJavaScript(js);
}


async function playHighlightedReadSessionInAvatarWindow({ extensionSessionId, backendSessionId, chunks }) {
  if (!avatarWindow || avatarWindow.isDestroyed()) {
    throw new Error('Avatar window not ready');
  }

  const js = `
    (async () => {
      const BACKEND_URL = ${JSON.stringify(BACKEND_URL)};
      const extensionSessionId = ${JSON.stringify(extensionSessionId)};
      const backendSessionId = ${JSON.stringify(backendSessionId)};
      const chunks = ${JSON.stringify(chunks)};
      const { ipcRenderer } = require('electron');

      window.__externalReadAloudActive = true;

      function emitHighlightSync(message) {
        ipcRenderer.send('highlight-sync', message);
      }

      function sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
      }

      function keepGreenDotOnTop() {
        const statusLed = document.getElementById('statusLed');
        if (!statusLed) return;
        statusLed.style.display = 'block';
        statusLed.style.visibility = 'visible';
        statusLed.style.opacity = '0.95';
        statusLed.style.zIndex = '100000';
        statusLed.style.pointerEvents = 'none';
      }

      function showIdleAvatar() {
        const externalVideo = document.getElementById('externalReadAloudVideo');
        if (externalVideo) {
          try {
            externalVideo.pause();
            externalVideo.removeAttribute('src');
            externalVideo.load();
          } catch {}
        }

        const idleVideo = document.getElementById('v_idle');
        if (idleVideo) {
          idleVideo.style.opacity = '1';
          idleVideo.style.zIndex = '5';
          idleVideo.loop = true;
          idleVideo.muted = true;
          idleVideo.defaultMuted = true;
          idleVideo.volume = 0;
          try { idleVideo.play().catch(() => {}); } catch {}
        }

        keepGreenDotOnTop();
      }

      function getVideoElement() {
        const container = document.querySelector('.video-container');
        if (!container) throw new Error('No .video-container found');

        let video = document.getElementById('externalReadAloudVideo');
        if (!video) {
          video = document.createElement('video');
          video.id = 'externalReadAloudVideo';
          const statusLed = document.getElementById('statusLed');
          if (statusLed && statusLed.parentElement === container) {
            container.insertBefore(video, statusLed);
          } else {
            container.appendChild(video);
          }
        }

        video.autoplay = false;
        video.loop = false;
        video.muted = false;
        video.defaultMuted = false;
        video.volume = 1.0;
        video.controls = false;
        video.playsInline = true;
        video.setAttribute('playsinline', 'true');
        video.removeAttribute('muted');

        video.style.position = 'absolute';
        video.style.inset = '0';
        video.style.width = '100%';
        video.style.height = '100%';
        video.style.objectFit = 'cover';
        video.style.zIndex = '50';
        video.style.display = 'block';
        video.style.visibility = 'visible';
        video.style.opacity = '1';
        video.style.pointerEvents = 'none';
        video.style.background = 'transparent';
        video.style.borderRadius = '5px';

        const idleVideo = document.getElementById('v_idle');
        if (idleVideo) {
          idleVideo.style.opacity = '0';
          idleVideo.style.zIndex = '1';
        }

        keepGreenDotOnTop();
        return video;
      }

      async function waitForChunk(chunkIndex) {
        const url = BACKEND_URL + '/wait_for_chunk/' + backendSessionId + '/' + chunkIndex;

        while (true) {
          const response = await fetch(url);
          const data = await response.json();

          if (data.ready && data.url) {
            return {
              url: BACKEND_URL + data.url,
              duration: data.duration
            };
          }

          if (data.error && data.error !== 'Timeout') {
            throw new Error(data.error);
          }

          await sleep(300);
        }
      }

      function playVideoForChunk(chunk, videoUrl, knownDuration) {
        return new Promise((resolve, reject) => {
          const video = getVideoElement();
          let progressTimer = null;
          let started = false;

          function emitTime() {
            emitHighlightSync({
              type: 'chunkTime',
              sessionId: extensionSessionId,
              chunkId: chunk.chunkId,
              currentTime: video.currentTime || 0,
              duration: video.duration || knownDuration || chunk.duration || 0
            });
          }

          function stopTimer() {
            if (progressTimer) {
              clearInterval(progressTimer);
              progressTimer = null;
            }
          }

          video.pause();
          video.loop = false;
          video.muted = false;
          video.defaultMuted = false;
          video.volume = 1.0;
          video.autoplay = false;
          video.controls = false;
          video.playsInline = true;
          video.removeAttribute('muted');

          video.onloadedmetadata = () => {
            const duration = video.duration || knownDuration || chunk.duration || 0;
            emitHighlightSync({
              type: 'chunkReady',
              sessionId: extensionSessionId,
              chunkId: chunk.chunkId,
              duration
            });
          };

          video.onplay = () => {
            if (started) return;
            started = true;

            emitHighlightSync({
              type: 'chunkStart',
              sessionId: extensionSessionId,
              chunkId: chunk.chunkId
            });

            emitTime();
            progressTimer = setInterval(emitTime, 100);
          };

          video.onended = () => {
            stopTimer();
            emitTime();
            emitHighlightSync({
              type: 'chunkEnd',
              sessionId: extensionSessionId,
              chunkId: chunk.chunkId
            });
            resolve();
          };

          video.onerror = () => {
            stopTimer();
            reject(new Error('Video playback failed: ' + videoUrl));
          };

          video.oncanplay = async () => {
            try {
              await video.play();
            } catch (err) {
              reject(err);
            }
          };

          video.src = videoUrl;
          video.load();
        });
      }

      try {
        if (window.setLedState) window.setLedState('playing');

        for (const chunk of chunks) {
          const ready = await waitForChunk(chunk.index);
          await playVideoForChunk(chunk, ready.url, ready.duration);
        }
      } catch (err) {
        console.error('❌ Highlight read playback failed:', err);
      } finally {
        window.__externalReadAloudActive = false;
        showIdleAvatar();
        if (window.setLedState) window.setLedState('listening');
      }
    })();
  `;

  await avatarWindow.webContents.executeJavaScript(js);
}


const avatarPlaybackQueue = [];
let avatarPlaybackQueueRunning = false;

function enqueueAvatarPlayback(sessionId, totalChunks, label = 'read-aloud') {
  avatarPlaybackQueue.push({
    sessionId,
    totalChunks,
    label,
    enqueuedAt: Date.now()
  });

  console.log(
    `[avatar-playback-queue] queued ${label}: ${sessionId} (${totalChunks} chunks). Queue length: ${avatarPlaybackQueue.length}`
  );

  runAvatarPlaybackQueue().catch(error => {
    console.error('[avatar-playback-queue] worker error:', error);
  });
}

async function runAvatarPlaybackQueue() {
  if (avatarPlaybackQueueRunning) {
    return;
  }

  avatarPlaybackQueueRunning = true;

  try {
    while (avatarPlaybackQueue.length > 0) {
      const item = avatarPlaybackQueue.shift();

      console.log(
        `[avatar-playback-queue] playing ${item.label}: ${item.sessionId} (${item.totalChunks} chunks)`
      );

      try {
        await playGeneratedSessionInAvatarWindow(
          item.sessionId,
          item.totalChunks
        );

        console.log(
          `[avatar-playback-queue] finished ${item.label}: ${item.sessionId}`
        );
      } catch (error) {
        console.error(
          `[avatar-playback-queue] playback failed for ${item.sessionId}:`,
          error
        );
      }
    }
  } finally {
    avatarPlaybackQueueRunning = false;
  }
}


function startExternalTextServer() {
  const expressApp = express();

  expressApp.use(express.json());

  expressApp.use((req, res, next) => {
    res.header('Access-Control-Allow-Origin', '*');
    res.header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.header('Access-Control-Allow-Headers', 'Content-Type');

    if (req.method === 'OPTIONS') {
      return res.sendStatus(204);
    }

    next();
  });


  expressApp.post('/start_read_session', async (req, res) => {
    setLedState('processing');

    try {
      const extensionSessionId = String(req.body.sessionId || '').trim();
      const text = String(req.body.selectedText || req.body.text || '').trim();
      const voice = String(req.body.voice || currentVoice || 'jonnydepp').trim();
      const targetLanguage = String(req.body.targetLanguage || req.body.target_language || '').trim();

console.log("🌍 Electron received targetLanguage:", targetLanguage || "(none)");
console.log("🌍 Raw /start_read_session body:", req.body);

      if (!extensionSessionId) {
        return res.status(400).json({ success: false, error: 'Missing sessionId' });
      }

      if (!text) {
        return res.status(400).json({ success: false, error: 'Missing selectedText' });
      }

      if (!avatarWindow || avatarWindow.isDestroyed()) {
        return res.status(503).json({ success: false, error: 'Avatar window not ready' });
      }

      console.log(`📖 Highlight read session requested: ${extensionSessionId}`);

      const formBody = new URLSearchParams();
      formBody.append('text', text);
      formBody.append('voice', voice);
      formBody.append('read_aloud', 'true');
      formBody.append('skip_avatar', 'false');
      formBody.append('client_session_id', extensionSessionId);

      if (targetLanguage) {
        formBody.append('target_language', targetLanguage);
      }

      const backendResponse = await fetch(`${BACKEND_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: formBody.toString()
      });

      const backendData = await backendResponse.json();

      if (!backendResponse.ok || !backendData.success) {
        console.error('❌ Avatar backend error:', backendData);
        return res.status(500).json({
          success: false,
          error: 'Avatar backend failed',
          backend: backendData
        });
      }

      const chunks = Array.isArray(backendData.chunks)
        ? backendData.chunks.map((chunk, index) => ({
            chunkId: chunk.chunkId || `${backendData.session_id}_${index}`,
            index,
            sourceText: chunk.sourceText || chunk.source_text || chunk.text || '',
            spokenText: chunk.spokenText || chunk.spoken_text || chunk.text || '',
            duration: chunk.duration || null
          }))
        : [{
            chunkId: `${backendData.session_id}_0`,
            index: 0,
            sourceText: text,
            spokenText: text,
            duration: null
          }];

      const readPlan = {
        type: 'readPlan',
        sessionId: extensionSessionId,
        backendSessionId: backendData.session_id,
        chunks
      };

      playHighlightedReadSessionInAvatarWindow({
        extensionSessionId,
        backendSessionId: backendData.session_id,
        chunks
      }).catch(error => {
        console.error('❌ Highlight playback error:', error);
      });

      return res.json(readPlan);
    } catch (error) {
      setLedState('listening');
      console.error('❌ /start_read_session error:', error);
      return res.status(500).json({ success: false, error: error.message });
    }
  });

  expressApp.post('/send_text', async (req, res) => {
    setLedState('processing');

try {
      const text = (req.body.text || '').trim();
      const voice = (req.body.voice || currentVoice || 'jonnydepp').trim();

      if (!text || text.length === 0) {
        return res.status(400).json({
          success: false,
          error: 'No text provided'
        });
      }

      if (!avatarWindow || avatarWindow.isDestroyed()) {
        return res.status(503).json({
          success: false,
          error: 'Avatar window not ready'
        });
      }

      console.log(`📨 External read-aloud text received: "${text}"`);

      const formBody = new URLSearchParams();
      formBody.append('text', text);
      formBody.append('voice', voice);
      formBody.append('read_aloud', 'true');
      formBody.append('skip_avatar', 'false');

      const backendResponse = await fetch(`${BACKEND_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded'
        },
        body: formBody.toString()
      });

      const backendData = await backendResponse.json();

      if (!backendResponse.ok || !backendData.success) {
        console.error('❌ Avatar backend error:', backendData);

        return res.status(500).json({
          success: false,
          error: 'Avatar backend failed',
          backend: backendData
        });
      }

      console.log(`✅ Read-aloud session created: ${backendData.session_id}`);
      console.log(`🔊 LLM used: ${backendData.llm_used}`);

      enqueueAvatarPlayback(
        backendData.session_id,
        backendData.total_chunks,
        'external-send-text'
      );

      return res.json({
        success: true,
        message: 'Read-aloud sent to avatar',
        session_id: backendData.session_id,
        total_chunks: backendData.total_chunks,
        response_text: backendData.response_text,
        display_text: backendData.display_text,
        llm_used: backendData.llm_used
      });
    } catch (error) {
      setLedState('listening');
console.error('❌ /send_text error:', error);

      return res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  const PORT = 8003;

  externalHttpServer = expressApp.listen(PORT, () => {
    console.log(`🔌 External text server listening on http://localhost:${PORT}`);
  });

  highlightWss = new WebSocket.Server({
    server: externalHttpServer,
    path: '/highlight-sync'
  });

  highlightWss.on('connection', socket => {
    console.log('🟨 Highlight sync client connected');
    highlightClients.add(socket);

    socket.on('close', () => {
      highlightClients.delete(socket);
    });
  });
}

function registerGlobalShortcuts() {
  globalShortcut.unregisterAll();

  console.log('========================================');
  console.log('🌍 REGISTERING GLOBAL SHORTCUTS (TOGGLE + DEBOUNCE)');
  console.log('========================================');

  let lastTriggerTime = 0;
  const DEBOUNCE_MS = 1000;

  const toggleShortcut = globalShortcut.register('Ctrl+Shift+Space', () => {
    const now = Date.now();

    if (now - lastTriggerTime < DEBOUNCE_MS) {
      console.log('⏱️ Debounced repeat keypress (Ctrl+Shift+Space)');
      return;
    }

    lastTriggerTime = now;

    if (!isRecording) {
      isRecording = true;
      setLedState('recording');
      console.log('🎤 Ctrl+Shift+Space - START recording');

      if (avatarWindow && !avatarWindow.isDestroyed()) {
        avatarWindow.webContents.send('voice-record-start');
      }
    } else {
      isRecording = false;
      setLedState('processing');
      console.log('🎤 Ctrl+Shift+Space - STOP recording');

      if (avatarWindow && !avatarWindow.isDestroyed()) {
        avatarWindow.webContents.send('voice-record-stop-auto');
      }
    }
  });

  console.log(`Ctrl+Shift+Space toggle (debounced): ${toggleShortcut ? '✅' : '❌'}`);

  if (!toggleShortcut) {
    globalShortcut.register('Alt+Space', () => {
      const now = Date.now();

      if (now - lastTriggerTime < DEBOUNCE_MS) {
        console.log('⏱️ Debounced repeat keypress (Alt+Space)');
        return;
      }

      lastTriggerTime = now;

      if (!isRecording) {
        isRecording = true;
        setLedState('recording');
        avatarWindow?.webContents.send('voice-record-start');
      } else {
        isRecording = false;
        setLedState('processing');
        avatarWindow?.webContents.send('voice-record-stop-auto');
      }
    });

    console.log('Alt+Space toggle registered as fallback (debounced)');
  }

  globalShortcut.register('Ctrl+Shift+A', () => {
    if (avatarWindow?.isVisible()) {
      avatarWindow.hide();
    } else {
      avatarWindow?.show();
    }
  });

  console.log('Ctrl+Shift+A: ✅');

  globalShortcut.register('Ctrl+Shift+X', () => {
    console.log('🛑 Ctrl+Shift+X - STOP avatar playback');

    avatarPlaybackQueue.length = 0;
    avatarPlaybackQueueRunning = false;
    console.log('🛑 Avatar playback queue cleared');

    if (!avatarWindow || avatarWindow.isDestroyed()) {
      return;
    }

    avatarWindow.webContents.executeJavaScript(`
      try {
        window.__externalReadAloudActive = false;
        document.body.classList.remove('playing-video');

        const externalVideo = document.getElementById('externalReadAloudVideo');
        if (externalVideo) {
          try {
            externalVideo.pause();
            externalVideo.loop = false;
            externalVideo.removeAttribute('src');
            externalVideo.load();
            externalVideo.remove();
          } catch (err) {
            console.warn('Could not remove external video:', err);
          }
        }

        document.querySelectorAll('video').forEach(video => {
          try {
            video.pause();
            video.loop = false;

            if (video.id === 'v_idle') {
              video.style.opacity = '1';
              video.style.zIndex = '5';
              video.style.display = 'block';
              video.style.visibility = 'visible';
              video.loop = true;
              video.muted = true;
              video.defaultMuted = true;
              video.volume = 0;

              try {
                video.play().catch(() => {});
              } catch (err) {
                console.warn('Could not play idle video:', err);
              }

              return;
            }

            if (video.id === 'v1' || video.id === 'v2') {
              video.removeAttribute('src');
              video.load();
              video.classList.remove('active');
              video.style.opacity = '0';
              video.style.zIndex = '2';
              video.style.display = 'block';
              video.style.visibility = 'visible';
            }
          } catch (err) {
            console.warn('Could not stop video:', err);
          }
        });

        const statusLed = document.getElementById('statusLed');
        if (statusLed) {
          statusLed.style.display = 'block';
          statusLed.style.visibility = 'visible';
          statusLed.style.opacity = '0.95';
          statusLed.style.zIndex = '100000';
        }

        document.querySelectorAll('img, canvas, .avatar-image, #avatarImage, #avatar, .base-avatar').forEach(el => {
          try {
            if (el.dataset.externalReadAloudPrevVisibility !== undefined) {
              el.style.visibility = el.dataset.externalReadAloudPrevVisibility;
              delete el.dataset.externalReadAloudPrevVisibility;
            } else {
              el.style.visibility = 'visible';
            }

            if (el.dataset.externalReadAloudPrevOpacity !== undefined) {
              el.style.opacity = el.dataset.externalReadAloudPrevOpacity;
              delete el.dataset.externalReadAloudPrevOpacity;
            } else {
              el.style.opacity = '1';
            }
          } catch (err) {
            console.warn('Could not restore avatar:', err);
          }
        });

        console.log('🛑 Avatar playback stopped by hotkey');
      } catch (err) {
        console.error('Failed inside Ctrl+Shift+X injected stop script:', err);
      }
    `).catch(err => {
      console.error('❌ Failed to stop avatar playback:', err);
    });
  });

  console.log('Ctrl+Shift+X stop playback: ✅');

  console.log('========================================');
  console.log('🎤 PUSH-TO-TOGGLE: Press Ctrl+Shift+Space once');
  console.log('   First press = start recording');
  console.log('   Second press = stop and transcribe');
  console.log('========================================');
}

ipcMain.on('recording-started', () => {
  setLedState('recording');
  console.log('🔊 Renderer confirmed recording started');
});

ipcMain.on('recording-stopped', () => {
  setLedState('processing');
  console.log('🔇 Renderer confirmed recording stopped');
});

app.whenReady().then(async () => {
  const { session } = require('electron');

  session.defaultSession.clearCache();

  console.log('🧹 Cache cleared');
  console.log('========================================');
  console.log('🎬 ELECTRON AVATAR - WITH CLICK FIXES + READ-ALOUD PLAYBACK');
  console.log('========================================');

  createWindow();
  startExternalTextServer();
});

app.on('will-quit', () => {
  globalShortcut.unregisterAll();
  console.log('👋 Global shortcuts unregistered');

  if (visibilityCheckInterval) {
    clearInterval(visibilityCheckInterval);
  }

  if (highlightWss) {
    try {
      highlightWss.close();
    } catch {}
  }

  if (externalHttpServer) {
    externalHttpServer.close(() => {
      console.log('🔌 External text server closed');
    });
  }

  console.log('👋 Cleanup complete');
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (avatarWindow === null) {
    createWindow();
  }
});