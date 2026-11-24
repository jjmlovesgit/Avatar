// background.js
// Phase 1 Avatar webpage highlighting router.
// Adds multilingual right-click menu support.

const ELECTRON_HTTP = "http://127.0.0.1:8003";
const ELECTRON_WS = "ws://127.0.0.1:8003/highlight-sync";

const MENU_ROOT_ID = "avatar-read-root";
const MENU_ORIGINAL_ID = "avatar-read-original";
const MENU_TRANSLATE_PREFIX = "avatar-translate-";

const COMMAND_ID = "send-selected-text";

const DEBUG = false;

const SUPPORTED_LANGUAGES = {
  english: "English",
  french: "French",
  german: "German",
  spanish: "Spanish",
  portuguese: "Portuguese",
  italian: "Italian"
};

const sessionsById = new Map();

let highlightSocket = null;
let reconnectTimer = null;

function debugLog(...args) {
  if (DEBUG) {
    console.log("[AvatarStudio]", ...args);
  }
}

function debugWarn(...args) {
  if (DEBUG) {
    console.warn("[AvatarStudio]", ...args);
  }
}

// -----------------------------
// Install / context menu
// -----------------------------

chrome.runtime.onInstalled.addListener(() => {
  createContextMenu();
});

chrome.runtime.onStartup.addListener(() => {
  createContextMenu();
});

function createContextMenu() {
  chrome.contextMenus.removeAll(() => {
    chrome.contextMenus.create({
      id: MENU_ROOT_ID,
      title: "Read with avatar",
      contexts: ["selection"]
    });

    chrome.contextMenus.create({
      id: MENU_ORIGINAL_ID,
      parentId: MENU_ROOT_ID,
      title: "Original",
      contexts: ["selection"]
    });

    for (const [languageKey, languageLabel] of Object.entries(SUPPORTED_LANGUAGES)) {
      chrome.contextMenus.create({
        id: `${MENU_TRANSLATE_PREFIX}${languageKey}`,
        parentId: MENU_ROOT_ID,
        title: `Translate to ${languageLabel}`,
        contexts: ["selection"]
      });
    }
  });
}

// -----------------------------
// Context menu click
// -----------------------------

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (!info || !info.menuItemId) return;

  const isOriginal = info.menuItemId === MENU_ORIGINAL_ID;
  const isTranslate = String(info.menuItemId).startsWith(MENU_TRANSLATE_PREFIX);

  if (!isOriginal && !isTranslate) {
    return;
  }

  if (!tab || typeof tab.id !== "number" || tab.id < 0) {
    debugWarn("No valid tab id for Avatar Studio action:", tab);
    return;
  }

  let targetLanguage = null;

  if (isTranslate) {
    const languageKey = String(info.menuItemId).replace(MENU_TRANSLATE_PREFIX, "");
    targetLanguage = SUPPORTED_LANGUAGES[languageKey] || null;
  }

  chrome.tabs.sendMessage(
    tab.id,
    {
      type: "captureAndStartReadSession",
      selectedText: info.selectionText || "",
      targetLanguage
    },
    () => {
      const err = chrome.runtime.lastError;

      if (err) {
        debugWarn("Could not message content script:", err.message);
      }
    }
  );
});

// -----------------------------
// Keyboard command
// -----------------------------
// Keyboard command keeps original-language behavior.

chrome.commands.onCommand.addListener(async (command) => {
  if (command !== COMMAND_ID) return;

  const [tab] = await chrome.tabs.query({
    active: true,
    currentWindow: true
  });

  if (!tab || typeof tab.id !== "number" || tab.id < 0) {
    debugWarn("No valid tab id for Avatar Studio action:", tab);
    return;
  }

  chrome.tabs.sendMessage(
    tab.id,
    {
      type: "captureAndStartReadSession",
      targetLanguage: null
    },
    () => {
      const err = chrome.runtime.lastError;

      if (err) {
        debugWarn("Could not message content script:", err.message);
      }
    }
  );
});

// -----------------------------
// Content script messages
// -----------------------------

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (!message || message.type !== "startReadSession") {
    return false;
  }

  const tabId =
    sender.tab && typeof sender.tab.id === "number" && sender.tab.id >= 0
      ? sender.tab.id
      : null;

  if (tabId === null) {
    sendResponse({
      success: false,
      error: "No sender tab found."
    });

    return true;
  }

  const sessionId = message.sessionId;

  if (!sessionId) {
    sendResponse({
      success: false,
      error: "Missing sessionId."
    });

    return true;
  }

  sessionsById.set(sessionId, {
    tabId,
    sessionId
  });

  ensureHighlightSocket();

  startReadSessionInElectron({
    sessionId,
    tabId,
    pageUrl: message.pageUrl || sender.tab.url || "",
    selectedText: message.selectedText || "",
    targetLanguage: message.targetLanguage || null
  })
    .then((readPlan) => {
      chrome.tabs.sendMessage(tabId, readPlan, () => {
        const err = chrome.runtime.lastError;

        if (err) {
          debugWarn("Could not send readPlan to content script:", err.message);
        }
      });

      sendResponse({
        success: true,
        readPlan
      });
    })
    .catch((error) => {
      console.error("startReadSession failed:", error);

      chrome.tabs.sendMessage(
        tabId,
        {
          type: "readSessionError",
          sessionId,
          error: String(error && error.message ? error.message : error)
        },
        () => {}
      );

      sendResponse({
        success: false,
        error: String(error && error.message ? error.message : error)
      });
    });

  return true;
});

async function startReadSessionInElectron({
  sessionId,
  tabId,
  pageUrl,
  selectedText,
  targetLanguage
}) {
  const response = await fetch(`${ELECTRON_HTTP}/start_read_session`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      type: "startReadSession",
      sessionId,
      tabId,
      pageUrl,
      selectedText,
      targetLanguage: targetLanguage || null
    })
  });

  let data = null;

  try {
    data = await response.json();
  } catch {
    data = null;
  }

  if (!response.ok) {
    throw new Error(
      data && data.error
        ? data.error
        : `Electron bridge failed with status ${response.status}`
    );
  }

  if (!data || data.type !== "readPlan") {
    throw new Error("Electron bridge did not return a readPlan.");
  }

  return data;
}

// -----------------------------
// Electron playback sync WebSocket
// -----------------------------

function ensureHighlightSocket() {
  if (
    highlightSocket &&
    (
      highlightSocket.readyState === WebSocket.OPEN ||
      highlightSocket.readyState === WebSocket.CONNECTING
    )
  ) {
    return;
  }

  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }

  try {
    highlightSocket = new WebSocket(ELECTRON_WS);
  } catch {
    scheduleReconnect();
    return;
  }

  highlightSocket.onopen = () => {
    debugLog("Avatar highlight sync connected.");
  };

  highlightSocket.onmessage = (event) => {
    let message;

    try {
      message = JSON.parse(event.data);
    } catch {
      debugWarn("Invalid highlight sync message.");
      return;
    }

    routeHighlightMessage(message);
  };

  highlightSocket.onerror = () => {
    debugWarn("Avatar highlight sync WebSocket error.");
  };

  highlightSocket.onclose = () => {
    highlightSocket = null;
    scheduleReconnect();
  };
}

function scheduleReconnect() {
  if (reconnectTimer) return;

  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;

    if (sessionsById.size > 0) {
      ensureHighlightSocket();
    }
  }, 1000);
}

function routeHighlightMessage(message) {
  if (!message || !message.sessionId) {
    return;
  }

  const route = sessionsById.get(message.sessionId);

  if (!route) {
    debugWarn("No tab route for highlight message:", message);
    return;
  }

  if (typeof route.tabId !== "number" || route.tabId < 0) {
    debugWarn("Invalid tab route for highlight message:", route);
    return;
  }

  chrome.tabs.sendMessage(route.tabId, message, () => {
    const err = chrome.runtime.lastError;

    if (err) {
      debugWarn("Could not route highlight message:", err.message);
    }
  });

  if (
    message.type === "readSessionEnded" ||
    message.type === "readSessionCancelled"
  ) {
    sessionsById.delete(message.sessionId);
  }
}