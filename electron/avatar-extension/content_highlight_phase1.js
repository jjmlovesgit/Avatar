// content_highlight_phase1.js
// Phase 1 optimized overlay-based chunk/sentence highlighting.
// No DOM wrapping. No Whisper. No word timings.
// Uses mapped DOM Range + fixed underline overlays + left-to-right sweep.

const DEBUG = false;

let activeSession = null;

function debugLog(...args) {
  if (DEBUG) console.log("🟨 [AvatarHighlight]", ...args);
}

function debugWarn(...args) {
  if (DEBUG) console.warn("🟨 [AvatarHighlight]", ...args);
}

function createSessionId() {
  return `read_${Date.now()}_${Math.random().toString(36).slice(2)}`;
}

function normalizeForMapping(text) {
  return String(text || "")
    .replace(/\u00a0/g, " ")
    .replace(/[“”]/g, '"')
    .replace(/[‘’]/g, "'")
    .replace(/\s+/g, " ")
    .trim();
}

function normalizeToken(token) {
  return String(token || "")
    .toLowerCase()
    .replace(/[“”]/g, '"')
    .replace(/[‘’]/g, "'")
    .replace(/[^\p{L}\p{N}]+/gu, "");
}

function snapForwardToWordBoundary(text, offset) {
  if (!text || text.length === 0) return 0;

  let i = Math.max(0, Math.min(text.length - 1, offset));

  while (
    i < text.length - 1 &&
    /\S/.test(text[i]) &&
    /\S/.test(text[i - 1] || " ")
  ) {
    i += 1;
  }

  while (i < text.length - 1 && /\s/.test(text[i])) {
    i += 1;
  }

  return i;
}

function snapForwardToSentenceOrWordBoundary(text, offset) {
  if (!text || text.length === 0) return 0;

  const start = Math.max(0, Math.min(text.length, offset));
  const maxLookahead = 180;
  const lookaheadEnd = Math.min(text.length, start + maxLookahead);

  for (let i = start; i < lookaheadEnd; i++) {
    if (/[.!?]/.test(text[i])) {
      return Math.min(text.length, i + 1);
    }
  }

  for (let i = start; i < lookaheadEnd; i++) {
    if (/\s/.test(text[i])) {
      return i;
    }
  }

  return start;
}

// ---------------------------------------------------------------------
// CSS
// ---------------------------------------------------------------------

function injectHighlightCss() {
  if (document.getElementById("avatar-reading-highlight-css")) return;

  const style = document.createElement("style");
  style.id = "avatar-reading-highlight-css";

  style.textContent = `
    .avatar-highlight-overlay-layer {
      position: fixed;
      inset: 0;
      pointer-events: none;
      z-index: 2147483646;
    }

    .avatar-highlight-rect {
      position: fixed;
      pointer-events: none;
      border-radius: 999px;
      height: 4px !important;
      background: rgba(255, 210, 80, 0.28);
      box-shadow: 0 0 0 1px rgba(255, 210, 80, 0.18);
      overflow: hidden;
      transition:
        top 80ms ease,
        left 80ms ease,
        width 80ms ease,
        opacity 80ms ease;
      opacity: 1;
    }

    .avatar-highlight-rect::before {
      content: "";
      position: absolute;
      inset: 0;
      width: var(--read-progress, 0%);
      background: rgba(255, 145, 35, 0.95);
      border-radius: 999px;
      transition: width 80ms linear;
    }

    .avatar-highlight-rect.complete {
      background: rgba(120, 220, 120, 0.18);
      box-shadow: 0 0 0 1px rgba(100, 210, 120, 0.18);
      opacity: 0.35;
    }

    @media (prefers-reduced-motion: reduce) {
      .avatar-highlight-rect,
      .avatar-highlight-rect::before {
        transition: none;
      }
    }
  `;

  document.documentElement.appendChild(style);
}

function ensureOverlayLayer(session) {
  if (session.overlayLayer && document.documentElement.contains(session.overlayLayer)) {
    return session.overlayLayer;
  }

  const layer = document.createElement("div");
  layer.className = "avatar-highlight-overlay-layer";
  document.documentElement.appendChild(layer);
  session.overlayLayer = layer;

  return layer;
}

// ---------------------------------------------------------------------
// Selection capture
// ---------------------------------------------------------------------

function captureSelectedRange() {
  const selection = window.getSelection();

  if (!selection || selection.rangeCount === 0) {
    return null;
  }

  const selectedText = selection.toString();

  if (!selectedText || !selectedText.trim()) {
    return null;
  }

  return {
    selectedText,
    range: selection.getRangeAt(0).cloneRange()
  };
}

function clearNativeSelectionSoon() {
  try {
    setTimeout(() => {
      try {
        window.getSelection()?.removeAllRanges();
      } catch {}
    }, 100);

    setTimeout(() => {
      try {
        window.getSelection()?.removeAllRanges();
      } catch {}
    }, 250);
  } catch {}
}















function startReadSessionFromSelection(options = {}) {
  const captured = captureSelectedRange();

  if (!captured) {
    debugWarn("No selected DOM text/range found.");
    return;
  }

  cleanupActiveSession();

  injectHighlightCss();

  const sessionId = createSessionId();

  activeSession = {
    sessionId,
    selectedText: captured.selectedText,
    targetLanguage: options.targetLanguage || null,
    originalRange: captured.range,
    chunks: [],
    activeChunkId: null,
    activeProgress: 0,
    readPlanMapped: false,
    syncMessageCount: 0,
    overlayLayer: null,
    overlayRects: [],
    overlayChunkId: null,
    startedAt: Date.now(),
    cleanup: null
  };

  ensureOverlayLayer(activeSession);
  attachLayoutListeners(activeSession);

  chrome.runtime.sendMessage(
    {
      type: "startReadSession",
      sessionId,
      pageUrl: location.href,
      selectedText: captured.selectedText,
      targetLanguage: options.targetLanguage || null
    },
    () => {
      const err = chrome.runtime.lastError;
      if (err) debugWarn("chrome.runtime.sendMessage failed:", err.message);
    }
  );
}

// ---------------------------------------------------------------------
// Message listener
// ---------------------------------------------------------------------

chrome.runtime.onMessage.addListener((message) => {
  if (!message || !message.type) return;

  if (message.type === "captureAndStartReadSession") {
    startReadSessionFromSelection({
      targetLanguage: message.targetLanguage || null
    });
    return;
  }

  if (!activeSession) {
    debugWarn("Message received but no activeSession", message.type, message.sessionId || "");
    return;
  }

  if (message.sessionId && message.sessionId !== activeSession.sessionId) {
    debugWarn(
      "Ignoring different session",
      "got",
      message.sessionId,
      "expected",
      activeSession.sessionId,
      "type",
      message.type
    );
    return;
  }

  switch (message.type) {
    case "readPlan":
      handleReadPlan(activeSession, message);
      break;

    case "chunkReady":
      activeSession.syncMessageCount += 1;
      break;

    case "chunkStart":
      activeSession.syncMessageCount += 1;
      handleChunkStart(activeSession, message.chunkId);
      break;

    case "chunkTime":
      activeSession.syncMessageCount += 1;
      handleChunkTime(
        activeSession,
        message.chunkId,
        Number(message.currentTime || 0),
        Number(message.duration || 0)
      );
      break;

    case "chunkEnd":
      activeSession.syncMessageCount += 1;
      handleChunkEnd(activeSession, message.chunkId);
      break;

    case "readSessionEnded":
      break;

    case "readSessionCancelled":
    case "readSessionError":
      debugWarn(message.type, message.error || "");
      break;

    default:
      debugWarn("Unhandled message type", message.type);
      break;
  }
});

// ---------------------------------------------------------------------
// Read-plan mapping
// ---------------------------------------------------------------------


function scrollChunkToReadingPosition(chunk, behavior = "auto") {
  if (!chunk || !chunk.sourceRange) return false;
  return scrollRangeToReadingPosition(chunk.sourceRange, behavior);
}


function getElementForRangeStart(range) {
  if (!range) return null;

  let node = range.startContainer;

  if (!node) return null;

  if (node.nodeType === Node.TEXT_NODE) {
    return node.parentElement;
  }

  if (node.nodeType === Node.ELEMENT_NODE) {
    return node;
  }

  return node.parentElement || null;
}

function getScrollableAncestor(el) {
  let node = el;

  while (node && node !== document.body && node !== document.documentElement) {
    try {
      const style = window.getComputedStyle(node);
      const overflowY = style.overflowY;
      const canScroll =
        (overflowY === "auto" || overflowY === "scroll" || overflowY === "overlay") &&
        node.scrollHeight > node.clientHeight + 4;

      if (canScroll) {
        return node;
      }
    } catch {}

    node = node.parentElement;
  }

  return document.scrollingElement || document.documentElement;
}

function scrollElementToReadingPosition(el, behavior = "auto") {
  if (!el) return false;

  const scroller = getScrollableAncestor(el);

  try {
    const rect = el.getBoundingClientRect();

    if (!rect) return false;

    if (
      scroller === document.scrollingElement ||
      scroller === document.documentElement ||
      scroller === document.body
    ) {
      const targetY = rect.top + window.scrollY - window.innerHeight * 0.35;

      window.scrollTo({
        top: Math.max(0, targetY),
        behavior
      });

      return true;
    }

    const scrollerRect = scroller.getBoundingClientRect();
    const targetTop =
      scroller.scrollTop +
      rect.top -
      scrollerRect.top -
      scroller.clientHeight * 0.35;

    scroller.scrollTo({
      top: Math.max(0, targetTop),
      behavior
    });

    return true;
  } catch {
    try {
      el.scrollIntoView({
        block: "center",
        inline: "nearest",
        behavior
      });

      return true;
    } catch {
      return false;
    }
  }
}

function scrollRangeToReadingPosition(range, behavior = "auto") {
  const el = getElementForRangeStart(range);

  if (el && scrollElementToReadingPosition(el, behavior)) {
    return true;
  }

  try {
    const rects = Array.from(range.getClientRects()).filter(
      (r) => r.width > 0 && r.height > 0
    );

    const rect = rects[0];
    if (!rect) return false;

    const targetY = rect.top + window.scrollY - window.innerHeight * 0.35;

    window.scrollTo({
      top: Math.max(0, targetY),
      behavior
    });

    return true;
  } catch {
    return false;
  }
}

function scrollChunkToReadingPosition(chunk, behavior = "auto") {
  if (!chunk || !chunk.sourceRange) return false;
  return scrollRangeToReadingPosition(chunk.sourceRange, behavior);
}

function redrawChunkAfterScroll(session, chunkId, delays = [0, 80, 180, 320]) {
  for (const delay of delays) {
    setTimeout(() => {
      if (!activeSession || activeSession !== session) return;
      if (session.activeChunkId && session.activeChunkId !== chunkId) return;

      const chunk = session.chunks.find((c) => c.chunkId === chunkId);
      if (!chunk) return;

      renderOverlayForChunk(session, chunk, session.activeProgress || 0);
    }, delay);
  }
}

function handleReadPlan(session, readPlan) {
  if (session.readPlanMapped) {
    debugWarn("Duplicate readPlan ignored for session", session.sessionId);
    return;
  }

  if (!Array.isArray(readPlan.chunks) || readPlan.chunks.length === 0) {
    debugWarn("readPlan has no chunks");
    return;
  }

  try {
    session.chunks = mapChunksToDomRanges(session.originalRange, readPlan.chunks);
    session.readPlanMapped = true;

    clearNativeSelectionSoon();

    // Immediately return to the first mapped sentence after mapping.
    // This is before/around the avatar starting playback.
    if (session.chunks[0]) {
      scrollChunkToReadingPosition(session.chunks[0], "auto");

      setTimeout(() => {
        if (!activeSession || activeSession !== session) return;
        if (session.activeChunkId) return;

        scrollChunkToReadingPosition(session.chunks[0], "auto");
        renderOverlayForChunk(session, session.chunks[0], 0);
      }, 80);

      setTimeout(() => {
        if (!activeSession || activeSession !== session) return;
        if (session.activeChunkId) return;

        scrollChunkToReadingPosition(session.chunks[0], "auto");
        renderOverlayForChunk(session, session.chunks[0], 0);
      }, 220);
    }
  } catch (err) {
    console.error("Avatar highlight failed to map readPlan:", err);
  }
}

function buildRangeTextIndex(range) {
  let normalizedText = "";
  const positions = [];
  let previousWasSpace = true;

  function appendTextNode(textNode, start, end) {
    if (!textNode || !textNode.data) return;

    start = Math.max(0, Math.min(textNode.data.length, start));
    end = Math.max(start, Math.min(textNode.data.length, end));

    for (let offset = start; offset < end; offset++) {
      const rawChar = textNode.data[offset];
      const normalizedChar = normalizeChar(rawChar);

      if (!normalizedChar) continue;

      if (/\s/.test(normalizedChar)) {
        if (previousWasSpace) continue;

        normalizedText += " ";
        positions.push({ node: textNode, offset });
        previousWasSpace = true;
      } else {
        normalizedText += normalizedChar;
        positions.push({ node: textNode, offset });
        previousWasSpace = false;
      }
    }
  }

  if (range.commonAncestorContainer.nodeType === Node.TEXT_NODE) {
    const textNode = range.commonAncestorContainer;

    appendTextNode(
      textNode,
      range.startContainer === textNode ? range.startOffset : 0,
      range.endContainer === textNode ? range.endOffset : textNode.data.length
    );

    return { normalizedText, positions };
  }

  const walker = document.createTreeWalker(
    range.commonAncestorContainer,
    NodeFilter.SHOW_TEXT,
    {
      acceptNode(node) {
        if (!range.intersectsNode(node)) {
          return NodeFilter.FILTER_REJECT;
        }

        if (!node.textContent || !node.textContent.trim()) {
          return NodeFilter.FILTER_REJECT;
        }

        const parent = node.parentElement;
        if (parent && parent.closest("script, style, noscript")) {
          return NodeFilter.FILTER_REJECT;
        }

        return NodeFilter.FILTER_ACCEPT;
      }
    }
  );

  while (walker.nextNode()) {
    const textNode = walker.currentNode;

    let start = 0;
    let end = textNode.data.length;

    if (textNode === range.startContainer) {
      start = range.startOffset;
    }

    if (textNode === range.endContainer) {
      end = range.endOffset;
    }

    appendTextNode(textNode, start, end);
  }

  return { normalizedText, positions };
}

function normalizeChar(char) {
  if (char === "\u00a0") return " ";
  if (char === "“" || char === "”") return '"';
  if (char === "‘" || char === "’") return "'";
  if (/\s/.test(char)) return " ";
  return char;
}

function buildTokenIndex(text) {
  const tokens = [];
  const regex = /[\p{L}\p{N}]+(?:['’-][\p{L}\p{N}]+)?/gu;

  let match;

  while ((match = regex.exec(text))) {
    tokens.push({
      raw: match[0],
      norm: normalizeToken(match[0]),
      start: match.index,
      end: match.index + match[0].length
    });
  }

  return tokens.filter((t) => t.norm.length > 0);
}

function mapChunksToDomRanges(originalRange, chunks) {
  const index = buildRangeTextIndex(originalRange);
  const selectedNorm = index.normalizedText;

  let cursor = 0;

  return chunks.map((chunk, i) => {
    const sourceText = chunk.sourceText || chunk.source_text || chunk.text || "";
    const spokenText = chunk.spokenText || chunk.spoken_text || sourceText;
    const normalizedChunk = normalizeForMapping(sourceText);

    let start = -1;
    let end = -1;
    let mappingMode = "none";

    if (normalizedChunk) {
      const exactStart = selectedNorm.indexOf(normalizedChunk, cursor);

      if (exactStart !== -1) {
        start = exactStart;
        end = exactStart + normalizedChunk.length;
        mappingMode = "ordered-exact";
      }
    }

    if (start === -1 && normalizedChunk) {
      const relaxed = findRelaxedChunkMatch(selectedNorm, normalizedChunk, cursor);

      if (relaxed) {
        start = relaxed.start;
        end = relaxed.end;
        mappingMode = "ordered-relaxed";
      }
    }

    if (start === -1 && normalizedChunk) {
      const tokenMatch = findSequentialTokenChunkRange(selectedNorm, normalizedChunk, cursor);

      if (tokenMatch) {
        start = tokenMatch.start;
        end = tokenMatch.end;
        mappingMode = "ordered-token";
      }
    }

    if (start === -1 || end === -1 || end <= start) {
      const approx = approximateChunkRangeFromCursor(selectedNorm, chunks, i, cursor);

      start = approx.start;
      end = approx.end;
      mappingMode = "ordered-approx";
    }

    cursor = Math.max(cursor, end);

    let sourceRange;

    try {
      sourceRange = normalizedOffsetsToRange(index, start, end);
    } catch {
      sourceRange = originalRange.cloneRange();
      sourceRange.collapse(true);
      mappingMode = "collapsed-fallback";
    }

    return {
      ...chunk,
      chunkId: chunk.chunkId || chunk.chunk_id || `chunk_${i}`,
      index: typeof chunk.index === "number" ? chunk.index : i,
      sourceText,
      spokenText,
      sourceRange,
      mappingMode
    };
  });
}

function findRelaxedChunkMatch(selectedNorm, chunkNorm, cursor) {
  const selectedTail = selectedNorm.slice(cursor);

  const relaxedChunk = relaxForChunkMatching(chunkNorm);
  if (!relaxedChunk) return null;

  const relaxedSelected = relaxForChunkMatching(selectedTail);
  const relaxedIndex = relaxedSelected.indexOf(relaxedChunk);

  if (relaxedIndex === -1) return null;

  const originalStartInTail = relaxedOffsetToOriginalOffset(selectedTail, relaxedIndex);
  const originalEndInTail = relaxedOffsetToOriginalOffset(
    selectedTail,
    relaxedIndex + relaxedChunk.length
  );

  return {
    start: cursor + originalStartInTail,
    end: cursor + Math.max(originalStartInTail + 1, originalEndInTail)
  };
}

function relaxForChunkMatching(text) {
  return String(text || "")
    .toLowerCase()
    .replace(/[“”]/g, '"')
    .replace(/[‘’]/g, "'")
    .replace(/[:;,.!?]+/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function relaxedOffsetToOriginalOffset(original, relaxedOffset) {
  let relaxedCount = 0;

  for (let i = 0; i < original.length; i++) {
    const ch = original[i];

    if (/[:;,.!?]/.test(ch)) {
      continue;
    }

    const contributes = /\s/.test(ch) ? " " : ch.toLowerCase();

    if (contributes) {
      if (relaxedCount >= relaxedOffset) {
        return i;
      }

      relaxedCount += 1;
    }
  }

  return original.length;
}

function findSequentialTokenChunkRange(selectedNorm, chunkNorm, cursor) {
  const selectedTokens = buildTokenIndex(selectedNorm);
  const chunkTokens = buildTokenIndex(chunkNorm)
    .map((t) => t.norm)
    .filter(Boolean);

  if (!selectedTokens.length || !chunkTokens.length) {
    return null;
  }

  const cursorTokenIndex = selectedTokens.findIndex((t) => t.end >= cursor);
  const searchStart = Math.max(0, cursorTokenIndex === -1 ? 0 : cursorTokenIndex - 2);

  const anchors = chunkTokens
    .filter((t) => t.length >= 3)
    .slice(0, 5);

  if (!anchors.length) {
    return null;
  }

  for (let i = searchStart; i < selectedTokens.length; i++) {
    if (selectedTokens[i].norm !== anchors[0]) {
      continue;
    }

    let selectedPos = i;
    let anchorPos = 0;

    while (selectedPos < selectedTokens.length && anchorPos < anchors.length) {
      if (selectedTokens[selectedPos].norm === anchors[anchorPos]) {
        anchorPos += 1;
      }

      selectedPos += 1;
    }

    if (anchorPos === anchors.length) {
      const startToken = i;
      const endToken = Math.min(
        selectedTokens.length - 1,
        startToken + Math.max(1, chunkTokens.length) - 1
      );

      return {
        start: selectedTokens[startToken].start,
        end: selectedTokens[endToken].end
      };
    }
  }

  return null;
}

function approximateChunkRangeFromCursor(selectedNorm, chunks, index, cursor) {
  const total = selectedNorm.length;
  const remainingChunks = Math.max(1, chunks.length - index);
  const remainingChars = Math.max(1, total - cursor);

  const sourceText =
    chunks[index]?.sourceText ||
    chunks[index]?.source_text ||
    chunks[index]?.text ||
    "";

  const normalizedSource = normalizeForMapping(sourceText);

  let estimatedLength = normalizedSource.length;

  if (!estimatedLength) {
    estimatedLength = Math.ceil(remainingChars / remainingChunks);
  }

  let start = snapForwardToWordBoundary(selectedNorm, cursor);
  let end = Math.min(total, start + estimatedLength);

  end = snapForwardToSentenceOrWordBoundary(selectedNorm, end);

  if (end <= start) {
    end = Math.min(
      total,
      start + Math.max(1, Math.ceil(remainingChars / remainingChunks))
    );
  }

  return {
    start: Math.max(0, Math.min(total - 1, start)),
    end: Math.max(1, Math.min(total, end))
  };
}

function normalizedOffsetsToRange(index, start, end) {
  const range = document.createRange();

  const startPos = index.positions[Math.max(0, start)];
  const endPos = index.positions[
    Math.min(index.positions.length - 1, Math.max(start, end - 1))
  ];

  if (!startPos || !endPos) {
    throw new Error("Could not resolve normalized offsets to DOM range");
  }

  range.setStart(startPos.node, startPos.offset);
  range.setEnd(endPos.node, endPos.offset + 1);

  return range;
}

// ---------------------------------------------------------------------
// Overlay drawing
// ---------------------------------------------------------------------

function getRangeRects(range) {
  if (!range) return [];

  return Array.from(range.getClientRects())
    .filter((rect) => rect.width > 0 && rect.height > 0)
    .map((rect) => ({
      left: rect.left,
      top: rect.top,
      width: rect.width,
      height: rect.height,
      right: rect.right,
      bottom: rect.bottom
    }));
}

function renderOverlayForChunk(session, chunk, progress, options = {}) {
  clearOverlayRects(session);

  const layer = ensureOverlayLayer(session);
  const rects = getRangeRects(chunk.sourceRange);

  if (!rects.length) {
    debugWarn("No visible rects for chunk", chunk.chunkId);
    return;
  }

  for (const rect of rects) {
    const el = document.createElement("div");
    el.className = "avatar-highlight-rect";

    if (options.complete) {
      el.classList.add("complete");
    }

    const padX = 2;
    const underlineHeight = 4;

    el.style.left = `${rect.left - padX}px`;
    el.style.top = `${rect.bottom - underlineHeight + 1}px`;
    el.style.width = `${rect.width + padX * 2}px`;
    el.style.height = `${underlineHeight}px`;

    layer.appendChild(el);
    session.overlayRects.push({
      el,
      width: rect.width
    });
  }

  session.overlayChunkId = chunk.chunkId;
  updateOverlayProgress(session, progress, options);
}

function updateOverlayProgress(session, progress, options = {}) {
  if (!session.overlayRects.length) return;

  const clamped = Math.max(0, Math.min(1, progress));
  const totalWidth = session.overlayRects.reduce((sum, item) => sum + item.width, 0);
  const targetWidth = totalWidth * clamped;

  let consumedWidth = 0;

  for (const item of session.overlayRects) {
    const remaining = targetWidth - consumedWidth;
    let localProgress = Math.max(0, Math.min(1, remaining / item.width));

    if (options.complete) {
      localProgress = 1;
      item.el.classList.add("complete");
    }

    item.el.style.setProperty("--read-progress", `${localProgress * 100}%`);

    consumedWidth += item.width;
  }
}

function clearOverlayRects(session) {
  for (const item of session.overlayRects || []) {
    try {
      const el = item.el || item;
      el.remove();
    } catch {}
  }

  session.overlayRects = [];
  session.overlayChunkId = null;
}

function attachLayoutListeners(session) {
  let raf = 0;

  const refresh = () => {
    cancelAnimationFrame(raf);

    raf = requestAnimationFrame(() => {
      if (!activeSession || activeSession !== session) return;
      if (!session.activeChunkId) return;

      const chunk = session.chunks.find((c) => c.chunkId === session.activeChunkId);
      if (!chunk) return;

      renderOverlayForChunk(session, chunk, session.activeProgress || 0);
    });
  };

  // document capture catches scrolls on nested scroll containers better than window alone.
  document.addEventListener("scroll", refresh, true);
  window.addEventListener("scroll", refresh, { passive: true });
  window.addEventListener("resize", refresh);

  session.cleanup = () => {
    document.removeEventListener("scroll", refresh, true);
    window.removeEventListener("scroll", refresh);
    window.removeEventListener("resize", refresh);
    cancelAnimationFrame(raf);
  };
}

// ---------------------------------------------------------------------
// Sync handlers
// ---------------------------------------------------------------------


// ---------------------------------------------------------------------
// Reading-position scroll helpers
// ---------------------------------------------------------------------

function getChunkFirstRect(chunk) {
  const rects = getRangeRects(chunk.sourceRange);
  return rects[0] || null;
}

function isChunkComfortablyVisible(chunk) {
  const rect = getChunkFirstRect(chunk);
  if (!rect) return false;

  const topComfort = window.innerHeight * 0.25;
  const bottomComfort = window.innerHeight * 0.75;

  return rect.top >= topComfort && rect.bottom <= bottomComfort;
}

function scrollChunkIntoReadingPosition(chunk, behavior = "smooth") {
  const rect = getChunkFirstRect(chunk);
  if (!rect) return false;

  const targetY =
    rect.top + window.scrollY - window.innerHeight * 0.38;

  window.scrollTo({
    top: Math.max(0, targetY),
    behavior
  });

  return true;
}

function redrawActiveChunkAfterScroll(session, chunkId, delays = [80, 180, 350]) {
  for (const delay of delays) {
    setTimeout(() => {
      if (!activeSession || activeSession !== session) return;
      if (session.activeChunkId !== chunkId) return;

      const chunk = session.chunks.find((c) => c.chunkId === chunkId);
      if (!chunk) return;

      renderOverlayForChunk(session, chunk, session.activeProgress || 0);
    }, delay);
  }
}

function handleChunkStart(session, chunkId) {
  const chunk = session.chunks.find((c) => c.chunkId === chunkId);

  if (!chunk) {
    debugWarn("chunkStart received but no matching chunk:", chunkId);
    return;
  }

  session.activeChunkId = chunkId;
  session.activeProgress = 0;

  // Move the real text/range into the reading position first.
  scrollChunkToReadingPosition(chunk, "auto");

  // Draw immediately and redraw after layout catches up.
  renderOverlayForChunk(session, chunk, 0);
  redrawChunkAfterScroll(session, chunkId, [40, 120, 260]);
}

function handleChunkTime(session, chunkId, currentTime, duration) {
  const chunk = session.chunks.find((c) => c.chunkId === chunkId);

  if (!chunk) {
    debugWarn("chunkTime received but no matching chunk:", chunkId);
    return;
  }

  const progress =
    duration > 0 ? Math.max(0, Math.min(1, currentTime / duration)) : 0;

  session.activeChunkId = chunkId;
  session.activeProgress = progress;

  if (session.overlayChunkId !== chunkId || !session.overlayRects.length) {
    renderOverlayForChunk(session, chunk, progress);
  } else {
    updateOverlayProgress(session, progress);
  }
}

function handleChunkEnd(session, chunkId) {
  const chunk = session.chunks.find((c) => c.chunkId === chunkId);

  if (!chunk) {
    debugWarn("chunkEnd received but no matching chunk:", chunkId);
    return;
  }

  session.activeProgress = 1;

  if (session.overlayChunkId !== chunkId || !session.overlayRects.length) {
    renderOverlayForChunk(session, chunk, 1, { complete: true });
  } else {
    updateOverlayProgress(session, 1, { complete: true });
  }
}

// ---------------------------------------------------------------------
// Cleanup
// ---------------------------------------------------------------------

function cleanupActiveSession() {
  if (!activeSession) return;

  if (typeof activeSession.cleanup === "function") {
    activeSession.cleanup();
  }

  clearOverlayRects(activeSession);

  if (activeSession.overlayLayer) {
    activeSession.overlayLayer.remove();
  }

  activeSession = null;
}

if (DEBUG) {
  window.__avatarHighlightDebug = {
    startReadSessionFromSelection,
    cleanupActiveSession,
    getActiveSession: () => activeSession
  };

  console.log("Avatar highlight content script loaded.");
}