from fastapi import FastAPI, UploadFile, File, Form, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.concurrency import run_in_threadpool
from contextlib import asynccontextmanager
from imtalker_core import InferenceAgent
import os
import uuid
import asyncio
import time
import shutil
import re
import subprocess
import struct
import wave
import io
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import deque
from PIL import Image
import base64
import httpx
import torch
import threading

VENUS_AVATAR_ONLY = os.getenv("VENUS_AVATAR_ONLY", "").lower() in ("1", "true", "yes")

# WebRTC VAD is needed by the avatar server's own microphone WebSocket, not
# by the Venus audio-to-video endpoint.
try:
    import webrtcvad
except ImportError:
    webrtcvad = None

# Wake Word - Using livekit-wakeword for better accuracy
if VENUS_AVATAR_ONLY:
    LIVEKIT_WAKEWORD_AVAILABLE = False
else:
    try:
        from livekit.wakeword import WakeWordModel
        LIVEKIT_WAKEWORD_AVAILABLE = True
        print("✅ livekit-wakeword loaded")
    except ImportError:
        LIVEKIT_WAKEWORD_AVAILABLE = False
        print("⚠️ livekit-wakeword not installed. Run: pip install livekit-wakeword")

# Whisper for speech-to-text
if VENUS_AVATAR_ONLY:
    WHISPER_AVAILABLE = False
else:
    try:
        from faster_whisper import WhisperModel
        WHISPER_AVAILABLE = True
    except ImportError:
        WHISPER_AVAILABLE = False
        print("⚠️ faster-whisper not installed. Run: pip install faster-whisper")

# ============================================
# CONFIGURATION - AVATAR PROFILES & DUAL LLM
# ============================================
AVATAR_PROFILE = os.getenv("AVATAR_PROFILE", "").strip().lower()
_PROFILE_DEFAULTS = {
    "puck": {
        "avatar": "puck_avatar.png",
        "idle": "idle_puck.mp4",
        "tts_backend": "sapi",
        "sapi_voice": "Paradee am_puck",
    },
}
_ACTIVE_PROFILE = _PROFILE_DEFAULTS.get(AVATAR_PROFILE, {})

DEFAULT_AVATAR = os.getenv("VENUS_AVATAR_IMAGE", _ACTIVE_PROFILE.get("avatar", "Johnny-Depp.png"))
DEFAULT_IDLE_VIDEO = os.getenv("AVATAR_IDLE_VIDEO", _ACTIVE_PROFILE.get("idle", "idle4.mp4"))
OUTPUT_DIR = "generated_clips"
UPLOAD_DIR = "uploads"
CHUNK_DIR = "chunks"
AVATAR_SIZE = (256, 256)

# LM Studio (Default - Conversational)
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
LM_STUDIO_MODEL = "qwen2.5-7b-instruct"
#"nvidia/nemotron-3-nano-4b" #"qwen3.5-35b-a3b"#"glm-4.6v-flash" #"gemma-4-31b-it-claude-opus-distill-v2" #"gemma-3n-e4b-it-text" #gemma-4-e4b-it"

OPENCLAW_URL = "http://127.0.0.1:18789/v1/chat/completions"
OPENCLAW_TOKEN = os.getenv("OPENCLAW_TOKEN", "")
if not OPENCLAW_TOKEN:
    print("⚠️ Warning: OPENCLAW_TOKEN environment variable not set. OpenClaw will not work.")
OPENCLAW_MODEL = "openclaw"

# TTS Backend ("pockettts" HTTP server or "sapi" local Windows SAPI / Kokoro Paradee neural voices)
TTS_BACKEND = os.getenv("TTS_BACKEND", _ACTIVE_PROFILE.get("tts_backend", "pockettts")).strip().lower()
SAPI_VOICE = os.getenv("SAPI_VOICE", _ACTIVE_PROFILE.get("sapi_voice", "Paradee am_puck")).strip()
SAPI_RATE = max(-10, min(10, int(os.getenv("SAPI_RATE", "0"))))

# PocketTTS
POCKET_TTS_URL = "http://localhost:8000"
POCKET_TTS_VOICE = "jonnydepp"

# Whisper configuration
WHISPER_MODEL_SIZE = "small"
WHISPER_COMPUTE_TYPE = "float16"

# Wake Word Configuration
WAKE_WORD_ENABLED = True
WAKE_WORD_MODEL = "hey_jarvis_v0.1.onnx"
WAKE_WORD_THRESHOLD = 0.5
WAKE_WORD_DEBOUNCE = 2.0
WAKE_WORD_TIMEOUT = 10.0  # Auto-deactivate after 10 seconds of silence
WHISPER_WAKE_ALIAS_ENABLED = os.getenv("WHISPER_WAKE_ALIAS", "1").lower() not in ("0", "false", "no")
WHISPER_WAKE_PHRASE_RE = re.compile(
    r"\bhey[\s,.\-!]*(?:jo+h?n+y|jo+n+ie|jo+n+i|puck|jarvis)\b[\s,.\-!:;]*",
    re.IGNORECASE,
)

# VAD Configuration
VAD_AGGRESSIVENESS = 2  # 0-3, higher = more aggressive
VAD_FRAME_SIZE = 480  # 30ms at 16kHz
VAD_MIN_SPEECH_FRAMES = 10  # ~300ms minimum speech
VAD_SILENCE_TIMEOUT_FRAMES = 30  # ~900ms silence timeout
VAD_NOISE_GATE_THRESHOLD = 150

# GPU Sequential Processing
GPU_SEQUENTIAL_MODE = True  # Set to False to disable GPU queue
MSE_STREAMING_ENABLED = os.getenv("IMTALKER_MSE_STREAMING", "1").lower() not in ("0", "false", "no")
MSE_FMP4_MIME = 'video/mp4; codecs="avc1.42E01F, mp4a.40.2"'
LLM_STREAMING_ENABLED = os.getenv("IMTALKER_LLM_STREAMING", "1").lower() not in ("0", "false", "no")
LLM_MAX_TOKENS = max(150, int(os.getenv("IMTALKER_LLM_MAX_TOKENS", "1000")))

# System Prompts for each LLM

SYSTEM_PROMPT_LM_STUDIO = """You are a real-time conversational avatar speaking to a user.

Your response will be streamed live and lip-synced across consecutive video clips.

Critical speaking rules:
1. For normal questions, answer in 3 complete, natural spoken sentences.
2. If the user asks for a longer explanation, story, or extended monologue, speak at length for as many sentences as requested.
3. Make each sentence about 10 to 18 words long so every sentence streams and speaks cleanly in roughly 3 seconds.
4. Start directly with a useful first sentence with no filler openers like "Certainly" or "Great question."
5. Do not use bullet points, markdown, numbered lists, headings, emojis, or special formatting.
6. Do not explain your reasoning.
7. Do not mention that you are an AI or an avatar.
8. Speak directly to the user in plain, conversational English.
9. Always end every sentence with a period, question mark, or exclamation point.

Conversation memory:
- Use recent conversation history to resolve words like he, she, it, that, they, his, her, and their.
- If the user asks a follow-up, answer based on the most recent relevant topic.
- If the reference is still unclear, ask one short clarification question.

Output goal:
Sound like a calm, engaging person speaking naturally in real time.

Accuracy:
- When answering factual questions about people, dates, or history, be precise.
- If unsure, say you are unsure instead of guessing.
- Do not invent names, titles, or roles."""

SYSTEM_PROMPT_OPENCLAW = """You are a helpful voice assistant. Answer concisely in 1-2 short sentences.

Question: {prompt}

Answer:"""

# ============================================
# GLOBAL STATE
# ============================================
app = FastAPI()
agent = None
avatar_image = None
whisper_model = None
warmup_complete = False
wakeword_model = None
ui_websockets = set()

# GPU Management
gpu_lock = asyncio.Lock()
gpu_task_queue = asyncio.Queue()
gpu_worker_active = False

# Active sessions
chunk_status: Dict[str, Dict] = {}
active_sessions: Dict[str, 'RealtimeSession'] = {}
avatar_audio_jobs: Dict[str, Dict] = {}

# ============================================
# CONVERSATION HISTORY (GLOBAL, SHARED)
# ============================================
conversation_history: List[Dict[str, str]] = []
MAX_HISTORY_MESSAGES = 12  # keep small for speed

def get_history():
    return conversation_history[-MAX_HISTORY_MESSAGES:]

def add_to_history(user_text: str, assistant_text: str):
    user_text = user_text.strip()

    # Skip junk / fragments
    if len(user_text) < 8:
        return

    conversation_history.append({"role": "user", "content": user_text})
    conversation_history.append({"role": "assistant", "content": assistant_text})

    # Trim
    conversation_history[:] = conversation_history[-MAX_HISTORY_MESSAGES:]
    
# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
for _runtime_dir in (OUTPUT_DIR, UPLOAD_DIR, CHUNK_DIR):
    os.makedirs(_runtime_dir, exist_ok=True)
app.mount("/clips", StaticFiles(directory=OUTPUT_DIR), name="clips")
app.mount("/chunks", StaticFiles(directory=CHUNK_DIR), name="chunks")

# ============================================
# GPU WORKER FOR SEQUENTIAL PROCESSING
# ============================================
async def gpu_worker():
    """Worker that processes GPU tasks sequentially"""
    global gpu_worker_active
    
    print("🎮 GPU Worker started")
    
    while True:
        try:
            # Get next task from queue
            task = await gpu_task_queue.get()
            if task is None:  # Shutdown signal
                break
            
            async with gpu_lock:
                print(f"🔒 GPU lock acquired for: {task['name']}")
                
                # Execute the task
                result = await task['func'](*task.get('args', []), **task.get('kwargs', {}))
                
                # Set result in future if provided
                if 'future' in task:
                    task['future'].set_result(result)
                
                # Optional VRAM flush (disabled by default to avoid CUDA allocator churn on 5090)
                if os.getenv("CLEAR_CUDA_CACHE_EACH_TASK", "0") == "1" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                print(f"🔓 GPU lock released for: {task['name']}")
                
        except Exception as e:
            print(f"❌ GPU Worker error: {e}")
            if 'future' in task:
                task['future'].set_exception(e)
    
    print("🎮 GPU Worker stopped")

async def queue_gpu_task(name: str, func, *args, **kwargs):
    """Queue a GPU task and wait for completion"""
    if not GPU_SEQUENTIAL_MODE:
        # Run immediately without queue
        return await func(*args, **kwargs)
    
    future = asyncio.Future()
    
    await gpu_task_queue.put({
        'name': name,
        'func': func,
        'args': args,
        'kwargs': kwargs,
        'future': future
    })
    
    return await future

async def start_gpu_worker():
    global gpu_worker_active
    if not gpu_worker_active:
        gpu_worker_active = True
        asyncio.create_task(gpu_worker())

async def stop_gpu_worker():
    global gpu_worker_active
    if gpu_worker_active:
        await gpu_task_queue.put(None)  # Shutdown signal
        gpu_worker_active = False


# ============================================
# PERFORMANCE LOGGER
# ============================================
class PerformanceLogger:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.start_time = time.time()
        self.markers = {}
        self.chunk_times = []
        self.first_audio_time = None
        self.ttfa = None
        
    def mark(self, name: str):
        elapsed = time.time() - self.start_time
        self.markers[name] = elapsed
        print(f"⏱️  [{self.session_id}] {name}: {elapsed:.3f}s")
        return elapsed
    
    def mark_first_audio(self, chunk_index: int):
        if self.first_audio_time is None:
            self.first_audio_time = time.time() - self.start_time
            self.ttfa = self.first_audio_time
            print(f"\n{'='*60}")
            print(f"🔊 [{self.session_id}] 🎵 TIME TO FIRST AUDIO: {self.ttfa:.3f}s (Chunk {chunk_index + 1})")
            print(f"{'='*60}\n")
    
    def chunk_mark(self, chunk_index: int, name: str):
        elapsed = time.time() - self.start_time
        self.chunk_times.append({
            "chunk": chunk_index,
            "step": name,
            "time": elapsed
        })
        print(f"⏱️  [{self.session_id}] Chunk {chunk_index} - {name}: {elapsed:.3f}s")
    
    def summary(self):
        print(f"\n{'='*60}")
        print(f"📊 PERFORMANCE SUMMARY [{self.session_id}]")
        print(f"{'='*60}")
        
        if self.ttfa:
            print(f"\n🎯 CRITICAL METRIC - TIME TO FIRST AUDIO: {self.ttfa:.3f}s")
            print(f"{'─'*50}")
        
        prev_time = 0
        for name, time_val in sorted(self.markers.items()):
            duration = time_val - prev_time
            print(f"   {name:35s}: {duration:6.3f}s (total: {time_val:6.3f}s)")
            prev_time = time_val
        
        print(f"{'='*60}\n")


# ============================================
# THINKING MODEL RESPONSE PARSER (FINAL ANSWER ONLY)
# ============================================
def extract_final_answer(raw_text: str) -> str:
    """
    Extract the final answer from a thinking model response.
    Always returns only the final answer (what the avatar should speak).
    """
    if not raw_text:
        return ""
    
    final_answer = raw_text.strip()
    
    # Look for end-of-thinking markers
    markers = ['<channel|>', '<|channel|>', '</channel|>']
    
    for marker in markers:
        if marker in raw_text:
            parts = raw_text.split(marker)
            final_answer = parts[-1].strip()
            break
    
    # Clean up any remaining tags
    final_answer = re.sub(r'<[^>]+>', '', final_answer).strip()
    final_answer = re.sub(r'<\|[^>]+\|>', '', final_answer).strip()
    
    # If final answer is empty after cleaning, use original
    if not final_answer:
        final_answer = raw_text.strip()
    
    return final_answer


# ============================================
# WAKE WORD DETECTOR CLASS (livekit-wakeword)
# ============================================
class JarvisWakeWordDetector:
    """Wake word detector for 'Hey Jarvis' using livekit-wakeword"""
    
    def __init__(self, model_path=WAKE_WORD_MODEL, threshold=WAKE_WORD_THRESHOLD):
        self.threshold = threshold
        self.model_path = model_path
        
        if LIVEKIT_WAKEWORD_AVAILABLE:
            try:
                self.model = WakeWordModel(models=[model_path])
                print(f"✅ Wake word model loaded with livekit: {model_path}")
            except Exception as e:
                print(f"❌ Failed to load wake word model: {e}")
                self.model = None
        else:
            self.model = None
        
        self.buffer_duration = 2.0
        self.buffer_size = int(16000 * self.buffer_duration)
        self.audio_buffer = deque(maxlen=self.buffer_size)
        self.last_detection_time = 0
        self.debounce_seconds = WAKE_WORD_DEBOUNCE
        self.detection_count = 0
        self.frame_counter = 0
        
    def process_audio_chunk(self, audio_chunk):
        if self.model is None:
            return {"detected": False, "confidence": 0.0}
        
        if isinstance(audio_chunk, bytes):
            audio_np = np.frombuffer(audio_chunk, dtype=np.int16)
        else:
            audio_np = audio_chunk
            
        self.audio_buffer.extend(audio_np)
        
        self.frame_counter += 1
        if self.frame_counter % 5 != 0:
            return {"detected": False, "confidence": 0.0}
        
        if len(self.audio_buffer) < self.buffer_size:
            return {"detected": False, "confidence": 0.0}
        
        audio_window = np.array(list(self.audio_buffer))[-self.buffer_size:]
        audio_float = audio_window.astype(np.float32) / 32768.0
        scores = self.model.predict(audio_float)
        score = scores.get("hey_jarvis_v0.1", 0.0)
        
        current_time = datetime.now().timestamp()
        if score > self.threshold:
            if current_time - self.last_detection_time > self.debounce_seconds:
                self.last_detection_time = current_time
                self.detection_count += 1
                return {
                    "detected": True,
                    "word": "hey_jarvis",
                    "confidence": float(score),
                    "count": self.detection_count
                }
        
        return {"detected": False, "confidence": float(score)}


# ============================================
# VAD SESSION WITH WAKE WORD INTEGRATION
# ============================================
class RealtimeSession:
    def __init__(self, session_id: str, voice: str, websocket: WebSocket):
        if webrtcvad is None:
            raise RuntimeError("The microphone WebSocket requires webrtcvad")
        self.session_id = session_id
        self.voice = voice
        self.websocket = websocket
        self.audio_buffer = bytearray()
        self.processing_queue = asyncio.Queue()
        self.is_active = True
        self.current_utterance_id = 0
        self.use_openclaw = False      # LLM toggle
        self.skip_avatar = False       # Skip avatar video, play TTS only
        
        # VAD settings
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.min_speech_frames = VAD_MIN_SPEECH_FRAMES
        self.silence_timeout_frames = VAD_SILENCE_TIMEOUT_FRAMES
        self.frame_size = VAD_FRAME_SIZE
        self.noise_gate_threshold = VAD_NOISE_GATE_THRESHOLD
        
        # Speech state
        self.is_speaking = False
        self.speech_frames = []
        self.silence_frames = 0
        self.last_audio_level = 0
        
        # Wake word state
        self.wakeword_detector = wakeword_model if WAKE_WORD_ENABLED else None
        self.is_wakeword_activated = not WAKE_WORD_ENABLED
        self.last_activation_time = 0
        self.wakeword_timeout = WAKE_WORD_TIMEOUT
        self.wakeword_buffer = deque(maxlen=32000)

        # Sleep-mode VAD state for Whisper wake-word alias ("Hey Jonny" / "Hey Johnny")
        self.sleep_is_speaking = False
        self.sleep_speech_frames = []
        self.sleep_silence_frames = 0
        self.sleep_silence_timeout_frames = 15  # ~450ms silence ends wake-phrase burst
        self.sleep_min_speech_frames = 6        # ~180ms minimum speech
        self.sleep_max_speech_frames = 140      # ~4.2s max for one-breath "Hey Jonny, <question>"
        self.whisper_wake_busy = False
        self.last_whisper_wake_time = 0.0
        
        # Worker task
        self.worker_task = None
        
    def get_audio_level(self, frame):
        try:
            samples = struct.unpack(f'{len(frame)//2}h', frame)
            rms = sum(s*s for s in samples) / len(samples)
            return rms ** 0.5
        except:
            return 0
    
    def is_voice_with_noise_gate(self, frame):
        if len(frame) != self.frame_size * 2:
            return False
            
        level = self.get_audio_level(frame)
        self.last_audio_level = level
        if level < self.noise_gate_threshold:
            return False
        
        try:
            return self.vad.is_speech(frame, 16000)
        except Exception as e:
            print(f"VAD error: {e}")
            return False
    
    def check_wakeword(self, frame):
        if self.wakeword_detector is None:
            return False
        
        audio_np = np.frombuffer(frame, dtype=np.int16)
        self.wakeword_buffer.extend(audio_np)
        
        if len(self.wakeword_buffer) >= 32000:
            buffer_array = np.array(list(self.wakeword_buffer))
            result = self.wakeword_detector.process_audio_chunk(buffer_array)
            if result.get("detected", False):
                self.wakeword_buffer.clear()
                return True
        
        return False
    
    async def activate_wakeword(self, label: str = "Hey Jarvis"):
        self.is_wakeword_activated = True
        self.last_activation_time = time.time()
        self.speech_frames = []
        self.silence_frames = 0
        self.sleep_is_speaking = False
        self.sleep_speech_frames = []
        self.sleep_silence_frames = 0
        self.current_utterance_id = 0
        
        await self.websocket.send_json({
            "type": "wakeword_activated",
            "status": "active",
            "message": f"🎯 {label} detected! I'm listening...",
            "timestamp": time.time(),
            "timeout": self.wakeword_timeout
        })
        print(f"🎯 [{self.session_id}] Wake word activated ({label})!")
    
    async def deactivate_wakeword(self):
        self.is_wakeword_activated = False
        self.speech_frames = []
        self.current_utterance_id = 0
        
        await self.websocket.send_json({
            "type": "wakeword_deactivated",
            "status": "idle",
            "message": "😴 Listening mode ended. Say 'Hey Jarvis' to wake me again.",
            "timestamp": time.time()
        })
        print(f"⏰ [{self.session_id}] Wake word deactivated (timeout)")
    
    def check_timeout(self):
        if self.is_wakeword_activated and not self.is_speaking:
            if time.time() - self.last_activation_time > self.wakeword_timeout:
                return True
        return False
    
    async def stop(self):
        self.is_active = False
        while not self.processing_queue.empty():
            try:
                self.processing_queue.get_nowait()
            except:
                pass


# ============================================
# UTILITY FUNCTIONS
# ============================================
def resize_avatar(image: Image.Image, size: Tuple[int, int] = (256, 256)) -> Image.Image:
    return image.resize(size, Image.Resampling.LANCZOS)

def split_text_into_chunks(text: str, target_duration_seconds: float = 3.0) -> List[str]:
    """
    Split assistant text into natural playback chunks for pseudo-streamed MP4 avatar clips.

    Goals:
    - Avoid tiny first clips like "Hi there!" that create stilted playback.
    - Keep chunks near the IMTalker-friendly 2-3 second range.
    - Merge very short sentences forward into the next sentence when possible.
    - Preserve sentence punctuation so TTS still sounds natural.
    """
    if not text:
        return []

    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return []

    # Split on sentence endings, keeping punctuation attached.
    # This handles: "Hi there! CRISPR is..." as two sentence units.
    sentences = re.findall(r'[^.!?]+[.!?]+|[^.!?]+$', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return [text]

    # Approximate speech timing. Existing logs show roughly 18-22 chars/sec.
    chars_per_second = 20.0

    # Tune these first. They are intentionally conservative.
    min_chunk_seconds = 1.20       # prevents 0.5s clips like "Hi there!"
    ideal_chunk_seconds = target_duration_seconds
    max_chunk_seconds = max(3.8, target_duration_seconds + 0.8)

    min_chunk_chars = int(min_chunk_seconds * chars_per_second)
    ideal_chunk_chars = int(ideal_chunk_seconds * chars_per_second)
    max_chunk_chars = int(max_chunk_seconds * chars_per_second)

    def estimated_seconds(chunk: str) -> float:
        return max(0.1, len(chunk) / chars_per_second)

    # Pass 1: merge tiny sentence fragments forward.
    # Example: "Hi there!" + "CRISPR is..." becomes one first clip.
    merged_sentences = []
    carry = ""

    for sentence in sentences:
        if sentence and sentence[-1] not in '.!?':
            sentence += '.'

        if carry:
            candidate = (carry + " " + sentence).strip()
        else:
            candidate = sentence

        # If candidate is still too short, keep carrying it forward.
        if len(candidate) < min_chunk_chars:
            carry = candidate
            continue

        merged_sentences.append(candidate)
        carry = ""

    if carry:
        if merged_sentences:
            # Attach trailing tiny fragment to previous chunk instead of making a tiny final clip.
            merged_sentences[-1] = (merged_sentences[-1] + " " + carry).strip()
        else:
            merged_sentences.append(carry)

    # Pass 2: pack sentence units into chunks near the target duration.
    chunks = []
    current_chunk = ""

    for sentence in merged_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if not current_chunk:
            current_chunk = sentence
            continue

        candidate = (current_chunk + " " + sentence).strip()

        # If adding the sentence stays near the ideal/max range, keep building.
        if len(candidate) <= ideal_chunk_chars or estimated_seconds(current_chunk) < min_chunk_seconds:
            current_chunk = candidate
        elif len(candidate) <= max_chunk_chars and len(current_chunk) < min_chunk_chars:
            current_chunk = candidate
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    # Pass 3: final cleanup. Merge any remaining tiny chunks with a neighbor.
    cleaned = []
    for chunk in chunks:
        if not chunk:
            continue

        if len(chunk) < min_chunk_chars:
            if cleaned:
                cleaned[-1] = (cleaned[-1] + " " + chunk).strip()
            else:
                cleaned.append(chunk)
        else:
            cleaned.append(chunk)

    # If the first chunk is still too short and there is a second chunk, merge it forward.
    if len(cleaned) >= 2 and len(cleaned[0]) < min_chunk_chars:
        cleaned[1] = (cleaned[0] + " " + cleaned[1]).strip()
        cleaned = cleaned[1:]

    return cleaned if cleaned else [text]


async def transcribe_audio(audio_path: str, perf: PerformanceLogger) -> str:
    global whisper_model
    
    if whisper_model is None:
        return ""
    
    try:
        segments, info = whisper_model.transcribe(
            audio_path,
            beam_size=1, #was 5
            language="en",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500, threshold=0.5)
        )
        
        transcript = " ".join([segment.text for segment in segments])
        return transcript.strip()
        
    except Exception as e:
        print(f"Whisper error: {e}")
        return ""

async def query_llm(prompt: str, use_openclaw: bool = False, perf: PerformanceLogger = None) -> Dict[str, str]:
    """Query LLM and return TTS text (final answer only)"""

    normalized = prompt.strip().lower()

    if any(phrase in normalized for phrase in [
        "new topic",
        "start new topic",
        "clear history",
        "reset conversation",
        "forget that",
        "change topic",
        "switch topic",
        "start over",
        "something else",
        "let's switch"
    ]):
        conversation_history.clear()
        prompt = "We are starting a new topic. Please acknowledge briefly and ask what the user wants to discuss next."

    if use_openclaw:
        print(f"   🔷 Using OpenClaw...")
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    OPENCLAW_URL,
                    json={
                        "model": OPENCLAW_MODEL,
                        "messages": [
                            {"role": "user", "content": SYSTEM_PROMPT_OPENCLAW.format(prompt=prompt)}
                        ],
                        "temperature": 0.5,
                        "top_p": 0.9,
                        "repeat_penalty": 1.1,
                        "max_tokens": 250,
                        "stream": False
                    },
                    headers={
                        "Authorization": f"Bearer {OPENCLAW_TOKEN}",
                        "Content-Type": "application/json",
                        "Cache-Control": "no-cache"
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    raw_result = data['choices'][0]['message']['content'].strip()
                    final_answer = extract_final_answer(raw_result).strip()

                    if not final_answer or final_answer == "...":
                        final_answer = "I’m not sure how to answer that yet."

                    add_to_history(prompt, final_answer)

                    print(f"   🆕 OpenClaw Response: {final_answer[:50]}...")
                    return {"tts_text": final_answer, "display_text": final_answer}

                error_msg = "I'm having trouble connecting to OpenClaw right now."
                return {"tts_text": error_msg, "display_text": error_msg}

        except Exception as e:
            print(f"OpenClaw error: {e}")
            error_msg = "I'm having trouble connecting to OpenClaw right now."
            return {"tts_text": error_msg, "display_text": error_msg}

    else:
        print(f"   🤖 Using LM Studio...")
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                response = await client.post(
                    LM_STUDIO_URL,
                    json={
                        "model": LM_STUDIO_MODEL,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT_LM_STUDIO},
                            *get_history(),
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.5,
                        "top_p": 0.9,
                        "repeat_penalty": 1.1,
                        "max_tokens": 250,
                        "stream": False
                    },
                    headers={"Cache-Control": "no-cache"}
                )

                if response.status_code == 200:
                    data = response.json()
                    raw_result = data['choices'][0]['message']['content'].strip()
                    final_answer = extract_final_answer(raw_result).strip()

                    if not final_answer or final_answer == "...":
                        final_answer = "I’m not sure how to answer that yet."

                    add_to_history(prompt, final_answer)

                    print(f"   🆕 LM Studio Response: {final_answer[:50]}...")
                    return {"tts_text": final_answer, "display_text": final_answer}

                error_msg = "I'm having trouble connecting to LM Studio right now."
                return {"tts_text": error_msg, "display_text": error_msg}

        except Exception as e:
            print(f"LM Studio error: {e}")
            error_msg = "I'm having trouble connecting to LM Studio right now."
            return {"tts_text": error_msg, "display_text": error_msg}

_llm_http_client: Optional[httpx.AsyncClient] = None
_tts_http_client: Optional[httpx.AsyncClient] = None


def _get_llm_client() -> httpx.AsyncClient:
    global _llm_http_client
    if _llm_http_client is None or _llm_http_client.is_closed:
        _llm_http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=10.0),
            limits=httpx.Limits(max_keepalive_connections=4, max_connections=8),
        )
    return _llm_http_client


def _get_tts_client() -> httpx.AsyncClient:
    global _tts_http_client
    if _tts_http_client is None or _tts_http_client.is_closed:
        _tts_http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0),
            limits=httpx.Limits(max_keepalive_connections=4, max_connections=8),
        )
    return _tts_http_client


def _clean_spoken_chunk(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"<\|[^>]+\|>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _pop_stream_sentence(buf: str, min_chars: int = 28, max_chars: int = 165, first_chunk: bool = False) -> Tuple[Optional[str], str]:
    """Extract one ready spoken chunk from a streaming token buffer."""
    # Prefer a complete sentence boundary (including optional closing quote/paren) followed by whitespace.
    for m in re.finditer(r'[.!?]+["”’\'\)\]]*(?:\s+)', buf):
        end_idx = m.end()
        candidate = buf[:end_idx].strip()
        if len(candidate) >= min_chars:
            return candidate, buf[end_idx:].lstrip()

    # For Chunk 0, allow an early natural clause split (comma/semicolon/colon) so TTFA drops
    # without ever breaking awkwardly mid-phrase on a bare space.
    if first_chunk:
        for m in re.finditer(r'(?:[,;:]|—|\s-\s)\s+', buf):
            end_idx = m.end()
            candidate = buf[:end_idx].strip()
            if len(candidate) >= 22:
                return candidate, buf[end_idx:].lstrip()
        first_max = 110
        if len(buf) >= first_max:
            window = buf[:first_max]
            split_at = window.rfind(" ")
            if split_at >= 36:
                return buf[:split_at].strip(), buf[split_at + 1:].lstrip()
        return None, buf

    # Safety valve for long run-on sentences without punctuation
    if len(buf) >= max_chars:
        window = buf[:max_chars]
        split_at = max(window.rfind(", "), window.rfind("; "), window.rfind(": "))
        if split_at < min_chars:
            split_at = window.rfind(" ")
        if split_at >= min_chars:
            return buf[:split_at + 1].strip(), buf[split_at + 1:].lstrip()

    return None, buf


async def stream_llm_lm_studio(prompt: str, cancel_event: Optional[asyncio.Event] = None):
    """Stream spoken sentence chunks directly from LM Studio SSE (`stream: True`)."""
    normalized = prompt.strip().lower()
    effective_prompt = prompt
    if any(phrase in normalized for phrase in [
        "new topic", "start new topic", "clear history", "reset conversation",
        "forget that", "change topic", "switch topic", "start over",
        "something else", "let's switch"
    ]):
        conversation_history.clear()
        effective_prompt = "We are starting a new topic. Please acknowledge briefly and ask what the user wants to discuss next."

    print("   🤖 Streaming from LM Studio (SSE)...")
    buf = ""
    full_chunks = []
    emitted_count = 0
    in_think_tag = False

    try:
        client = _get_llm_client()
        async with client.stream(
            "POST",
            LM_STUDIO_URL,
            json={
                "model": LM_STUDIO_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT_LM_STUDIO},
                    *get_history(),
                    {"role": "user", "content": effective_prompt},
                ],
                "temperature": 0.5,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "max_tokens": LLM_MAX_TOKENS,
                "stream": True,
            },
            headers={"Cache-Control": "no-cache"},
        ) as response:
            if response.status_code != 200:
                yield "I'm having trouble connecting to LM Studio right now."
                return

            async for raw_line in response.aiter_lines():
                if cancel_event is not None and cancel_event.is_set():
                    print("   🛑 LM Studio SSE stream canceled (interrupted)")
                    return
                if not raw_line or not raw_line.startswith("data:"):
                    continue
                data_str = raw_line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    payload = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choices = payload.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                token = delta.get("content") or ""
                if not token:
                    continue

                # Skip inline <think>...</think> blocks if a reasoning model emits them
                if "<think>" in token:
                    in_think_tag = True
                    token = token.split("<think>", 1)[0]
                if in_think_tag:
                    if "</think>" in token:
                        in_think_tag = False
                        token = token.split("</think>", 1)[1]
                    else:
                        continue

                # Strip channel markers from thinking models
                for marker in ("<channel|>", "<|channel|>", "</channel|>"):
                    if marker in token:
                        token = token.split(marker)[-1]
                        buf = ""

                buf += token
                while True:
                    is_first = (emitted_count == 0)
                    min_chars = 16 if is_first else 36
                    ready, buf = _pop_stream_sentence(buf, min_chars=min_chars, first_chunk=is_first)
                    if not ready:
                        break
                    clean = _clean_spoken_chunk(ready)
                    if clean:
                        emitted_count += 1
                        full_chunks.append(clean)
                        yield clean

        tail = _clean_spoken_chunk(buf)
        if tail and (cancel_event is None or not cancel_event.is_set()):
            full_chunks.append(tail)
            yield tail

        full_text = " ".join(full_chunks).strip()
        if full_text and (cancel_event is None or not cancel_event.is_set()):
            add_to_history(prompt, full_text)
        elif not full_chunks and (cancel_event is None or not cancel_event.is_set()):
            yield "I'm not sure how to answer that yet."

    except Exception as exc:
        print(f"LM Studio stream error: {exc}")
        if not full_chunks:
            yield "I'm having trouble connecting to LM Studio right now."


def prune_old_stream_chunks(max_age_seconds: int = 600):
    """Delete old stream_*.mp4 files so extended monologues don't accumulate disk usage."""
    try:
        cutoff = time.time() - max_age_seconds
        for name in os.listdir(CHUNK_DIR):
            if name.startswith("stream_") and name.endswith(".mp4"):
                p = os.path.join(CHUNK_DIR, name)
                if os.path.getmtime(p) < cutoff:
                    os.remove(p)
    except Exception:
        pass


class SapiTtsWorker:
    """Dedicated STA COM worker for ultra-fast local Windows SAPI / Paradee Kokoro neural TTS."""

    def __init__(self):
        import queue as _q
        self._queue = _q.Queue()
        self._thread = None
        self._lock = threading.Lock()
        self._voices: List[str] = []

    def _ensure_started(self):
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            ready = threading.Event()
            self._init_error = None
            self._thread = threading.Thread(
                target=self._run_sta, args=(ready,), name="sapi-tts-sta", daemon=True
            )
            self._thread.start()
            ready.wait(timeout=10.0)
            if self._init_error is not None:
                raise RuntimeError(f"SAPI init failed: {self._init_error}")

    def _run_sta(self, ready: threading.Event):
        try:
            import pythoncom
            import win32com.client
            pythoncom.CoInitialize()
            sp = win32com.client.Dispatch("SAPI.SpVoice")
            voice_tokens = {}
            for tok in sp.GetVoices():
                desc = tok.GetDescription()
                voice_tokens[desc] = tok
            self._voices = list(voice_tokens.keys())
            ready.set()
        except Exception as exc:
            self._init_error = exc
            ready.set()
            return

        current_voice_desc = None
        try:
            while True:
                job = self._queue.get()
                if job is None:
                    break
                text, output_path, voice_query, rate, loop, fut = job
                try:
                    # Resolve voice token (exact match first, then substring match)
                    chosen_desc = None
                    q_low = (voice_query or SAPI_VOICE).strip().lower()
                    for desc in voice_tokens:
                        if desc.lower() == q_low:
                            chosen_desc = desc
                            break
                    if chosen_desc is None:
                        for desc in voice_tokens:
                            if q_low in desc.lower():
                                chosen_desc = desc
                                break
                    if chosen_desc is not None and chosen_desc != current_voice_desc:
                        sp.Voice = voice_tokens[chosen_desc]
                        current_voice_desc = chosen_desc
                    sp.Rate = rate

                    fs = win32com.client.Dispatch("SAPI.SpFileStream")
                    fs.Open(os.path.abspath(output_path), 3)  # SSFMCreateForWrite = 3
                    try:
                        sp.AudioOutputStream = fs
                        sp.Speak(text)
                    finally:
                        fs.Close()

                    ok = os.path.isfile(output_path) and os.path.getsize(output_path) > 44
                    loop.call_soon_threadsafe(fut.set_result, ok)
                except Exception as exc:
                    loop.call_soon_threadsafe(fut.set_exception, exc)
        finally:
            try:
                pythoncom.CoUninitialize()
            except Exception:
                pass

    async def synthesize(self, text: str, output_path: str, voice_name: Optional[str] = None, rate: int = SAPI_RATE) -> bool:
        self._ensure_started()
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self._queue.put((text, output_path, voice_name or SAPI_VOICE, rate, loop, fut))
        return await fut


_sapi_tts_worker = SapiTtsWorker()


async def generate_speech_sapi(text: str, output_path: str, voice_name: Optional[str] = None) -> bool:
    try:
        return await _sapi_tts_worker.synthesize(text, output_path, voice_name=voice_name, rate=SAPI_RATE)
    except Exception as exc:
        print(f"⚠️ SAPI TTS error: {exc}")
        return False


async def generate_speech_pockettts(text: str, output_path: str, voice: str, perf: PerformanceLogger, chunk_idx: int = None) -> bool:
    # Route to local Windows SAPI when TTS_BACKEND=sapi OR when voice starts with "sapi:"
    use_sapi = (TTS_BACKEND == "sapi") or (isinstance(voice, str) and voice.lower().startswith("sapi:"))
    if use_sapi:
        sapi_voice_name = voice[5:].strip() if (isinstance(voice, str) and voice.lower().startswith("sapi:")) else SAPI_VOICE
        if await generate_speech_sapi(text, output_path, voice_name=sapi_voice_name):
            return True
        if TTS_BACKEND == "sapi":
            print("⚠️ SAPI TTS failed; falling back to PocketTTS HTTP...")

    try:
        client = _get_tts_client()
        response = await client.post(
            f"{POCKET_TTS_URL}/v1/audio/speech",
            json={"input": text, "voice": voice, "response_format": "wav"}
        )

        if response.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(response.content)
            return True
        return False
    except Exception as e:
        print(f"PocketTTS error: {e}")
        return False


async def render_wav_to_chunk_video(
    wav_path: str,
    chunk_index: int,
    session_id: str,
    perf: PerformanceLogger,
    fast: bool = False,
    fragmented: bool = False,
    motion_state: Optional[dict] = None,
) -> Optional[str]:
    """Render an already-synthesized WAV file into a video chunk on the GPU."""
    chunk_filename = f"stream_{session_id}_{chunk_index}.mp4"
    chunk_path = os.path.join(CHUNK_DIR, chunk_filename)

    if os.path.exists(chunk_path):
        if os.path.exists(wav_path):
            os.remove(wav_path)
        return chunk_path

    if fast:
        await run_in_threadpool(
            agent.generate_fast,
            img_pil=avatar_image,
            aud_path=wav_path,
            output_path=chunk_path,
            fragmented=fragmented,
            motion_state=motion_state,
        )
    else:
        std_nfe = max(2, int(os.getenv("IMTALKER_STD_NFE", "6")))
        await run_in_threadpool(
            agent.generate,
            img_pil=avatar_image,
            aud_path=wav_path,
            output_path=chunk_path,
            nfe=std_nfe,
            fragmented=fragmented,
            motion_state=motion_state,
        )

    if os.path.exists(wav_path):
        os.remove(wav_path)

    if os.getenv("CLEAR_CUDA_CACHE_EACH_TASK", "0") == "1" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if chunk_index == 0:
        perf.mark_first_audio(chunk_index)

    return chunk_path

async def generate_chunk_video(
    text: str,
    chunk_index: int,
    session_id: str,
    voice: str,
    perf: PerformanceLogger,
    fast: bool = False,
    fragmented: bool = False,
    motion_state: Optional[dict] = None,
) -> Optional[str]:
    """Generate video chunk with optional fMP4 (MSE) muxing and continuous motion state."""
    chunk_filename = f"stream_{session_id}_{chunk_index}.mp4"
    chunk_path = os.path.join(CHUNK_DIR, chunk_filename)
    
    if os.path.exists(chunk_path):
        return chunk_path
    
    temp_wav = os.path.join(CHUNK_DIR, f"temp_{session_id}_{chunk_index}.wav")
    
    success = await generate_speech_pockettts(text, temp_wav, voice, perf, chunk_index)
    
    if not success:
        return None
    
    if fast:
        await run_in_threadpool(
            agent.generate_fast,
            img_pil=avatar_image,
            aud_path=temp_wav,
            output_path=chunk_path,
            fragmented=fragmented,
            motion_state=motion_state,
        )
    else:
        std_nfe = max(2, int(os.getenv("IMTALKER_STD_NFE", "6")))
        await run_in_threadpool(
            agent.generate,
            img_pil=avatar_image,
            aud_path=temp_wav,
            output_path=chunk_path,
            nfe=std_nfe,
            fragmented=fragmented,
            motion_state=motion_state,
        )
    
    if os.path.exists(temp_wav):
        os.remove(temp_wav)
    
    if os.getenv("CLEAR_CUDA_CACHE_EACH_TASK", "0") == "1" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    if chunk_index == 0:
        perf.mark_first_audio(chunk_index)
    
    return chunk_path



def make_silent_wav(path: str, seconds: float = 10.0, sample_rate: int = 16000):
    """Create a mono 16 kHz silent WAV file for idle/baseline animation."""
    samples = np.zeros(int(seconds * sample_rate), dtype=np.int16)

    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())


def normalize_audio_to_wav(input_path: str, output_path: str, seconds: float = 10.0):
    """Convert .wav/.mp3 input into the mono 16 kHz WAV format expected by IMTalker."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-t",
            str(seconds),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-af",
            "apad",
            output_path
        ],
        check=True,
        capture_output=True
    )


async def generate_idle_clip_task(
    audio_path: str,
    output_path: str,
    driver_video_path: Optional[str] = None,
    nfe: int = 3
):
    """Generate the idle MP4 using the current avatar, an audio file, and optional guide video."""
    return await run_in_threadpool(
        agent.generate_with_driver,
        img_pil=avatar_image,
        aud_path=audio_path,
        driver_video_path=driver_video_path,
        output_path=output_path,
        nfe=nfe
    )

async def warmup_pockettts_voice(voice: str):
    print(f"\n🔥 Warming up PocketTTS voice: {voice}...")
    try:
        test_text = "Hello."
        test_wav = os.path.join(CHUNK_DIR, f"warmup_{voice}.wav")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{POCKET_TTS_URL}/v1/audio/speech",
                json={"input": test_text, "voice": voice, "response_format": "wav"}
            )
            
            if response.status_code == 200:
                with open(test_wav, "wb") as f:
                    f.write(response.content)
                print(f"✅ Voice '{voice}' warmed up successfully")
                if os.path.exists(test_wav):
                    os.remove(test_wav)
                return True
    except Exception as e:
        print(f"⚠️ Voice warmup error: {e}")
        return False

async def warmup_wakeword():
    global wakeword_model
    if WAKE_WORD_ENABLED and LIVEKIT_WAKEWORD_AVAILABLE:
        print("\n🔥 Warming up Wake Word detector (livekit-wakeword)...")
        try:
            wakeword_model = JarvisWakeWordDetector()
            dummy_audio = np.zeros(32000, dtype=np.int16)
            wakeword_model.process_audio_chunk(dummy_audio)
            print("✅ Wake Word detector ready! (livekit-wakeword)")
        except Exception as e:
            print(f"⚠️ Wake Word warmup failed: {e}")


# ============================================
# QUEUE WORKER FOR PROCESSING UTTERANCES
# ============================================
def match_whisper_wake_phrase(text: str) -> Tuple[Optional[str], str]:
    """Match 'Hey Jonny' / 'Hey Johnny' (or 'Hey Jarvis') and return (label, trailing_question)."""
    if not text:
        return None, ""
    match = WHISPER_WAKE_PHRASE_RE.search(text)
    if not match:
        return None, ""
    matched_raw = match.group(0).lower()
    if "jarvis" in matched_raw:
        label = "Hey Jarvis"
    elif "puck" in matched_raw:
        label = "Hey Puck"
    else:
        label = "Hey Jonny"
    remainder = text[match.end():].strip(" ,.!?:;-")
    return label, remainder


def _transcribe_pcm16_sync(pcm_bytes: bytes) -> str:
    if whisper_model is None or not pcm_bytes:
        return ""
    audio_f32 = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    segments, _ = whisper_model.transcribe(
        audio_f32,
        beam_size=1,
        language="en",
        vad_filter=False,
        condition_on_previous_text=False,
    )
    return " ".join(segment.text for segment in segments).strip()


async def check_whisper_wakeword(session: RealtimeSession, pcm_bytes: bytes):
    """Background Whisper check for 'Hey Jonny' while sleeping; spoofs the Hey Jarvis wake event."""
    if not session.is_active or session.is_wakeword_activated or whisper_model is None:
        return
    if session.whisper_wake_busy:
        return
    if time.time() - session.last_whisper_wake_time < WAKE_WORD_DEBOUNCE:
        return

    session.whisper_wake_busy = True
    try:
        transcript = await run_in_threadpool(_transcribe_pcm16_sync, pcm_bytes)
        if not transcript or not session.is_active or session.is_wakeword_activated:
            return

        label, remainder = match_whisper_wake_phrase(transcript)
        if not label:
            return

        session.last_whisper_wake_time = time.time()
        print(f"🎯 [{session.session_id}] Whisper wake alias matched '{label}' from: {transcript!r}")
        await session.activate_wakeword(label=label)
        await session.websocket.send_json({
            "type": "wakeword_detected",
            "status": "activated",
            "word": "hey_jarvis",
            "alias": label.lower().replace(" ", "_"),
            "message": f"🎯 {label} detected! I'm listening...",
            "timestamp": time.time(),
        })

        # If the user asked a question in the same breath ("Hey Jonny, why is the sky blue?"),
        # process the trailing question immediately without requiring a second utterance.
        if len(remainder) >= 3:
            print(f"🎤 [{session.session_id}] One-breath question after '{label}': {remainder}")
            await session.processing_queue.put({"text": remainder})
            await session.websocket.send_json({
                "type": "processing_started",
                "status": "processing",
                "message": f"🤔 Thinking... (via {'OpenClaw' if session.use_openclaw else 'LM Studio'})",
                "timestamp": time.time(),
            })
    except Exception as exc:
        print(f"⚠️ [{session.session_id}] Whisper wake alias error: {exc}")
    finally:
        session.whisper_wake_busy = False


async def process_utterance(session: RealtimeSession, payload, utterance_id: int):
    start_time = time.time()
    unique_id = f"{session.session_id}_{utterance_id}_{int(start_time * 1000)}"
    perf = PerformanceLogger(session.session_id)
    
    try:
        if not session.is_active:
            return
        
        print(f"\n🔄 Processing utterance {utterance_id} [{unique_id}] - LLM: {'OpenClaw' if session.use_openclaw else 'LM Studio'} | Skip Avatar: {session.skip_avatar}")

        if isinstance(payload, dict) and "text" in payload:
            user_input = str(payload["text"]).strip()
        else:
            await session.websocket.send_json({
                "type": "status_update",
                "status": "transcribing",
                "message": "📝 Transcribing...",
                "utterance": utterance_id,
                "timestamp": time.time()
            })

            temp_audio = os.path.join(CHUNK_DIR, f"audio_{unique_id}.wav")
            with wave.open(temp_audio, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(payload)

            user_input = await transcribe_audio(temp_audio, perf)

            if os.path.exists(temp_audio):
                os.remove(temp_audio)
        
        # VALIDATION: Check if speech is too short
        if not user_input or len(user_input.strip()) < 2:
            print(f"⚠️ [{utterance_id}] Speech too short - NOT interrupting playback")
            await session.websocket.send_json({
                "type": "error",
                "status": "error",
                "message": "❌ Speech too short",
                "timestamp": time.time()
            })
            session.is_speaking = False
            session.speech_frames = []
            session.silence_frames = 0
            return
        
        # VALID SPEECH - NOW clear video and process
        print(f"✅ [{utterance_id}] Valid speech detected - interrupting playback")
        
        await session.websocket.send_json({
            "type": "clear_video",
            "utterance": utterance_id,
            "timestamp": time.time()
        })
        
        print(f"📝 [{utterance_id}] User said: {user_input}")
        
        # --- IMMEDIATE UI UPDATE WITH USER QUESTION ---
        await session.websocket.send_json({
            "type": "transcript",
            "utterance": utterance_id,
            "text": user_input,
            "response": "",           # empty placeholder
            "num_chunks": 0,
            "partial": True           # indicates this is just the user part
        })
        
        await session.websocket.send_json({
            "type": "status_update",
            "status": "generating",
            "message": f"💬 Generating response... (via {'OpenClaw' if session.use_openclaw else 'LM Studio'})",
            "utterance": utterance_id,
            "text": user_input,
            "timestamp": time.time()
        })
        
        utterance_session_id = f"{session.session_id}_utt{utterance_id}"

        def should_cancel_utterance() -> bool:
            return (not session.is_active) or (not session.processing_queue.empty())

        # --- 3-STAGE STREAMING PIPELINE: LM Studio SSE -> Parallel PocketTTS -> IMTalker MSE ---
        if (not session.use_openclaw) and (not session.skip_avatar) and LLM_STREAMING_ENABLED:
            sentence_queue: asyncio.Queue = asyncio.Queue()
            tts_queue: asyncio.Queue = asyncio.Queue(maxsize=4)
            cancel_event = asyncio.Event()
            streamed_sentences: List[str] = []

            async def _llm_producer():
                s_idx = 0
                try:
                    async for sentence in stream_llm_lm_studio(user_input, cancel_event=cancel_event):
                        if cancel_event.is_set() or should_cancel_utterance():
                            cancel_event.set()
                            break
                        streamed_sentences.append(sentence)
                        await sentence_queue.put((s_idx, sentence, time.time() - start_time))
                        s_idx += 1
                finally:
                    await sentence_queue.put(None)

            async def _tts_producer():
                """Synthesize PocketTTS WAVs OUTSIDE gpu_lock so Chunk k+1 TTS overlaps with Chunk k render."""
                try:
                    while True:
                        item = await sentence_queue.get()
                        if item is None:
                            break
                        s_idx, sentence, llm_ready_s = item
                        if cancel_event.is_set() or should_cancel_utterance():
                            cancel_event.set()
                            break
                        temp_wav = os.path.join(CHUNK_DIR, f"temp_{utterance_session_id}_{s_idx}.wav")
                        t_tts = time.perf_counter()
                        ok = await generate_speech_pockettts(sentence, temp_wav, session.voice, perf, s_idx)
                        tts_ms = (time.perf_counter() - t_tts) * 1000.0
                        if not ok or cancel_event.is_set() or should_cancel_utterance():
                            if os.path.exists(temp_wav):
                                os.remove(temp_wav)
                            if cancel_event.is_set() or should_cancel_utterance():
                                cancel_event.set()
                                break
                            continue
                        await tts_queue.put((s_idx, sentence, temp_wav, llm_ready_s, tts_ms))
                finally:
                    await tts_queue.put(None)

            llm_task = asyncio.create_task(_llm_producer())
            tts_task = asyncio.create_task(_tts_producer())
            utterance_motion_state = {}
            use_fmp4 = MSE_STREAMING_ENABLED
            chunk_idx = 0
            interrupted = False

            try:
                while True:
                    ready_item = await tts_queue.get()
                    if ready_item is None:
                        break
                    s_idx, sentence, temp_wav, llm_ready_s, tts_ms = ready_item
                    if should_cancel_utterance():
                        interrupted = True
                        cancel_event.set()
                        if os.path.exists(temp_wav):
                            os.remove(temp_wav)
                        print(f"🛑 [{utterance_id}] Interrupted before chunk {chunk_idx} — switching to new utterance")
                        break

                    if chunk_idx == 0:
                        print(f"🤖 [{utterance_id}] Chunk 0 ({len(sentence)}c): {sentence[:60]}...")

                    t_ren = time.perf_counter()
                    chunk_path = await queue_gpu_task(
                        f"Video_chunk_{chunk_idx}_utt{utterance_id}",
                        render_wav_to_chunk_video,
                        temp_wav,
                        chunk_idx,
                        utterance_session_id,
                        perf,
                        fast=(chunk_idx == 0),
                        fragmented=use_fmp4,
                        motion_state=utterance_motion_state,
                    )
                    render_ms = (time.perf_counter() - t_ren) * 1000.0

                    if should_cancel_utterance():
                        interrupted = True
                        cancel_event.set()
                        print(f"🛑 [{utterance_id}] Interrupted after chunk {chunk_idx} — aborting remaining chunks")
                        break

                    if chunk_path and os.path.exists(chunk_path):
                        filename = os.path.basename(chunk_path)
                        elapsed_s = time.time() - start_time
                        print(f"⚡ [{utterance_id}] Chunk {chunk_idx} ready @ {elapsed_s:.2f}s (llm@{llm_ready_s:.2f}s, tts={tts_ms:.0f}ms, render={render_ms:.0f}ms)")
                        await session.websocket.send_json({
                            "type": "chunk_ready",
                            "utterance": utterance_id,
                            "chunk": chunk_idx,
                            "total": None,
                            "url": f"/chunks/{filename}",
                            "format": "fmp4" if use_fmp4 else "mp4",
                            "mime": MSE_FMP4_MIME,
                            "duration": round(elapsed_s, 2)
                        })
                        if chunk_idx == 0:
                            perf.mark_first_audio(chunk_idx)
                        chunk_idx += 1
            finally:
                cancel_event.set()
                # Drain any prefetched WAVs left in tts_queue so nothing leaks on disk
                while not tts_queue.empty():
                    try:
                        leftover = tts_queue.get_nowait()
                        if leftover and leftover[2] and os.path.exists(leftover[2]):
                            os.remove(leftover[2])
                    except Exception:
                        break
                try:
                    await llm_task
                except Exception:
                    pass
                try:
                    await tts_task
                except Exception:
                    pass

            full_response_text = " ".join(streamed_sentences).strip()
            total_time = time.time() - start_time

            if not interrupted:
                await session.websocket.send_json({
                    "type": "transcript",
                    "utterance": utterance_id,
                    "text": user_input,
                    "response": full_response_text,
                    "num_chunks": chunk_idx,
                    "partial": False
                })
                await session.websocket.send_json({
                    "type": "status_update",
                    "status": "active" if session.is_wakeword_activated else "idle",
                    "message": "🟢 Listening..." if session.is_wakeword_activated else "💤 Say 'Hey Jarvis'",
                    "total_time": round(total_time, 2),
                    "timestamp": time.time()
                })
                await session.websocket.send_json({
                    "type": "complete",
                    "utterance": utterance_id,
                    "total_time": round(total_time, 2)
                })
                print(f"✅ [{utterance_id}] Stream complete in {total_time:.2f}s with {chunk_idx} chunks ({len(full_response_text)} chars)")
            else:
                print(f"🛑 [{utterance_id}] Stopped after {chunk_idx} chunks for interruption")
                if chunk_idx > 0:
                    try:
                        await session.websocket.send_json({
                            "type": "complete",
                            "utterance": utterance_id,
                            "interrupted": True,
                            "total_time": round(total_time, 2)
                        })
                    except Exception:
                        pass

            prune_old_stream_chunks()
            session.is_speaking = False
            session.speech_frames = []
            session.silence_frames = 0
            session.last_activation_time = time.time()
            return

        # --- NON-STREAMING FALLBACK (OpenClaw or Skip Avatar or IMTALKER_LLM_STREAMING=0) ---
        llm_response = await queue_gpu_task(
            f"LLM_query_utt{utterance_id}",
            query_llm,
            user_input,
            use_openclaw=session.use_openclaw,
            perf=perf
        )
        
        response_text = llm_response["tts_text"]
        display_text = llm_response["display_text"]
        
        print(f"🤖 [{utterance_id}] TTS Response: {response_text[:50]}...")
        
        # --- SKIP AVATAR MODE: Generate TTS only and send audio URL ---
        if session.skip_avatar:
            tts_filename = f"tts_{utterance_session_id}.wav"
            tts_path = os.path.join(CHUNK_DIR, tts_filename)
            success = await generate_speech_pockettts(response_text, tts_path, session.voice, perf)
            if success:
                await session.websocket.send_json({
                    "type": "audio_response",
                    "utterance": utterance_id,
                    "url": f"/chunks/{tts_filename}",
                    "text": display_text
                })
                # Also send complete transcript for UI
                await session.websocket.send_json({
                    "type": "transcript",
                    "utterance": utterance_id,
                    "text": user_input,
                    "response": display_text,
                    "partial": False
                })
            else:
                await session.websocket.send_json({"type": "error", "message": "TTS failed"})
            
            session.is_speaking = False
            session.speech_frames = []
            session.silence_frames = 0
            session.last_activation_time = time.time()
            return
        
        # --- AVATAR MODE: Generate video chunks as usual ---
        chunks = split_text_into_chunks(response_text, target_duration_seconds=3.0)
        
        # --- SEND COMPLETE TRANSCRIPT ---
        await session.websocket.send_json({
            "type": "transcript",
            "utterance": utterance_id,
            "text": user_input,
            "response": display_text,
            "num_chunks": len(chunks),
            "partial": False
        })
        
        utterance_motion_state = {}
        use_fmp4 = MSE_STREAMING_ENABLED
        interrupted = False

        for chunk_idx, sentence in enumerate(chunks):
            if should_cancel_utterance():
                interrupted = True
                break

            fast_mode = (chunk_idx == 0)

            chunk_path = await queue_gpu_task(
                f"Video_chunk_{chunk_idx}_utt{utterance_id}",
                generate_chunk_video,
                sentence,
                chunk_idx,
                utterance_session_id,
                session.voice,
                perf,
                fast=fast_mode,
                fragmented=use_fmp4,
                motion_state=utterance_motion_state,
            )

            if should_cancel_utterance():
                interrupted = True
                break

            if chunk_path and os.path.exists(chunk_path):
                filename = os.path.basename(chunk_path)

                await session.websocket.send_json({
                    "type": "chunk_ready",
                    "utterance": utterance_id,
                    "chunk": chunk_idx,
                    "total": len(chunks),
                    "url": f"/chunks/{filename}",
                    "format": "fmp4" if use_fmp4 else "mp4",
                    "mime": MSE_FMP4_MIME,
                    "duration": round(time.time() - start_time, 2)
                })

                if chunk_idx == 0:
                    perf.mark_first_audio(chunk_idx)

        total_time = time.time() - start_time
        
        if not interrupted:
            await session.websocket.send_json({
                "type": "status_update",
                "status": "active" if session.is_wakeword_activated else "idle",
                "message": "🟢 Listening..." if session.is_wakeword_activated else "💤 Say 'Hey Jarvis'",
                "total_time": round(total_time, 2),
                "timestamp": time.time()
            })
            
            await session.websocket.send_json({
                "type": "complete",
                "utterance": utterance_id,
                "total_time": round(total_time, 2)
            })
            
            print(f"✅ [{utterance_id}] Complete in {total_time:.2f}s with {len(chunks)} chunks")
        else:
            print(f"🛑 [{utterance_id}] Stopped early for interruption")
        prune_old_stream_chunks()
        
        session.is_speaking = False
        session.speech_frames = []
        session.silence_frames = 0
        session.last_activation_time = time.time()
        
    except Exception as e:
        print(f"❌ Error processing utterance {utterance_id}: {e}")
        session.is_speaking = False
        session.speech_frames = []
        session.silence_frames = 0
        try:
            await session.websocket.send_json({
                "type": "error",
                "status": "error",
                "message": f"❌ Error: {str(e)}",
                "timestamp": time.time()
            })
        except:
            pass


async def queue_worker(session: RealtimeSession):
    """Worker that processes utterances from the queue sequentially"""
    utterance_counter = 0
    try:
        while session.is_active:
            try:
                audio_data = await asyncio.wait_for(session.processing_queue.get(), timeout=1.0)
                utterance_counter += 1
                session.current_utterance_id = utterance_counter
                
                await process_utterance(session, audio_data, utterance_counter)
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Queue worker error: {e}")
                break
    finally:
        session.is_active = False
        print(f"🔚 Queue worker ended for session {session.session_id}")


# ============================================
# WEBSOCKET ENDPOINT WITH WAKE WORD & VAD
# ============================================
@app.websocket("/ws/realtime/{voice}")
async def websocket_realtime(websocket: WebSocket, voice: str):
    await websocket.accept()
    if VENUS_AVATAR_ONLY:
        await websocket.send_json({"type": "error", "message": "Microphone WebSocket is disabled in VENUS_AVATAR_ONLY mode"})
        await websocket.close()
        return
    
    session_id = str(uuid.uuid4())[:8]
    wakeword_status = "enabled" if WAKE_WORD_ENABLED else "disabled"
    print(f"🔊 Connected: {session_id} (Wake Word: {wakeword_status})")
    
    session = RealtimeSession(session_id, voice, websocket)
    active_sessions[session_id] = session
    session.worker_task = asyncio.create_task(queue_worker(session))
    
    await websocket.send_json({
        "type": "ready",
        "status": "idle",
        "message": "🎤 Say 'Hey Jarvis' to wake me up" if WAKE_WORD_ENABLED else "🟢 Always listening...",
        "wakeword_enabled": WAKE_WORD_ENABLED,
        "wakeword": "hey_jarvis" if WAKE_WORD_ENABLED else None,
        "timestamp": time.time()
    })
    
    async def send_status_updates():
        while session.is_active:
            await asyncio.sleep(0.5)
            
            if not session.is_active:
                break
                
            if WAKE_WORD_ENABLED:
                if session.is_wakeword_activated:
                    if session.is_speaking:
                        status = "speaking"
                        message = "🎙️ Listening to you..."
                        time_left = int(session.wakeword_timeout - (time.time() - session.last_activation_time))
                        extra = {"time_left": max(0, time_left)}
                    else:
                        time_left = int(session.wakeword_timeout - (time.time() - session.last_activation_time))
                        if time_left <= 3:
                            status = "warning"
                            message = f"⚠️ Listening mode ends in {time_left}s"
                            extra = {"time_left": time_left, "warning": True}
                        else:
                            status = "active"
                            message = f"🟢 Active - listening... ({time_left}s)"
                            extra = {"time_left": time_left}
                else:
                    status = "idle"
                    message = "💤 Sleeping - Say 'Hey Jarvis'"
                    extra = {"sleeping": True}
            else:
                if session.is_speaking:
                    status = "speaking"
                    message = "🎙️ Listening..."
                    extra = {}
                else:
                    status = "listening"
                    message = "🟢 Always listening..."
                    extra = {}
            
            try:
                await websocket.send_json({
                    "type": "status_update",
                    "status": status,
                    "message": message,
                    "audio_level": session.last_audio_level,
                    "is_speaking": session.is_speaking,
                    "is_activated": session.is_wakeword_activated if WAKE_WORD_ENABLED else True,
                    "timestamp": time.time(),
                    **extra
                })
            except:
                break
    
    status_task = asyncio.create_task(send_status_updates())
    
    try:
        while True:
            # Receive either text (JSON) or bytes (audio)
            try:
                message = await websocket.receive()
            except WebSocketDisconnect:
                print(f"🔇 Disconnected: {session_id}")
                break
            except RuntimeError as e:
                if "disconnect" in str(e).lower():
                    print(f"🔇 Disconnected (RuntimeError): {session_id}")
                    break
                raise
            
            # Handle JSON control messages
            if "text" in message:
                try:
                    data = json.loads(message["text"])
                    if data.get("type") == "set_llm":
                        session.use_openclaw = data.get("use_openclaw", False)
                        print(f"🔧 [{session_id}] LLM switched to: {'OpenClaw' if session.use_openclaw else 'LM Studio'}")
                        await websocket.send_json({
                            "type": "llm_updated",
                            "use_openclaw": session.use_openclaw,
                            "message": f"LLM: {'OpenClaw' if session.use_openclaw else 'LM Studio'}"
                        })
                    elif data.get("type") == "set_skip_avatar":
                        session.skip_avatar = data.get("skip_avatar", False)
                        print(f"🔧 [{session_id}] Skip Avatar: {session.skip_avatar}")
                        await websocket.send_json({
                            "type": "skip_avatar_updated",
                            "skip_avatar": session.skip_avatar
                        })
                    continue
                except:
                    pass
            
            # Handle binary audio data
            if "bytes" in message:
                data = message["bytes"]
            else:
                continue
            
            session.audio_buffer.extend(data)
            
            while len(session.audio_buffer) >= session.frame_size * 2:
                frame_bytes = session.frame_size * 2
                frame = bytes(session.audio_buffer[:frame_bytes])
                session.audio_buffer = session.audio_buffer[frame_bytes:]
                
                if len(frame) != frame_bytes:
                    continue
                
                if WAKE_WORD_ENABLED and not session.is_wakeword_activated:
                    # 1. Existing ONNX wake word model ("Hey Jarvis") stays primary & instant
                    if session.check_wakeword(frame):
                        await session.activate_wakeword(label="Hey Jarvis")
                        
                        await websocket.send_json({
                            "type": "wakeword_detected",
                            "status": "activated",
                            "message": "🎯 Hey Jarvis detected! I'm listening...",
                            "timestamp": time.time()
                        })
                        continue

                    # 2. Parallel Whisper wake alias ("Hey Jonny" / "Hey Johnny")
                    if WHISPER_WAKE_ALIAS_ENABLED and whisper_model is not None:
                        is_sleep_speech = session.is_voice_with_noise_gate(frame)
                        if is_sleep_speech:
                            session.sleep_is_speaking = True
                            session.sleep_speech_frames.append(frame)
                            session.sleep_silence_frames = 0
                            if len(session.sleep_speech_frames) >= session.sleep_max_speech_frames:
                                burst = b"".join(session.sleep_speech_frames)
                                session.sleep_is_speaking = False
                                session.sleep_speech_frames = []
                                session.sleep_silence_frames = 0
                                asyncio.create_task(check_whisper_wakeword(session, burst))
                        elif session.sleep_is_speaking:
                            session.sleep_speech_frames.append(frame)
                            session.sleep_silence_frames += 1
                            if session.sleep_silence_frames >= session.sleep_silence_timeout_frames:
                                valid_count = len(session.sleep_speech_frames) - session.sleep_silence_frames
                                if valid_count >= session.sleep_min_speech_frames:
                                    burst = b"".join(session.sleep_speech_frames[:valid_count])
                                    asyncio.create_task(check_whisper_wakeword(session, burst))
                                session.sleep_is_speaking = False
                                session.sleep_speech_frames = []
                                session.sleep_silence_frames = 0
                    continue
                
                if WAKE_WORD_ENABLED and session.check_timeout():
                    await session.deactivate_wakeword()
                    
                    await websocket.send_json({
                        "type": "wakeword_timeout",
                        "status": "idle",
                        "message": "😴 Going to sleep... Say 'Hey Jarvis' to wake me",
                        "timestamp": time.time()
                    })
                    continue
                
                if session.is_wakeword_activated:
                    is_speech = session.is_voice_with_noise_gate(frame)
                    
                    if is_speech and not session.is_speaking:
                        session.is_speaking = True
                        session.speech_frames = [frame]
                        session.silence_frames = 0
                        print(f"🎤 [{session_id}] Speech started")
                        
                        await websocket.send_json({
                            "type": "speech_started",
                            "status": "speaking",
                            "message": "🎙️ I hear you...",
                            "timestamp": time.time()
                        })
                        
                    elif is_speech and session.is_speaking:
                        session.speech_frames.append(frame)
                        session.silence_frames = 0
                        session.last_activation_time = time.time()
                        
                    elif not is_speech and session.is_speaking:
                        session.speech_frames.append(frame)
                        session.silence_frames += 1
                        
                        if session.silence_frames >= session.silence_timeout_frames:
                            session.is_speaking = False
                            
                            speech_frames_count = len(session.speech_frames) - session.silence_frames
                            if speech_frames_count > 0:
                                speech_audio = b''.join(session.speech_frames[:speech_frames_count])
                                
                                if len(session.speech_frames) >= session.min_speech_frames:
                                    print(f"📊 [{session_id}] Processing: {len(session.speech_frames)} frames")
                                    await session.processing_queue.put(speech_audio)
                                    
                                    await websocket.send_json({
                                        "type": "processing_started",
                                        "status": "processing",
                                        "message": f"🤔 Thinking... (via {'OpenClaw' if session.use_openclaw else 'LM Studio'})",
                                        "timestamp": time.time()
                                    })
                                else:
                                    print(f"⚠️ [{session_id}] Too short: {len(session.speech_frames)} frames")
                            
                            session.speech_frames = []
                            session.silence_frames = 0
                    
    except WebSocketDisconnect:
        print(f"🔇 Disconnected: {session_id}")
    finally:
        status_task.cancel()
        await session.stop()
        if session.worker_task:
            session.worker_task.cancel()
        if session_id in active_sessions:
            del active_sessions[session_id]
        try:
            await status_task
        except:
            pass
        
# ============================================
# UI WEBSOCKET ENDPOINT
# ============================================
@app.websocket("/ws/ui")
async def websocket_ui(websocket: WebSocket):
    await websocket.accept()
    ui_websockets.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except Exception:
        pass
    finally:
        ui_websockets.discard(websocket)

# ============================================
# HTTP ENDPOINTS
# ============================================
@app.get("/voices")
async def list_voices():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{POCKET_TTS_URL}/v1/audio/voices", timeout=5)
            if response.status_code == 200:
                voices_data = response.json()
                voices = []
                
                if 'voices' in voices_data:
                    for voice in voices_data['voices']:
                        voice_id = voice.get('voice_id') or voice.get('id')
                        voice_name = voice.get('name') or voice_id
                        voices.append({"id": voice_id, "name": voice_name})
                elif isinstance(voices_data, list):
                    for voice in voices_data:
                        voice_id = voice.get('voice_id') or voice.get('id')
                        voice_name = voice.get('name') or voice_id
                        voices.append({"id": voice_id, "name": voice_name})
                
                built_in = ["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"]
                existing_ids = [v['id'] for v in voices]
                for voice_id in built_in:
                    if voice_id not in existing_ids:
                        voices.insert(0, {"id": voice_id, "name": voice_id.capitalize()})
                
                voices.sort(key=lambda x: 0 if x['id'] == 'jonnydepp' else 1)
                
                return {"voices": voices}
    except Exception as e:
        print(f"Error fetching voices: {e}")
    
    return {"voices": [
        {"id": "jonnydepp", "name": "Jonny Depp"},
        {"id": "alba", "name": "Alba"},
        {"id": "marius", "name": "Marius"},
        {"id": "javert", "name": "Javert"},
        {"id": "jean", "name": "Jean"},
        {"id": "fantine", "name": "Fantine"},
        {"id": "cosette", "name": "Cosette"},
        {"id": "eponine", "name": "Eponine"},
        {"id": "azelma", "name": "Azelma"}
    ]}

@app.get("/avatar/profile")
async def avatar_profile():
    """Return active avatar portrait and idle loop so Electron switches automatically."""
    return {
        "profile": AVATAR_PROFILE or "johnny",
        "avatar_file": os.path.basename(DEFAULT_AVATAR),
        "idle_file": os.path.basename(DEFAULT_IDLE_VIDEO),
        "avatar_url": "/avatar/image",
        "idle_url": "/avatar/idle",
        "tts_backend": TTS_BACKEND,
        "voice": SAPI_VOICE if TTS_BACKEND == "sapi" else POCKET_TTS_VOICE,
    }


@app.get("/avatar/image")
async def avatar_image_file():
    if os.path.isfile(DEFAULT_AVATAR):
        return FileResponse(DEFAULT_AVATAR)
    raise HTTPException(status_code=404, detail="Avatar image not found")


@app.get("/avatar/idle")
async def avatar_idle_file():
    if os.path.isfile(DEFAULT_IDLE_VIDEO):
        return FileResponse(DEFAULT_IDLE_VIDEO, media_type="video/mp4")
    raise HTTPException(status_code=404, detail="Idle video not found")


@app.get("/wakeword/status")
async def wakeword_status():
    return {
        "enabled": WAKE_WORD_ENABLED,
        "model": WAKE_WORD_MODEL,
        "threshold": WAKE_WORD_THRESHOLD,
        "loaded": wakeword_model is not None,
        "library": "livekit-wakeword" if LIVEKIT_WAKEWORD_AVAILABLE else "none"
    }

@app.get("/llm/status")
async def llm_status():
    """Get current LLM configuration status"""
    return {
        "lm_studio": {
            "url": LM_STUDIO_URL,
            "model": LM_STUDIO_MODEL
        },
        "openclaw": {
            "url": OPENCLAW_URL,
            "model": OPENCLAW_MODEL
        }
    }

@app.get("/gpu/status")
async def gpu_status():
    """Get GPU worker status"""
    return {
        "sequential_mode": GPU_SEQUENTIAL_MODE,
        "worker_active": gpu_worker_active,
        "queue_size": gpu_task_queue.qsize() if gpu_task_queue else 0,
        "cuda_available": torch.cuda.is_available()
    }

async def notify_ui_play_session(
    session_id: str,
    total_chunks: int,
    response_text: str,
    display_text: str,
    voice: str
):
    dead_sockets = []

    for ws in list(ui_websockets):
        try:
            await ws.send_json({
                "type": "play_session",
                "session_id": session_id,
                "total_chunks": total_chunks,
                "response_text": response_text,
                "display_text": display_text,
                "voice": voice
            })
        except Exception:
            dead_sockets.append(ws)

    for ws in dead_sockets:
        ui_websockets.discard(ws)


async def render_venus_audio_job(job_id: str, wav_path: str, video_path: str, fast: bool):
    """Render Venus's existing speech without asking PocketTTS to speak again."""
    job = avatar_audio_jobs[job_id]
    job["status"] = "generating"
    started = time.perf_counter()
    try:
        async def render():
            if fast:
                return await run_in_threadpool(agent.generate_fast, avatar_image, wav_path, video_path)
            return await run_in_threadpool(agent.generate, avatar_image, wav_path, video_path, nfe=3)

        result = await queue_gpu_task(f"Venus_audio_{job_id}", render)
        if not result or not os.path.isfile(video_path) or os.path.getsize(video_path) == 0:
            raise RuntimeError("Avatar renderer did not produce an MP4")
        job["status"] = "ready"
        job["video_url"] = f"/chunks/{os.path.basename(video_path)}"
        job["generation_seconds"] = round(time.perf_counter() - started, 3)
    except Exception as exc:
        job["status"] = "error"
        job["error"] = str(exc)
        print(f"Venus avatar job {job_id} failed: {exc}")
        if os.path.exists(video_path):
            os.remove(video_path)
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


@app.post("/avatar/audio", status_code=202)
async def avatar_from_audio(
    request: Request,
    audio: UploadFile = File(...),
    turn_id: str = Form(""),
    chunk_index: int = Form(0),
    fast: bool = Form(True),
):
    """Accept one Venus speech chunk as mono PCM16 WAV and render an avatar MP4."""
    if request.client is None or request.client.host not in ("127.0.0.1", "::1"):
        raise HTTPException(status_code=403, detail="Venus audio rendering is local-only")
    if agent is None or avatar_image is None or (GPU_SEQUENTIAL_MODE and not gpu_worker_active):
        raise HTTPException(status_code=503, detail="Avatar renderer is not ready")
    if chunk_index < 0 or len(turn_id) > 80:
        raise HTTPException(status_code=422, detail="Invalid turn_id or chunk_index")
    pending = sum(job["status"] in ("queued", "generating") for job in avatar_audio_jobs.values())
    if pending >= 8:
        raise HTTPException(status_code=429, detail="Avatar render queue is full")

    raw = await audio.read(2_000_001)
    if not raw or len(raw) > 2_000_000:
        raise HTTPException(status_code=413, detail="WAV must be between 1 byte and 2 MB")
    try:
        with wave.open(io.BytesIO(raw), "rb") as wav:
            sample_rate = wav.getframerate()
            seconds = wav.getnframes() / sample_rate if sample_rate else 0
            valid = (wav.getnchannels() == 1 and wav.getsampwidth() == 2
                     and 8_000 <= sample_rate <= 48_000 and 0.1 <= seconds <= 15)
    except (wave.Error, EOFError) as exc:
        raise HTTPException(status_code=422, detail="Audio must be a PCM WAV") from exc
    if not valid:
        raise HTTPException(status_code=422, detail="Expected mono PCM16 WAV, 8–48 kHz, 0.1–15 seconds")

    os.makedirs(CHUNK_DIR, exist_ok=True)
    job_id = uuid.uuid4().hex
    wav_path = os.path.join(CHUNK_DIR, f"venus_{job_id}.wav")
    video_path = os.path.join(CHUNK_DIR, f"venus_{job_id}.mp4")
    with open(wav_path, "wb") as destination:
        destination.write(raw)
    job = {
        "job_id": job_id, "status": "queued", "turn_id": turn_id,
        "chunk_index": chunk_index, "audio_seconds": round(seconds, 3),
        "fast": fast, "video_url": None,
    }
    avatar_audio_jobs[job_id] = job
    asyncio.create_task(render_venus_audio_job(job_id, wav_path, video_path, fast))
    return {**job, "status_url": f"/avatar/audio/{job_id}"}


@app.get("/avatar/audio/{job_id}")
async def avatar_audio_status(job_id: str):
    job = avatar_audio_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Avatar job not found")
    return job


@app.delete("/avatar/audio/{job_id}")
async def delete_avatar_audio(job_id: str):
    job = avatar_audio_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Avatar job not found")
    if job["status"] in ("queued", "generating"):
        raise HTTPException(status_code=409, detail="Avatar job is still running")
    video_path = os.path.join(CHUNK_DIR, f"venus_{job_id}.mp4")
    if os.path.exists(video_path):
        os.remove(video_path)
    del avatar_audio_jobs[job_id]
    return {"deleted": True}


@app.post("/chat")
async def chat(
    audio: UploadFile = File(None),
    text: str = Form(None),
    voice: str = Form(POCKET_TTS_VOICE),
    system_prompt: str = Form(None),
    use_openclaw: bool = Form(False),
    skip_avatar: bool = Form(False),
    transcribe_only: bool = Form(False),
    read_aloud: str = Form("false")
):
    session_id = str(uuid.uuid4())[:8]
    perf = PerformanceLogger(session_id)
    user_input = ""

    print(f"\n{'='*60}")
    print(
        f"🎬 NEW REQUEST [{session_id}] - "
        f"LLM: {'OpenClaw' if use_openclaw else 'LM Studio'} | "
        f"Skip Avatar: {skip_avatar} | "
        f"Transcribe Only: {transcribe_only} | "
        f"Read Aloud: {read_aloud}"
    )
    print(f"{'='*60}")

    perf.mark("🚀 Request received")

    if audio:
        print(f"\n🎤 Received audio: {audio.filename}")

        temp_audio = os.path.join(UPLOAD_DIR, f"audio_{session_id}.webm")
        with open(temp_audio, "wb") as f:
            shutil.copyfileobj(audio.file, f)

        perf.mark("📁 Audio file saved")

        wav_path = temp_audio.replace(".webm", ".wav")
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                temp_audio,
                "-ar",
                "16000",
                "-ac",
                "1",
                "-y",
                wav_path
            ],
            capture_output=True
        )

        perf.mark("🔄 FFmpeg conversion complete")

        user_input = await transcribe_audio(wav_path, perf)

        if not user_input:
            return {
                "success": False,
                "error": "Failed to transcribe audio"
            }

        print(f"   📝 Transcript ({len(user_input)} chars): {user_input[:100]}...")

        os.remove(temp_audio)
        os.remove(wav_path)

        perf.mark("🗑️  Temp files cleaned")

    elif text and text.strip():
        user_input = text.strip()
        print(f"\n📝 User text ({len(user_input)} chars): {user_input[:100]}...")
        perf.mark("📝 Text input received")

    else:
        return {
            "success": False,
            "error": "No audio or text input provided"
        }

    if transcribe_only:
        perf.mark("✅ Transcription complete (transcribe_only)")
        return {
            "success": True,
            "transcript": user_input,
            "transcribe_only": True
        }

    read_aloud_flag = str(read_aloud).strip().lower() in (
        "true",
        "1",
        "yes",
        "on"
    )

    print(f"🔊 Read aloud mode: {read_aloud_flag}")

    if read_aloud_flag:
        response_text = user_input
        display_text = user_input
        print(
            f"🔊 READ ALOUD MODE - skipping LLM. "
            f"Text ({len(response_text)} chars): {response_text[:100]}..."
        )
    else:
        llm_response = await queue_gpu_task(
            f"LLM_query_{session_id}",
            query_llm,
            user_input,
            use_openclaw=use_openclaw,
            perf=perf
        )

        response_text = llm_response["tts_text"]
        display_text = llm_response["display_text"]

    print(f"   🤖 TTS Response ({len(response_text)} chars): {response_text[:100]}...")

    if skip_avatar:
        tts_filename = f"tts_{session_id}.wav"
        tts_path = os.path.join(CHUNK_DIR, tts_filename)

        success = await generate_speech_pockettts(
            response_text,
            tts_path,
            voice,
            perf
        )

        if success:
            perf.mark("✅ TTS audio generated")

            return {
                "success": True,
                "skip_avatar": True,
                "audio_url": f"/chunks/{tts_filename}",
                "response_text": response_text,
                "display_text": display_text,
                "voice": voice,
                "llm_used": "Read Aloud" if read_aloud_flag else (
                    "OpenClaw" if use_openclaw else "LM Studio"
                )
            }

        return {
            "success": False,
            "error": "TTS generation failed"
        }

    chunks = split_text_into_chunks(
        response_text,
        target_duration_seconds=3.0
    )

    perf.mark(f"📝 Response split into {len(chunks)} chunks")

    chunk_status[session_id] = {
        "total_chunks": len(chunks),
        "ready": [False] * len(chunks),
        "paths": [None] * len(chunks),
        "text_chunks": chunks,
        "response_text": response_text,
        "display_text": display_text,
        "generation_start": time.time()
    }

    async def generate_progressively():
        chat_motion_state = {}
        for i, chunk_text in enumerate(chunks):
            start_time = time.time()
            fast_mode = i == 0

            chunk_path = await queue_gpu_task(
                f"Video_chunk_{i}_{session_id}",
                generate_chunk_video,
                chunk_text,
                i,
                session_id,
                voice,
                perf,
                fast=fast_mode,
                fragmented=False,
                motion_state=chat_motion_state,
            )

            if chunk_path:
                chunk_status[session_id]["paths"][i] = chunk_path
                chunk_status[session_id]["ready"][i] = True

                gen_time = time.time() - start_time
                estimated_duration = len(chunk_text) / 20

                print(
                    f"   🚀 Chunk {i + 1}/{len(chunks)} ready "
                    f"in {gen_time:.2f}s "
                    f"(speaks for ~{estimated_duration:.1f}s)"
                )
            else:
                print(f"   ❌ Chunk {i + 1}/{len(chunks)} failed")

    asyncio.create_task(generate_progressively())

    perf.mark("✨ Response ready, starting video generation")

    result = {
        "success": True,
        "session_id": session_id,
        "total_chunks": len(chunks),
        "transcript": user_input if audio else None,
        "response_text": response_text,
        "display_text": display_text,
        "voice": voice,
        "llm_used": "Read Aloud" if read_aloud_flag else (
            "OpenClaw" if use_openclaw else "LM Studio"
        ),
        "skip_avatar": False,
        "message": f"Generated {len(chunks)} video chunks"
    }

    if read_aloud_flag:
        await notify_ui_play_session(
            session_id=session_id,
            total_chunks=len(chunks),
            response_text=response_text,
            display_text=display_text,
            voice=voice
        )

    return result


@app.post("/generate_idle")
async def generate_idle(
    audio: Optional[UploadFile] = File(None),
    driver_video: Optional[UploadFile] = File(None),
    seconds: float = Form(10.0),
    nfe: int = Form(3)
):
    """
    Generate /chunks/idle.mp4 for idle playback.

    - If audio is omitted, the server creates a silent WAV.
    - If audio is provided, it may be .wav or .mp3.
    - If driver_video is provided, it is passed to generate_with_driver.
    """
    if agent is None or avatar_image is None:
        return {
            "success": False,
            "error": "Avatar agent is not ready."
        }

    if seconds <= 0:
        return {
            "success": False,
            "error": "seconds must be greater than 0."
        }

    os.makedirs(CHUNK_DIR, exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    job_id = str(uuid.uuid4())[:8]
    wav_path = os.path.join(UPLOAD_DIR, f"idle_{job_id}.wav")
    output_path = os.path.join(CHUNK_DIR, "idle.mp4")
    raw_audio_path = None
    driver_path = None

    try:
        if audio is None:
            make_silent_wav(wav_path, seconds=seconds)
            print(f"🎧 Idle generation using generated silent WAV: {wav_path}")
        else:
            audio_ext = os.path.splitext(audio.filename or "")[1].lower()

            if audio_ext not in [".wav", ".mp3"]:
                return {
                    "success": False,
                    "error": "Audio must be .wav or .mp3."
                }

            raw_audio_path = os.path.join(UPLOAD_DIR, f"idle_input_{job_id}{audio_ext}")

            with open(raw_audio_path, "wb") as f:
                shutil.copyfileobj(audio.file, f)

            normalize_audio_to_wav(raw_audio_path, wav_path, seconds=seconds)
            print(f"🎧 Idle generation using uploaded audio converted to WAV: {wav_path}")

        if driver_video is not None:
            driver_ext = os.path.splitext(driver_video.filename or "")[1].lower()

            if driver_ext not in [".mp4", ".mov", ".webm"]:
                return {
                    "success": False,
                    "error": "Driver video must be .mp4, .mov, or .webm."
                }

            driver_path = os.path.join(UPLOAD_DIR, f"idle_driver_{job_id}{driver_ext}")

            with open(driver_path, "wb") as f:
                shutil.copyfileobj(driver_video.file, f)

            print(f"🎬 Idle generation using guide video: {driver_path}")
        else:
            print("ℹ️ Idle generation has no uploaded guide video. Agent may auto-detect default driver video.")

        result = await queue_gpu_task(
            "Generate_idle_clip",
            generate_idle_clip_task,
            wav_path,
            output_path,
            driver_path,
            nfe
        )

        if not result or not os.path.exists(output_path):
            return {
                "success": False,
                "error": "Idle video generation failed."
            }

        return {
            "success": True,
            "url": "/chunks/idle.mp4",
            "seconds": seconds,
            "used_uploaded_audio": audio is not None,
            "used_uploaded_driver_video": driver_video is not None
        }

    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "error": f"ffmpeg failed: {e.stderr.decode(errors='ignore')}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        for temp_path in [raw_audio_path, wav_path]:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass


@app.get("/chunk_status/{session_id}")
async def get_chunk_status(session_id: str):
    if session_id not in chunk_status:
        return {
            "error": "Session not found"
        }

    return {
        "total_chunks": chunk_status[session_id]["total_chunks"],
        "ready": chunk_status[session_id]["ready"],
        "response_text": chunk_status[session_id].get("response_text", ""),
        "display_text": chunk_status[session_id].get("display_text", "")
    }


@app.get("/wait_for_chunk/{session_id}/{chunk_index}")
async def wait_for_chunk(session_id: str, chunk_index: int):
    if session_id not in chunk_status:
        return {
            "error": "Session not found"
        }

    start_time = time.time()
    timeout = 120

    while time.time() - start_time < timeout:
        if (
            chunk_index < len(chunk_status[session_id]["ready"])
            and chunk_status[session_id]["ready"][chunk_index]
        ):
            return {
                "ready": True,
                "url": f"/chunks/stream_{session_id}_{chunk_index}.mp4"
            }

        await asyncio.sleep(0.5)

    return {
        "ready": False,
        "error": "Timeout"
    }


@app.get("/")
async def index(request: Request):
    avatar_bg = ""

    if os.path.exists(DEFAULT_AVATAR):
        with open(DEFAULT_AVATAR, "rb") as f:
            avatar_bg = base64.b64encode(f.read()).decode()

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "AVATAR_BASE64": avatar_bg
        }
    )


@app.post("/send_text")
async def send_text(data: dict):
    text = data.get("text", "").strip()
    voice = data.get("voice", POCKET_TTS_VOICE)

    if not text:
        return {
            "success": False,
            "error": "No text"
        }

    print(f"📥 /send_text received read-aloud text ({len(text)} chars)")

    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.post(
            "http://127.0.0.1:8002/chat",
            data={
                "text": text,
                "voice": voice,
                "read_aloud": "true",
                "skip_avatar": "false"
            }
        )

    return response.json()

# ============================================
# LIFESPAN
# ============================================
def start_gpu_keep_alive():
    """Background thread to generate silent videos every 30 seconds when GPU idle."""
    def keep_alive_worker():
        import soundfile as sf
        import tempfile
        while True:
            time.sleep(300)
            try:
                # Only run if GPU is not actively processing a task (simple check: queue size)
                if gpu_task_queue.qsize() > 0:
                    continue  # skip if busy
                
                # Use global avatar image
                if avatar_image is None:
                    continue
                
                # Generate 1 second of silence
                silent_audio = np.zeros(1 * 16000, dtype=np.float32)
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as fa:
                    audio_path = fa.name
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as fv:
                    video_path = fv.name
                
                sf.write(audio_path, silent_audio, 16000)
                # Use generate_fast for minimal overhead
                agent.generate_fast(avatar_image, audio_path, video_path)
                os.unlink(audio_path)
                os.unlink(video_path)
                print("🟢 GPU keep‑alive: silent video generated")
            except Exception as e:
                print(f"⚠️ GPU keep‑alive error: {e}")
    
    thread = threading.Thread(target=keep_alive_worker, daemon=True)
    thread.start()
    print("🟢 GPU keep‑alive thread started (every 30s)")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent, avatar_image, whisper_model, wakeword_model, warmup_complete

    print("\n" + "="*60)
    print("🚀 AVATAR STUDIO - GPU SEQUENTIAL + SKIP AVATAR TOGGLE")
    print("="*60)

    # Create directories FIRST (critical!)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(CHUNK_DIR, exist_ok=True)

    agent = InferenceAgent()

    if os.path.exists(DEFAULT_AVATAR):
        original = Image.open(DEFAULT_AVATAR).convert('RGB')
        avatar_image = resize_avatar(original, AVATAR_SIZE)
        print(f"✅ Avatar loaded and resized to {AVATAR_SIZE[0]}x{AVATAR_SIZE[1]}")

    # ========== DEEP WARMUP & PRE‑COMPUTE LATENTS ==========
    agent.warmup(avatar_path=DEFAULT_AVATAR)
    if avatar_image is not None:
        agent.set_avatar(avatar_image)
    # ======================================================

    # ---------- SAFE AUDIO PIPELINE WARMUP (fixed) ----------
    try:
        import soundfile as sf
        import tempfile
        import subprocess

        # 1. Check that ffmpeg is installed and reachable
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)

        # 2. Generate a clean sine wave (2 seconds, 440 Hz)
        duration = 2.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        warmup_audio = 0.1 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            warmup_wav = f.name
        sf.write(warmup_wav, warmup_audio, sample_rate)

        dummy_video = os.path.join(CHUNK_DIR, "audio_warmup.mp4")

        # 3. Try video generation (low NFE = faster warmup)
        agent.generate(avatar_image, warmup_wav, dummy_video, nfe=2)

        # 4. Cleanup
        os.unlink(warmup_wav)
        os.unlink(dummy_video)
        print("✅ Audio pipeline warmed up successfully")

    except Exception as e:
        print(f"⚠️ Deep warmup failed (ignored): {e}")
        print("   First real request may be slightly slower, but server will run normally.")
    # -------------------------------------------------------

    try:
        if agent.try_safe_compile():
            print("✅ GPU Active")
    except:
        pass

    if VENUS_AVATAR_ONLY:
        await start_gpu_worker()
        print("✅ Venus avatar-only server ready on port 8002; POST /avatar/audio")
        try:
            yield
        finally:
            await stop_gpu_worker()
        return

    if WHISPER_AVAILABLE:
        print(f"\n🎤 Loading Whisper {WHISPER_MODEL_SIZE} model...")
        try:
            load_start = time.time()
            whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cuda", compute_type=WHISPER_COMPUTE_TYPE, download_root="./whisper_models")
            print(f"✅ Whisper {WHISPER_MODEL_SIZE} loaded on GPU in {time.time()-load_start:.2f}s")
        except Exception as e:
            print(f"⚠️ Failed to load faster-whisper: {e}")
            whisper_model = None

    # Check LM Studio, OpenClaw, PocketTTS
    try:
        async with httpx.AsyncClient() as client:
            if (await client.get("http://localhost:1234/v1/models", timeout=5)).status_code == 200:
                print("✅ LM Studio connected")
    except:
        print("⚠️ LM Studio not responding")
    try:
        async with httpx.AsyncClient() as client:
            if (await client.get("http://127.0.0.1:18789/v1/models", timeout=5)).status_code == 200:
                print("✅ OpenClaw connected")
    except:
        print("⚠️ OpenClaw not responding (will use LM Studio as fallback)")
    if TTS_BACKEND == "sapi":
        print(f"\n🔥 Warming up Windows SAPI TTS voice: {SAPI_VOICE}...")
        warm_wav = os.path.join(CHUNK_DIR, "warmup_sapi.wav")
        if await generate_speech_sapi("Hello.", warm_wav, voice_name=SAPI_VOICE):
            print(f"✅ Windows SAPI voice '{SAPI_VOICE}' warmed up (local neural TTS active)")
            if os.path.exists(warm_wav):
                os.remove(warm_wav)
        else:
            print(f"⚠️ SAPI warmup failed for '{SAPI_VOICE}'")
    else:
        try:
            async with httpx.AsyncClient() as client:
                if (await client.get(f"{POCKET_TTS_URL}/health", timeout=5)).status_code == 200:
                    print(f"✅ PocketTTS connected on port {POCKET_TTS_URL}")
        except:
            print(f"⚠️ PocketTTS not responding")

        await warmup_pockettts_voice(POCKET_TTS_VOICE)
    await warmup_wakeword()
    await start_gpu_worker()
    start_gpu_keep_alive()

    print("\n" + "="*60)
    print("✅ Server Ready!")
    print("📍 http://localhost:8002")
    print("📍 WebSocket: ws://localhost:8002/ws/realtime/{voice}")
    print(f"🎤 VAD: Aggressiveness={VAD_AGGRESSIVENESS}, Silence=900ms")
    print(f"🎯 Wake Word: {'Enabled (Hey Jarvis + Hey Jonny)' if WAKE_WORD_ENABLED else 'Disabled'}")
    print(f"📚 Wake Word Library: {'livekit-wakeword' if LIVEKIT_WAKEWORD_AVAILABLE else 'none'}")
    print(f"🔊 TTS Backend: {TTS_BACKEND} ({SAPI_VOICE if TTS_BACKEND == 'sapi' else POCKET_TTS_VOICE})")
    print(f"🤖 LLM: LM Studio (SSE Stream) + OpenClaw (Toggle)")
    print(f"🎮 GPU Worker: {'Sequential' if GPU_SEQUENTIAL_MODE else 'Parallel'} mode")
    print(f"🎬 Skip Avatar Toggle: Available (Voice/Text can bypass video)")
    print("="*60 + "\n")

    yield

    await stop_gpu_worker()
    for session in active_sessions.values():
        await session.stop()
    active_sessions.clear()
    for client in (_llm_http_client, _tts_http_client):
        if client is not None and not client.is_closed:
            await client.aclose()

app.router.lifespan_context = lifespan


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("🎬 AVATAR STUDIO - GPU SEQUENTIAL + SKIP AVATAR TOGGLE")
    print("="*60)
    print("\n✅ Whisper Small")
    print("✅ LM Studio (Gemma) - Conversational")
    print("✅ OpenClaw - Task-Oriented (Toggle)")
    print("✅ PocketTTS for voice")
    print("✅ WebRTC VAD (Voice Activity Detection)")
    print("✅ livekit-wakeword (Hey Jarvis)")
    print("✅ Real-time WebSocket streaming")
    print("✅ WebSocket LLM toggle support")
    print("✅ GPU Sequential Worker (prevents race conditions)")
    print("✅ Skip Avatar Toggle (Voice Only mode)")
    print("\n🌐 Open: http://localhost:8002")
    print("🔌 WebSocket: ws://localhost:8002/ws/realtime/jonnydepp")
    print("\n📊 Features:")
    print("   - Toggle between LM Studio and OpenClaw")
    print("   - Toggle between Avatar Video and Voice Only (TTS)")
    print("   - Avatar ALWAYS speaks only the final answer")
    print("   - GPU tasks queued sequentially (no more 100% GPU peg!)")
    print("   - Text input always available (no listening required)")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8002)
