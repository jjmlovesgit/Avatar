# Real-Time Local Talking-Head Avatar Stack (RTX 5090 + Electron + MSE fMP4)

A low-latency, 100% local conversational talking-head avatar stack built around **IMTalker** on NVIDIA RTX 5090 (Blackwell SM120), **Windows SAPI / Kokoro neural TTS** (or PocketTTS), **LM Studio SSE streaming**, **faster-whisper + OpenWakeWord**, an always-on-top **Electron desktop client** with a single-stream **MediaSource (`fMP4`)** player, and a **Chrome MV3 read-aloud extension**.

## Measured Performance (RTX 5090, 512×512 @ 25 fps)

| Stage / Metric | Verified Result | Notes |
| --- | --- | --- |
| **IMTalker 3.0 s clip render** | **~695 ms** (`RTF = 0.23`) | Hybrid BF16 (`SynthesisNetwork`) + FP32 encoders/attention/motion (`48.22 dB PSNR`, `0.65 MouthMAE`), `render_batch=8`, SDPA, `h264_nvenc`, ~4.86 GB peak VRAM |
| **Local Neural TTS (Windows SAPI)** | **80–92 ms** | In-process STA COM worker using Kokoro SAPI (`Paradee am_puck`) — no HTTP TTS server needed |
| **End-to-End Time to First Audio (TTFA)** | **~1.18 s** | `331 ms` LM Studio SSE first chunk + `92 ms` SAPI neural TTS + `754 ms` IMTalker `fMP4` + NVENC (down from `2.16 s` with PocketTTS HTTP) |
| **Chunk-to-Chunk Motion Continuity** | **4.28 MAE** (vs `12.08` stateless) | Continuous `(prev_x, prev_a)` motion state threaded through `FMGenerator.sample()` + single-stream `MediaSource` (`SourceBuffer.mode = 'sequence'`) |

---

## Repository Structure

```text
Avatar/
├── backend/
│   ├── python_avatar_server_electron.py  # FastAPI + WebSocket server (LLM SSE, SAPI/PocketTTS, Whisper wake alias, fMP4 MSE chunks, /avatar/profile)
│   ├── python_venus_avatar_server.py     # Focused audio-to-avatar launcher for Realtime-Venus-Audio (/avatar/audio)
│   ├── imtalker_core.py                  # Unified IMTalker engine, GPU uint8->NVENC fMP4 encoder, continuous motion state
│   ├── imtalker_frontend_rtx.py          # Hybrid FP32/BF16 RTX 5090 renderer + multi-avatar latent cache
│   ├── generator/                        # Flow-matching audio-to-motion generator (FM.py, FMT.py, wav2vec2.py)
│   ├── renderer/                         # IMTRenderer (models.py, SDPA attention_modules.py, lia_resblocks.py)
│   ├── templates/                        # Browser web UI fallback
│   ├── tests/                            # Automated endpoint tests (pytest)
│   ├── driver_video/driver_video.mp4     # Reference blink driver for idle-loop generation
│   ├── hey_jarvis_v0.1.onnx              # OpenWakeWord ONNX detector
│   ├── puck_avatar.png / idle_puck.mp4   # Puck profile portrait + seamless 5s blinking idle loop
│   └── Johnny-Depp.png / idle4.mp4       # Default Johnny profile portrait + idle loop
├── electron/
│   ├── main.js                           # Always-on-top desktop avatar window, global PTT hotkey, read-aloud highlight bridge
│   ├── bridge.js                         # Preload / IPC bridge
│   ├── avatar.html                       # Single-stream MSE fMP4 player + EOF stall watchdog + profile sync + mic streamer
│   ├── package.json                      # Electron dependencies
│   └── avatar-extension/                 # Chrome MV3 "Send to Avatar Studio" + webpage highlight sync extension
├── scripts/
│   └── build_blinking_idle.py            # Generates seamless, practically-still, closed-lip blinking idle MP4s from any 512x512 portrait
└── docs/
    └── VENUS_AUDIO_ENDPOINT.md           # Operational runbook (Venus Audio + Desktop Avatar Studio + all env vars)
```

---

## Prerequisites & Model Checkpoints

Model weights are excluded from Git. Place the IMTalker checkpoints inside `backend/checkpoints/`:

```text
backend/checkpoints/
├── generator.safetensors          # (or generator.ckpt)
├── renderer.safetensors           # (or renderer.ckpt)
└── wav2vec2-base-960h/
    ├── config.json
    ├── model.safetensors
    └── preprocessor_config.json
```

System requirements:
- Windows 10/11 with NVIDIA RTX GPU (tuned for RTX 5090 Blackwell SM120; on RTX 30/40 GPUs set `IMTALKER_FORCE_LOCAL=1`)
- Python 3.11 + PyTorch with CUDA (`2.7.0+cu128` verified on RTX 5090)
- `ffmpeg` (with `h264_nvenc` support) on `PATH`
- Node.js 18+ (for `electron/`)
- LM Studio running locally on `http://127.0.0.1:1234` (OpenAI-compatible SSE endpoint)
- Optional local neural SAPI voices (e.g., Kokoro Paradee `Paradee am_puck` / `Paradee af_heart`) + `pywin32`, or PocketTTS on `http://127.0.0.1:8000`

---

## Quick Start

### 1. Install Backend Dependencies

```powershell
cd backend
conda create -n avatar python=3.11 -y
conda activate avatar
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install pywin32  # Required for in-process Windows SAPI neural TTS
```

### 2. Launch the Avatar Backend (Puck Profile + Local SAPI Neural Voice)

```powershell
cd backend
conda activate avatar
$env:AVATAR_PROFILE = 'puck'
$env:TTS_BACKEND = 'sapi'
$env:SAPI_VOICE = 'Paradee am_puck'
$env:IMTALKER_MSE_STREAMING = '1'
$env:IMTALKER_LLM_STREAMING = '1'
python python_avatar_server_electron.py
```

- Switch back to the default Johnny portrait & idle loop at any time by omitting `AVATAR_PROFILE` (or setting `$env:AVATAR_PROFILE = 'johnny'`).
- For focused Venus audio-only rendering (`/avatar/audio` without Whisper/LLM/TTS), run `python python_venus_avatar_server.py` — see [docs/VENUS_AUDIO_ENDPOINT.md](docs/VENUS_AUDIO_ENDPOINT.md).

### 3. Launch the Electron Desktop Avatar

```powershell
cd electron
npm install
npm start
```

The Electron client automatically calls `GET http://localhost:8002/avatar/profile` on startup to load the active portrait and seamless blinking idle loop, connects to `/ws/realtime/jonnydepp` for wake-word (`Hey Jarvis` / `Hey Jonny` / `Hey Puck`) and VAD streaming, and plays continuous fragmented MP4 chunks over a single `MediaSource` element with automatic EOF recovery back to idle.

### 4. (Optional) Chrome Read-Aloud Extension

1. Open `chrome://extensions` and enable **Developer mode**.
2. Click **Load unpacked** and select `electron/avatar-extension`.
3. Select text on any page and right-click **Send to Avatar Studio** (or press `Ctrl+Shift+S`).

---

## Generating a New Seamless Blinking Idle Loop

To create a practically-still, closed-lip, naturally-blinking, exact-seam `512x512` idle loop for any aligned portrait:

```powershell
python scripts/build_blinking_idle.py `
  --avatar backend/puck_avatar.png `
  --driver backend/driver_video/driver_video.mp4 `
  --output backend/idle_puck.mp4
```

---

## Key Environment Variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `AVATAR_PROFILE` | `johnny` | Active avatar preset (`johnny` or `puck`; exposed via `GET /avatar/profile`) |
| `TTS_BACKEND` | `pockettts` | `sapi` (in-process Windows SAPI / Kokoro neural voices, ~80–90 ms) or `pockettts` (`localhost:8000`) |
| `SAPI_VOICE` | `Paradee am_puck` | Voice name or substring when `TTS_BACKEND=sapi` |
| `IMTALKER_FAST_NFE` / `IMTALKER_STD_NFE` | `6` / `6` | Flow-matching ODE solver steps for chunk 0 and chunks 1+ |
| `IMTALKER_BF16` | `1` | Hybrid BF16 (`SynthesisNetwork` only) with FP32 encoders, `MotionDecoder`, and `CrossAttention` |
| `IMTALKER_RENDER_BATCH` | `8` | GPU frame decode micro-batch size |
| `IMTALKER_NVENC` | `1` | Hardware `h264_nvenc` encoding (`0` = CPU `libx264`) |
| `IMTALKER_MSE_STREAMING` | `1` | Fragmented MP4 (`fMP4`) streaming for Electron's single-stream MSE player |
| `IMTALKER_LLM_STREAMING` | `1` | Sentence-by-sentence LM Studio SSE streaming (`IMTALKER_LLM_MAX_TOKENS=1000`) |
| `WHISPER_WAKE_ALIAS` | `1` | Enables `"Hey Jonny"` / `"Hey Johnny"` / `"Hey Puck"` Whisper wake alias alongside `hey_jarvis_v0.1.onnx` |
| `IMTALKER_FORCE_LOCAL` | `0` | Set `1` to enable local CUDA rendering on non-Blackwell (RTX 30/40) GPUs |
