# Realtime-Venus-Audio + FlashIMTalkerV2 Runbook

Single reference for running `C:\Projects\Realtime-Venus-Audio` (Omni Movie UI or Audio-only UI) together with the local avatar renderer in `C:\Projects\FlashIMTalkerV2` on RTX 5090.

---

## 1. Folders & Conda Environments

| Role | Working Folder | Conda Environment | Verified Stack | Port |
| --- | --- | --- | --- | --- |
| **Avatar Video Server** | `C:\Projects\FlashIMTalkerV2` | `imtalker-rtx5090` *(also works in `flashrtRTC`)* | Python 3.11, PyTorch `2.7.0+cu128`, RTX 5090 | `127.0.0.1:8002` |
| **Venus Omni / Audio UI** | `C:\Projects\Realtime-Venus-Audio` | `venus-audio` | Python 3.10, PyTorch `2.11.0+cu128`, RTX 5090 | *(desktop UI)* |

> **VRAM note:** Make sure only **one** avatar server (`8002`) and **one** Venus UI (`omni_mic_ui.py` **or** `live_mic_ui.py`, not both) are running at a time. If `C:\Projects\LFM3\lfm3_avatar_server.py` (port `8003`) or an older `G:\FlashIMTalker` server is still open, stop it first to keep GPU memory free.

---

## 2. Launch Commands (2 PowerShell Windows)

### Window 1 — Start the Avatar Server (`FlashIMTalkerV2`)

```powershell
Set-Location C:\Projects\FlashIMTalkerV2
conda activate imtalker-rtx5090
python python_venus_avatar_server.py
```

What this does:
- Sets `VENUS_AVATAR_ONLY=1` (skips Whisper, wake word, PocketTTS, LM Studio/OpenClaw, and GPU keep-alive; does not require `webrtcvad`).
- Loads the Venus portrait from `C:\Projects\Realtime-Venus-Audio\assets\avatar.png` via `VENUS_AVATAR_IMAGE` (leaving `Johnny-Depp.png` untouched for the full studio app and LFM3).
- Starts the focused HTTP API at `http://127.0.0.1:8002/avatar/audio`.
- Look for `IMTalker RTX 5090 Pipeline Online (synth_bf16=True, render_batch=8)` and `Uvicorn running on http://127.0.0.1:8002` before starting Venus.

*(Alternative manual launch without the focused wrapper: set `$env:VENUS_AVATAR_ONLY='1'`, `$env:VENUS_AVATAR_IMAGE='C:\Projects\Realtime-Venus-Audio\assets\avatar.png'`, `$env:PYTHONIOENCODING='utf-8'`, and run `python python_avatar_server_electron.py`.)*

### Window 2 — Start Realtime-Venus-Audio

**Option A — Omni Movie Conversation UI (`omni_mic_ui.py`):**

```powershell
Set-Location C:\Projects\Realtime-Venus-Audio
conda activate venus-audio
python omni_mic_ui.py
```

**Option B — Audio-only Voice UI (`live_mic_ui.py`):**

```powershell
Set-Location C:\Projects\Realtime-Venus-Audio
conda activate venus-audio
python live_mic_ui.py
```

In the Venus UI:
1. Check **Enable avatar video**.
2. Choose **Playback timing**:
   - `sync` — holds each speech chunk until its rendered video is ready, then plays voice and video together.
   - `voice_first` — plays Venus voice immediately and shows video as each chunk finishes.
3. Set **First chunk (s)** (e.g., `1.0`–`3.0`; `1.0`–`1.5` gives the fastest first-turn response).
4. Click **Load Omni** (or start the session in `live_mic_ui.py`). Wait for **READY**, then click **Play video** or speak/type a question.
5. Uncheck **Conversation only** in **Session activity** whenever you want to inspect `[avatar] ready ... render=...s wall=...s` timing or debug logs.

---

## 3. Environment Variables (Optional Tuning)

Set any of these in **Window 1** *before* running `python python_venus_avatar_server.py`:

| Variable | Default | Purpose |
| --- | --- | --- |
| `VENUS_AVATAR_IMAGE` | `C:\Projects\Realtime-Venus-Audio\assets\avatar.png` | Portrait image animated by the avatar server. |
| `VENUS_AVATAR_PORT` | `8002` | Local bind port (Venus `avatar_bridge.py` defaults to `8002`). |
| `IMTALKER_FAST_NFE` | `6` | Flow-matching ODE steps for `fast=true` chunks (higher = smoother/crisper lip-sync; `6`–`8` is ~700 ms per 3.0 s clip). |
| `IMTALKER_BF16` | `1` | Uses BF16 on `SynthesisNetwork` only while keeping face encoders, `MotionDecoder`, and `CrossAttention` in FP32 (~48.2 dB PSNR). Set `0` for all-FP32. |
| `IMTALKER_RENDER_BATCH` | `8` | GPU micro-batch size for frame decoding (`8` is optimal on RTX 5090 at ~4.9 GB peak VRAM). |
| `IMTALKER_NVENC` | `1` | Uses RTX 5090 `h264_nvenc` hardware encoding (`0` falls back to CPU `libx264`). |
| `IMTALKER_NVENC_PRESET` | `p4` | NVENC preset (`p1` fastest, `p4` balanced/sharp). |
| `IMTALKER_NVENC_CQ` | `16` | NVENC constant quality (lower = sharper video). |
| `CLEAR_CUDA_CACHE_EACH_TASK` | `0` | Set `1` to force `torch.cuda.empty_cache()` after every chunk if VRAM is tight. |
| `IMTALKER_STD_NFE` | `6` | Flow-matching ODE steps for follow-up chunks (`1+`) in full studio mode. |
| `IMTALKER_MSE_STREAMING` | `1` | Sends continuous `fMP4` chunks (`avc1.42E01F, mp4a.40.2`) to Electron's single-stream MSE player (`0` = standard MP4). |
| `IMTALKER_LLM_STREAMING` | `1` | Streams LM Studio responses sentence-by-sentence via SSE (`0` = wait for full reply). |
| `IMTALKER_LLM_MAX_TOKENS` | `1000` | Max tokens for LM Studio replies (supports extended monologues). |
| `AVATAR_PROFILE` | *(unset / johnny)* | Preset profile: `gretchen`, `puck`, or unset (`johnny`). Auto-syncs Electron via `/avatar/profile`. |
| `AVATAR_IDLE_VIDEO` | *(profile default)* | Custom idle loop filename/path (e.g. `gretchen.mp4`, `idle_puck.mp4`, `idle4.mp4`). |
| `AVATAR_PROMPT_CFG` | `prompt.cfg` | System prompt config file (`SYSTEM_PROMPT_LM_STUDIO`, hot-reloads on edit without restarting). |
| `LM_STUDIO_MODEL` | `dolphin3.0-llama3.1-8b-abliterated` | LM Studio model ID used for SSE streaming & chat completions. |
| `WHISPER_WAKE_ALIAS` | `1` | Enables `"Hey Gretchen"` / `"Hey Puck"` / `"Hey Jonny"` / `"Hey Johnny"` Whisper wake alias alongside `hey_jarvis_v0.1.onnx` (`0` = ONNX only). |
| `TTS_BACKEND` | `pockettts` | `sapi` for local in-process Windows SAPI / Offline MS Natural / Paradee voices (~56–90 ms, no TTS server needed), or `pockettts` for HTTP (`localhost:8000`). |
| `SAPI_VOICE` | `Paradee am_puck` | Voice name/substring when `TTS_BACKEND=sapi` (e.g. `Jenny(Natural)`, `Aria (Natural)`, `Andrew (Natural HD)`, `Ryan (Natural)`, `Paradee am_puck`, `Paradee bf_isabella`). |
| `SAPI_RATE` | `0` | Windows SAPI speaking rate (`-10` to `+10`). |
| `IMTALKER_POSE_MODE` | `calm` | Head movement conditioning: `calm` (slow, smooth breathing/sway), `still` (centered zero-pose), or `free` (original unconstrained head movement). |
| `IMTALKER_POSE_WEIGHT` | `0.45` | Pose conditioning blend weight (`0.0–1.0`): `0.45` calms head speed by ~35% while keeping full wide lip articulation (avoid `1.0`, which dampens mouth opening). |
| `IMTALKER_CFG_SCALE` | `2.6` *(calm)* / `2.0` *(free)* | Audio lip-sync CFG guidance scale (higher = crisper, wider mouth opening). |
| `IMTALKER_POSE_PITCH_AMP` | `0.025` | Max head pitch (up/down nod) amplitude in radians when `IMTALKER_POSE_MODE=calm` (`0.025` ≈ ±1.4°). |
| `IMTALKER_POSE_YAW_AMP` | `0.035` | Max head yaw (left/right turn) amplitude in radians when `IMTALKER_POSE_MODE=calm` (`0.035` ≈ ±2.0°). |
| `IMTALKER_POSE_ROLL_AMP` | `0.012` | Max head roll (tilt) amplitude in radians when `IMTALKER_POSE_MODE=calm` (`0.012` ≈ ±0.7°). |
| `IMTALKER_POSE_FREQ` | `0.20` | Head sway frequency in Hz when `IMTALKER_POSE_MODE=calm` (`0.20` = one gentle sway every 5 seconds; lower = slower). |
| `IMTALKER_GAZE_LOCK` | `0` | Set `1` to pass centered camera gaze (`[0, 0]`) into `FMGenerator` (`0` = natural audio-driven gaze). |
| `IMTALKER_SPEAK_BLINK` | `1` | Injects natural eyelid-only blinks (~every 3.4 s) during speech via `driver_video/blink_delta.pt` (+0 ms latency, `0` = off). |
| `IMTALKER_BLINK_STRENGTH` | `1.35` | Speaking blink eyelid-closure depth (`1.35` = full relaxed closure for both Puck & Gretchen). |
| `IMTALKER_BLINK_INTERVAL_SEC` | `3.4` | Average seconds between blinks while speaking (with organic `±0.8 s` jitter). |
| `VAD_AGGRESSIVENESS` | `3` | WebRTC VAD strictness (`0–3`; `3` rejects chair creaks and ambient clicks best). |
| `VAD_NOISE_GATE_THRESHOLD` | `350` | Minimum mic RMS level to trigger VAD (raise to `500` if background noise still triggers). |
| `VAD_MIN_SPEECH_FRAMES` | `12` | Minimum speech frames (`30 ms` each; `12` ≈ `360 ms`) before queueing an utterance. |

Example overrides before launch:

```powershell
# Calm, ultra-natural Gretchen with default offline Jenny(Natural) voice
$env:AVATAR_PROFILE     = 'gretchen'
$env:TTS_BACKEND        = 'sapi'
$env:SAPI_VOICE         = 'Jenny(Natural)'
$env:IMTALKER_POSE_MODE = 'calm'   # or 'still' for minimum movement, 'free' for original
python python_avatar_server_electron.py
```

---

## 4. API Contract (`http://127.0.0.1:8002`)

`POST /avatar/audio` accepts localhost multipart form data:

| Field | Required | Value |
| --- | --- | --- |
| `audio` | Yes | Mono PCM16 WAV, 8–48 kHz, 0.1–15 seconds, up to 2 MB (Venus 24 kHz WAV works directly). |
| `turn_id` | No | Caller-defined reply ID (up to 80 chars). |
| `chunk_index` | No | Nonnegative chunk index (default `0`). |
| `fast` | No | `true` (default, uses `IMTALKER_FAST_NFE`) or `false` (`nfe=3`). |

- Returns **HTTP 202** with `job_id`, `status_url`, `status`, `turn_id`, `chunk_index`, `audio_seconds`, and `fast`.
- Poll `GET /avatar/audio/{job_id}` until `status` is `ready` or `error` (`ready` includes `video_url` under `/chunks/` and `generation_seconds`).
- Call `DELETE /avatar/audio/{job_id}` after downloading/playing to remove the MP4 and job record (`409` if still rendering; input WAV is deleted automatically).
- `avatar_bridge.py` handles upload, polling, download, cleanup, and muted frame playback in the Venus UI automatically.

---

## 5. Quick Standalone Test (Without Starting Venus)

With Window 1 running, test with the newest saved Venus reply WAV in another PowerShell window:

```powershell
$wav = (Get-ChildItem 'C:\Projects\Realtime-Venus-Audio\logs\reply-*.wav' |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
curl.exe -sS -F "audio=@$wav;type=audio/wav" -F "turn_id=manual-test" `
    http://127.0.0.1:8002/avatar/audio
```

Check status (replace `JOB_ID` from the response):

```powershell
curl.exe -sS http://127.0.0.1:8002/avatar/audio/JOB_ID
```

Clean up the test clip when done:

```powershell
curl.exe -sS -X DELETE http://127.0.0.1:8002/avatar/audio/JOB_ID
```

---

## 6. Full Desktop Avatar Studio (`electronJon` + LM Studio + Offline Natural SAPI / PocketTTS)

Use this stack when running the floating desktop avatar (`C:\Projects\electronJon`) with wake words (**"Hey Jarvis"** via ONNX + **"Hey Gretchen" / "Hey Puck" / "Hey Jonny" / "Hey Johnny"** via Whisper, including one-breath questions like *"Hey Gretchen, why is the sky blue?"*), SSE LLM streaming, hot-reloaded `prompt.cfg`, continuous `(prev_x, prev_a)` motion state, and single-stream `fMP4` MSE playback.

> Stop `python_venus_avatar_server.py` before starting `python_avatar_server_electron.py` (both use port `8002`).
> Always set `$env:TTS_BACKEND = 'sapi'` explicitly when launching SAPI profiles so an earlier shell setting (`pockettts`) cannot override the profile default.

### Option A — Gretchen (Default Approved Settings: Offline `Jenny(Natural)` + Calm Pose + Articulated Lips)

All tuned defaults (`SAPI_VOICE=Jenny(Natural)`, `IMTALKER_POSE_MODE=calm`, `IMTALKER_POSE_WEIGHT=0.45`, `IMTALKER_CFG_SCALE=2.6`, `VAD_AGGRESSIVENESS=3`, `VAD_NOISE_GATE_THRESHOLD=350`, `VAD_MIN_SPEECH_FRAMES=12`, `LM_STUDIO_MODEL=dolphin3.0-llama3.1-8b-abliterated`) are now built into the `gretchen` profile.

1. Start **LM Studio** on `localhost:1234` with `dolphin3.0-llama3.1-8b-abliterated` loaded.
2. *(Optional)* Edit `C:\Projects\FlashIMTalkerV2\prompt.cfg` (`SYSTEM_PROMPT_LM_STUDIO`) — changes hot-reload automatically on the next spoken turn with no restart needed.
3. **Window 1 — Avatar Backend (`C:\Projects\FlashIMTalkerV2`):**
   ```powershell
   Set-Location C:\Projects\FlashIMTalkerV2
   conda activate flashrtRTC
   Remove-Item Env:VENUS_AVATAR_IMAGE, Env:AVATAR_IDLE_VIDEO, Env:SAPI_VOICE -ErrorAction SilentlyContinue
   $env:AVATAR_PROFILE = 'gretchen'
   $env:TTS_BACKEND    = 'sapi'
   python python_avatar_server_electron.py
   ```
   *(Full explicit settings equivalent: `SAPI_VOICE='Jenny(Natural)'` [or `'Aria (Natural)'` / `'Paradee bf_isabella'`], `SAPI_RATE='0'`, `IMTALKER_POSE_MODE='calm'`, `IMTALKER_POSE_WEIGHT='0.45'`, `IMTALKER_CFG_SCALE='2.6'`, `VAD_AGGRESSIVENESS='3'`, `VAD_NOISE_GATE_THRESHOLD='350'`, `VAD_MIN_SPEECH_FRAMES='12'`.)*
4. **Window 2 — Electron Desktop UI (`C:\Projects\electronJon`):**
   ```powershell
   Set-Location C:\Projects\electronJon
   npm start
   ```
   *(Or press **`Ctrl+R`** in an already-open Electron window after switching profiles; wake with **"Hey Gretchen"** or **"Hey Jarvis"**.)*

### Option B — Puck (Default Approved Settings: Offline `Andrew (Natural HD)` + Calm Pose + Articulated Lips)

All tuned defaults (`SAPI_VOICE=Andrew (Natural HD)`, `IMTALKER_POSE_MODE=calm`, `IMTALKER_POSE_WEIGHT=0.45`, `IMTALKER_CFG_SCALE=2.6`, `VAD_AGGRESSIVENESS=3`, `VAD_NOISE_GATE_THRESHOLD=350`, `VAD_MIN_SPEECH_FRAMES=12`) are now built into the `puck` profile.

1. Start **LM Studio** on `localhost:1234`.
2. **Window 1 — Avatar Backend (`C:\Projects\FlashIMTalkerV2`):**
   ```powershell
   Set-Location C:\Projects\FlashIMTalkerV2
   conda activate flashrtRTC
   Remove-Item Env:VENUS_AVATAR_IMAGE, Env:AVATAR_IDLE_VIDEO, Env:SAPI_VOICE -ErrorAction SilentlyContinue
   $env:AVATAR_PROFILE = 'puck'
   $env:TTS_BACKEND    = 'sapi'
   python python_avatar_server_electron.py
   ```
   *(Alternative male voices: `$env:SAPI_VOICE = 'Paradee am_puck'` or `'Ryan (Natural)'`. Loads `puck_avatar.png` + seamless blinking `idle_puck.mp4` and wakes with **"Hey Puck"** / **"Hey Jarvis"** / **"Hey Jonny"**.)*
3. **Window 2 — Electron Desktop UI (`C:\Projects\electronJon`):** `npm start` (or `Ctrl+R`).

**Installed local offline SAPI voices on this PC** *(bridged via `C:\Projects\NaturalVoiceSAPIAdapter` v0.2.11, measured warm on 1 sentence)*:
- `Microsoft Jenny(Natural) - English (United States)` — Offline MS Natural Female (**~56 ms**, full `<rate>`/`<pitch>`/`<volume>` XML)
- `Microsoft Aria (Natural) - English (United States)` — Offline MS Natural Female (**~62 ms**, full `<rate>`/`<pitch>`/`<volume>` XML)
- `Microsoft Ryan (Natural) - English (United Kingdom)` — Offline MS Natural UK Male (**~67 ms**, full `<rate>`/`<pitch>`/`<volume>` XML)
- `Microsoft Andrew (Natural HD) - English (United States)` — Offline MS Natural HD Male (**~73 ms**, full `<rate>`/`<pitch>`/`<volume>` XML)
- `Paradee am_puck` / `Paradee af_heart` / `Paradee bf_isabella` / `Paradee bm_fable` — Kokoro neural SAPI (**~80–96 ms**, rate only)
- `Pockettts_Sloan_2` / `Pockettts_Elle_mono` / `Alba (Pocket TTS)` — PocketTTS SAPI (**~142 ms**)
- `Microsoft Mark` / `Microsoft Zira` — Classic Windows SAPI (**~11 ms**)
- *(Avoid `Microsoft ... Online (Natural)` voices for live avatar chat — cloud round-trip is **1,200–2,200 ms** per sentence.)*

### Option C — Johnny + PocketTTS HTTP Voice Server (~1.8–2.2 s TTFA, `jonnydepp` voice)

1. Start **LM Studio** on `localhost:1234`.
2. **Window 1 — PocketTTS Server (`localhost:8000`):**
   ```powershell
   Set-Location C:\Projects\pocket-tts-openai_streaming_server
   conda activate pocket_tts
   python pocket_tts_api.py
   ```
3. **Window 2 — Avatar Backend (`C:\Projects\FlashIMTalkerV2`):**
   ```powershell
   Set-Location C:\Projects\FlashIMTalkerV2
   conda activate flashrtRTC
   Remove-Item Env:AVATAR_PROFILE -ErrorAction SilentlyContinue
   $env:TTS_BACKEND = 'pockettts'
   python python_avatar_server_electron.py
   ```
4. **Window 3 — Electron Desktop UI (`C:\Projects\electronJon`):**
   ```powershell
   Set-Location C:\Projects\electronJon
   npm start
   ```

### Option D — Dual Avatar Mode (Puck + Gretchen Simultaneously, 2 Independent Floating Windows + Shared Room Memory)

Runs **one single backend** (pre-caching both `puck_avatar.png` and `gretchen.png` latents at startup for 0 ms switching and zero extra VRAM) and **two independent draggable Electron windows** (Puck owns the single mic stream; Gretchen connects display-only to `/ws/display/gretchen`). Both share the same room memory (`[Puck]:` / `[Gretchen]:` turns) and load their own hot-reloaded persona prompts (`SYSTEM_PROMPT_PUCK` and `SYSTEM_PROMPT_GRETCHEN`) from `prompt.cfg`.

1. Start **LM Studio** on `localhost:1234` with `dolphin3.0-llama3.1-8b-abliterated` loaded.
2. **Window 1 — Avatar Backend (`C:\Projects\FlashIMTalkerV2`):**
   ```powershell
   Set-Location C:\Projects\FlashIMTalkerV2
   conda activate flashrtRTC
   Remove-Item Env:VENUS_AVATAR_IMAGE, Env:AVATAR_IDLE_VIDEO, Env:SAPI_VOICE -ErrorAction SilentlyContinue
   $env:AVATAR_PROFILE = 'dual'
   $env:TTS_BACKEND    = 'sapi'
   python python_avatar_server_electron.py
   ```
3. **Window 2 — Electron Dual Windows (`C:\Projects\electronJon`):**
   ```powershell
   Set-Location C:\Projects\electronJon
   npm run start:dual   # (plain `npm start` also auto-detects dual_mode=true from backend)
   ```
   - Drag Puck and Gretchen independently to any screen corners.
   - Say **"Hey Puck, ..."** or **"Hey Gretchen, ..."** (from sleep OR mid-conversation) to address either one, or click either avatar window to switch the active speaker.

### Verified Performance Summary (RTX 5090)
- **Offline MS Natural SAPI (`Jenny(Natural)` / `Aria (Natural)`)**: **`56–62 ms`** plain, **`67–69 ms`** with full `<rate>`/`<pitch>`/`<volume>` XML tags (zero latency penalty).
- **Local SAPI (`Paradee am_puck`) + LM Studio SSE + IMTalker `fMP4` MSE**: **`1.18 s` TTFA** (`331 ms` LLM + `92 ms` SAPI + `754 ms` IMTalker).
- **PocketTTS HTTP (`jonnydepp`) + LM Studio SSE + IMTalker `fMP4` MSE**: **`~1.8–2.2 s` TTFA**, 3-stage pipelined PocketTTS prefetch overlaps TTS with rendering on follow-up chunks.
- **3.0 s IMTalker render (`synth_bf16=True`, `render_batch=8`, `nfe=6`, `h264_nvenc p4 cq 16`)**: **`~695 ms`** (`RTF = 0.23`, ~4.9 GB VRAM, 48.2 dB PSNR vs. FP32).

### Rollback Backups
- Pre-MSE streaming backups: `C:\Projects\FlashIMTalkerV2\backup_pre_mse_20261123\` and `C:\Projects\electronJon\backup_pre_mse_20261123\`
- Pre-LLM SSE streaming backups: `C:\Projects\FlashIMTalkerV2\backup_pre_llm_stream_20261123\` and `C:\Projects\electronJon\backup_pre_llm_stream_20261123\`
- Pre-speedup `G:\FlashIMTalker` backup: `G:\FlashIMTalker\backup_pre_speedup_20261123\`
