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
| `WHISPER_WAKE_ALIAS` | `1` | Enables `"Hey Jonny"` / `"Hey Johnny"` Whisper wake alias alongside `hey_jarvis_v0.1.onnx` (`0` = ONNX only). |
| `TTS_BACKEND` | `pockettts` | `sapi` for local in-process Windows SAPI / Kokoro Paradee neural voices (~80–90 ms, no TTS server needed), or `pockettts` for HTTP (`localhost:8000`). |
| `SAPI_VOICE` | `Paradee am_puck` | Voice name/substring when `TTS_BACKEND=sapi` (e.g. `Paradee am_puck`, `Paradee af_heart`, `Pockettts_Sloan_2`, `Microsoft Mark`). |
| `SAPI_RATE` | `0` | Windows SAPI speaking rate (`-10` to `+10`). |

Example override before launch (extra-crisp lip-sync):

```powershell
$env:IMTALKER_FAST_NFE = '8'
python python_venus_avatar_server.py
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

## 6. Full Desktop Avatar Studio (`electronJon` + LM Studio + SAPI / PocketTTS)

Use this stack when running the floating desktop avatar (`C:\Projects\electronJon`) with wake words (**"Hey Jarvis"** via ONNX + **"Hey Jonny" / "Hey Johnny"** via Whisper, including one-breath questions like *"Hey Jonny, give me a 10 sentence monologue on CRISPR"*), SSE LLM streaming, continuous `(prev_x, prev_a)` motion state, and single-stream `fMP4` MSE playback.

> Stop `python_venus_avatar_server.py` before starting `python_avatar_server_electron.py` (both use port `8002`).

### Option A — Fastest (~1.18 s TTFA, Puck Avatar + Local Windows SAPI Neural Voice, No PocketTTS Server Needed)

1. Start **LM Studio** on `localhost:1234` with `qwen2.5-7b-instruct` loaded.
2. **Window 1 — Avatar Backend (`C:\Projects\FlashIMTalkerV2`):**
   ```powershell
   Set-Location C:\Projects\FlashIMTalkerV2
   conda activate flashrtRTC
   $env:AVATAR_PROFILE = 'puck'
   python python_avatar_server_electron.py
   ```
   *(Setting `AVATAR_PROFILE=puck` automatically loads `puck_avatar.png` + seamless `idle_puck.mp4`, enables `TTS_BACKEND=sapi` with `Paradee am_puck`, enables **"Hey Puck"** alongside **"Hey Jonny"** / **"Hey Jarvis"**, and auto-syncs Electron's portrait & idle loop via `/avatar/profile`. Leave `AVATAR_PROFILE` unset for Johnny + `idle4.mp4`.)*
3. **Window 2 — Electron Desktop UI (`C:\Projects\electronJon`):**
   ```powershell
   Set-Location C:\Projects\electronJon
   npm start
   ```
   *(Or press `Ctrl+R` in an open Electron window after switching backend profiles, or test in a browser at `http://localhost:8002`.)*

**Installed local Windows SAPI voices on this PC** (measured on 1 sentence):
- `Paradee am_puck` — Kokoro neural Male (**~80–90 ms**)
- `Paradee af_heart` — Kokoro neural Female (**~88 ms**)
- `Paradee bm_fable` / `Paradee bf_isabella` — British English Kokoro neural
- `Pockettts_Sloan_2` / `Pockettts_Elle_mono` / `Alba (Pocket TTS)` — PocketTTS SAPI (**~142 ms**)
- `Microsoft Mark` / `Microsoft Zira` — Windows SAPI (**~11 ms**)

### Option B — PocketTTS HTTP Voice Server (~1.8–2.2 s TTFA, `jonnydepp` voice)

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
   python python_avatar_server_electron.py
   ```
4. **Window 3 — Electron Desktop UI (`C:\Projects\electronJon`):**
   ```powershell
   Set-Location C:\Projects\electronJon
   npm start
   ```

### Verified Performance Summary (RTX 5090)
- **Local SAPI (`Paradee am_puck`) + LM Studio SSE + IMTalker `fMP4` MSE**: **`1.18 s` TTFA** (`331 ms` LLM + `92 ms` SAPI + `754 ms` IMTalker)
- **PocketTTS HTTP (`jonnydepp`) + LM Studio SSE + IMTalker `fMP4` MSE**: **`~1.8–2.2 s` TTFA**, 3-stage pipelined PocketTTS prefetch overlaps TTS with rendering on follow-up chunks.
- **3.0 s IMTalker render (`synth_bf16=True`, `render_batch=8`, `nfe=6`, `h264_nvenc p4 cq 16`)**: **`~695 ms`** (`RTF = 0.23`, ~4.9 GB VRAM, 48.2 dB PSNR vs. FP32).

### Rollback Backups
- Pre-MSE streaming backups: `C:\Projects\FlashIMTalkerV2\backup_pre_mse_20261123\` and `C:\Projects\electronJon\backup_pre_mse_20261123\`
- Pre-LLM SSE streaming backups: `C:\Projects\FlashIMTalkerV2\backup_pre_llm_stream_20261123\` and `C:\Projects\electronJon\backup_pre_llm_stream_20261123\`
- Pre-speedup `G:\FlashIMTalker` backup: `G:\FlashIMTalker\backup_pre_speedup_20261123\`
