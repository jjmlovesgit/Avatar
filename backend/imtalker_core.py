"""
IMTalker Unified Client - FLASHVLA ENGINE (Blackwell SM120)
- Local inference ONLY on RTX 5090
- All other GPUs → FastAPI server fallback
- Uses optimized .safetensors and custom C++ NVFP4 kernels
"""

import os
import sys
import time
import json
import base64
import io
import subprocess
import numpy as np
from PIL import Image

# Try to import torch for GPU detection
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ Torch not available - using FastAPI mode")

# ==========================================
# 🚀 IMTALKER LOCAL ENGINE IMPORT
# ==========================================
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from imtalker_frontend_rtx import IMTalkerTorchFrontendRtx
    LOCAL_IMTALKER_AVAILABLE = True
except ImportError as e:
    LOCAL_IMTALKER_AVAILABLE = False
    print(f"⚠️ IMTalker local frontend not found: {e} - using FastAPI mode")

# Optional FlashVLA plugin registration (not required for pure PyTorch RTX execution)
if LOCAL_IMTALKER_AVAILABLE:
    try:
        import flash_vla  # noqa: F401
        from flash_vla.hardware import _PIPELINE_MAP
        _PIPELINE_MAP[("imtalker", "torch", "rtx_sm120")] = (
            "imtalker_frontend_rtx", "IMTalkerTorchFrontendRtx"
        )
        print("✅ IMTalker Architecture Registered for Blackwell SM120")
    except ImportError:
        print("✅ IMTalker PyTorch RTX Frontend ready (standalone mode)")

# ==========================================
# GPU DETECTION – BLACKWELL (SM120) OR IMTALKER_FORCE_LOCAL=1
# ==========================================
def detect_gpu_type():
    """Detect GPU – local inference on Blackwell (SM120) or any CUDA GPU with IMTALKER_FORCE_LOCAL=1"""
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return {"name": "CPU", "use_local": False, "vram_gb": 0, "capability": (0, 0)}
    
    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0) # Returns (12, 0) for 5090
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    is_blackwell = capability[0] >= 12
    force_local = os.getenv("IMTALKER_FORCE_LOCAL", "0").lower() in ("1", "true", "yes")
    use_local = (is_blackwell or force_local) and LOCAL_IMTALKER_AVAILABLE
    
    return {
        "name": device_name,
        "is_blackwell": is_blackwell,
        "vram_gb": round(vram_gb, 1),
        "capability": capability, # <--- The missing key
        "use_local": use_local
    }
    
# Note: Ensure LocalInferenceAgent uses self.model = IMTalkerTorchFrontendRtx(checkpoint_dir="./checkpoints")

# ==========================================
# DEFAULT DRIVER VIDEO PATH
# ==========================================
def get_default_driver_path():
    default_path = os.path.join("driver_video", "driver_video.mp4")
    if os.path.isfile(default_path):
        return default_path
    return None


# ==========================================
# 1. CONFIGURATION
# ==========================================
class AppConfig:
    def __init__(self):
        gpu_info = detect_gpu_type()
        self.gpu_info = gpu_info
        
        # ========== Model Architecture ==========
        self.input_nc = 3
        self.audio_marcing = 2
        self.wav2vec_sec = 2.0
        self.attention_window = 5
        self.only_last_features = True
        self.audio_dropout_prob = 0.1
        self.style_dim = 512
        self.dim_a = 512
        self.dim_h = 512
        self.dim_e = 7
        self.dim_motion = 32
        self.dim_c = 32
        self.dim_w = 32
        self.fmt_depth = 8
        self.num_heads = 8
        self.mlp_ratio = 4.0
        self.no_learned_pe = False
        self.num_prev_frames = 10
        self.max_grad_norm = 1.0
        self.ode_atol = 1e-5
        self.ode_rtol = 1e-5
        self.torchdiffeq_ode_method = 'euler'
        self.swin_res_threshold = 128
        self.window_size = 8
        self.fix_noise_seed = False
        # =========================================

        self.use_local_inference = gpu_info["use_local"]
        
        if self.use_local_inference:
            print(f"🎮 Blackwell (SM120) detected – using LOCAL inference (RTX 5090)")
            self._init_local_config()
        else:
            print(f"🖥️ {gpu_info['name']} (SM{''.join(map(str,gpu_info['capability']))}) – not Blackwell, using FastAPI server")
            self._init_api_config()
    
    def _init_local_config(self):
        """Local inference configuration (RTX 5090 only)"""
        self.device = "cuda"
        self.rank = "cuda"
        
        # 🚀 Now pointing to the perfectly contiguous Safetensors!
        self.renderer_path = "./checkpoints/renderer.safetensors"
        self.generator_path = "./checkpoints/generator.safetensors"
        self.wav2vec_model_path = "./checkpoints/wav2vec2-base-960h"
        
        self.crop = True
        self.seed = 42
        self.nfe = 4                     # 5090 is fast
        self.a_cfg_scale = 2.0
        self.input_size = 256
        self.fps = 25.0
        self.video_resolution = (512, 512)
        
        # Audio processing
        self.sampling_rate = 16000
        self.audio_smoothing_window = 3
        self.audio_fade_duration = 0.05
        
        # FFmpeg (pipe encoding)
        self.fast_ffmpeg = True
        self.ffmpeg_crf = 23
        self.ffmpeg_preset = 'ultrafast'
        self.ffmpeg_audio_bitrate = '128k'
        
        self.driver_cache_enabled = True
        self.torch_compile = True
    
    def _init_api_config(self):
        """FastAPI client configuration (all non‑Blackwell GPUs)"""
        self.device = "cuda"
        self.rank = "cuda"
        
        self.api_url = "http://127.0.0.1:7861"
        self.generate_endpoint = "/generate"
        self.health_endpoint = "/health"
        
        self.crop = True
        self.seed = 42
        self.nfe = 8
        self.a_cfg_scale = 2.0
        self.input_size = 256
        self.fps = 25.0
        self.video_resolution = (512, 512)


# ==========================================
# 2. DATA PROCESSOR
# ==========================================
class DataProcessor:
    def __init__(self, opt):
        self.opt = opt
        self.sampling_rate = getattr(opt, 'sampling_rate', 16000)
        
        if opt.use_local_inference:
            import face_alignment
            import torchvision.transforms as transforms
            from transformers import Wav2Vec2FeatureExtractor
            
            self.fa = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D, device='cpu', flip_input=False
            )
            if os.path.exists(opt.wav2vec_model_path):
                self.wav2vec_preprocessor = Wav2Vec2FeatureExtractor.from_pretrained(
                    opt.wav2vec_model_path, local_files_only=True
                )
            else:
                self.wav2vec_preprocessor = Wav2Vec2FeatureExtractor.from_pretrained(
                    "facebook/wav2vec2-base-960h"
                )
            self.transform = transforms.Compose([
                transforms.Resize((opt.input_size, opt.input_size)),
                transforms.ToTensor()
            ])
    
    def process_img(self, img: Image.Image) -> Image.Image:
        if not self.opt.use_local_inference:
            return img
        
        import cv2
        img_arr = np.array(img)
        if img_arr.ndim == 2:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)
        elif img_arr.shape[2] == 4:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2RGB)
        
        h, w = img_arr.shape[:2]
        try:
            bboxes = self.fa.face_detector.detect_from_image(img_arr)
            valid_bboxes = [(int(x1), int(y1), int(x2), int(y2), score) 
                           for (x1, y1, x2, y2, score) in bboxes if score > 0.5]
        except:
            valid_bboxes = []
        
        if valid_bboxes:
            x1, y1, x2, y2, _ = valid_bboxes[0]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            half_side = int(max(x2-x1, y2-y1) * 0.8)
            x1_new = max(0, cx - half_side)
            y1_new = max(0, cy - half_side)
            x2_new = min(w, cx + half_side)
            y2_new = min(h, cy + half_side)
            side = min(x2_new - x1_new, y2_new - y1_new)
            crop_img = img_arr[y1_new:y1_new+side, x1_new:x1_new+side]
        else:
            side = min(w, h)
            crop_img = img_arr[:side, :side]
        
        return Image.fromarray(crop_img).resize((self.opt.input_size, self.opt.input_size))
    
    def process_audio(self, path: str):
        if not self.opt.use_local_inference:
            return None
        
        import librosa
        speech_array, sr = librosa.load(path, sr=self.sampling_rate)
        
        # Normalize, Soft clip, Smooth, Fade
        speech_array = speech_array - np.mean(speech_array)
        speech_array = np.clip(speech_array, -0.95, 0.95)
        window = np.ones(3)/3
        if len(speech_array) > 3:
            speech_array = np.convolve(speech_array, window, mode='same')
        fade_samples = int(0.05 * self.sampling_rate)
        if len(speech_array) > 2*fade_samples:
            fade_in = np.linspace(0,1,fade_samples)
            fade_out = np.linspace(1,0,fade_samples)
            speech_array[:fade_samples] *= fade_in
            speech_array[-fade_samples:] *= fade_out
        
        return self.wav2vec_preprocessor(
            speech_array, sampling_rate=self.sampling_rate, return_tensors='pt'
        ).input_values[0]
    
    def transform_tensor(self, img):
        if self.opt.use_local_inference:
            return self.transform(img)
        return img


# ==========================================
# 3. LOCAL INFERENCE AGENT (FLASHVLA ENGINE)
# ==========================================
class LocalInferenceAgent:
    def __init__(self, opt):
        self.opt = opt
        self.device = opt.device
        self.data_processor = DataProcessor(opt)
        
        print("📦 Loading FlashVLA Blackwell Engine (RTX 5090)...")
        
        # 🚀 THE ENGINE SWAP 🚀
        # Force the engine to use the IMTalker frontend class directly
        print("🚀 Forcing IMTalker Frontend (Bypassing Pi0.5 Default)...")
        self.model = IMTalkerTorchFrontendRtx(checkpoint_dir="./checkpoints")
        self._avatar_tensor_cache = {}
        
        print("✅ Blackwell SM120 Engine Online")
    
    def set_avatar(self, avatar_pil):
        """Pass the avatar to the FlashVLA engine to pre-compute reference latents."""
        s_pil = self.data_processor.process_img(avatar_pil)
        s = self.data_processor.transform_tensor(s_pil).unsqueeze(0).to(self.device)
        self.cached_avatar_tensor = s
        if not hasattr(self, "_avatar_tensor_cache"):
            self._avatar_tensor_cache = {}
        self._avatar_tensor_cache[id(avatar_pil)] = s

        if hasattr(self.model, 'set_avatar'):
            self.model.set_avatar(s)
            print("✅ Pre‑computed reference latents for avatar (FlashVLA cache)")
        else:
            print("✅ Avatar tensor cached for inference")
            
    def warmup(self, avatar_path=None):
        """Deep warmup to fully initialize CUDA Graphs and custom kernels."""
        import tempfile
        import soundfile as sf
        
        print("🔥 Warming up RTX 5090 inference (FlashVLA Engine)...")
        start = time.time()
        
        if avatar_path and os.path.exists(avatar_path):
            avatar_img = Image.open(avatar_path).convert('RGB')
        else:
            avatar_img = Image.new('RGB', (256,256), color=(128,128,128))
            
        silent_audio = np.zeros(2 * self.opt.sampling_rate, dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fa:
            temp_audio = fa.name
        sf.write(temp_audio, silent_audio, self.opt.sampling_rate)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as fv:
            temp_video = fv.name
            
        try:
            self.generate(avatar_img, temp_audio, temp_video, nfe=2)
            print(f"✅ Deep warmup complete in {time.time()-start:.2f}s")
        except Exception as e:
            print(f"⚠️ Deep warmup failed: {e}")
        finally:
            for p in [temp_audio, temp_video]:
                if os.path.exists(p):
                    os.remove(p)
    
    def _build_ffmpeg_cmd(self, w, h, aud_path, audio_duration, output_path, use_nvenc=None, include_audio=True, fragmented=False, target_video_sec=None):
        if use_nvenc is None:
            use_nvenc = os.getenv("IMTALKER_NVENC", "1").lower() not in ("0", "false", "no")

        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{w}x{h}', '-pix_fmt', 'rgb24', '-r', str(self.opt.fps),
            '-i', 'pipe:0',
        ]
        if include_audio:
            cmd.extend(['-i', aud_path])

        if use_nvenc:
            nvenc_cq = os.getenv("IMTALKER_NVENC_CQ", "16")
            nvenc_preset = os.getenv("IMTALKER_NVENC_PRESET", "p4")
            cmd.extend([
                '-c:v', 'h264_nvenc', '-preset', nvenc_preset, '-tune', 'll',
                '-rc', 'vbr', '-cq', str(nvenc_cq), '-b:v', '0',
                '-profile:v', 'baseline', '-level:v', '3.1', '-bf', '0', '-g', '25',
                '-pix_fmt', 'yuv420p',
            ])
        else:
            cmd.extend([
                '-c:v', 'libx264', '-profile:v', 'baseline', '-level:v', '3.1',
                '-bf', '0', '-g', '25',
                '-pix_fmt', 'yuv420p', '-preset', self.opt.ffmpeg_preset,
                '-crf', str(self.opt.ffmpeg_crf),
            ])

        if include_audio:
            if fragmented:
                # Match exact video frame duration & fixed 48 kHz mono AAC-LC so MSE
                # SourceBuffer('sequence') stitches consecutive fMP4 chunks with zero A/V drift.
                dur = target_video_sec if target_video_sec and target_video_sec > 0 else audio_duration
                fade_d = min(0.015, max(0.005, dur * 0.1))
                af_chain = f'apad,atrim=0:{dur:.6f},afade=t=in:ss=0:d={fade_d:.4f},afade=t=out:st={max(0.0, dur - fade_d):.6f}:d={fade_d:.4f}'
                cmd.extend(['-c:a', 'aac', '-ar', '48000', '-ac', '1', '-b:a', self.opt.ffmpeg_audio_bitrate, '-af', af_chain, '-shortest'])
            else:
                fade_duration = getattr(self.opt, 'audio_fade_duration', 0.05)
                afade = f'afade=t=in:ss=0:d={fade_duration},afade=t=out:st={max(0, audio_duration-fade_duration)}:d={fade_duration}'
                cmd.extend(['-c:a', 'aac', '-b:a', self.opt.ffmpeg_audio_bitrate, '-af', afade, '-shortest'])
        else:
            cmd.append('-an')

        if fragmented:
            cmd.extend(['-movflags', 'frag_keyframe+empty_moov+default_base_moof', output_path, '-loglevel', 'error'])
        else:
            cmd.extend(['-movflags', '+faststart', output_path, '-loglevel', 'error'])
        return cmd

    def generate_fast(self, img_pil, aud_path, output_path, fragmented=False, motion_state=None, include_audio=True):
        """Fast generation with clean lip-sync (default NFE=6; override with IMTALKER_FAST_NFE)."""
        fast_nfe = max(2, int(os.getenv("IMTALKER_FAST_NFE", "6")))
        return self.generate(
            img_pil, aud_path, output_path, nfe=fast_nfe, is_first_chunk=True,
            include_audio=include_audio, fragmented=fragmented, motion_state=motion_state
        )
    
    def generate(self, img_pil, aud_path, output_path, nfe=None, is_first_chunk=False, include_audio=True, fragmented=False, motion_state=None):
        import math
        import time
        import queue
        import threading
        import subprocess as sp
        
        if nfe is None:
            nfe = self.opt.nfe

        t0 = time.time()
        
        # Use per-image cached tensor if available (supports instant multi-avatar switching)
        tensor_cache = getattr(self, "_avatar_tensor_cache", None)
        if tensor_cache is not None and img_pil is not None and id(img_pil) in tensor_cache:
            s = tensor_cache[id(img_pil)]
        elif img_pil is not None and tensor_cache is not None and len(tensor_cache) > 0:
            s_pil = self.data_processor.process_img(img_pil)
            s = self.data_processor.transform_tensor(s_pil).unsqueeze(0).to(self.device)
            if len(tensor_cache) >= 8:
                tensor_cache.pop(next(iter(tensor_cache)))
            tensor_cache[id(img_pil)] = s
        elif hasattr(self, 'cached_avatar_tensor'):
            s = self.cached_avatar_tensor
        else:
            s_pil = self.data_processor.process_img(img_pil)
            s = self.data_processor.transform_tensor(s_pil).unsqueeze(0).to(self.device)
            
        t1 = time.time()
        a = self.data_processor.process_audio(aud_path).unsqueeze(0).to(self.device)
        t2 = time.time()
        print(f"⏱️  [profile] img process: {t1-t0:.3f}s, audio: {t2-t1:.3f}s")
        
        with torch.no_grad():
            w, h = self.opt.video_resolution
            audio_duration = a.shape[-1] / self.opt.sampling_rate
            target_video_sec = math.ceil(a.shape[-1] * self.opt.fps / self.opt.sampling_rate) / self.opt.fps
            cmd = self._build_ffmpeg_cmd(
                w, h, aud_path, audio_duration, output_path,
                include_audio=include_audio, fragmented=fragmented, target_video_sec=target_video_sec
            )
            proc = sp.Popen(cmd, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)
            
            torch.manual_seed(self.opt.seed)
            if hasattr(self.model, 'infer_uint8_batches'):
                write_q = queue.Queue(maxsize=4)
                writer_err = [None]

                def _pipe_writer():
                    try:
                        while True:
                            chunk = write_q.get()
                            if chunk is None:
                                break
                            proc.stdin.write(memoryview(chunk))
                    except Exception as exc:
                        writer_err[0] = exc
                    finally:
                        try:
                            proc.stdin.close()
                        except Exception:
                            pass

                writer = threading.Thread(target=_pipe_writer, daemon=True)
                writer.start()
                try:
                    for batch_u8 in self.model.infer_uint8_batches(
                        image=s,
                        audio=a,
                        nfe=nfe,
                        cfg_scale=self.opt.a_cfg_scale,
                        seed=self.opt.seed,
                        motion_state=motion_state,
                    ):
                        write_q.put(batch_u8)
                finally:
                    write_q.put(None)
                    writer.join()
                _, stderr_bytes = proc.communicate()
                if proc.returncode != 0 or writer_err[0] is not None:
                    err_txt = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else str(writer_err[0])
                    print(f"⚠️ FFmpeg encode error (rc={proc.returncode}): {err_txt}")
            else:
                generated_frames = self.model.predict(
                    image=s,
                    audio=a,
                    nfe=nfe,
                    cfg_scale=self.opt.a_cfg_scale,
                    seed=self.opt.seed,
                    motion_state=motion_state,
                )
                for frame_tensor in generated_frames:
                    frame_np = frame_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    frame_np = np.clip(frame_np, 0, 1) * 255
                    proc.stdin.write(frame_np.astype(np.uint8).tobytes())
                proc.stdin.close()
                proc.communicate()

            return output_path if os.path.exists(output_path) else None
    
    def generate_with_driver(self, img_pil, aud_path, driver_video_path, output_path, nfe=None, is_first_chunk=False):
        import time
        import subprocess as sp
        import inspect
        
        if nfe is None:
            nfe = self.opt.nfe

        if driver_video_path is None or not os.path.exists(driver_video_path):
            print("ℹ️ Driver video path missing or unavailable – using audio-only generation")
            return self.generate(img_pil, aud_path, output_path, nfe, is_first_chunk)

        print(f"🎬 Using driver video for guided idle/baseline motion: {driver_video_path}")

        t0 = time.time()
        
        # Use per-image cached tensor if available (supports instant multi-avatar switching)
        tensor_cache = getattr(self, "_avatar_tensor_cache", None)
        if tensor_cache is not None and img_pil is not None and id(img_pil) in tensor_cache:
            s = tensor_cache[id(img_pil)]
        elif img_pil is not None and tensor_cache is not None and len(tensor_cache) > 0:
            s_pil = self.data_processor.process_img(img_pil)
            s = self.data_processor.transform_tensor(s_pil).unsqueeze(0).to(self.device)
            if len(tensor_cache) >= 8:
                tensor_cache.pop(next(iter(tensor_cache)))
            tensor_cache[id(img_pil)] = s
        elif hasattr(self, 'cached_avatar_tensor'):
            s = self.cached_avatar_tensor
        else:
            s_pil = self.data_processor.process_img(img_pil)
            s = self.data_processor.transform_tensor(s_pil).unsqueeze(0).to(self.device)
            
        t1 = time.time()
        a = self.data_processor.process_audio(aud_path).unsqueeze(0).to(self.device)
        t2 = time.time()
        print(f"⏱️  [profile] img process: {t1-t0:.3f}s, audio: {t2-t1:.3f}s")
        
        with torch.no_grad():
            w, h = self.opt.video_resolution
            audio_duration = a.shape[-1] / self.opt.sampling_rate
            fade_duration = getattr(self.opt, 'audio_fade_duration', 0.05)
            afade = f'afade=t=in:ss=0:d={fade_duration},afade=t=out:st={max(0, audio_duration-fade_duration)}:d={fade_duration}'
            
            cmd = [
                'ffmpeg', '-y',
                '-f', 'rawvideo', '-vcodec', 'rawvideo',
                '-s', f'{w}x{h}', '-pix_fmt', 'rgb24', '-r', str(self.opt.fps),
                '-i', 'pipe:0',
                '-i', aud_path,
                '-c:v', 'libx264', '-profile:v', 'baseline', '-level:v', '3.0',
                '-pix_fmt', 'yuv420p', '-preset', self.opt.ffmpeg_preset,
                '-crf', str(self.opt.ffmpeg_crf), '-c:a', 'aac',
                '-b:a', self.opt.ffmpeg_audio_bitrate, '-af', afade,
                '-shortest', '-movflags', '+faststart', output_path, '-loglevel', 'error'
            ]
            proc = sp.Popen(cmd, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)
            
            # 🚀 FLASHVLA PREDICT WITH OPTIONAL DRIVER VIDEO
            # This keeps normal audio-only generation untouched and only adds
            # driver-video kwargs when the loaded frontend exposes them.
            torch.manual_seed(self.opt.seed)

            predict_kwargs = {
                "image": s,
                "audio": a,
                "nfe": nfe,
                "cfg_scale": self.opt.a_cfg_scale,
                "seed": self.opt.seed
            }

            try:
                predict_signature = inspect.signature(self.model.predict)
                predict_parameters = predict_signature.parameters

                if "driver_video_path" in predict_parameters:
                    predict_kwargs["driver_video_path"] = driver_video_path
                elif "driver_path" in predict_parameters:
                    predict_kwargs["driver_path"] = driver_video_path
                elif "driving_video_path" in predict_parameters:
                    predict_kwargs["driving_video_path"] = driver_video_path
                elif "guide_video_path" in predict_parameters:
                    predict_kwargs["guide_video_path"] = driver_video_path
                elif "source_video_path" in predict_parameters:
                    predict_kwargs["source_video_path"] = driver_video_path
                else:
                    print("⚠️ Loaded IMTalker frontend predict() has no driver-video parameter – using audio-only predict()")
            except Exception as e:
                print(f"⚠️ Could not inspect IMTalker predict() signature: {e}")

            if hasattr(self.model, 'generate_with_driver'):
                generated_frames = self.model.generate_with_driver(
                    image=s,
                    audio=a,
                    driver_video_path=driver_video_path,
                    nfe=nfe,
                    cfg_scale=self.opt.a_cfg_scale,
                    seed=self.opt.seed
                )
            elif hasattr(self.model, 'predict_with_driver'):
                generated_frames = self.model.predict_with_driver(
                    image=s,
                    audio=a,
                    driver_video_path=driver_video_path,
                    nfe=nfe,
                    cfg_scale=self.opt.a_cfg_scale,
                    seed=self.opt.seed
                )
            else:
                generated_frames = self.model.predict(**predict_kwargs)
            
            # Write frames directly to FFmpeg pipe
            for frame_tensor in generated_frames:
                # Assuming engine yields standard (1, 3, H, W) tensors
                frame_np = frame_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                frame_np = np.clip(frame_np, 0, 1) * 255
                proc.stdin.write(frame_np.astype(np.uint8).tobytes())
            
            proc.stdin.close()
            proc.communicate()
            return output_path if os.path.exists(output_path) else None


# ==========================================
# 4. FASTAPI CLIENT (for non‑5090 GPUs)
# ==========================================
class FastAPIClient:
    def __init__(self, opt):
        self.opt = opt
        self._check_connection()
    
    def _check_connection(self):
        try:
            import requests
            r = requests.get(f"{self.opt.api_url}{self.opt.health_endpoint}", timeout=5)
            if r.status_code == 200:
                print(f"✅ Connected to FastAPI at {self.opt.api_url}")
                return True
        except Exception as e:
            print(f"⚠️ FastAPI server unreachable: {e}")
        return False
    
    def generate(self, img_pil, aud_path, output_path, nfe=None, is_first_chunk=False):
        import requests
        nfe = nfe or self.opt.nfe
        img_bytes = io.BytesIO()
        img_pil.save(img_bytes, format='PNG')
        img_b64 = base64.b64encode(img_bytes.getvalue()).decode()
        with open(aud_path, 'rb') as f:
            aud_b64 = base64.b64encode(f.read()).decode()
        try:
            resp = requests.post(
                f"{self.opt.api_url}{self.opt.generate_endpoint}",
                json={
                    "image_base64": img_b64,
                    "audio_base64": aud_b64,
                    "crop": self.opt.crop,
                    "seed": self.opt.seed,
                    "nfe": nfe,
                    "cfg_scale": self.opt.a_cfg_scale
                },
                timeout=120
            )
            if resp.status_code == 200:
                data = resp.json()
                if 'video_url' in data:
                    vid_resp = requests.get(f"{self.opt.api_url}{data['video_url']}", timeout=60)
                    if vid_resp.status_code == 200:
                        with open(output_path, 'wb') as f:
                            f.write(vid_resp.content)
                        return output_path
                elif 'video_base64' in data:
                    with open(output_path, 'wb') as f:
                        f.write(base64.b64decode(data['video_base64']))
                    return output_path
            print(f"❌ API error {resp.status_code}")
        except Exception as e:
            print(f"❌ API exception: {e}")
        return None
    
    def generate_fast(self, img_pil, aud_path, output_path):
        return self.generate(img_pil, aud_path, output_path, nfe=6)
    
    def generate_with_driver(self, img_pil, aud_path, driver_video_path, output_path, nfe=None, is_first_chunk=False):
        print("⚠️ Driver video not supported in API mode – using audio-only")
        return self.generate(img_pil, aud_path, output_path, nfe, is_first_chunk)
    
    def warmup(self, avatar_path=None):
        print("🌐 FastAPI warmup not needed")
        return True
    
    def set_avatar(self, avatar_pil):
        print("⚠️ FastAPI client does not support precomputed avatar latents")


# ==========================================
# 5. UNIFIED INFERENCE AGENT
# ==========================================
class InferenceAgent:
    def __init__(self, opt=None):
        if opt is None:
            opt = AppConfig()
        self.opt = opt
        gpu_info = opt.gpu_info
        print("\n" + "="*60)
        print(f"🎮 GPU: {gpu_info['name']} ({gpu_info['vram_gb']} GB, SM{''.join(map(str,gpu_info['capability']))})")
        if opt.use_local_inference:
            print("🚀 Using LOCAL INFERENCE (RTX 5090 / FlashVLA)")
            self.engine = LocalInferenceAgent(opt)
        else:
            print("🌐 Using FASTAPI SERVER (non‑Blackwell GPU)")
            self.engine = FastAPIClient(opt)
        self.data_processor = DataProcessor(opt)
        print("="*60 + "\n")
    
    def warmup(self, avatar_path=None):
        return self.engine.warmup(avatar_path)
    
    def set_avatar(self, avatar_pil):
        """Pre‑compute reference latents for the avatar (local inference only)."""
        if self.opt.use_local_inference:
            self.engine.set_avatar(avatar_pil)
        else:
            print("⚠️ FastAPI client does not support precomputed avatar latents")
    
    def generate(self, img_pil, aud_path, output_path, nfe=None, is_first_chunk=False, **kwargs):
        return self.engine.generate(img_pil, aud_path, output_path, nfe, is_first_chunk, **kwargs)
    
    def generate_fast(self, img_pil, aud_path, output_path, **kwargs):
        return self.engine.generate_fast(img_pil, aud_path, output_path, **kwargs)
    
    def generate_with_driver(self, img_pil, aud_path, driver_video_path=None, output_path=None, nfe=None, is_first_chunk=False):
        if driver_video_path is None:
            driver_video_path = get_default_driver_path()
            if driver_video_path:
                print(f"🎬 Auto-detected driver video: {driver_video_path}")
            else:
                print("ℹ️ No default driver video found – using audio-only generation")
        return self.engine.generate_with_driver(img_pil, aud_path, driver_video_path, output_path, nfe, is_first_chunk)
    
    def try_safe_compile(self):
        return False


# ==========================================
# 6. MAIN (example usage)
# ==========================================
if __name__ == "__main__":
    opt = AppConfig()
    agent = InferenceAgent(opt)
    avatar = "avatar.png" if os.path.exists("avatar.png") else None
    agent.warmup(avatar)
    print("\n✅ IMTalker FlashVLA Engine Ready!")
    print(f"   Mode: {'LOCAL (5090 / FlashVLA)' if opt.use_local_inference else 'FASTAPI'}")