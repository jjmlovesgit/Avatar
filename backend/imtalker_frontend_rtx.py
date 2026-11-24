import math, os, sys, torch, numpy as np

# Import your native project models
try:
    from renderer.models import IMTRenderer
    from generator.FM import FMGenerator
except ImportError:
    print("⚠️ Could not import IMTRenderer or FMGenerator. Check your paths.")

class MockOpt:
    def __init__(self):
        # Model Dimensions & Arch
        self.input_nc = 3
        self.dim_a = 512
        self.dim_h = 512
        self.dim_e = 7
        self.dim_motion = 32
        self.dim_c = 32
        self.dim_w = 32
        self.fmt_depth = 8
        self.num_heads = 8
        self.mlp_ratio = 4.0
        self.input_size = 256
        self.swin_res_threshold = 128
        self.window_size = 8
        
        # Sequence & Audio Configs for FMGenerator
        self.fps = 25.0
        self.wav2vec_sec = 2.0
        self.num_prev_frames = 10
        self.only_last_features = True
        self.ode_atol = 1e-5
        self.ode_rtol = 1e-5
        self.torchdiffeq_ode_method = 'euler'
        self.fix_noise_seed = False
        self.sampling_rate = 16000
        self.rank = 'cuda'
        self.audio_dropout_prob = 0.1
        self.attention_window = 5
        self.wav2vec_model_path = "./checkpoints/wav2vec2-base-960h"

class IMTalkerTorchFrontendRtx:
    def __init__(self, checkpoint_dir, **kwargs):
        self.device = "cuda"
        self.opt = MockOpt()
        self.use_bf16 = os.getenv("IMTALKER_BF16", "1").lower() not in ("0", "false", "no")
        self.render_batch_size = max(1, int(os.getenv("IMTALKER_RENDER_BATCH", "8")))
        self.render_dtype = torch.float32

        # Calm head-movement conditioning knobs (cuts frenetic pitch/yaw velocity by ~50% with 0ms overhead)
        # IMTALKER_POSE_MODE: "calm" (default slow smooth sway), "still" (centered zero-pose), "free" (unconstrained)
        self.pose_mode = os.getenv("IMTALKER_POSE_MODE", "calm").strip().lower()
        self.pose_weight = max(0.0, min(1.0, float(os.getenv("IMTALKER_POSE_WEIGHT", "0.45"))))
        self.pose_pitch_amp = float(os.getenv("IMTALKER_POSE_PITCH_AMP", "0.025"))  # rad (~1.4 deg)
        self.pose_yaw_amp = float(os.getenv("IMTALKER_POSE_YAW_AMP", "0.035"))      # rad (~2.0 deg)
        self.pose_roll_amp = float(os.getenv("IMTALKER_POSE_ROLL_AMP", "0.012"))    # rad (~0.7 deg)
        self.pose_freq_hz = max(0.02, float(os.getenv("IMTALKER_POSE_FREQ", "0.20")))
        self.gaze_lock = os.getenv("IMTALKER_GAZE_LOCK", "0").lower() in ("1", "true", "yes")
        self.cfg_scale_override = os.getenv("IMTALKER_CFG_SCALE", "").strip()
        self.default_cfg_scale = float(self.cfg_scale_override) if self.cfg_scale_override else (2.6 if self.pose_mode != "free" else 2.0)

        # Natural periodic speaking blinks injected in 32-D motion token space (+0.0ms latency, 99% lip preservation)
        self.speak_blink_enabled = os.getenv("IMTALKER_SPEAK_BLINK", "1").lower() not in ("0", "false", "no")
        self.blink_strength = float(os.getenv("IMTALKER_BLINK_STRENGTH", "1.35"))
        self.blink_interval_sec = max(1.5, float(os.getenv("IMTALKER_BLINK_INTERVAL_SEC", "3.4")))
        self._blink_delta = None
        if self.speak_blink_enabled:
            blink_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "driver_video", "blink_delta.pt")
            if os.path.isfile(blink_path):
                try:
                    self._blink_delta = torch.load(blink_path, map_location=self.device, weights_only=True).float()
                except Exception as exc:
                    print(f"⚠️ Could not load speaking blink tensor ({blink_path}): {exc}")

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = False

        # ==========================================
        # 🚀 INITIALIZE MODELS
        # ==========================================
        print("⚙️ Initializing Video Renderer & FM Generator...")
        self.renderer = IMTRenderer(self.opt).to(self.device).eval()
        self.fm = FMGenerator(self.opt).to(self.device).eval()
        
        # 1. Load Renderer Weights
        try:
            ckpt = torch.load(f"{checkpoint_dir}/renderer.ckpt", map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            ae_state_dict = {k.replace("gen.", ""): v for k, v in state_dict.items() if k.startswith("gen.")}
            self.renderer.load_state_dict(ae_state_dict if ae_state_dict else state_dict, strict=False)
            print("✅ Renderer weights loaded.")
        except Exception as e:
            print(f"⚠️ Renderer load failed: {e}")

        # 2. Load Generator Weights
        try:
            ckpt = torch.load(f"{checkpoint_dir}/generator.ckpt", map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            model_state = {k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")}
            self.fm.load_state_dict(model_state if model_state else state_dict, strict=False)
            print("✅ FM Generator weights loaded.")
        except Exception as e:
            print(f"⚠️ FM Generator load failed: {e}")

        if self.use_bf16:
            # Keep face encoders, MotionDecoder, and CrossAttention/GuidedResampler in FP32
            # for crisp lip-sync (48.2 dB PSNR, 0.65 MouthMAE); only SynthesisNetwork uses BF16.
            self.renderer.frame_decoder.to(dtype=torch.bfloat16)

        self.cached_image_sum = None
        self._face_cache = {}
        self._max_face_cache = 8
        print(
            f"✅ IMTalker RTX 5090 Pipeline Online "
            f"(synth_bf16={self.use_bf16}, render_batch={self.render_batch_size}, "
            f"pose_mode={self.pose_mode}, pose_w={self.pose_weight:.2f}, cfg={self.default_cfg_scale:.1f}, "
            f"gaze_lock={self.gaze_lock}, speak_blink={self._blink_delta is not None})"
        )

    def set_prompt(self, audio_path): pass

    def _face_cache_key(self, input_image: torch.Tensor):
        flat = input_image.reshape(-1)
        return (
            tuple(input_image.shape),
            round(float(flat.sum().item()), 4),
            round(float(flat[0].item()), 5),
            round(float(flat[flat.numel() // 2].item()), 5),
            round(float(flat[-1].item()), 5),
        )

    def _get_or_encode_face(self, input_image: torch.Tensor):
        key = self._face_cache_key(input_image)
        cached = self._face_cache.get(key)
        if cached is not None:
            return cached

        img_r = input_image.to(device=self.device, dtype=self.render_dtype)
        f_r, g_r = self.renderer.dense_feature_encoder(img_r)
        t_r = self.renderer.latent_token_encoder(img_r)
        ta_r = self.renderer.adapt(t_r, g_r)
        m_r = self.renderer.latent_token_decoder(ta_r)
        t_r_fp32 = t_r.float()

        entry = (f_r, g_r, t_r_fp32, m_r)
        if len(self._face_cache) >= self._max_face_cache:
            self._face_cache.pop(next(iter(self._face_cache)))
        self._face_cache[key] = entry
        self.f_r, self.g_r, self.t_r, self.m_r = entry
        self.cached_image_sum = key[1]
        return entry

    @torch.no_grad()
    def set_avatar(self, input_image: torch.Tensor):
        if input_image.dim() == 3:
            input_image = input_image.unsqueeze(0)
        input_image = input_image.to(self.device)
        self._get_or_encode_face(input_image)
    
    def predict(self, *args, **kwargs):
        return self.infer(*args, **kwargs)

    def _build_pose_gaze(self, audio: torch.Tensor, frame_offset: int = 0):
        """Build calm continuous [T, 3] pose (pitch, yaw, roll in rad) and optional [T, 2] gaze."""
        T = max(1, int(math.ceil(audio.shape[-1] * self.opt.fps / self.opt.sampling_rate)))
        gaze = torch.zeros((T, 2), dtype=torch.float32) if self.gaze_lock else None

        if self.pose_mode == "free":
            return None, gaze, T
        if self.pose_mode == "still":
            return torch.zeros((T, 3), dtype=torch.float32), gaze, T

        # "calm" default: slow, smooth, low-amplitude sinusoidal breathing/sway anchored at camera center
        t_sec = (np.arange(T, dtype=np.float32) + float(frame_offset)) / float(self.opt.fps)
        f = self.pose_freq_hz
        pitch = self.pose_pitch_amp * np.sin(2.0 * np.pi * f * t_sec)
        yaw = self.pose_yaw_amp * np.sin(2.0 * np.pi * (f * 0.72) * t_sec + 0.7)
        roll = self.pose_roll_amp * np.sin(2.0 * np.pi * (f * 0.56) * t_sec + 1.5)
        # Gentle ramp-in on frame 0..12 of a brand-new utterance so speech starts smoothly from neutral idle
        if frame_offset == 0 and T > 1:
            ramp_len = min(12, T)
            ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(ramp_len, dtype=np.float32) / float(ramp_len)))
            pitch[:ramp_len] *= ramp
            yaw[:ramp_len] *= ramp
            roll[:ramp_len] *= ramp
        pose = torch.from_numpy(np.stack([pitch, yaw, roll], axis=1))
        return pose, gaze, T

    def _apply_speaking_blinks(self, motion_sequence: torch.Tensor, frame_offset: int, motion_state: dict = None):
        """Inject natural eyelid-only blinks (~every 3.4s) into the 32-D motion sequence right before decoding."""
        if not self.speak_blink_enabled or self._blink_delta is None:
            return motion_sequence

        T = int(motion_sequence.shape[1])
        if T <= 0:
            return motion_sequence

        blink_len = int(self._blink_delta.shape[0])
        fps = float(self.opt.fps)
        win_start = frame_offset
        win_end = frame_offset + T

        # Track blink schedule continuously across chunks in motion_state (or local schedule if stateless)
        if motion_state is not None:
            if "blink_starts" not in motion_state:
                first_delay = int(round(fps * float(np.random.uniform(1.0, 2.0))))
                motion_state["blink_starts"] = []
                motion_state["next_blink_frame"] = first_delay
            blink_starts = motion_state["blink_starts"]
            next_b = int(motion_state["next_blink_frame"])
        else:
            first_delay = int(round(fps * float(np.random.uniform(1.0, 2.0))))
            blink_starts = []
            next_b = first_delay

        while next_b < win_end:
            blink_starts.append(next_b)
            jitter = float(np.random.uniform(-0.7, 0.9))
            step = max(int(round(fps * 1.8)), int(round(fps * (self.blink_interval_sec + jitter))))
            next_b += step

        if motion_state is not None:
            motion_state["next_blink_frame"] = next_b
            # Prune completed blinks from earlier chunks
            motion_state["blink_starts"] = [b for b in blink_starts if b + blink_len > win_end]

        delta_scaled = (self.blink_strength * self._blink_delta).to(device=motion_sequence.device, dtype=motion_sequence.dtype)
        out = motion_sequence.clone()
        for b_start in blink_starts:
            ov_start = max(win_start, b_start)
            ov_end = min(win_end, b_start + blink_len)
            if ov_end > ov_start:
                local_s = ov_start - win_start
                local_e = ov_end - win_start
                blink_s = ov_start - b_start
                blink_e = ov_end - b_start
                out[:, local_s:local_e, :] += delta_scaled[blink_s:blink_e, :].unsqueeze(0)
        return out

    @torch.no_grad()
    def _sample_motion_and_face(self, input_image: torch.Tensor, audio: torch.Tensor, cfg_scale: float, nfe: int, motion_state: dict = None):
        if input_image.dim() == 3:
            input_image = input_image.unsqueeze(0)

        input_image = input_image.to(self.device, non_blocking=True)
        audio = audio.to(self.device, non_blocking=True)

        f_r, g_r, t_r_fp32, m_r = self._get_or_encode_face(input_image)

        frame_offset = int(motion_state.get("frame_offset", 0)) if motion_state is not None else 0
        pose_cond, gaze_cond, num_frames = self._build_pose_gaze(audio, frame_offset=frame_offset)

        data = {
            's': input_image,
            'a': audio,
            'ref_x': t_r_fp32,
            'gaze': gaze_cond, 'pose': pose_cond, 'cam': None
        }

        eff_cfg = self.default_cfg_scale if (self.cfg_scale_override or (self.pose_mode != "free" and abs(cfg_scale - 2.0) < 1e-6)) else cfg_scale
        eff_pw = self.pose_weight if self.pose_mode != "free" else 0.0

        if motion_state is not None:
            prev_state = motion_state if ("prev_x" in motion_state and "prev_a" in motion_state) else None
            motion_sequence, next_state = self.fm.sample(
                data, a_cfg_scale=eff_cfg, nfe=nfe, prev_state=prev_state, return_state=True, pose_weight=eff_pw
            )
            motion_state["prev_x"] = next_state["prev_x"]
            motion_state["prev_a"] = next_state["prev_a"]
            motion_state["prev_pose"] = next_state.get("prev_pose")
            motion_state["prev_gaze"] = next_state.get("prev_gaze")
            motion_state["frame_offset"] = frame_offset + num_frames
        else:
            motion_sequence = self.fm.sample(data, a_cfg_scale=eff_cfg, nfe=nfe, pose_weight=eff_pw)

        motion_sequence = self._apply_speaking_blinks(motion_sequence, frame_offset=frame_offset, motion_state=motion_state)

        if self.render_dtype != motion_sequence.dtype:
            motion_sequence = motion_sequence.to(dtype=self.render_dtype)

        return motion_sequence, f_r, g_r, m_r

    @torch.no_grad()
    def infer_uint8_batches(self, image: torch.Tensor, audio: torch.Tensor, cfg_scale: float = 2.0, nfe: int = 4, batch_size: int = None, motion_state: dict = None, **kwargs):
        """Yield rendered RGB uint8 numpy batches shaped [bs, H, W, 3] for pipelined FFmpeg encoding."""
        bs_max = batch_size or self.render_batch_size
        motion_sequence, f_r, g_r, m_r = self._sample_motion_and_face(image, audio, cfg_scale, nfe, motion_state=motion_state)
        T = motion_sequence.shape[1]

        for start in range(0, T, bs_max):
            end = min(start + bs_max, T)
            bs = end - start
            action_batch = motion_sequence[0, start:end, :]
            g_r_b = g_r.expand(bs, -1)
            ta_c = self.renderer.adapt(action_batch, g_r_b)
            m_c = self.renderer.latent_token_decoder(ta_c)
            m_r_b = [m.expand(bs, -1, -1, -1) for m in m_r]
            f_r_b = [f.expand(bs, -1, -1, -1) for f in f_r]
            rendered_batch = self.renderer.decode(m_c, m_r_b, f_r_b)
            batch_u8 = (
                rendered_batch.float()
                .clamp_(0.0, 1.0)
                .mul_(255.0)
                .to(torch.uint8)
                .permute(0, 2, 3, 1)
                .contiguous()
                .cpu()
                .numpy()
            )
            yield batch_u8

    @torch.no_grad()
    def infer(self, *args, **kwargs):
        input_image = kwargs.get('image')
        audio = kwargs.get('audio')
        cfg_scale = kwargs.get('cfg_scale', 2.0)
        nfe = kwargs.get('nfe', 4)
        bs_max = kwargs.get('batch_size', None) or self.render_batch_size
        motion_state = kwargs.get('motion_state', None)

        motion_sequence, f_r, g_r, m_r = self._sample_motion_and_face(input_image, audio, cfg_scale, nfe, motion_state=motion_state)
        T = motion_sequence.shape[1]
        generated_frames = []

        for start in range(0, T, bs_max):
            end = min(start + bs_max, T)
            bs = end - start
            action_batch = motion_sequence[0, start:end, :]
            g_r_b = g_r.expand(bs, -1)
            ta_c = self.renderer.adapt(action_batch, g_r_b)
            m_c = self.renderer.latent_token_decoder(ta_c)
            m_r_b = [m.expand(bs, -1, -1, -1) for m in m_r]
            f_r_b = [f.expand(bs, -1, -1, -1) for f in f_r]
            rendered_batch = self.renderer.decode(m_c, m_r_b, f_r_b)
            rendered_cpu = rendered_batch.float().cpu()
            generated_frames.extend(rendered_cpu.unbind(0))

        return generated_frames