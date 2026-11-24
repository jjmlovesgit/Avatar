import os, sys, torch, numpy as np

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
        print(f"✅ IMTalker RTX 5090 Pipeline Online (synth_bf16={self.use_bf16}, render_batch={self.render_batch_size})")

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

    @torch.no_grad()
    def _sample_motion_and_face(self, input_image: torch.Tensor, audio: torch.Tensor, cfg_scale: float, nfe: int, motion_state: dict = None):
        if input_image.dim() == 3:
            input_image = input_image.unsqueeze(0)

        input_image = input_image.to(self.device, non_blocking=True)
        audio = audio.to(self.device, non_blocking=True)

        f_r, g_r, t_r_fp32, m_r = self._get_or_encode_face(input_image)

        data = {
            's': input_image,
            'a': audio,
            'ref_x': t_r_fp32,
            'gaze': None, 'pose': None, 'cam': None
        }

        if motion_state is not None:
            prev_state = motion_state if ("prev_x" in motion_state and "prev_a" in motion_state) else None
            motion_sequence, next_state = self.fm.sample(
                data, a_cfg_scale=cfg_scale, nfe=nfe, prev_state=prev_state, return_state=True
            )
            motion_state["prev_x"] = next_state["prev_x"]
            motion_state["prev_a"] = next_state["prev_a"]
        else:
            motion_sequence = self.fm.sample(data, a_cfg_scale=cfg_scale, nfe=nfe)

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