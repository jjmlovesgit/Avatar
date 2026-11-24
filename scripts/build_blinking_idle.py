"""
Generate a seamless, practically-still, closed-lip, naturally-blinking idle loop
for any aligned 512x512 avatar portrait using IMTalker's motion-token reenactment.

Example:
    python scripts/build_blinking_idle.py \
        --avatar backend/puck_avatar.png \
        --driver backend/driver_video/driver_video.mp4 \
        --output backend/idle_puck.mp4
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
os.environ.setdefault("IMTALKER_BF16", "0")
sys.path.insert(0, str(BACKEND_DIR))


def build_idle(
    avatar_path: Path,
    driver_path: Path,
    output_path: Path,
    base_gain: float = 0.12,
    blink_gain: float = 1.40,
    start_frame: int = 80,
    num_frames: int = 125,
):
    orig_cwd = os.getcwd()
    os.chdir(str(BACKEND_DIR))
    try:
        from imtalker_core import AppConfig, InferenceAgent

        opt = AppConfig()
        agent = InferenceAgent(opt)
        dp = agent.engine.data_processor
        fe = agent.engine.model
        r = fe.renderer

        avatar_256 = Image.open(avatar_path).convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
        s_pil = dp.process_img(avatar_256)
        s = dp.transform_tensor(s_pil).unsqueeze(0).cuda()

        raw = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-i", str(driver_path),
                "-vf", "fps=25,scale=512:512", "-frames:v", str(start_frame + num_frames + 15),
                "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
            ],
            capture_output=True,
            check=True,
        ).stdout
        drv_512 = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 512, 512, 3)

        # Single fixed face crop matching IMTalker's portrait framing so blink magnitude transfers cleanly
        lm = dp.fa.get_landmarks(drv_512[min(135, len(drv_512) - 1)])[0]
        xs, ys = lm[:, 0], lm[:, 1]
        cx, cy = float(xs.mean()), float(ys.mean())
        size = int(max(float(xs.max() - xs.min()), float(ys.max() - ys.min())) * 1.95)
        x1 = max(0, int(cx - size / 2))
        y1 = max(0, int(cy - size * 0.56))
        x2 = min(512, x1 + size)
        y2 = min(512, y1 + size)

        cropped = np.stack(
            [
                np.array(Image.fromarray(f[y1:y2, x1:x2]).resize((256, 256), Image.Resampling.LANCZOS))
                for f in drv_512
            ],
            axis=0,
        )
        drv_t = torch.from_numpy(cropped.copy()).permute(0, 3, 1, 2).float().div_(255.0).cuda()

        with torch.no_grad():
            f_r, g_r, t_r, m_r = fe._get_or_encode_face(s)

            t_drv_list = [r.mot_encode(drv_t[i : i + 16]) for i in range(0, len(drv_t), 16)]
            t_drv_all = torch.cat(t_drv_list, dim=0)

            t_slice = t_drv_all[start_frame : start_frame + num_frames]
            open_anchor = torch.cat([t_slice[54:58], t_slice[67:71]], dim=0).mean(dim=0, keepdim=True)
            delta = t_slice - open_anchor

            # Raised-cosine endpoint envelope -> exact matching start & end frames
            fade = 22.0
            env = np.ones(num_frames, dtype=np.float32)
            for i in range(num_frames):
                if i < fade:
                    env[i] = 0.5 * (1.0 - np.cos(np.pi * i / fade))
                elif i > num_frames - 1 - fade:
                    env[i] = 0.5 * (1.0 - np.cos(np.pi * (num_frames - 1 - i) / fade))
            env[0] = 0.0
            env[-1] = 0.0

            # Smooth gain bell over the driver blink window (local frames 59..65)
            blink_bell = np.zeros(num_frames, dtype=np.float32)
            for i in range(num_frames):
                if 54 <= i < 59:
                    blink_bell[i] = 0.5 * (1.0 - np.cos(np.pi * (i - 54) / 5.0))
                elif 59 <= i <= 65:
                    blink_bell[i] = 1.0
                elif 65 < i <= 70:
                    blink_bell[i] = 0.5 * (1.0 - np.cos(np.pi * (70 - i) / 5.0))

            gain_curve = (base_gain + (blink_gain - base_gain) * blink_bell) * env
            t_target = t_r + torch.from_numpy(gain_curve).cuda().unsqueeze(1) * delta

            out_batches = []
            for i in range(0, num_frames, 8):
                chunk = t_target[i : i + 8]
                bs = len(chunk)
                ta = r.adapt(chunk, g_r.expand(bs, -1))
                mc = r.latent_token_decoder(ta)
                out = r.decode(
                    mc,
                    [m.expand(bs, -1, -1, -1) for m in m_r],
                    [f.expand(bs, -1, -1, -1) for f in f_r],
                )
                u8 = out.float().clamp_(0, 1).mul_(255).to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()
                out_batches.append(u8)
            frames = np.concatenate(out_batches, axis=0)
            frames[-1] = frames[0]

        h, w = frames.shape[1], frames.shape[2]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}", "-pix_fmt", "rgb24", "-r", "25",
            "-i", "pipe:0",
            "-c:v", "libx264", "-profile:v", "baseline", "-level:v", "3.1",
            "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "medium",
            "-g", "25", "-an", "-movflags", "+faststart",
            str(output_path),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        proc.stdin.write(frames.tobytes())
        proc.stdin.close()
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {ret}")
        print(f"Saved seamless {num_frames/25:.2f}s idle loop ({w}x{h}) -> {output_path}")
    finally:
        os.chdir(orig_cwd)


def main():
    parser = argparse.ArgumentParser(description="Build a seamless blinking idle MP4 for an avatar portrait")
    parser.add_argument("--avatar", type=Path, required=True, help="Path to 512x512 avatar PNG")
    parser.add_argument("--driver", type=Path, default=BACKEND_DIR / "driver_video" / "driver_video.mp4")
    parser.add_argument("--output", type=Path, required=True, help="Output MP4 path")
    parser.add_argument("--base-gain", type=float, default=0.12)
    parser.add_argument("--blink-gain", type=float, default=1.40)
    args = parser.parse_args()
    build_idle(args.avatar.resolve(), args.driver.resolve(), args.output.resolve(), args.base_gain, args.blink_gain)


if __name__ == "__main__":
    main()
