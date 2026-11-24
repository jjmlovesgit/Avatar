"""Run only the local avatar renderer needed by Realtime-Venus-Audio."""

import os
import sys

# Ensure emoji/status logs never fail on cp1252 Windows consoles.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Must be set before importing the server and its optional voice stack.
os.environ["VENUS_AVATAR_ONLY"] = "1"
os.environ.setdefault(
    "VENUS_AVATAR_IMAGE",
    r"C:\Projects\Realtime-Venus-Audio\assets\avatar.png",
)

from python_avatar_server_electron import app  # noqa: E402

# Keep only the Venus audio-to-video routes and /chunks static video serving.
_ROUTES = {
    "/avatar/audio", "/avatar/audio/{job_id}", "/chunks",
    "/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc",
}
app.router.routes[:] = [route for route in app.router.routes if route.path in _ROUTES]


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("VENUS_AVATAR_PORT", "8002"))
    print(f"Venus avatar video endpoint: http://127.0.0.1:{port}/avatar/audio", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port)
