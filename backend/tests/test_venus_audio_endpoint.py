"""Exercise the Venus audio bridge without loading the avatar checkpoint."""

import asyncio
import io
from pathlib import Path
import tempfile
import unittest
import wave

import httpx

import python_avatar_server_electron as server


def pcm16_wav(channels=1, sample_rate=24_000, seconds=0.25):
    audio = io.BytesIO()
    with wave.open(audio, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\0\0" * int(sample_rate * seconds) * channels)
    return audio.getvalue()


class FakeAgent:
    def generate_fast(self, image, audio_path, video_path):
        assert Path(audio_path).is_file()
        Path(video_path).write_bytes(b"fake mp4")
        return video_path


class VenusAudioEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.saved = (server.agent, server.avatar_image, server.GPU_SEQUENTIAL_MODE,
                      server.CHUNK_DIR, server.avatar_audio_jobs)
        self.temp_dir = tempfile.TemporaryDirectory()
        server.agent = FakeAgent()
        server.avatar_image = object()
        server.GPU_SEQUENTIAL_MODE = False
        server.CHUNK_DIR = self.temp_dir.name
        server.avatar_audio_jobs = {}
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://testserver"
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        (server.agent, server.avatar_image, server.GPU_SEQUENTIAL_MODE,
         server.CHUNK_DIR, server.avatar_audio_jobs) = self.saved
        self.temp_dir.cleanup()

    async def test_upload_status_and_cleanup(self):
        response = await self.client.post(
            "/avatar/audio", files={"audio": ("speech.wav", pcm16_wav(), "audio/wav")},
            data={"turn_id": "venus-turn-1", "chunk_index": "2", "fast": "true"},
        )
        self.assertEqual(response.status_code, 202, response.text)
        job = response.json()
        self.assertEqual(job["turn_id"], "venus-turn-1")
        self.assertEqual(job["chunk_index"], 2)
        for _ in range(100):
            status = (await self.client.get(job["status_url"])).json()
            if status["status"] == "ready":
                break
            await asyncio.sleep(0.01)
        self.assertEqual(status["status"], "ready", status)
        self.assertTrue(Path(self.temp_dir.name, f"venus_{job['job_id']}.mp4").is_file())
        self.assertFalse(Path(self.temp_dir.name, f"venus_{job['job_id']}.wav").exists())
        self.assertEqual((await self.client.delete(job["status_url"])).status_code, 200)
        self.assertEqual((await self.client.get(job["status_url"])).status_code, 404)

    async def test_rejects_stereo_and_unready_renderer(self):
        response = await self.client.post(
            "/avatar/audio", files={"audio": ("stereo.wav", pcm16_wav(channels=2), "audio/wav")},
        )
        self.assertEqual(response.status_code, 422)
        server.agent = None
        response = await self.client.post(
            "/avatar/audio", files={"audio": ("speech.wav", pcm16_wav(), "audio/wav")},
        )
        self.assertEqual(response.status_code, 503)

    async def test_rejects_nonlocal_upload(self):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app, client=("198.51.100.2", 12345)),
            base_url="http://testserver",
        ) as remote:
            response = await remote.post(
                "/avatar/audio", files={"audio": ("speech.wav", pcm16_wav(), "audio/wav")},
            )
        self.assertEqual(response.status_code, 403)


if __name__ == "__main__":
    unittest.main()
