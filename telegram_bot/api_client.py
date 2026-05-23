"""Тонкий клиент над FastAPI gateway. Используется ботом."""
from __future__ import annotations

import asyncio
from typing import Any

import httpx

from config import bot_settings


class FightVisionAPI:
    def __init__(self, base_url: str | None = None) -> None:
        self.base = (base_url or bot_settings.fv_backend_url).rstrip("/")
        self._timeout = httpx.Timeout(connect=10.0, read=300.0, write=300.0, pool=10.0)

    async def upload_video(self, file_bytes: bytes, filename: str) -> str:
        """Загружает видео в gateway. Возвращает job_id."""
        url = f"{self.base}/api/analyze"
        files = {"file": (filename, file_bytes, "video/mp4")}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            r = await client.post(url, files=files)
            r.raise_for_status()
            data = r.json()
            return data["job_id"]

    async def get_status(self, job_id: str) -> dict[str, Any]:
        url = f"{self.base}/api/jobs/{job_id}"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            r = await client.get(url)
            r.raise_for_status()
            return r.json()

    async def get_result(self, job_id: str) -> dict[str, Any]:
        url = f"{self.base}/api/jobs/{job_id}/result"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            r = await client.get(url)
            r.raise_for_status()
            return r.json()

    async def download(self, media_path: str) -> bytes:
        """media_path вида '/media/results/{job}/highlights.mp4'."""
        url = f"{self.base}{media_path}"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            r = await client.get(url)
            r.raise_for_status()
            return r.content

    async def wait_for_completion(
        self,
        job_id: str,
        on_progress=None,
    ) -> dict[str, Any]:
        """Опрашивает статус до status=done|failed. Вызывает on_progress(status_dict) на каждом тике."""
        elapsed = 0
        last_progress_int = -1

        while elapsed < bot_settings.fv_bot_poll_timeout:
            status = await self.get_status(job_id)

            if on_progress is not None:
                # вызываем callback только если процент существенно изменился
                progress_int = int(status.get("progress", 0.0) * 100)
                if progress_int != last_progress_int:
                    try:
                        await on_progress(status)
                    except Exception as exc:  # noqa: BLE001
                        # не ломаем pipeline если callback упал
                        print(f"[bot] on_progress callback failed: {exc}")
                    last_progress_int = progress_int

            if status["status"] == "done":
                return await self.get_result(job_id)
            if status["status"] == "failed":
                raise RuntimeError(status.get("error") or "Processing failed")

            await asyncio.sleep(bot_settings.fv_bot_poll_interval)
            elapsed += bot_settings.fv_bot_poll_interval

        raise TimeoutError(f"Job {job_id} not completed within {bot_settings.fv_bot_poll_timeout}s")
