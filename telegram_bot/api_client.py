"""Тонкий async-клиент над FastAPI gateway. Используется Telegram-ботом."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx

from config import bot_settings


ProgressCallback = Callable[[dict[str, Any]], Awaitable[None]]


class FightVisionAPI:
    def __init__(self, base_url: str | None = None) -> None:
        self.base = (base_url or bot_settings.fv_backend_url).rstrip("/")
        self._timeout = httpx.Timeout(connect=15.0, read=600.0, write=600.0, pool=15.0)

    async def health(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
            r = await client.get(f"{self.base}/health")
            r.raise_for_status()
            return r.json()

    async def upload_video_path(self, path: Path, filename: str) -> str:
        """Загружает видео в gateway из файла. Возвращает job_id."""
        url = f"{self.base}/api/analyze"
        content_type = "video/mp4"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            with path.open("rb") as fh:
                files = {"file": (filename, fh, content_type)}
                r = await client.post(url, files=files)
            r.raise_for_status()
            return r.json()["job_id"]

    async def get_status(self, job_id: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            r = await client.get(f"{self.base}/api/jobs/{job_id}")
            r.raise_for_status()
            return r.json()

    async def get_result(self, job_id: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            r = await client.get(f"{self.base}/api/jobs/{job_id}/result")
            r.raise_for_status()
            return r.json()

    async def download_to_path(self, media_path: str, out_path: Path) -> int:
        """Скачивает /media/... в файл. Возвращает размер в байтах."""
        url = f"{self.base}{media_path}"
        size = 0
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream("GET", url) as r:
                r.raise_for_status()
                with out_path.open("wb") as fh:
                    async for chunk in r.aiter_bytes():
                        if chunk:
                            size += len(chunk)
                            fh.write(chunk)
        return size

    async def wait_for_completion(
        self,
        job_id: str,
        on_progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        """Опрашивает статус до done/failed. on_progress вызывается при изменении процента."""
        elapsed = 0
        last_progress_int = -1
        last_stage = ""

        while elapsed < bot_settings.fv_bot_poll_timeout:
            status = await self.get_status(job_id)
            progress_int = int(float(status.get("progress", 0.0)) * 100)
            stage = str(status.get("stage", ""))

            if on_progress is not None and (progress_int != last_progress_int or stage != last_stage):
                try:
                    await on_progress(status)
                except Exception as exc:  # noqa: BLE001
                    print(f"[bot] progress callback failed: {exc}")
                last_progress_int = progress_int
                last_stage = stage

            if status.get("status") == "done":
                return await self.get_result(job_id)
            if status.get("status") == "failed":
                raise RuntimeError(status.get("error") or "Processing failed")

            await asyncio.sleep(bot_settings.fv_bot_poll_interval)
            elapsed += bot_settings.fv_bot_poll_interval

        raise TimeoutError(f"Job {job_id} не завершился за {bot_settings.fv_bot_poll_timeout} секунд")
