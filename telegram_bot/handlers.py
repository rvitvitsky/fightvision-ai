"""Хендлеры Telegram-бота FightVision AI."""
from __future__ import annotations

import contextlib
import html
import tempfile
from pathlib import Path
from typing import Any

from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import FSInputFile, Message

from api_client import FightVisionAPI
from config import bot_settings


router = Router()
api = FightVisionAPI()

WELCOME = (
    "🥊 <b>FightVision AI</b>\n\n"
    "Пришли видео боксёрского раунда — я найду удары, соберу нарезку и сводку.\n\n"
    "<b>Ограничения:</b>\n"
    f"• До <b>{bot_settings.fv_bot_max_video_mb} МБ</b> через Telegram\n"
    "• Форматы: mp4, mov, mkv, webm\n"
    "• Лучше короткий статичный раунд из-за ринга\n\n"
    "Команды: /help, /health"
)

STAGE_LABELS = {
    "queued": "⏳ В очереди",
    "starting": "🚀 Запуск",
    "pose_extraction": "👤 Извлечение поз YOLO",
    "features": "📊 Расчёт фич",
    "event_detection": "🥊 Поиск ударов",
    "visual_rerank": "👁 Визуальная переоценка",
    "mlp_rerank": "🧠 MLP-переоценка",
    "attributes": "🏷 Атрибуты ударов",
    "postprocess": "✂️ Финальный отбор",
    "highlights_render": "🎬 Сборка нарезки",
    "finished": "✅ Готово",
    "error": "❌ Ошибка",
}


def _mb(num_bytes: int | None) -> float:
    return float(num_bytes or 0) / (1024 * 1024)


def _progress_bar(value: float, length: int = 14) -> str:
    value = max(0.0, min(1.0, float(value or 0.0)))
    filled = int(round(value * length))
    return "▓" * filled + "░" * (length - filled)


def _stage_text(status: dict[str, Any]) -> str:
    stage = str(status.get("stage") or "")
    label = STAGE_LABELS.get(stage, stage or "Обработка")
    progress = float(status.get("progress") or 0.0)
    pct = int(progress * 100)
    return f"{label}\n<code>{_progress_bar(progress)}</code> <b>{pct}%</b>"


def _summary_text(result: dict[str, Any]) -> str:
    summary = result.get("summary", {}) or {}
    red = summary.get("red", {}) or {}
    blue = summary.get("blue", {}) or {}
    events = result.get("events", []) or []
    model_info = result.get("model_info", {}) or {}

    def fighter(title: str, st: dict[str, Any]) -> str:
        accuracy = int(float(st.get("accuracy", 0) or 0) * 100)
        return (
            f"{title}\n"
            f"  Ударов: <b>{st.get('punches', 0)}</b>\n"
            f"  Точность: <b>{accuracy}%</b>\n"
            f"  Попало: {st.get('landed', 0)}\n"
            f"  Блок: {st.get('blocked', 0)}\n"
            f"  Мимо: {st.get('missed', 0)}"
        )

    parts = [
        "📊 <b>Сводка боя</b>",
        f"Всего событий: <b>{len(events)}</b>",
        "",
        fighter("🔴 <b>RED</b>", red),
        "",
        fighter("🔵 <b>BLUE</b>", blue),
    ]

    if model_info.get("event_model") == "heuristic":
        parts += ["", "⚠️ Модели не подгружены, использована эвристика — точность ниже."]

    return "\n".join(parts)


def _short_timeline(result: dict[str, Any], limit: int = 12) -> str:
    events = result.get("events", []) or []
    if not events:
        return ""

    lines = ["", "🧾 <b>Первые события</b>"]
    for ev in events[:limit]:
        t = float(ev.get("time", 0) or 0)
        fighter = str(ev.get("fighter", "?"))
        punch_type = str(ev.get("punch_type", "?"))
        hand = str(ev.get("hand", "?"))
        target = str(ev.get("target", "?"))
        eff = str(ev.get("effectiveness", "?"))
        lines.append(
            f"• {t:6.1f}s — <b>{html.escape(fighter)}</b>, "
            f"{html.escape(hand)} {html.escape(punch_type)}, "
            f"{html.escape(target)}, {html.escape(eff)}"
        )
    if len(events) > limit:
        lines.append(f"…и ещё {len(events) - limit}")
    return "\n".join(lines)


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer(WELCOME)


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(WELCOME)


@router.message(Command("health"))
async def cmd_health(message: Message) -> None:
    try:
        data = await api.health()
        await message.answer(f"✅ Backend на связи\n<pre>{html.escape(str(data))}</pre>")
    except Exception as exc:  # noqa: BLE001
        await message.answer(f"❌ Backend недоступен:\n<code>{html.escape(str(exc))}</code>")


@router.message(F.video | F.document)
async def handle_video(message: Message) -> None:
    video = message.video or message.document
    if video is None:
        await message.answer("Прикрепи видео файлом или отправь как видео.")
        return

    if message.document:
        mime = (message.document.mime_type or "").lower()
        name = (message.document.file_name or "").lower()
        ok_ext = name.endswith((".mp4", ".mov", ".mkv", ".webm", ".avi"))
        if not mime.startswith("video/") and not ok_ext:
            await message.answer("Этот документ не похож на видео. Поддержка: mp4, mov, mkv, webm, avi.")
            return

    file_size_mb = _mb(video.file_size)
    if file_size_mb > bot_settings.fv_bot_max_video_mb:
        await message.answer(
            f"Файл <b>{file_size_mb:.1f} МБ</b> больше лимита бота "
            f"({bot_settings.fv_bot_max_video_mb} МБ). Залей его через сайт."
        )
        return

    status_msg = await message.answer("⬇️ Скачиваю видео из Telegram...")
    tmp_dir_obj = tempfile.TemporaryDirectory(prefix="fightvision_bot_")
    tmp_dir = Path(tmp_dir_obj.name)

    try:
        file_info = await message.bot.get_file(video.file_id)
        filename = getattr(video, "file_name", None) or f"{video.file_id}.mp4"
        input_path = tmp_dir / filename

        await message.bot.download_file(file_info.file_path, destination=input_path)

        await status_msg.edit_text("⬆️ Отправляю видео в FightVision backend...")
        job_id = await api.upload_video_path(input_path, filename)

        await status_msg.edit_text(f"🆔 Job <code>{job_id[:8]}</code>\n⏳ В очереди...")

        async def on_progress(status: dict[str, Any]) -> None:
            with contextlib.suppress(Exception):
                await status_msg.edit_text(f"🆔 Job <code>{job_id[:8]}</code>\n\n{_stage_text(status)}")

        result = await api.wait_for_completion(job_id, on_progress=on_progress)
        events = result.get("events", []) or []
        highlights_url = result.get("highlights_url", "")

        if not events or not highlights_url:
            await status_msg.edit_text(
                "✅ Анализ завершён, но удары не найдены.\n"
                "Попробуй другой фрагмент или проверь видео через сайт."
            )
            return

        await status_msg.edit_text("📥 Скачиваю нарезку...")
        out_path = tmp_dir / "fightvision_highlights.mp4"
        out_size = await api.download_to_path(highlights_url, out_path)
        out_size_mb = _mb(out_size)

        await status_msg.delete()
        caption = f"🎬 <b>Нарезка ударов</b> — найдено <b>{len(events)}</b>"

        if out_size_mb <= bot_settings.fv_bot_max_result_mb:
            try:
                await message.answer_video(FSInputFile(out_path), caption=caption)
            except Exception:
                await message.answer_document(FSInputFile(out_path), caption=caption)
        else:
            await message.answer(
                f"🎬 Нарезка готова, но файл <b>{out_size_mb:.1f} МБ</b> — слишком большой для отправки ботом.\n"
                "Открой результат на сайте или забери файл из папки <code>media/results</code>."
            )

        await message.answer(_summary_text(result) + _short_timeline(result))

    except TimeoutError:
        await status_msg.edit_text("⌛ Превышено время ожидания. Попробуй видео покороче или через сайт.")
    except Exception as exc:  # noqa: BLE001
        safe = html.escape(str(exc))
        with contextlib.suppress(Exception):
            await status_msg.edit_text(f"❌ Ошибка обработки:\n<code>{safe}</code>")
    finally:
        tmp_dir_obj.cleanup()


@router.message()
async def fallback(message: Message) -> None:
    await message.answer("🎥 Пришли видео раунда — я разберу его на удары. Команды: /help, /health")
