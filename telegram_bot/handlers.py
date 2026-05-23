"""Хендлеры команд и сообщений Telegram-бота."""
from __future__ import annotations

import io
from typing import Any

from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import BufferedInputFile, Message

from api_client import FightVisionAPI
from config import bot_settings


router = Router()
api = FightVisionAPI()


# --- text labels ------------------------------------------------------------

WELCOME = (
    "🥊 *FightVision AI*\n\n"
    "Пришли видео боксёрского раунда — я найду удары, соберу нарезку и сводку.\n\n"
    "Ограничения:\n"
    f"• До *{bot_settings.max_video_mb} МБ* (лимит Telegram Bot API)\n"
    "• Форматы: mp4, mov, mkv, webm\n"
    "• Один статичный план из-за ринга\n\n"
    "Просто отправь видео и подожди — обработка занимает несколько минут."
)

STAGE_LABELS = {
    "queued": "⏳ В очереди",
    "starting": "🚀 Запуск",
    "pose_extraction": "👤 Извлечение поз (YOLO)",
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


# --- helpers ----------------------------------------------------------------


def _progress_bar(value: float, length: int = 14) -> str:
    value = max(0.0, min(1.0, value))
    filled = int(round(value * length))
    return "▓" * filled + "░" * (length - filled)


def _format_stage_message(status: dict[str, Any]) -> str:
    stage = status.get("stage", "")
    label = STAGE_LABELS.get(stage, stage or "Обработка")
    pct = int(status.get("progress", 0.0) * 100)
    return f"{label}\n`{_progress_bar(status.get('progress', 0.0))}` *{pct}%*"


def _format_summary(result: dict[str, Any]) -> str:
    summary = result.get("summary", {})
    red = summary.get("red", {})
    blue = summary.get("blue", {})
    model_info = result.get("model_info", {})

    def fmt_fighter(name: str, st: dict) -> str:
        return (
            f"{name}\n"
            f"  Ударов:     *{st.get('punches', 0)}*\n"
            f"  Точность:   *{int(st.get('accuracy', 0) * 100)}%*\n"
            f"  Попало:     {st.get('landed', 0)}\n"
            f"  Блок:       {st.get('blocked', 0)}\n"
            f"  Мимо:       {st.get('missed', 0)}"
        )

    parts = [
        "📊 *Сводка боя*",
        "",
        fmt_fighter("🔴 *RED*", red),
        "",
        fmt_fighter("🔵 *BLUE*", blue),
    ]

    mode = model_info.get("event_model", "unknown")
    if mode == "heuristic":
        parts.append("")
        parts.append("_⚠️ Модели не подгружены, использована эвристика — точность ниже._")

    return "\n".join(parts)


# --- handlers ---------------------------------------------------------------


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer(WELCOME, parse_mode="Markdown")


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(WELCOME, parse_mode="Markdown")


@router.message(Command("health"))
async def cmd_health(message: Message) -> None:
    """Диагностика связи с бекендом."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{bot_settings.fv_backend_url}/health")
            r.raise_for_status()
            data = r.json()
        await message.answer(
            f"✅ Backend на связи\n```\n{data}\n```",
            parse_mode="Markdown",
        )
    except Exception as exc:  # noqa: BLE001
        await message.answer(f"❌ Backend недоступен: `{exc}`", parse_mode="Markdown")


@router.message(F.video | F.document)
async def handle_video(message: Message) -> None:
    """Главный хендлер. Принимает видео, прогоняет через gateway, шлёт нарезку обратно."""
    video = message.video or message.document
    if video is None:
        await message.answer("Прикрепи видео файлом или просто шли как видео.")
        return

    # для документов — фильтруем по mime
    if message.document:
        mime = (message.document.mime_type or "").lower()
        if not mime.startswith("video/"):
            await message.answer("Этот документ не похож на видео. Поддержка: mp4, mov, mkv, webm.")
            return

    file_size_mb = (video.file_size or 0) / (1024 * 1024)
    if file_size_mb > bot_settings.max_video_mb:
        await message.answer(
            f"Файл *{file_size_mb:.1f} МБ* — это больше лимита Telegram Bot API "
            f"({bot_settings.max_video_mb} МБ). Сожми видео или залей на сайт.",
            parse_mode="Markdown",
        )
        return

    # начальное сообщение которое будем редактировать
    status_msg = await message.answer("⬇️ Скачиваю видео...")

    try:
        file_info = await message.bot.get_file(video.file_id)
        buf = io.BytesIO()
        await message.bot.download_file(file_info.file_path, destination=buf)
        buf.seek(0)
        file_bytes = buf.read()

        await status_msg.edit_text("⬆️ Отправляю на анализ...")
        filename = getattr(video, "file_name", None) or f"{video.file_id}.mp4"
        job_id = await api.upload_video(file_bytes, filename)

        await status_msg.edit_text(f"🆔 Job `{job_id[:8]}`\n⏳ В очереди...", parse_mode="Markdown")

        async def on_progress(status: dict[str, Any]) -> None:
            try:
                await status_msg.edit_text(
                    f"🆔 Job `{job_id[:8]}`\n\n{_format_stage_message(status)}",
                    parse_mode="Markdown",
                )
            except Exception:
                pass  # rate-limit или повтор того же текста — игнор

        result = await api.wait_for_completion(job_id, on_progress=on_progress)

        # отправляем нарезку
        highlights_url = result.get("highlights_url", "")
        events = result.get("events", [])

        if not events or not highlights_url:
            await status_msg.edit_text(
                "✅ Анализ завершён, но удары не найдены 🤔\n"
                "Возможно, видео слишком короткое или модели не подгружены.",
            )
            return

        await status_msg.edit_text("📥 Скачиваю нарезку...")
        video_bytes = await api.download(highlights_url)

        await status_msg.delete()
        await message.answer_video(
            BufferedInputFile(video_bytes, filename="highlights.mp4"),
            caption=f"🎬 *Нарезка ударов* — найдено *{len(events)}*",
            parse_mode="Markdown",
        )
        await message.answer(_format_summary(result), parse_mode="Markdown")

    except TimeoutError:
        await status_msg.edit_text("⌛ Превышено время ожидания. Попробуй видео покороче.")
    except Exception as exc:  # noqa: BLE001
        try:
            await status_msg.edit_text(f"❌ Ошибка обработки:\n`{exc}`", parse_mode="Markdown")
        except Exception:
            await message.answer(f"❌ Ошибка обработки: {exc}")


@router.message()
async def fallback(message: Message) -> None:
    """Любой другой текст — намекаем что нужно прислать видео."""
    await message.answer(
        "🎥 Пришли видео раунда — я разберу его на удары.\n"
        "Команды: /help, /health",
    )
