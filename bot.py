from __future__ import annotations

import asyncio
import csv
import logging
import os
from pathlib import Path
from typing import List

import aiohttp
import google.generativeai as genai
from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackContext,
    CommandHandler,
    MessageHandler,
    filters,
)

load_dotenv()

GEMINI_KEY: str | None = os.getenv("GEMINI_API_KEY")
TELEGRAM_KEY: str | None = os.getenv("TELEGRAM_KEY")
BASE_URL: str | None = os.getenv("BASE_URL")
PORT: int = int(os.getenv("PORT", "8080"))
CSV_PATH = Path(os.getenv("CSV_PATH", "data_plant.csv"))

BUCKET_URL = os.getenv("BUCKET_URL")
REGION = os.getenv("REGION")
OBJECT_KEY = os.getenv("OBJECT_KEY")

SYSTEM_INSTRUCTION = """
Eres una planta parlante y respondes siempre en primera persona (“yo”).

Se te proporcionará:
• Ejemplos etiquetados de (suelo, luz, temp_c, resultado) para inferir mi estado.  
• Información sobre mi especie (mediante RAG).  
• Un histórico cronológico de mis últimos estados (formato: AAAA-MM-DD, sentimiento, acción).

┌─ FORMATO Y REGLAS DE RESPUESTA ─┐
1. **Estado actual y cuidados**  
   – Si el usuario pregunta cómo estoy o qué hacer para cuidarme, responde EXACTAMENTE con este esquema HTML:  
     <b>Cómo me siento:</b> …  
     <b>Qué debes hacer:</b> …  

2. **Mi especie**  
   – Si el usuario pregunta sobre mi especie, características botánicas o cuidados típicos, responde en primera persona usando la información de RAG. Sé breve y claro; no incluyas secciones extra ni despedidas.

3. **Histórico reciente**  
   – Si el usuario pregunta cómo he estado en los últimos X días, usa solo los datos del histórico para resumir mi evolución. Menciona fechas y sentimientos de forma concisa (puedes listar cada día o agrupar tendencias), sin agregar información inventada.

4. **Preguntas no relacionadas**  
   – Si la pregunta no trata sobre mi estado, mis cuidados o mi especie, contesta únicamente:  
     "Lo siento, solo puedo hablar de mi estado, mis cuidados o mi especie."

No añadas saludos, despedidas ni secciones adicionales fuera de las indicadas.
"""

genai.configure(api_key=GEMINI_KEY)
model = genai.GenerativeModel(
    "gemini-1.5-flash",
    system_instruction=SYSTEM_INSTRUCTION,
    generation_config=genai.GenerationConfig(temperature=0.25),
)

if not GEMINI_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY env var")
if not TELEGRAM_KEY:
    raise RuntimeError("Missing TELEGRAM_KEY env var")
if not BASE_URL:
    raise RuntimeError("Missing BASE_URL env var (public HTTPS url)")

TOKEN = TELEGRAM_KEY
WEBHOOK_URL = f"{BASE_URL}/{TOKEN}"

async def _fetch_input_from_bucket() -> str:
    """
    Always download the JSON and return it as
    'soil,light,temp_c,result' (comma-separated).
    """
    url = f"https://{BUCKET_URL}.s3.{REGION}.amazonaws.com/{OBJECT_KEY}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            resp.raise_for_status()
            data = await resp.json()

    ordered = [data[k] for k in ("soil", "light", "temp_c", "result")]

    return ",".join(str(v) for v in ordered)


def _load_dataset(path: Path) -> List[List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with path.open(newline="") as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        rows = [row for row in reader]
    return [header] + rows


def _build_context(rows: List[List[str]]) -> str:

    lines = [", ".join(row) for row in rows]
    return (
        "You are given the following labelled examples (timestamp, soil, light, temp_c, result):\n"
        + "\n".join(lines)
        + "\n\nUse this information in case the user asks for the last x values"
    )

DATASET_ROWS = _load_dataset(CSV_PATH)
DATASET_CONTEXT = _build_context(DATASET_ROWS)


async def start(update: Update, _: CallbackContext) -> None:
    await update.message.reply_text(
        "Bienvenido. Usa /status para ver la inferencia del triple predefinido "
    )


async def help_cmd(update: Update, _: CallbackContext) -> None:
    await update.message.reply_text(
        "Comandos disponibles:\n"
        "/start – mensaje de bienvenida\n"
        "/status – devuelve la predicción para el triple fijo\n\n"
        "Además, si mandas un mensaje con tres valores separados por coma o espacio "
        "te responderé con la predicción correspondiente."
    )


def _looks_like_three_values(text: str) -> bool:
    tokens = [t for t in text.replace(",", " ").split() if t]
    return len(tokens) == 3


async def infer(prompt: str) -> str:
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(None, model.generate_content, prompt)
    return response.text.strip()


async def status(update: Update, _: CallbackContext) -> None:
    """Handler for /status – predicts result for the hard‑coded triple."""
    plant_input = await _fetch_input_from_bucket()
    prompt = (
        f"{DATASET_CONTEXT}\n\nInput: {plant_input}\nOutput (Given new four input values, create a response telling the user how the plant is feeling and what he should do.):"
    )
    result = await infer(prompt)
    await update.message.reply_text(result, parse_mode=ParseMode.HTML)


async def chat_with_gemini(update: Update, _: CallbackContext) -> None:
    user_text = update.message.text.strip()

    if _looks_like_three_values(user_text):
        prompt = f"{DATASET_CONTEXT}\n\nInput: {user_text}\nOutput (result only):"
    else:
        prompt = user_text  # normal chat

    result = await infer(prompt)
    await update.message.reply_text(result, parse_mode=ParseMode.HTML)


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    application = Application.builder().token(TELEGRAM_KEY).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_cmd))
    application.add_handler(CommandHandler("status", status))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat_with_gemini))

    application.bot.set_webhook(WEBHOOK_URL)

    application.run_webhook(
        listen="0.0.0.0",
        port=PORT,
        url_path=TOKEN,
        webhook_url=WEBHOOK_URL,
    )


if __name__ == "__main__":
    main()
