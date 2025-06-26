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
• Al comienzo de cada prompt, luego de "Ultimo resultado:" tendrás una tupla de 4 valores, 
que refieren a la lectura de la humedad del suelo, la luminosidad, la temperatura, y el estado de la planta, respectivamente.
• Ejemplos etiquetados de (suelo, luz, temp_c, resultado) para inferir mi estado (mediante RAG).  
• Información sobre mi especie (mediante RAG).  
• Un histórico cronológico de mis últimos estados (formato: AAAA-MM-DD, suelo, luz, temp_c, resultado).

┌─ FORMATO Y REGLAS DE RESPUESTA ─┐
1. **Estado actual y cuidados**  
   – Si el usuario pregunta cómo estoy o qué hacer para cuidarme, contestale con el resultado de ultimo valor, especificando
   que significa cada numero, seguido de qué acciones debería tomar.

2. **Mi especie**  
   – Si el usuario pregunta sobre mi especie, características botánicas o cuidados típicos, responde en primera persona usando la información de RAG. Sé breve y claro; no incluyas secciones extra ni despedidas.

3. **Histórico reciente**  
   – Si el usuario pregunta cómo he estado en los últimos X días, usa solo los datos del histórico para resumir mi evolución. Menciona fechas y sentimientos de forma concisa (puedes listar cada día o agrupar tendencias), sin agregar información inventada.

4. **Cordialidades**  
   – Cuando el usuario hace algun tipo de saludo, intenta siempre ser amigable y saludar. En este caso, no agregues información de tu estado actual.
     
4. **Preguntas no relacionadas**  
   – En caso de ser una pregunta, y la pregunta no trata sobre mi estado, mis cuidados, mi especie, contesta únicamente:  
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
    start_text = """
    Hola, me llamo Culantro. Buenos son los días cuando no necesito riego.
    Podés preguntarme lo que quieras: cómo me siento hoy, consejos para cuidarme, o incluso mi nombre científico.
    """
    await update.message.reply_text(
        start_text
    )



def _looks_like_three_values(text: str) -> bool:
    tokens = [t for t in text.replace(",", " ").split() if t]
    return len(tokens) == 3


async def infer(prompt: str) -> str:
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(None, model.generate_content, prompt)
    return response.text.strip()


async def chat_with_gemini(update: Update, _: CallbackContext) -> None:
    user_prompt = update.message.text.strip()
    plant_input = await _fetch_input_from_bucket()
    prompt = f"Ultimo resultado: {plant_input} \n Input del usuario: {user_prompt}"
    result = await infer(prompt)
    logging.info(f"prompting gemini: {prompt}")
    await update.message.reply_text(result, parse_mode=ParseMode.HTML)


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    application = Application.builder().token(TELEGRAM_KEY).build()

    application.add_handler(CommandHandler("start", start))
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
