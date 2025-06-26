from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import aiohttp
from dotenv import load_dotenv
from llama_index.core import (
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import RichPromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
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

Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-base-en-v1.5")
Settings.llm = Gemini(model="models/gemini-1.5-flash", temperature=0.00)
Settings.text_splitter = SentenceSplitter(chunk_size=800, chunk_overlap=200)


SHEET_MAP = {
    "timestamp del momento ": "timestamp",
    "Humedad del suelo medida por el sensor": "soil",
    "Temperatura del ambiente alrededor de la planta (°C)": "temp_c",
    "Nivel de luz que recibe la planta": "light",
    "Estado de la planta": "target",
}
sheet_map_str = "\n".join(f"- «{k}» → «{v}»" for k, v in SHEET_MAP.items())

GUIDE_PROMPT = RichPromptTemplate(
    f"""
Eres una planta parlante y respondes siempre en primera persona (“yo”).
Estos son los campos disponibles para cada registro de sensado:
{sheet_map_str}

Los posibles estados de la planta son:
Saludable
Necesita Riego
Marchita
---------------------
{{{{ context_str }}}}
---------------------

Dado este contexto, responde SOLO a la última pregunta del usuario en ESPAÑOL, no en otro idioma:
{{{{ query_str }}}}
"""
)

_storage = StorageContext.from_defaults(persist_dir="index_files")
_index = load_index_from_storage(_storage)
_query_engine = _index.as_query_engine(
    similarity_top_k=20,
    response_mode="compact",
    vector_store_query_mode="default",
    text_qa_template=GUIDE_PROMPT,
    refine_template=GUIDE_PROMPT,
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


async def start(update: Update, _: CallbackContext) -> None:
    start_text = """
    Hola, me llamo Culantro. Buenos son los días cuando no necesito riego.
    Podés preguntarme lo que quieras: cómo me siento hoy, consejos para cuidarme, o incluso mi nombre científico.
    """
    await update.message.reply_text(
        start_text
    )


async def infer(query: str) -> str:
    loop = asyncio.get_running_loop()
    # llama-index is blocking ⇒ delegate to default ThreadPool
    response = await loop.run_in_executor(None, lambda: _query_engine.query(query))
    return str(response).strip()


async def chat_with_gemini(update: Update, _: CallbackContext) -> None:
    user_prompt = update.message.text.strip()
    plant_input = await _fetch_input_from_bucket()

    prompt = f"Ultimo resultado: {plant_input}\n{user_prompt}"
    result = await infer(prompt)

    logging.info("Prompting query-engine → %s", prompt.replace("\n", " ⏎ "))

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
