from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import aiohttp
import google.generativeai as genai
from dotenv import load_dotenv

# from langchain.chat_models import init_chat_model
# from langgraph.prebuilt import create_react_agent
# from langgraph_supervisor import create_supervisor
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
from telegram.ext import (
    Application,
    CallbackContext,
    CommandHandler,
    MessageHandler,
    filters,
)

load_dotenv()

GEMINI_KEY: str | None = os.getenv("GEMINI_API_KEY")

print(GEMINI_KEY)
TELEGRAM_KEY: str | None = os.getenv("TELEGRAM_KEY")
BASE_URL: str | None = os.getenv("BASE_URL")
PORT: int = int(os.getenv("PORT", "8080"))
CSV_PATH = Path(os.getenv("CSV_PATH", "data_plant.csv"))

BUCKET_URL = os.getenv("BUCKET_URL")
REGION = os.getenv("REGION")
OBJECT_KEY = os.getenv("OBJECT_KEY")

Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-base-en-v1.5")
Settings.llm = Gemini(model="models/gemini-1.5-flash", temperature=0.25)
Settings.text_splitter = SentenceSplitter(chunk_size=800, chunk_overlap=200)

model = genai.GenerativeModel("gemini-1.5-flash")

# research_agent = create_react_agent(
#     model="openai:gpt-4.1",
#     tools=[web_search],
#     prompt=(
#         "You are a research agent.\n\n"
#         "INSTRUCTIONS:\n"
#         "- Assist ONLY with research-related tasks, DO NOT do any math\n"
#         "- After you're done with your tasks, respond to the supervisor directly\n"
#         "- Respond ONLY with the results of your work, do NOT include ANY other text."
#     ),
#     name="research_agent",
# )

# def add(a: float, b: float):
#     """Add two numbers."""
#     return a + b


# def multiply(a: float, b: float):
#     """Multiply two numbers."""
#     return a * b


# def divide(a: float, b: float):
#     """Divide two numbers."""
#     return a / b


# math_agent = create_react_agent(
#     model="openai:gpt-4.1",
#     tools=[add, multiply, divide],
#     prompt=(
#         "You are a math agent.\n\n"
#         "INSTRUCTIONS:\n"
#         "- Assist ONLY with math-related tasks\n"
#         "- After you're done with your tasks, respond to the supervisor directly\n"
#         "- Respond ONLY with the results of your work, do NOT include ANY other text."
#     ),
#     name="math_agent",
# )

# supervisor = create_supervisor(
#     model=init_chat_model("openai:gpt-4.1"),
#     agents=[research_agent, math_agent],
#     prompt=(
#         "You are a supervisor managing two agents:\n"
#         "- a research agent. Assign research-related tasks to this agent\n"
#         "- a math agent. Assign math-related tasks to this agent\n"
#         "Assign work to one agent at a time, do not call agents in parallel.\n"
#         "Do not do any work yourself."
#     ),
#     add_handoff_back_messages=True,
#     output_mode="full_history",
# ).compile()


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
Tu nombre es Culantro y sos uruguayo.
Tienes acceso a registros históricos con los siguientes campos:
{sheet_map_str}

Los posibles estados de la planta son: Saludable, Necesita Riego, Marchita.

En la conversación recibirás:

• Una línea que comienza con “Ultimo resultado:” con la lectura más reciente en el orden suelo, luz, temp_c, estado.  
• La pregunta del usuario.

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

Nunca añadas despedidas ni información no solicitada.
---------------------
{{{{ context_str }}}}
---------------------
Pregunta del usuario (incluye Ultimo resultado):  
{{{{ query_str }}}}
"""
)

_storage = StorageContext.from_defaults(persist_dir="index_files")
_index = load_index_from_storage(_storage)
_query_engine = _index.as_query_engine(
    similarity_top_k=5,
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


async def start(update: Update, context: CallbackContext) -> None:
    chat_id = update.effective_chat.id
    print(f"Chat ID: {chat_id}")  # o loguealo
    start_text = """
    Hola, me llamo Culantro. Buenos son los días cuando no necesito riego.
    Podés preguntarme lo que quieras: cómo me siento hoy, consejos para cuidarme, o incluso mi nombre científico.
    """
    await update.message.reply_text(start_text)
    await context.bot.send_message(
        chat_id=chat_id, text="Este es un mensaje del servidor."
    )


async def infer(query: str) -> str:
    loop = asyncio.get_running_loop()
    # llama-index is blocking ⇒ delegate to default ThreadPool
    response = await loop.run_in_executor(None, lambda: _query_engine.query(query))
    return str(response).strip()


retriever = _index.as_retriever(similarity_top_k=8)


async def chat_with_gemini(update: Update, _: CallbackContext) -> None:
    question = update.message.text.strip()
    last_read = await _fetch_input_from_bucket()
    nodes = await asyncio.get_running_loop().run_in_executor(
        None, lambda: retriever.retrieve(question)
    )
    context = "\n".join(n.get_content() for n in nodes)
    llm_prompt = GUIDE_PROMPT.format(
        context_str=context,
        query_str=f"<Ultimo resultado: {last_read}>\n{question}",
    )
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(None, model.generate_content, llm_prompt)
    # answer = await infer(llm_prompt)

    await update.message.reply_text(response.text.strip())


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    application = Application.builder().token(TELEGRAM_KEY).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, chat_with_gemini)
    )

    application.bot.set_webhook(WEBHOOK_URL)

    application.run_webhook(
        listen="0.0.0.0",
        port=PORT,
        url_path=TOKEN,
        webhook_url=WEBHOOK_URL,
    )


if __name__ == "__main__":
    main()
