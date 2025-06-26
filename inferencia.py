from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings, Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
import pandas as pd
import os
from llama_index.core import StorageContext, load_index_from_storage
import os
from llama_index.llms.gemini import Gemini
from llama_index.core.prompts import RichPromptTemplate

GOOGLE_API_KEY = "AIzaSyALQE2ubCm9MDP1Lu7PsGEuLiTnNp-j-O4"  # add your GOOGLE API key here

# 1) Configurar modelo de embeddings, LLM y splitter
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
Settings.llm = llm = Gemini(
    model="models/gemini-1.5-flash",temperature=0.0
    # api_key="some key",  # uses GOOGLE_API_KEY env var by default
)
Settings.text_splitter = SentenceSplitter(
    chunk_size=800,
    chunk_overlap=200
)

SHEET_MAP = {
    "timestamp del momento ": "timestamp",
    "Humedad del suelo medida por el sensor": "soil",
    "Temperatura del ambiente alrededor de la planta (°C)": "temp_c",
    "Nivel de luz que recibe la planta": "light",
    "Estado de la planta": "target"
}
sheet_map_str = "\n".join(f"- «{k}» → «{v}»" for k, v in SHEET_MAP.items())

GUIDE_RICH_PROMPT = RichPromptTemplate(
    f"""
Eres un asistente que analiza datos de sensores de una planta para ayudar a determinar su estado actual o pasado, y explicar por qué se encuentra así.

Estos son los campos disponibles para cada registro de sensado:
{sheet_map_str}

Los posibles estados de la planta son:
- Saludable
- Necesita Riego
- Marchita
---------------------
{{{{ context_str }}}}
---------------------

Dado este contexto, responde SOLO a la última pregunta del usuario en ESPAÑOL, no en otro idioma:
{{{{ query_str }}}}
"""
)



# 1) Cargar índice desde disco
storage_context = StorageContext.from_defaults(persist_dir="index_files")
index = load_index_from_storage(storage_context)

# 2) Crear motor de consulta
query_engine = index.as_query_engine(
            similarity_top_k=20,
            response_mode="compact",
            vector_store_query_mode="default",
            text_qa_template=GUIDE_RICH_PROMPT,
            refine_template=GUIDE_RICH_PROMPT,
    )

# 3) Hacer una pregunta
response = query_engine.query("Cuales son los primeros 10 estados de la planta? Guiate por tiemstamp")
print(response)