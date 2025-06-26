
import pandas as pd
from llama_index.core import (
    Document,
    Settings,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1) Configurar modelo de embeddings, LLM y splitter
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.text_splitter = SentenceSplitter(
    chunk_size=800,
    chunk_overlap=200
)

print("Embedder activo:", Settings.embed_model)

# 2) Leer el CSV local
df = pd.read_csv("data_plant.csv")
docs = []

# 3) Crear documentos por fila
for idx, row in df.iterrows():
    parts = [f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])]
    if parts:
        docs.append(Document(
            text="\n".join(parts),
            metadata={"row": idx}
        ))

# (opcional) Documentos por columna
for col in df.columns:
    vals = df[col].dropna().astype(str).tolist()
    if vals:
        docs.append(Document(
            text=f"Columna: {col}\n" + "\n".join(vals),
            metadata={"col": col}
        ))

print(f"Total de documentos creados: {len(docs)}")

doc_culantrillo = Document(
    text="""Adiantum raddianum, conocida también como helecho “Delta” o de cabellera de Venus, es un helecho perenne originario de Sudamérica que alcanza unos 40–50 cm de altura y forma delicadas frondas triangulares sostenidas por pecíolos negros que le aportan elegancia. Para prosperar, requiere luz indirecta brillante —ideal cerca de ventanas orientadas al norte o al este— y nunca sol directo, que quemaría sus frondas. Necesita suelo bien drenado pero constantemente húmedo, rico en materia orgánica (una mezcla de turba, perlita y compost funciona bien), evitando que se seque o se encharque. La temperatura ideal está entre 18 °C y 27 °C, sin bajar de unos 10–15 °C, y la humedad debe mantenerse alta (al menos 60 %), lo que se logra con nebulización, bandejas de guijarros húmedos o incluso cultivándolo en baños o terrarios . Aunque crece lento, vive hasta unos 15 años y rara vez necesita fertilizantes; se recomienda abonar con fertilizante líquido equilibrado diluido a mitad de fuerza una vez al mes en primavera verano. Es aconsejable replantar cada 1–2 años cuando la planta se congestione, aprovechando para dividir el cepellón si se desea propagarla. Entre sus problemas comunes destacan el amarilleo por exceso de riego y las puntas marrones por baja humedad, así como la posible aparición de plagas como cochinillas o escamas, que se controlan con jabón insecticida . En resumen, replicar su hábitat natural —luz suave, suelo húmedo, calor y humedad alta— asegura que este helecho luzca frondas saludables y exhuberantes.""",  # tu texto completo
    metadata={"especie": "culantrillo", "tipo": "guía de cuidado"}
)

# Lo agregás junto con otros documentos antes de construir el índice
docs.append(doc_culantrillo)

# 4) Crear índice
index = VectorStoreIndex.from_documents(docs)
print("Índice creado con éxito.")

# 5) Guardar el índice localmente
index.storage_context.persist(persist_dir="index_files")
print("Índice guardado en ./index_files")

