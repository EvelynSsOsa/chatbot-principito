
   
# Este script requiere las siguientes bibliotecas:
# pip install faiss-cpu numpy transformers torch sentence-transformers

import faiss
import pickle
import numpy as np
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# --- Cargar índice FAISS ---
print("Cargando índice FAISS...")
index = faiss.read_index("principito.index")

# --- Cargar los fragmentos de texto ---
print("Cargando fragmentos de texto...")
with open("principito_text_chunks.pkl", "rb") as f:
    text_chunks = pickle.load(f)

# --- Cargar modelo de embeddings ---
print("Cargando modelo de embeddings...")
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# --- Cargar modelo generativo ---
print("Cargando modelo generativo...")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
modelo_llm = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")

generator = pipeline(
    "text-generation",
    model=modelo_llm,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

# --- Función para responder preguntas ---
def responder_pregunta(pregunta_usuario, k=3):
    print(f"Buscando los {k} fragmentos más relevantes para: '{pregunta_usuario}'")
    
    # Codificar la pregunta
    query_embedding = embedding_model.encode(pregunta_usuario)
    query_embedding = np.array([query_embedding], dtype=np.float32)

    # Buscar los fragmentos más cercanos
    distances, indices = index.search(query_embedding, k)
    print(f"\n--- Recuperando los {k} fragmentos más relevantes ---")
    fragmentos = [text_chunks[idx] for idx in indices[0] if idx < len(text_chunks)]

    if not fragmentos:
        return "Error: No se recuperaron fragmentos."

    contexto = "\n\n".join(fragmentos)

    # Construir el prompt
    prompt = (
        f"Pregunta: {pregunta_usuario}\n\n"
        f"Contexto:\n{contexto}\n\n"
        f"Respuesta:"
    )

    print("\n--- Generando respuesta con modelo de texto ---")
    respuesta_obj = generator(
        prompt,
        max_new_tokens=80,
        num_return_sequences=1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    respuesta_generada = respuesta_obj[0]['generated_text']
    
    # Extraer la respuesta después de 'Respuesta:'
    partes = respuesta_generada.split("Respuesta:")
    if len(partes) > 1:
        return partes[-1].strip()
    else:
        return respuesta_generada.strip()

# --- Ejemplo de uso ---
if __name__ == "__main__":
    pregunta_ejemplo = "¿Qué aprendió el principito del zorro sobre domesticar?"
    print("\nPregunta:", pregunta_ejemplo)
    print("\nGenerando respuesta...")
    respuesta = responder_pregunta(pregunta_ejemplo, k=3)
    print("\nRespuesta FINAL del LLM:", respuesta)
