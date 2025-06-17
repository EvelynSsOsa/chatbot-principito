# Este script requiere las siguientes bibliotecas:
# pip install faiss-cpu numpy transformers torch sentence-transformers

import faiss
import pickle
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import torch

# Cargar el índice FAISS
print("Cargando índice FAISS...")
index = faiss.read_index("principito.index")

# Cargar los fragmentos de texto
print("Cargando fragmentos de texto...")
with open("principito_text_chunks.pkl", "rb") as f:
    text_chunks = pickle.load(f)

# Cargar el modelo de embeddings
print("Cargando modelo de embeddings...")
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Cargar el modelo generativo
print("Cargando modelo generativo...")
generator = pipeline(
    "text-generation",
    model="EleutherAI/gpt-neo-125M",
    tokenizer="EleutherAI/gpt-neo-125M",
    device=0 if torch.cuda.is_available() else -1
)

def responder_pregunta(pregunta_usuario, k=3):
    print(f"\nBuscando los {k} fragmentos más relevantes para: '{pregunta_usuario}'")

    # Obtener embedding de la pregunta
    query_embedding = embedding_model.encode(pregunta_usuario)
    query_embedding = np.array([query_embedding], dtype=np.float32)

    # Recuperar fragmentos más similares
    distances, indices = index.search(query_embedding, k)
    fragmentos = [text_chunks[idx] for idx in indices[0] if idx < len(text_chunks)]

    if not fragmentos:
        return "Error: No se recuperaron fragmentos relevantes."

    contexto_para_llm = "\n\n".join(fragmentos)

    # Construir el prompt
    prompt = f"""Contexto:
{contexto_para_llm}

Pregunta: {pregunta_usuario}

Respuesta:"""

    print("\n--- PROMPT ENVIADO AL LLM ---")
    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
    print("--- FIN DEL PROMPT ---")

    # Generar respuesta
    respuesta_obj = generator(
        prompt,
        max_new_tokens=80,
        num_return_sequences=1,
        do_sample=False,
        pad_token_id=generator.tokenizer.eos_token_id
    )

    respuesta_completa = respuesta_obj[0]['generated_text']
    respuesta_generada = respuesta_completa.split("Respuesta:")[-1].strip()

    # Validación simple para evitar repeticiones sin sentido
    if pregunta_usuario.lower() in respuesta_generada.lower() and len(respuesta_generada.split()) < len(pregunta_usuario.split()) + 6:
        print("ADVERTENCIA: La respuesta parece ser una repetición de la pregunta o muy corta.")
    
    return respuesta_generada

# Ejemplo de uso local
if __name__ == "__main__":
    pregunta_ejemplo = "¿Qué aprendió el principito del zorro sobre domesticar?"
    print("\nPregunta:", pregunta_ejemplo)
    print("\nGenerando respuesta...")
    respuesta = responder_pregunta(pregunta_ejemplo, k=3)
    print("\nRespuesta FINAL del LLM:", respuesta)
