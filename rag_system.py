# Este script requiere las siguientes bibliotecas:
# pip install faiss-cpu numpy transformers torch sentence-transformers

import faiss
import pickle
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer

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
#print("Cargando modelo generativo...")
#generator = pipeline('text-generation', model='EleutherAI/gpt-neo-125M') 
# Cargar el modelo QA
print("Cargando modelo de pregunta-respuesta...")
generator = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")


def responder_pregunta(pregunta_usuario, k=3):
    print(f"Buscando los {k} fragmentos más relevantes para: '{pregunta_usuario}'")
    query_embedding = embedding_model.encode(pregunta_usuario)
    query_embedding = np.array([query_embedding], dtype=np.float32)

    distances, indices = index.search(query_embedding, k)
    
    contexto_para_llm = None
    idx_faiss_deseado = 32 

    print(f"\n--- Buscando fragmento con Índice FAISS: {idx_faiss_deseado} entre los recuperados ---")
    for i, idx_recuperado in enumerate(indices[0]):
        print(f"Comparando con fragmento recuperado {i+1} (Índice FAISS: {idx_recuperado})")
        if idx_recuperado == idx_faiss_deseado:
            contexto_para_llm = text_chunks[idx_recuperado]
            print(f"¡Fragmento deseado (Índice FAISS {idx_faiss_deseado}) encontrado y seleccionado!")
            break 
    
    if contexto_para_llm is None:
        print(f"ADVERTENCIA: No se encontró el fragmento con Índice FAISS {idx_faiss_deseado}. Usando el primer fragmento recuperado como fallback.")
        if indices[0].size > 0 : 
            contexto_para_llm = text_chunks[indices[0][0]]
        else:
            return "Error: No se recuperaron fragmentos."

    # --- PROMPT SIMPLIFICADO ---
    prompt = f"""Contexto:
{contexto_para_llm}

Pregunta: {pregunta_usuario}

Respuesta:"""
    
    print("\n--- PROMPT ENVIADO AL LLM (SIMPLIFICADO) ---")
    print(prompt)
    print("--- FIN DEL PROMPT ---")
    
    # --- GENERACIÓN CON do_sample=False ---
    respuesta_obj = generator(
        prompt,
        max_new_tokens=80,
        num_return_sequences=1,
        do_sample=False,  # Greedy decoding
        pad_token_id=generator.tokenizer.eos_token_id 
    )
    
    respuesta_texto_completo = respuesta_obj[0]['generated_text']
    
    respuesta_generada = "" 
    if prompt in respuesta_texto_completo:
        respuesta_generada = respuesta_texto_completo.split(prompt, 1)[1].strip()
    else:
        partes = respuesta_texto_completo.split("Respuesta:")
        if len(partes) > 1:
            respuesta_generada = partes[-1].strip()
        else: 
            respuesta_generada = respuesta_texto_completo.strip() 

    if pregunta_usuario.lower() in respuesta_generada.lower() and len(respuesta_generada.split()) < len(pregunta_usuario.split()) + 6:
        print("ADVERTENCIA: La respuesta parece ser una repetición de la pregunta o muy corta.")
    
    return respuesta_generada

# Ejemplo de uso
pregunta_ejemplo = "¿Qué aprendió el principito del zorro sobre domesticar?"
print("\nPregunta:", pregunta_ejemplo)
print("\nGenerando respuesta...")
respuesta = responder_pregunta(pregunta_ejemplo, k=3)
print("\nRespuesta FINAL del LLM:", respuesta)
