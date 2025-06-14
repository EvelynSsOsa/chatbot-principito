# Este script requiere las siguientes bibliotecas:
# pip install numpy faiss-cpu sentence-transformers torch

import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# Cargar los embeddings y textos (asumiendo que están en chunks_with_embeddings)
# Si el script se ejecuta de forma independiente, cargar desde text_processor.py
try:
    from text_processor import chunks_with_embeddings
except ImportError:
    print("Error: No se encontró la lista chunks_with_embeddings")
    print("Asegúrate de ejecutar primero text_processor.py")
    exit(1)

# Extraer embeddings y convertirlos a array numpy
embeddings = np.array([chunk['embedding'] for chunk in chunks_with_embeddings], dtype=np.float32)
texts = [chunk['text'] for chunk in chunks_with_embeddings]

# Verificar la forma del array de embeddings
print(f"Forma del array de embeddings: {embeddings.shape}")

# Obtener la dimensionalidad
d = embeddings.shape[1]  # debería ser 384

# Crear el índice FAISS
index = faiss.IndexFlatL2(d)

# Añadir los embeddings al índice
index.add(embeddings)

# Guardar el índice FAISS
faiss.write_index(index, "principito.index")
print("\nÍndice FAISS guardado en 'principito.index'")

# Guardar los textos
with open("principito_text_chunks.pkl", "wb") as f:
    pickle.dump(texts, f)
print("Fragmentos de texto guardados en 'principito_text_chunks.pkl'")

# Realizar una búsqueda de ejemplo
print("\nRealizando una búsqueda de ejemplo...")

# Cargar el modelo para generar el embedding de la consulta
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Definir una pregunta de ejemplo
query = "¿Qué le dijo el zorro al principito sobre la amistad?"

# Generar el embedding para la pregunta
query_embedding = model.encode(query)
query_embedding = np.array([query_embedding], dtype=np.float32)

# Realizar la búsqueda
k = 3  # número de resultados a recuperar
distances, indices = index.search(query_embedding, k)

# Mostrar los resultados
print(f"\nResultados para la pregunta: '{query}'")
print("\nFragmentos más relevantes:")
for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
    print(f"\nResultado {i+1} (distancia: {distance:.4f}):")
    print(f"Texto: {texts[idx][:200]}...")  # Mostrar solo los primeros 200 caracteres 