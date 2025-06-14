# Este script requiere las siguientes bibliotecas:
# pip install sentence-transformers torch

import os
from sentence_transformers import SentenceTransformer
import numpy as np

# Definir el archivo de entrada
input_file = "principito_extraido.txt"

# Verificar si el archivo existe
if not os.path.exists(input_file):
    print(f"Error: No se encontró el archivo '{input_file}'")
    exit(1)

# Leer el contenido del archivo
with open(input_file, 'r', encoding='utf-8') as f:
    text = f.read()

# Dividir el texto en párrafos
paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

# Cargar el modelo de embeddings
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Generar embeddings para cada párrafo
chunks_with_embeddings = []
for paragraph in paragraphs:
    # Limpiar saltos de línea internos del párrafo
    cleaned_paragraph = paragraph.replace('\n', ' ').replace('  ', ' ')  # Reemplaza \n con espacio y luego múltiples espacios con uno solo
    
    # Generar el embedding usando el párrafo limpio
    embedding = model.encode(cleaned_paragraph)
    
    # Almacenar el párrafo original y su embedding
    chunks_with_embeddings.append({
        'text': paragraph,  # Mantenemos el texto original
        'embedding': embedding
    })

# Imprimir información sobre los resultados
print(f"\nNúmero total de fragmentos (chunks): {len(chunks_with_embeddings)}")
print(f"Dimensionalidad de los embeddings: {chunks_with_embeddings[0]['embedding'].shape}")

# Imprimir los primeros 3 fragmentos y una porción de sus embeddings
print("\nPrimeros 3 fragmentos y sus embeddings:")
for i, chunk in enumerate(chunks_with_embeddings[:3]):
    print(f"\nFragmento {i+1}:")
    print(f"Texto: {chunk['text'][:200]}...")  # Mostrar solo los primeros 200 caracteres
    print(f"Embedding (primeros 5 valores): {chunk['embedding'][:5]}")

# Guardar los resultados (opcional)
print("\nLos embeddings están almacenados en la lista 'chunks_with_embeddings'")
print("Cada elemento de la lista es un diccionario con las claves 'text' y 'embedding'") 