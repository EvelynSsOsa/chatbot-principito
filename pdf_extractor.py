# Este script requiere la siguiente biblioteca:
# pip install pdfplumber

import pdfplumber
import os

# Definir nombres de archivos
pdf_file = "el principito.pdf"
output_file = "principito_extraido.txt"

# Verificar si el archivo PDF existe
if not os.path.exists(pdf_file):
    print(f"Error: No se encontró el archivo '{pdf_file}'")
    exit(1)

try:
    # Abrir el PDF y extraer el texto
    with pdfplumber.open(pdf_file) as pdf:
        # Lista para almacenar el texto de cada página
        text_pages = []
        
        # Extraer texto de cada página
        for page in pdf.pages:
            text = page.extract_text()
            if text:  # Verificar si se extrajo texto
                text_pages.append(text)
        
        # Combinar todo el texto
        full_text = "\n\n".join(text_pages)
        
        # Guardar el texto en un archivo
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_text)
        
        print(f"¡Éxito! El texto ha sido extraído y guardado en '{output_file}'")

except Exception as e:
    print(f"Error al procesar el PDF: {str(e)}") 