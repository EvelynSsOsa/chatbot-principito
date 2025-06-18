import streamlit as st
from rag_system import responder_pregunta

st.set_page_config(page_title="Pregúntale al Principito", page_icon="🪐")

st.title("🪐 Pregúntale al Principito")

st.markdown("""
¿Tienes una duda sobre la historia de *El Principito* u otro PDF que hayas subido?
Escribe tu pregunta abajo y el sistema la contestará con base en el contenido procesado.
""")

pregunta = st.text_input("Escribe tu pregunta aquí:")

if pregunta:
    with st.spinner("Consultando al modelo..."):
        respuesta = responder_pregunta(pregunta)
    st.success("Respuesta:")
    st.write(respuesta)

