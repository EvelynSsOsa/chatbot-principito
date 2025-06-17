import streamlit as st
from rag_system import responder_pregunta

st.set_page_config(page_title="Pregúntale al Principito", page_icon="🪐")
st.title("🪐 Pregúntale al Principito")

st.markdown("""
¿Tienes una duda sobre la historia de *El Principito*?
Escribe tu pregunta abajo y el sistema te responderá usando un modelo de lenguaje real.
""")

pregunta = st.text_input("Escribe tu pregunta aquí:")

if pregunta:
    with st.spinner("Consultando al Principito..."):
        respuesta = responder_pregunta(pregunta)
    st.success("Respuesta:")
    st.write(respuesta)

