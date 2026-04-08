'''
Te dejo aqui mis propuestas para mejorar la visualizacion del proyecot, a ver que te parece 
import streamlit as st

val1 = st.radio("Tipo de contrato",["***1: Servicios***","***2: Suministros***","***3: Obras***","***4: Privado de administración pública***","***5: Gestión de servicio público***","***6: Concesión de servicios***"],index = None,)
st.write("Has seleccionado:", val1)

val2 = st.slider("Duracion del contrato", min_value = 0.0, max_value = 1095.0, step = 0.1)

val3 = st.text_input("Código CPV (deben ser 8 digitos)", "03451300")
st.write("El codigo actual és:", val3)


if st.button("Predict"):
    prediction = str(model.predict([[val1, val2, val3, val4]])[0])
    st.write("Predicción:", prediction)
'''