import csv

from flask import Flask, jsonify, request, render_template_string
import os
import joblib
import pickle
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

from utils.toolboxapp import *

os.chdir(os.path.dirname(__file__))

app = Flask(__name__)

# Nuevo endpoint para formulario de predicción
@app.route("/form_predict", methods=["GET", "POST"])
def form_predict():
    options = [
        "1: Servicios",
        "2: Suministros",
        "3: Obras",
        "4: Privado de administración pública",
        "5: Gestión de servicio público",
        "6: Concesión de servicios"
    ]
    if request.method == "POST":
        valores = {}
        valores['Tipus_de_contracte'] = request.form.get('Tipus_de_contracte')
        valores['Duracion_total'] = int(request.form.get('Duracion_total', 0))
        valores['CPV_def'] = request.form.get('CPV_def')
        # Llama a la función de predicción (ajusta según tu función real)
        resultado = prediccion(model, valores)
        return render_template_string('''
            <h2>Resultado de la predicción</h2>
            <p>{{ resultado }}</p>
            <a href="/form_predict">Volver</a>
        ''', resultado=resultado)
    return render_template_string('''
        <h2>Formulario de Predicción</h2>
        <form method="post" action="/form_predict">
            <label for="Tipus_de_contracte">Tipo de contrato:</label>
            <select name="Tipus_de_contracte" required>
                {% for opt in options %}
                <option value="{{ opt }}">{{ opt }}</option>
                {% endfor %}
            </select><br><br>
            <label for="Duracion_total">Duración del contrato:</label>
            <input type="number" step="0.1" min="0" max="1095" name="Duracion_total" required><br><br>
            <label for="CPV_def">Código CPV (8 dígitos):</label>
            <input type="text" name="CPV_def" pattern="\\d{8}" required><br><br>
            <input type="submit" value="Predecir">
        </form>
    ''', options=options)

# Carga el modelo
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


@app.route("/", methods=["GET"])
def hello():
    return welcome_message()


@app.route("/api/v1/predict", methods=["GET"])
def predict():
  
#   Columnas del fichero usado para entrenar el modelo: 
#   Tipus_de_contracte | CPV_def | Duracion_total
    
#   Recibimos los parámetros de la petición GET    
    valores = {}

    # valores['Tipus_de_contracte'] = request.args.get('Tipo', np.nan, type=float)
    # valores['CPV_def'] = request.args.get('CPV', np.nan, type=float)
    # valores['Duracion_total'] = request.args.get('Dur', np.nan, type=float)
    
    valores['Tipus_de_contracte'] = st.radio("Tipo de contrato",["1: Servicios","2: Suministros","3: Obras","4: Privado de administración pública","5: Gestión de servicio público","6: Concesión de servicios"],index = None,)
    st.write("Has seleccionado:", valores['Tipus_de_contracte'])

    valores['Duracion_total'] = st.slider("Duracion del contrato", min_value = 0.0, max_value = 1095.0, step = 0.1)

    valores['CPV_def'] = st.text_input("Código CPV (deben ser 8 digitos)", "03451300")
    st.write("El codigo actual és:", valores['CPV_def'])

    #response = prediccion(model, valores)
    prediccion(model, valores)

    #return jsonify(response)
    
@app.route("/api/v2/predict", methods=["POST"])
def predict_v2():

#   Columnas del fichero usado para entrenar el modelo: 
#   Tipus_de_contracte | CPV_def | Duracion_total
    
#   Recibimos los parámetros de la petición POST
    data = request.get_json(force=True)

    valores = {}
    valores['Tipus_de_contracte'] = data.get('Tipo', np.nan)
    valores['CPV_def'] = data.get('CPV', np.nan)
    valores['Duracion_total'] = data.get('Dur', np.nan)

    #response = prediccion(model, valores)
    prediccion(model, valores)

    #return jsonify(response)

@app.route("/Codigos_CPV", methods=["GET"])
def view_csv():
    """Render CSV as an HTML table."""

    try:
        with open("data/Codigos_CPV.csv", newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';')
            rows = list(reader)

        # Simple HTML table rendering
        html = """
        <html>
        <head><title>CSV Viewer</title></head>
        <body>
        <h2>CSV Data</h2>
        <table border="1" cellpadding="5">
        {% for row in rows %}
            <tr>
            {% for col in row %}
                <td>{{ col }}</td>
            {% endfor %}
            </tr>
        {% endfor %}
        </table>
        </body>
        </html>
        """
        return render_template_string(html, rows=rows)
    
    except FileNotFoundError:
        return "CSV file not found.", 404
    
@app.route("/api/v1/retrain", methods=["GET"])
def retrain():
    global model
    if os.path.exists("data/Advertising_new.csv"):
        data = pd.read_csv('data/Advertising_new.csv')
        data.columns = [col.lower() for col in data.columns]

        X = data.drop(columns=['sales'])
        y = data['sales']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

        model.fit(X_train, y_train)
        rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        mape = mean_absolute_percentage_error(y_test, model.predict(X_test))
        model.fit(X, y)

        # with open('ad_model.pkl', 'wb') as f:
        #     pickle.dump(model, f)

        return f"Model retrained. New evaluation metric RMSE: {str(rmse)}, MAPE: {str(mape)}"
    else:
        return "<h2>New data for retrain NOT FOUND. Nothing done!</h2>"


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
