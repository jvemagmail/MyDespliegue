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

# Carga el modelo
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route("/", methods=["GET"])
def hello():
    return welcome_message()

# Endpoint para el formulario de predicción
@app.route("/api/v1/predict", methods=["GET", "POST"])
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
        numero, texto = valores['Tipus_de_contracte'].split(":", 1)
        numero = numero.strip()
        valores['Tipus_de_contracte'] = int(numero)       
        
        valores['CPV_def'] = int(request.form.get('CPV_def'))
        valores['Duracion_total'] = int(request.form.get('Duracion_total', 0))

        # Llama a la función de predicción (ajusta según tu función real)
        resultado = prediccion(model, valores)
        return render_template_string('''
            <h2>Resultado de la predicción</h2>
            <p>{{ resultado }} EUR</p>
            <a href="/api/v1/predict">Volver</a>
        ''', resultado=resultado)
    
    return render_template_string('''
        <h2>Formulario de Predicción</h2>
        <form method="post">
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

    resultado = prediccion(model, valores)
    
    return jsonify({"resultado": resultado})

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
    

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
