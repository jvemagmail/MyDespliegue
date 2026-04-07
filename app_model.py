import csv

from flask import Flask, jsonify, request, render_template_string
import os
import joblib
import pickle
import numpy as np
import pandas as pd
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


@app.route("/api/v1/predict", methods=["GET"])
def predict():
  
#   Columnas del fichero usado para entrenar el modelo: 
#   Tipus_de_contracte | CPV_def | Duracion_total
    
#   Recibimos los parámetros de la petición GET    
    valores = {}

    valores['Tipus_de_contracte'] = request.args.get('Tipo', np.nan, type=float)
    valores['CPV_def'] = request.args.get('CPV', np.nan, type=float)
    valores['Duracion_total'] = request.args.get('Dur', np.nan, type=float)
    
    response = prediccion(model, valores)

    return jsonify(response)
    
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

    response = prediccion(model, valores)

    return jsonify(response)

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
