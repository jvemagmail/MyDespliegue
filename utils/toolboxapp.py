import numpy as np
import pandas as pd

def welcome_message():

    return """
        <!DOCTYPE html>
        <html lang="es">
        <head>
            <meta charset="UTF-8">
            <title>Información del Modelo de Predicción</title>
        </head>
        <body>
            <h1>Modelo de Predicción de Importe de Liquidación</h1>
            <p>
                Este modelo ha sido desarrollado para predecir el <strong>importe de la liquidación</strong> en licitaciones públicas, utilizando técnicas avanzadas de machine learning y un conjunto seleccionado de variables relevantes.
            </p>
            <ul>
                <li>El modelo utiliza un Random Forest optimizado mediante <em>Optuna</em> para ofrecer la mejor precisión posible.</li>
                <li>Las variables más importantes para la predicción son: <strong>Código CPV, Duración total, Tipo de contrato</strong>.</li>
                <li>El modelo ha sido entrenado y validado con datos reales de licitaciones.</li>
            </ul>
            <p>
                Para realizar una predicción, pulse el siguiente enlace:<br>
                <a href="https://mydespliegue.onrender.com/api/v1/predict" target="_blank">
                    https://mydespliegue.onrender.com/api/v1/predict
                </a>
            </p>

            <p>
                Para probar la predicción con Postman, puedes usar la siguiente configuración:<br>
                <strong>POST</strong> a <code>https://mydespliegue.onrender.com/api/v2/predict</code> <br>
                Body (JSON):
                <pre>
            {
            "Tipo": 1,
            "CPV": 33696500,
            "Dur": 12
            }
                </pre>
                O copia este comando <code>curl</code> y pégalo en tu terminal:
                <button onclick="navigator.clipboard.writeText('curl -X POST \"https://mydespliegue.onrender.com/api/v2/predict\" -H \"Content-Type: application/json\" -d \"{\\\"Tipo\\\":1,\\\"CPV\\\":33696500,\\\"Dur\\\":12}\"')">Copiar comando curl</button>
            </p>
            <p>
                <a href="https://learning.postman.com/docs/getting-started/introduction/" target="_blank">¿Cómo usar Postman?</a>
            </p>
        </body>
        </html>
    """

def prediccion(model, valores):

    response = {}

#   Aplicamos logaritmo a las variables numéricas para que el modelo pueda hacer la predicción correctamente
    valores = {key: np.log1p(val) for key, val in valores.items()}

    missing = [name for name, val in valores.items() if np.isnan(val)]

    input_data = pd.DataFrame(valores, index=[0])
    
    prediction = model.predict(input_data)

    response['Parámetros de entrada'] = valores
    response['Predicción'] = prediction[0]

    if missing:
        response['warning'] = f"Missing values imputed for: {', '.join(missing)}"

    return response