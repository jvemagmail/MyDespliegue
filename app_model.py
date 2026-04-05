from flask import Flask, jsonify, request
import os
import joblib
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np

os.chdir(os.path.dirname(__file__))

app = Flask(__name__)

# Carga el modelo
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


@app.route("/", methods=["GET"])
def hello():
    return "Bienvenido a mi API del modelo de predicción de importe de adjudicación de licitaciones públicas de cataluña"


@app.route("/api/v1/predict", methods=["GET"])
def predict():
  
# Columnas del fichero usado para entrenar el modelo: 
# ['Procediment_dadjudicacio', 'Tipus_de_contracte', 'Adjudicatari', 'CPV_Descripcion', 'Duracion_total']
  
    nproc = request.args.get('Proc', np.nan, type=bool)
    ntipo = request.args.get('Tipo', np.nan, type=float)
    nAdj = request.args.get('Adj', np.nan, type=float)
    nCPV = request.args.get('CPV', np.nan, type=float)
    ndur = request.args.get('Dur', np.nan, type=float)
    
    #missing = [name for name, val in [('CPV_Descripcion', nCPV), ('Duracion_total', ndur), ('Adjudicatari', nAdj), ('Procediment_dadjudicacio', nproc), ('Tipus_de_contracte', ntipo) ] if np.isnan(val)]

    input_data = pd.DataFrame({'Procediment_dadjudicacio': [nproc], 'Tipus_de_contracte': [ntipo], 'Adjudicatari': [nAdj], 'CPV_Descripcion': [nCPV], 'Duracion_total': [ndur], })
    #return jsonify(input_data.to_dict(orient="records"))
    prediction = model.predict(input_data)

    response = {'predictions': prediction[0]}
    
    
    
    #if missing:
    #    response['warning'] = f"Missing values imputed for: {', '.join(missing)}"
    
    

    return jsonify(response)
    


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
