
# Importar librerías
import seaborn as sns   
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from functools import reduce

import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
import pickle
import os

# Importar utils
import sys
sys.path.append('src')

from utils.bootcampviztools import *
from utils.toolbox import *
from utils.Pipeline_model import *

pd.options.mode.copy_on_write = True # CoW por defecto a partir de Pandas 3.0.0

'''Carga de datos'''
# Cargamos los datos
df = pd.read_parquet('data/datos_licitaciones.parquet', engine='fastparquet')

# Nuestro proyecto se centrará en predecir el importe de la liquidación, por lo que nos quedamos con esa variable como target.
target = 'Import de la liquidació'

# Eliminar los registros donde el valor de la variable target es menor a 0
# los trataremos una vez entrenado el modelo
df_copia = df[df[target] > 0]

# Aplicamos logaritmo a la variable target para reducir su asimetría
# Nuestra nueva variable target será target_log
target_log = target + '_log'
df_copia.loc[:, target_log] = np.log1p(df_copia[target])

X = df_copia.drop([target, target_log], axis=1)
y = df_copia[target_log]

'''Split de los datos'''
# Hacemos el split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

'''Preprocesamiento'''
pipeline_steps = []

X_train, pipeline_steps = pipeline_model(X_train)

'''Eliminamos las features con poca colinearidad con el target para hacer el despliegue con un número reducido de features'''
no_drop = ['CPV_Descripcion', 'Duracion_total', 'Adjudicatari', 'Procediment_dadjudicacio', 'Tipus_de_contracte']
drop_cols = [col for col in X_train.columns if col not in no_drop]

X_train.drop(drop_cols, axis=1, inplace=True)
pipeline_steps.append(lambda df, drop_cols=drop_cols: df.drop(drop_cols, axis=1))

print(X_train.columns)

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=10)

best_params = study.best_params

best_n_estimators = best_params["n_estimators"]
best_max_depth = best_params["max_depth"]
best_min_samples_split = best_params["min_samples_split"]
best_min_samples_leaf = best_params["min_samples_leaf"]
best_max_features = best_params["max_features"]
best_criterion = best_params["criterion"]
best_max_samples = best_params["max_samples"]

best_rf_model = RandomForestRegressor(
    n_estimators=best_n_estimators,
    max_depth=best_max_depth,
    min_samples_split=best_min_samples_split,
    min_samples_leaf=best_min_samples_leaf,
    max_features=best_max_features,
    criterion=best_criterion,
    max_samples=best_max_samples,
    random_state=42,
    n_jobs=-1
)

best_rf_model.fit(X_train,y_train)

# Aplicamos las modificaciones que hemos hecho sobre X_train a X
X = reduce(lambda acc, func: func(acc), pipeline_steps, X)

best_rf_model.fit(X,y)

with open('model.pkl', 'wb') as f:
    pickle.dump(best_rf_model, f)

# # Aplicamos las modificaciones que hemos hecho sobre X_train a X_test
# X_test = reduce(lambda acc, func: func(acc), pipeline_steps, X_test)
# predecir(best_rf_model, X_test, y_test)