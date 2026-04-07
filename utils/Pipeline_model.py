import pandas as pd
import numpy as np
import unicodedata

# Importar utils
import sys
sys.path.append('src')
from utils.bootcampviztools import *
from utils.toolbox import *

pd.options.mode.copy_on_write = True # CoW por defecto a partir de Pandas 3.0.0

def pipeline_model(X_train):

    # Inicializamos la lista de pasos del pipeline
    pipeline_steps = []

    # Eliminar todos los apóstrofes de los nombres de columnas y de los datos tipo string y pasar a minúsculas
    tratar_strings(X_train)
    pipeline_steps.append(lambda df: tratar_strings(df))

    # Vemos que hay columnas con un alto porcentaje de missings
    # Decidimos eliminar las columnas con más del 60% de valores nulos
    eliminar_columnas_nulas(X_train, pipeline_steps)

    # Ahora trataremos columnas con algún valor en blanco, eliminaremos las que tienen más del 70% en blanco
    # Para el resto informamos el valor 'Desconocido'
    tratar_columnas_con_algun_blanco(X_train, pipeline_steps, val='Desconocido')

    # Eliminar las columnas con cardinalidad 0 o mayor que 60%
    tratar_cardinalidad(X_train, pipeline_steps)

    if "Ambit_organitzatiu" in X_train.columns:

        clave_Ambit_organitzatiu = {
            "universitats": 1,
            "departaments i sector public de la generalitat de catalunya": 2,
            "entitats de ladministracio local": 3
        }

        X_train["Ambit_organitzatiu"] = X_train["Ambit_organitzatiu"].map(clave_Ambit_organitzatiu)

        pipeline_steps.append(lambda df: df.assign(Ambit_organitzatiu=df["Ambit_organitzatiu"].map(clave_Ambit_organitzatiu)))

    if "Identificador_agrupacio_organisme" in X_train.columns:
        identificador_agrupacio_organisme(X_train)
        pipeline_steps.append(lambda df: identificador_agrupacio_organisme(df))

    drop_cols = ["Agrupacio_organisme", "Identificador_organisme_contractant", "Organisme_contractant"]
    for col in drop_cols:
        if col in X_train.columns:
            X_train.drop(col, axis=1, inplace=True)
            pipeline_steps.append(lambda df, col=col: df.drop(col, axis=1))


    if "Procediment_dadjudicacio" in X_train.columns:
        X_train["Procediment_dadjudicacio"] = np.where(X_train["Procediment_dadjudicacio"] == "menor", True, False)
        pipeline_steps.append(lambda df: df.assign(Procediment_dadjudicacio=np.where(df["Procediment_dadjudicacio"] == "menor", True, False)))

    if "Tipus_de_contracte" in X_train.columns:
        tipus_de_contracte(X_train)
        pipeline_steps.append(lambda df: tipus_de_contracte(df))

    if "Codi_CPV" in X_train.columns:
        df_CPV = pd.read_csv('data/Codigos_CPV.csv', sep=';')
        df_CPV["CPV_def"] = df_CPV["CPV_def"].astype(str).str.zfill(8)

        X_train = codi_cpv(X_train, df_CPV)
        pipeline_steps.append(lambda df: codi_cpv(df, df_CPV))

        X_train.drop(columns=['Codi_CPV', "CPV_Descripcion"], inplace=True)
        pipeline_steps.append(lambda df, col=['Codi_CPV', "CPV_Descripcion"]: df.drop(col, axis=1))

        X_train['CPV_def'] = X_train['CPV_def'].astype(int)
        pipeline_steps.append(lambda df: df.assign(CPV_def=df['CPV_def'].astype(int)))

        # freq_CPV = X_train['CPV_def'].value_counts(normalize=True)
        # X_train['CPV_def'] = X_train['CPV_def'].map(freq_CPV)
        # pipeline_steps.append(lambda df: df.assign(CPV_def=df['CPV_def'].map(freq_CPV)))

        # X_train.drop(columns=['Codi_CPV', 'CPV_def'], inplace=True)
        # pipeline_steps.append(lambda df, col=['Codi_CPV', 'CPV_def']: df.drop(col, axis=1))

        # freq_CPV = X_train["CPV_Descripcion"].value_counts(normalize=True)
        # X_train["CPV_Descripcion"] = X_train['CPV_Descripcion'].map(freq_CPV)
        # pipeline_steps.append(lambda df: df.assign(CPV_Descripcion=df['CPV_Descripcion'].map(freq_CPV)))

    
    if "Tipo_de_contrato" in X_train.columns:
        freq_Tipo_de_contrato = X_train["Tipo_de_contrato"].value_counts(normalize=True)
        X_train["Tipo_de_contrato"] = X_train['Tipo_de_contrato'].map(freq_Tipo_de_contrato)
        pipeline_steps.append(lambda df: df.assign(Tipo_de_contrato=df['Tipo_de_contrato'].map(freq_Tipo_de_contrato)))

    if "Codi_CPV_div" in X_train.columns:
        freq_CPV_div = X_train["Codi_CPV_div"].value_counts(normalize=True)
        X_train["Codi_CPV_div"] = X_train['Codi_CPV_div'].map(freq_CPV_div)
        pipeline_steps.append(lambda df: df.assign(Codi_CPV_div=df['Codi_CPV_div'].map(freq_CPV_div)))

    if "Adjudicatari" in X_train.columns:
        freq_adj = X_train["Adjudicatari"].value_counts(normalize=True)
        X_train["Adjudicatari"] = X_train['Adjudicatari'].map(freq_adj)
        pipeline_steps.append(lambda df: df.assign(Adjudicatari=df['Adjudicatari'].map(freq_adj)))

    #similitud_con_target(X_train, y_train, ["Import d’adjudicació"], pipeline_steps)

    # Eliminamos la columna de "Import d'adjudicacio" dado que coincide en un 99% con nuestra target, y no aporta información adicional            df.drop(columns=[col], inplace=True)
    if "Import_dadjudicacio" in X_train.columns:
        X_train.drop('Import_dadjudicacio', axis = 1, inplace = True)
        pipeline_steps.append(lambda df, col='Import_dadjudicacio': df.drop(col, axis=1))
    
    fechas(X_train)
    pipeline_steps.append(lambda df: fechas(df))

    duracion(X_train)
    pipeline_steps.append(lambda df: duracion(df))

    if "Tipus_de_liquidacio" in X_train.columns:
        X_train["Tipus_de_liquidacio"] = np.where(X_train["Tipus_de_liquidacio"] =="compliment", True, False)
        pipeline_steps.append(lambda df: df.assign(Tipus_de_liquidacio=np.where(df["Tipus_de_liquidacio"] == "compliment", True, False)))

    # Separar columnas numéricas, objeto y fecha
    features_numericas = X_train.select_dtypes(include=['number']).columns
   
    X_train[features_numericas] = np.log1p(X_train[features_numericas])
    pipeline_steps.append(lambda df: df.assign(**{col: np.log1p(df[col]) for col in features_numericas}))

    return X_train, pipeline_steps
