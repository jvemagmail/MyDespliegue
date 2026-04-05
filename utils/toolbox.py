import optuna
import pandas as pd
import numpy as np
import unicodedata

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error

import optuna
import joblib

def data_report(df):

    '''
    Esta función nos devuelve un dataframe con un resumen de cada columna 
    del dataframe original, incluyendo su nombre, tipo de dato, porcentaje de valores 
    faltantes, número de valores únicos y porcentaje de cardinalidad.
    '''
    # Sacamos los NOMBRES
    cols = pd.DataFrame(df.columns.values, columns=["COL_N"])

    # Sacamos los TIPOS
    types = pd.DataFrame(df.dtypes.values, columns=["DATA_TYPE"])

    # Sacamos los MISSINGS
    percent_missing = round(df.isnull().sum() * 100 / len(df), 2)
    percent_missing_df = pd.DataFrame(percent_missing.values, columns=["MISSINGS (%)"])

    # Sacamos los VALORES UNICOS
    unicos = pd.DataFrame(df.nunique().values, columns=["UNIQUE_VALUES"])
    
    percent_cardin = round(unicos['UNIQUE_VALUES']*100/len(df), 2)
    percent_cardin_df = pd.DataFrame(percent_cardin.values, columns=["CARDIN (%)"])

    concatenado = pd.concat([cols, types, percent_missing_df, unicos, percent_cardin_df], axis=1, sort=False)
    concatenado.set_index('COL_N', drop=True, inplace=True)

    return concatenado

def quitar_acentos(texto):

    '''Esta función recibe un texto y devuelve el mismo texto sin acentos 
    ni caracteres especiales.'''
    
    return ''.join(
        c for c in unicodedata.normalize('NFKD', texto)
        if not unicodedata.combining(c)
    )

def tratar_strings(df):
    
    '''Esta función recibe un dataframe y devuelve el mismo dataframe con los nombres 
    de las columnas y los valores de tipo string tratados, eliminando acentos, 
    caracteres especiales y espacios.'''
    
    df.columns = [quitar_acentos(col) for col in df.columns]
    df.columns = df.columns.str.replace(r"['’`´]", "", regex=True)
    df.columns = df.columns.str.replace(r" ", "_", regex=True)

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.replace(r"['’`´]", "", regex=True).str.lower()
        df[col] = df[col].apply(quitar_acentos)
    
    return df

def similitud_con_target(df, y, cols, pipeline_steps):
    
    '''Esta función recibe un dataframe, una variable target, una lista de columnas 
    y una lista de pasos del pipeline.
    Para cada columna de la lista, calcula el porcentaje de coincidencia 
    entre los valores de la columna y la variable target.'''

    for col in cols:
        porcentaje = len(df[df[col] == y])*100/len(df)
        print(f"Porcentage coincidencia {col} / {y.name}: {porcentaje:.2f}%")

        if porcentaje > 95:
            print(f"Se debería eliminar la columna: {col}")
    

def eliminar_columnas_nulas(df, pipeline_steps, threshold=0.6):

    '''Esta función recibe un dataframe, una lista de pasos del pipeline y un umbral.
    Elimina las columnas que están completamente vacías o que tienen todos los valores
    como strings vacíos.'''

    # Encontrar columnas totalmente nulas o con todos los strings vacíos
    columnas_nulas = [col for col in df.columns 
                            if df[col].isnull().all() or (df[col].astype(str) == '').all()]
    
    if len(columnas_nulas) > 0:
        print(f"Total de columnas vacías: {len(columnas_nulas)}\n")
        for col in columnas_nulas:
            print(f"Se elimina columna: {col}")

        df.drop(columns=columnas_nulas, inplace=True)

        for col in columnas_nulas:
            pipeline_steps.append(lambda df, col=col: df.drop(col, axis=1))

def tratar_cardinalidad(df, pipeline_steps, threshold=0.6):
    
    '''Esta función recibe un dataframe, una lista de pasos del pipeline y un umbral.
    Elimina las columnas que tienen una cardinalidad del 0% o mayor que el umbral.'''

    l = len(df)

    # Encontrar columnas con cardinalidad 0 o mayor que el umbral
    df_report = data_report(df).reset_index()

    columnas_card_0= df_report['COL_N'][df_report['CARDIN (%)'] == 0].tolist()
    columnas_card_max = df_report['COL_N'][df_report['CARDIN (%)'] > threshold*100].tolist()

    columnas_cardinalidad = columnas_card_0 + columnas_card_max

    if len(columnas_cardinalidad) > 0:
        print(f"Total de columnas con cardinalidad 0 o mayor que {threshold*100}%: {len(columnas_cardinalidad)}\n")
        
        for col in columnas_cardinalidad:
            valores_unicos = df[col].nunique()
            porcentaje_cardin = valores_unicos*100/l
            print(f"Se elimina columna: {col} ({valores_unicos} valores únicos, {porcentaje_cardin:.2f}%)")
            pipeline_steps.append(lambda df, col=col: df.drop(col, axis=1))

        df.drop(columns=columnas_cardinalidad, inplace=True)

def otros(df, col, val='Otros'):

    '''Esta función recibe un dataframe, una columna y un valor. 
    Reemplaza los valores vacíos de la columna por el valor especificado.'''

    df[col] = df[col].apply(lambda x: val if str(x) == '' else x)

    return df

def tratar_columnas_con_algun_blanco(df, pipeline_steps, val='Otros'):

    '''Esta función recibe un dataframe, una lista de pasos del pipeline y un valor.
    Elimina las columnas que tienen más del 70% de valores en blanco o reemplaza 
    los valores en blanco por el valor especificado.'''

    l = len(df)
    columnas_objeto = df.select_dtypes(include=['object']).columns.tolist()

    if len(columnas_objeto) > 0:
        print(f"Columnas con algún blanco:\n")
   
        # columnas con algún valor en blanco
        for col in columnas_objeto:
            if (df[col].astype(str) == '').any():
                nulos = len(df[df[col].astype(str) == ''])
                porcentaje = nulos*100/l

                if porcentaje > 70:
                    print(f"Se elimina columna: {col} {nulos} {porcentaje:.2f}%")
                    df.drop(columns=col, inplace=True)
                    pipeline_steps.append(lambda df, col=col: df.drop(col, axis=1))
                else:
                    print(f"Columna con blancos: {col} {nulos} {porcentaje:.2f}%")
                    if porcentaje < 30:
                        otros(df, col, val)
                        pipeline_steps.append(lambda df, col=col: otros(df, col, val))

def tratar_columnas_con_algun_cero(df, pipeline_steps):

    '''Esta función recibe un dataframe, una lista de pasos del pipeline y un valor.
    Elimina las columnas que tienen más del 70% de valores en cero '''

    l = len(df)
    columnas_numericas = df.select_dtypes(include=['number']).columns.tolist()

    if len(columnas_numericas) > 0:
        print(f"Columnas con algún cero:\n")
   
        # columnas con algún valor cero
        for col in columnas_numericas:
            if (df[col] == 0).any():
                ceros = len(df[df[col] == 0])
                porcentaje = ceros*100/l

                if porcentaje > 70:
                    print(f"Se elimina columna: {col} {ceros} {porcentaje:.2f}%")
                    df.drop(columns=col, inplace=True)
                    pipeline_steps.append(lambda df, col=col: df.drop(col, axis=1))
                else:
                    print(f"Columna con ceros: {col} {ceros} {porcentaje:.2f}%")
                    


def fechas(df):

    '''Esta función recibe un dataframe y devuelve el mismo dataframe con las columnas 
    de fecha tratadas, convirtiendo las fechas a formato datetime, añadiendo columnas de día, 
    mes y año y eliminando las columnas originales.'''

    columnas_fecha = [col for col in df.columns if col.startswith('Data')]

    for col in columnas_fecha:
        df["datetime_" + col] = pd.to_datetime(df[col], format='%d/%m/%Y', errors='coerce')

        # Miramos si hay nulos en el nuevo campo
        mask = df["datetime_" + col].isnull()
        df_nulos = df[col][mask]
        l = len(df_nulos)

        if l > 0:
            indexs = df_nulos.index.to_list()

            print(f"\nFechas incongruentes {col}: \n{df_nulos}")
            
            # Solucionamos las columnas incongruentes
            for idx in indexs:
                source = df.at[idx, col]
                fixed = source.replace('/00', '/20')
                df.at[idx, col] = fixed
                df.at[idx, "datetime_" + col] = pd.to_datetime(fixed, format='%d/%m/%Y', errors='coerce')

            # Miramos si hay nulos en el nuevo campo
            mask = df["datetime_" + col].isnull()
            df_nulos = df[col][mask]
            l = len(df_nulos)

            if l == 0:
                print(f"\nFechas corregidas: {col}")
                for idx in indexs:
                    print(f"{df.at[idx, col]}")
            else:
                print('error')

        df[col] = df["datetime_" + col]
        
        columnas_dt(df, col)

        df.drop(columns=[col, "datetime_" + col], inplace=True)
        
    return df

def columnas_dt(df, col):

    '''Esta función recibe un dataframe y una columna de tipo datetime, 
    y devuelve el mismo dataframe con las columnas de día, 
    mes, año y unix timestamp añadidas.'''
    
    # Añadimos las columnas dia, mes y año de las fechas y eliminamos la columna fecha
    df["day_" + col] = df[col].dt.day
    df["month_" + col] = df[col].dt.month
    df["year_" + col] = df[col].dt.year
    df["unix_" + col] = df[col].astype("int64") // 10**9

    return df

def duracion(df):

    '''Esta función recibe un dataframe y devuelve el mismo dataframe con la columna 
    de duración total calculada a partir de las columnas de días, meses y años, 
    eliminando las columnas originales.'''
    
    df["Duracion_total"] = (df["Durada_dies"] + df["Durada_mesos"] * 30 + df["Durada_anys"] * 365) 

    df.drop(columns=['Durada_dies', 'Durada_mesos', 'Durada_anys'], inplace=True)

    return df


def similitud_con_exercici(df, pipeline_steps):

    '''Esta función recibe un dataframe y una lista de pasos del pipeline.
    Elimina las columnas que tienen una similitud del 100% con la columna de ejercicio'''

    col = 'Exercici'
    l = len(df)

    columnas_year = [col_year for col_year in df.columns if col_year.startswith('year_Data')]
    for col_year in columnas_year:
        coincidencias = len(df[df[col] == df[col_year]])
        porcentaje = coincidencias*100/l
        if porcentaje == 100:
            print(f"Se elimina la columna {col_year} Porcentage coincidencia con {col} : {porcentaje:.2f}%")
            df.drop(columns=[col_year], inplace=True)
            pipeline_steps.append(lambda df, col=col_year: df.drop(col, axis=1))
        # else:
        #     print(col_year, porcentaje)

def mediana(df, col):
    
    '''Esta función recibe un dataframe y una columna, 
    y devuelve el mismo dataframe con los valores negativos o cero reemplazados 
    por la mediana de la columna.'''

    if col in df.columns:
        mediana = df[col].median()
        df.loc[df[col] <= 0, col] = mediana
    
    return df


    
def identificador_agrupacio_organisme(df):

    '''Esta función recibe un dataframe y devuelve el mismo dataframe con la columna
    de Identificador_agrupacio_organisme tratada, creando una nueva columna de referencia
    a partir de los 2 primeros dígitos del identificador, y eliminando la columna original.'''
        
    df["Indice_Identificador_agrupacio_organisme"] = df["Identificador_agrupacio_organisme"].str[:2]
    df["Indice_Identificador_agrupacio_organisme"].value_counts()

    clave_referencia_licitacio = {
        "08": "Barcelona",
        "80": "Barcelona",
        "96": "Departamentos_Estudio",
        "15": "Salud",
        "43": "Tarragona",
        "17": "Girona",
        "14": "Cultura",
        "25": "Lleida",
        "00": "Sociales",
        "81": "Consejo_Comarcal",
        "79": "Fundacion_privada",
        "98": "Desarrollo",
        "99": "Varios",
        "70": "Varios",
        "10": "Varios",
        "90": "Varios",
        "82": "Varios",
        "95": "Varios"

    }

    df["Referencia_Licitacio"] = df["Indice_Identificador_agrupacio_organisme"].map(clave_referencia_licitacio)
    df["Referencia_Licitacio"] = np.where(
        df["Identificador_agrupacio_organisme"] =="8000",
        "Universidades",
        df["Referencia_Licitacio"]
    )

    '''
    Y ahora hacemos la transformación pertinente, para convertir la feature en numerica 
    '''

    clave_referencia_licitacio_numerico = {
        "Barcelona": 1,
        "Departamentos_Estudio": 2,
        "Salud": 3,
        "Tarragona": 4,
        "Universidades": 5,
        "Girona": 6,
        "Cultura": 7,
        "Lleida": 8,
        "Sociales": 9,
        "Consejo_Comarcal": 10,
        "Fundacion_privada": 11,
        "Desarrollo": 12,
        "Varios": 13,  
    }
    df["Referencia_Licitacio"] = df["Referencia_Licitacio"].map(clave_referencia_licitacio_numerico)

    '''
    Una vez resuelto esto, toca eliminar la columna del identificador, dada su alta cardinalidad
    '''
    df.drop("Identificador_agrupacio_organisme", axis = 1, inplace = True)
    df.drop("Indice_Identificador_agrupacio_organisme", axis = 1, inplace = True)

    return df
                    
def tipus_de_contracte(df):

    ''''Esta función recibe un dataframe y devuelve el mismo dataframe con la columna
    de Tipus_de_contracte tratada, creando una nueva columna de referencia a partir de los tipos de contrato,
    y eliminando la columna original.'''

    clave_tipus_contracte = {
    '5. serveis': 1,
    '3. subministraments': 2,
    '1. obres': 3,
    '10. privat dadministracio publica': 4,
    'subministraments': 2,
    'gestio de servei public': 5,
    '8. concessio de serveis': 6   
    }

    df["Tipus_de_contracte"] = df["Tipus_de_contracte"].map(clave_tipus_contracte)

    return df

def codi_cpv(df, df_cpv):

    '''Esta función recibe un dataframe y otro dataframe con la información de los códigos CPV,
    y devuelve el mismo dataframe con la columna de Codi_CPV tratada, 
    creando nuevas columnas de referencia a partir de los códigos CPV, y eliminando la columna original.
    '''
    # los 2 primeros dígitos son la división
    df["Codi_CPV_div"] = df["Codi_CPV"].str[:2].str.zfill(2)
    df["Codi_CPV"] = df["Codi_CPV"].str[:8].astype(str).str.zfill(8)
    
    df = df.merge(df_cpv[['CPV_def','CPV_Descripcion', 'Tipo_de_contrato']], left_on='Codi_CPV', right_on='CPV_def', how='left')
    
    return df

def predecir(model, X_test, y_test):

    '''Esta función recibe un modelo, un conjunto de datos de prueba y una variable target de prueba,
    y devuelve el RMSE del modelo en escala logarítmica.'''
    
    # Predecir sobre el test
    y_pred = model.predict(X_test)
    y_pred = np.expm1(y_pred)
    y_test = np.expm1(y_test)

    # Calcular RMSE en escala logarítmica
    rmse_log = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE modelo {model}: {rmse_log:.4f}")


def objective(trial, X_train, y_train):
    param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 70),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "criterion": trial.suggest_categorical("criterion", ["squared_error", "absolute_error"]),
        "max_samples": trial.suggest_float("max_samples", 0.5, 1.0)
    }

    model = RandomForestRegressor(**param_grid, random_state=42, n_jobs=-1)

    score = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    ).mean()

    return score
