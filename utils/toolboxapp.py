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
                <li>Las variables más importantes para la predicción son: <strong>CPV_Descripcion, Duracion_total, Adjudicatari, Procediment_dadjudicacio</strong> y <strong>Tipus_de_contracte</strong>.</li>
                <li>El modelo ha sido entrenado y validado con datos reales de licitaciones.</li>
            </ul>
            <p>
                Para realizar una predicción, pulse el siguiente enlace:<br>
                <a href="https://mydespliegue.onrender.com/api/v1/predict" target="_blank">
                    https://mydespliegue.onrender.com/api/v1/predict
                </a>
            </p>
        </body>
        </html>
    """