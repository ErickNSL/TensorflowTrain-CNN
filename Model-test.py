import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from dotenv import load_dotenv

# paths
load_dotenv()
modelo_path = 'modelo_figuras.h5'
ruta_imagen_nueva = os.getenv("TEST_IMG")

# load model
modelo = tf.keras.models.load_model(modelo_path)

# label mapping
mapeo_etiquetas = {0: "Círculo", 1: "Cuadrado"}


def predecir_imagen(ruta_imagen):
    # load and process image
    imagen = load_img(ruta_imagen, target_size=(190, 190))
    imagen_array = img_to_array(imagen)
    imagen_array = imagen_array / 255.0
    imagen_array = np.expand_dims(imagen_array, axis=0)  # add batch dimension

    # make prediction
    prediccion = modelo.predict(imagen_array, verbose=0)
    clase_predicha = int(np.argmax(prediccion))
    probabilidad = float(np.max(prediccion))

    # print result
    print(f"Predicción: {mapeo_etiquetas[clase_predicha]} (Probabilidad: {probabilidad*100:.2f}%)")

    return mapeo_etiquetas[clase_predicha], probabilidad


# call function
resultado, prob = predecir_imagen(ruta_imagen_nueva)
