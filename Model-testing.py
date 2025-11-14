import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from dotenv import load_dotenv




####################
### TEST THE MODEL AND COUNT THE OBJECTS
####################



# paths
modelo = tf.keras.models.load_model('modelo_figuras.h5')
load_dotenv()


def clasificar_imagen(ruta_imagen):
    # load the image
    img = image.load_img(ruta_imagen, target_size=(190, 190))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    
    # normalize the image
    img_array = img_array / 255.0

    # make prediction
    pred = modelo.predict(img_array, verbose=0)

    # get the label with highest probability
    return int(np.argmax(pred, axis=1)[0])


def contar_figuras(ruta_imagen):
    # initialize counters
    contador_circulos = 0
    contador_cuadrados = 0

    clase = clasificar_imagen(ruta_imagen)

    if clase == 0:
        contador_circulos = 1
    elif clase == 1:
        contador_cuadrados = 1

    return contador_circulos, contador_cuadrados


# image path to test
ruta_imagen = os.getenv("TEST_IMG")

# count figures in the image
circulos, cuadrados = contar_figuras(ruta_imagen)

# show results
print(f"Circulos: {circulos}")
print(f"Cuadrados: {cuadrados}")