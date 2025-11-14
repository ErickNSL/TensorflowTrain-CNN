import numpy as np
import sklearn, json
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models




#########################################################################
### Entrena el modelo solo con las imagenes normalizadas del data set ###
### Sin usar funciones de Agumentation                                ###
#########################################################################




### PATHS
ruta_imagenes = 'normalizedimg'
archivo_labels = 'labels.json'


### CARGA LAS IMAGENES Y ETIQUETAS EN ARRAYS NP
def cargar_datos_desde_json(archivo_json):
    with open(archivo_json, 'r') as f:
        data = json.load(f)

    imagenes = []
    etiquetas = []

    for ruta_imagen, etiqueta in data:
        print(f"Loading: {ruta_imagen}, Label: {etiqueta}")
        try:
            imagen = np.load(ruta_imagen)
            print(f"Loaded image shape: {imagen.shape}")
            imagenes.append(imagen)
            etiquetas.append(etiqueta)
        except Exception as e:
            print(f"Error loading {ruta_imagen}: {e}")
    
    return np.array(imagenes), np.array(etiquetas)

### GUARDA LOS ARRAY NP EN LAS VARIABLES
imagenes, etiquetas = cargar_datos_desde_json(archivo_labels)

### VERIFICA LOS CANALES DE LA IMAGEN Y AÑADE EL FALTANTE
if len(imagenes.shape) == 3:  
    imagenes = imagenes[..., np.newaxis]



# Dividir los datos en conjuntos de entrenamiento y validación
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(imagenes, etiquetas, test_size=0.2, random_state=42)



### CONSTRYE EL MODELO CON LAS REDES
modelo = models.Sequential([
    layers.Input(shape=(190, 190, 3)),  # Tamaño de la imagen (190x170) y 3 canales (RGB)
    layers.Conv2D(32, (3, 3), activation='relu'),  # Capa convolucional
    layers.MaxPooling2D((2, 2)),  # Pooling para reducir dimensiones
    layers.Conv2D(64, (3, 3), activation='relu'),  # Otra capa convolucional
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),  # Más profundidad en la red
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),  # Aplanar para pasar a capas densas
    layers.Dense(128, activation='relu'),  # Capa completamente conectada
    layers.Dense(3, activation='softmax')  # Capa de salida con 3 clases
])
### COMPILACION
modelo.compile( optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])



### PARAMETROS DE AJUSTE DEL ENTRENAMIENTO DEL MODELO, VUELTAS Y CANTIDADES
historial = modelo.fit(x_train, y_train, epochs=90, batch_size=7, validation_data=(x_val, y_val))

### GUARDA EL MODELO
modelo.save('modelo_figuras.h5')



### EVALUACION DEL MODELO
loss, accuracy = modelo.evaluate(x_val, y_val, verbose=2)
print(f"Model Accuracy: {accuracy:.2f}")







