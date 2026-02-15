"""
# 01_json_to_masks.py
-> Script utilizado en main.py para convertir los documentos json a un array de numpy

## OBJETIVO DEL SCRIPT
Rescatar los documentos .json obtenidos anteriormente de labelme y convertirlos a un array de numpy para trabajarlo en Python.
Este array contiene los pixeles de las imagenes.
"""
import json
import os
import numpy as np
import sys
import cv2
from PIL import Image  ### <- Junto con cv2, librería para el tratamiento de imágenes.  PIL es para operaciones más básicas a diferencia de OpenCV
from utils.labels import Etiquetas


def convert_json(json_dir, mask_dir, mode):
    os.makedirs(mask_dir, exist_ok=True)
    
    for json_file in os.listdir(json_dir):
        if not json_file.endswith(".json"): ### <- Con esta línea nos aseguramos de procesar unicamente archivos .json. De no tener dicha extensión, se ignora.
            continue
        json_path = os.path.join(json_dir, json_file)

        # Leemos el JSON
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f) ### <- Convierte el string del json a un diccionario de python. Contiene atributos como el tamaño de la imagen y el etiquetado hecho con labelme
    
    #OBSERVACIÓN
    #Los json generados por el labelme consisten en un diccionario "grande", sus llaves nos entregan elementos como "strings", "listas", "diccionarios".
    # Por ejemplo los puntos de las regiones hechas en el labelme se guardan con un llave denominada "points" cuyo valor es una lista de listas de floats.
    #Por tal motivo la variable que guarde el método json.load() de estos archivos será un diccionario.


    #Rescatamos la dimensión de la imagen
        height = data["imageHeight"]
        width  = data["imageWidth"]


    #Inicializamos la máscara con el color de la burbuja
        mask = np.full(
            (height, width),
            Etiquetas["burbuja"],
            dtype=np.uint8
        )  ### <- Se crea el arreglo de numpy 2D de height X weight. Como es np.full se rellena con lo que indique el segundo argumento, en este caso "Burbuja_ID"


        #Marcamos los polígonos
        for shape in data["shapes"]: ### <- data["shapes"] es una lista de diccionarios. Cada diccionario contiene las coordenadas de cada etiqueta puesta sobre la imagen

            etiqueta = shape["label"]

            # Validación adicional en caso de que no se haya hecho correctamente el etiquetado.
            if etiqueta not in Etiquetas:
                raise ValueError(
                    f"Etiqueta desconocida '{etiqueta}' en {json_file}. "
                    f"Etiquetas válidas: {list(Etiquetas.keys())}"
                )

            class_id = Etiquetas[etiqueta]

            # Puntos del polígono (Labelme to OpenCV)
            points = np.array(
                shape["points"],
                dtype=np.int32
            ) ### <- Transforma la lista de listas que contiene los puntos en un arreglo de numpy. El segundo argumento fuerza a que los elementos sean enteros, así se evitan bugs.


            # Rellenar polígono con el id correspondiente
            cv2.fillPoly(mask, [points], class_id) 


        # Guardamos la máscara
        mask_name = os.path.splitext(json_file)[0] + ".png"
        mask_path = os.path.join(mask_dir, mask_name)

        Image.fromarray(mask).save(mask_path)
        print(f"Máscara creada: {mask_name}")
