"""
# segmentation_metrics.py
-> Script para calcular métricas de desempeño del modelo de segmentación.

## OBJETIVO DEL SCRIPT
Compara máscaras anotadas manualmente con máscaras predichas por el modelo 

Métricas:
- IoU por clase (Intersección sobre Unión)
- Dice Similarity Coefficient (DSC) por clase
- Porcentaje de burbujas
"""

import os
import cv2
import sys
import numpy as np
from PIL import Image

# =======================
# FORZAR PATH AL PROYECTO
# =======================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)  ### <- Fija la carpeta base del programa. En este caso nos la última ubicación que contiene es "Estructuracion General"



from utils.labels import Etiquetas


# ===========================
# MÉTRICAS (IoU/DSC/Burbujas)
# ===========================
def IoU(y_verdadero, y_pred, class_id): ### <- Cuantifica el solapamiento entre la máscara real y la predicha
   
    #Este bloque ahorra la escritura de sentencias condicionales. 
    #En el caso de que y_verdadero , y_pred sean arrays de numpy, este bloque almacena en verdadero y pred arreglos de igual dimensiones con booleanos
    verdadero = (y_verdadero == class_id)
    pred = (y_pred == class_id)

    # True = 1 ; False = 0
    interseccion = np.logical_and(verdadero, pred).sum() ### <- Evalua los valores de verdad de verdadero y pred con un conector "and" y los suma todos
    union = np.logical_or(verdadero, pred).sum() ### <- Evalua los valores de verdad de verdadero y pred con un conector "or" y los suma todos

    if union == 0: ### <-Esto implica que verdadero=pred=0 en todas las posiciones, por ende la intersección es vacía. Sin embargo este caso genera un 0/0 que se arregla con un return forzado
        return np.nan ### <- Son completamente disjuntas. Pésima predicción

    return interseccion / union ### <- El caso teórico bueno es que retorne 1 exacto. En la pratica debe ser lo mas cercano a 1 puesto que será casi imposible que sea exactamente 1


def dice_score(y_verdadero, y_pred, class_id): ### <- Se obtiene por cada clase por separado
    verdadero = (y_verdadero == class_id)
    pred = (y_pred == class_id)

    interseccion = np.logical_and(verdadero, pred).sum()
    total = verdadero.sum() + pred.sum()

    if total == 0:
        return np.nan
    
    num = 2*interseccion

    return num / total


def Porcentaje_Burbuja(mask): ### <- Se puede calcular esto entre la máscara real y la predicha

    pixel_burbuja = (mask == Etiquetas["burbuja"]).sum()
    total = mask.size

    if total == 0:
        return np.nan

    return 100 * (pixel_burbuja / total)



def metrics():
    REAL_DIR = os.path.join(BASE_DIR, "inference","real_masks","masks_labels")
    PRED_DIR = os.path.join(BASE_DIR, "inference","outputs","masks")

    real_files = sorted(os.listdir(REAL_DIR))


    print("Métricas \n")
    mean = {}
    list_metricas = ["IoU","DSC"]

    for fname in real_files:

        real_path = os.path.join(REAL_DIR, fname)
        pred_path = os.path.join(PRED_DIR, fname)


        if not os.path.exists(pred_path):
            print(f"No hay predicción para {fname}")
            continue


        real = cv2.imread(real_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)


        if real is None or pred is None:
            print(f"No se pudo leer {fname}")
            continue

        print(f"Imagen: {fname}")


        for cname, cid in Etiquetas.items():

            # Calculo de métricas
            iou = IoU(real, pred, cid)
            dice = dice_score(real, pred, cid)

            print(f"  {cname:10s} | IoU: {iou:.3f} | Dice: {dice:.3f}")



        porcentaje_burbuja_pred = Porcentaje_Burbuja(pred)
        porcentaje_burbuja_real = Porcentaje_Burbuja(real)
        print(f"  % burbujas (pred): {porcentaje_burbuja_pred:.2f}%\n")
        print(f"  % burbujas (real): {porcentaje_burbuja_real:.2f}%\n")
        print(f"  % Error: {abs(porcentaje_burbuja_pred-porcentaje_burbuja_real):.2f}%\n")



"""
VALORES PARA EL DSC
0.80-1.00: Very high similarity 
0.60-0.79: High similarity 
0.40-0.59: Moderate similarity
0.20-0.39: Low similarity 
0.00-0.19: Very low similarity 
"""