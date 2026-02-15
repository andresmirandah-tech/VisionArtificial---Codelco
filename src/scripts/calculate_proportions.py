"""
# 04_calculate_proportions.py
-> Script utilizado en main.py para calcular los porcentajes de detección de cada elemento en la máscara predicha.

## OBJETIVO DEL SCRIPT
Calcular los porcentajes de detección de cada elemento con respecto al total de la imagen. Esto para las imágenes
que se encuentren en la carpeta de inferencia
"""

import os
import sys
import numpy as np
from PIL import Image


# =======================
# FORZAR PATH AL PROYECTO
# =======================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR) 

from utils.labels import Etiquetas


# ======================================================================================
# RUTAS DEL PROYECTO : Fijamos los directorios más importantes (Inferencia en este caso)
# ======================================================================================
MASK_DIR = os.path.join(BASE_DIR, "inference", "outputs", "masks")



def calculate_proportions(mask_dir):


    # ================================
    # RECORREMOS CADA IMAGEN PROCESADA
    # ================================
    for fname in os.listdir(mask_dir):

        if not fname.lower().endswith(".png"): ### <- Aseguramos que unicamente estemos trabajando con imágenes.
            continue


        path = os.path.join(MASK_DIR, fname)
        mask = np.array(Image.open(path))


        total_pixels = mask.size


        # Conteos
        n_borde    = np.sum(mask == Etiquetas["borde"])
        n_exterior = np.sum(mask == Etiquetas["exterior"])
        n_espuma   = np.sum(mask == Etiquetas["espuma"])
        n_burbuja  = np.sum(mask == Etiquetas["burbuja"])

        # Porcentajes directos
        p_burbuja_directo = n_burbuja / total_pixels

        # Porcentaje por complemento
        p_otras = (n_borde + n_exterior + n_espuma) / total_pixels
        p_burbuja_complemento = 1.0 - p_otras

        # =====================
        # OUTPUT
        # =====================
        print(f"{fname}")
        print(f"  % burbuja (directo)      : {p_burbuja_directo*100:.2f}%")
        print(f"  % burbuja (complemento)  : {p_burbuja_complemento*100:.2f}%")
        print(f"  diferencia      : {abs(p_burbuja_directo - p_burbuja_complemento)*100:.4f}%")
        print("-" * 50)

    #OBSERVACIÓN
    ## Se calcula de dos maneras distintas el porcentaje para chequear que coincidan ambos enfoques
    