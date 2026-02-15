"""
# 02_colorize_masks_for_inference.py
-> Script utilizado en main.py unicamente para corroborar que la conversión de json a masks fue correcta (Tal como fue construido en el labelme)
Y que además coincide con el diccionario de colores definido en labels.py

## OBJETIVO DEL SCRIPT
Colorear las imágenes obtenidas por "01_json_to_masks.py" unicamente para corroborar que la traducción con el diccionario de clases y colores es correcto.
El coloreado tambien será de utilidad para analizar las predicciones hechas por el modelo.

## OBSERVACIÓN
- LAS IMÁGENES COLOREADAS NO SE USAN PARA ENTRENAR EL MODELO, SOLO SON PARA VERIFICACIÓN
- Lo único que cambia respecto del script "02_colorize_masks" es el directorio en el cual busca las imágenes y donde las guarda.
Posteriormente se eliminará y conservará solo la primera versión, pensando en controlar estas diferencias en el "main.py" del programa.
"""

import os
import sys
import numpy as np
from PIL import Image

# =======================
# FORZAR PATH AL PROYECTO
# =======================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR) ### <- Fija la carpeta base del programa. En este caso nos la última ubicación que contiene es "Estructuracion General"

from utils.labels import Colores

# ============================================================
# RUTAS DEL PROYECTO : Fijamos los directorios más importantes
# ============================================================
MASKS_LABELS_DIR = os.path.join(BASE_DIR, "inference","outputs", "masks") ### <- Directorio de las masks
MASKS_COLOR_DIR  = os.path.join(BASE_DIR, "inference","outputs","colored") ### <- Directorio de guardado de las imágenes coloreadas

os.makedirs(MASKS_COLOR_DIR, exist_ok=True)


# ===============================
# Colorear máscaras
# ===============================
for mask_file in os.listdir(MASKS_LABELS_DIR):

    if not mask_file.endswith(".png"): ### <- Aseguramos de trabajar solo con imágenes
        continue 

    mask_path = os.path.join(MASKS_LABELS_DIR, mask_file)

    # Leer máscara numérica como un arreglo de numpy
    mask = np.array(Image.open(mask_path))

    # Incializamos el contenedor de la imagen RGB vacío
    print(mask.shape)
    h, w = mask.shape ### <- Obtenemos las dimensiones de la imagen
    color_mask = np.zeros((h, w, 3), dtype=np.uint8) ### <- Inicializamos el tensor. El tercer parámetro representa precisamente el canal RGB

    # Asignar colores por clase
    for class_id, color in Colores.items(): ### <- Ocupamos .items() que nos recupera una tupla : ("Llave",valor) del diccionario
        color_mask[mask == class_id] = color ### <- Reemplaza todas los pixeles que coincidan de mask con class_id y se les asigna el color

    # Guardar imagen coloreada
    out_path = os.path.join(MASKS_COLOR_DIR, mask_file)
    Image.fromarray(color_mask).save(out_path)

    print(f"máscara coloreada: {mask_file}")

print("Coloreado completada")

