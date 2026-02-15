"""
# 02_colorize_masks.py
-> Script utilizado en main.py unicamente para corroborar que la conversión de json a masks fue correcta (Tal como fue construido en el labelme)
Y que además coincide con el diccionario de colores definido en labels.py

## OBJETIVO DEL SCRIPT
Colorear las imágenes obtenidas por "01_json_to_masks.py" unicamente para corroborar que la traducción con el diccionario de clases y colores es correcto.
El coloreado tambien será de utilidad para analizar las predicciones hechas por el modelo.

## OBSERVACIÓN
- LAS IMÁGENES COLOREADAS NO SE USAN PARA ENTRENAR EL MODELO, SOLO SON PARA VERIFICACIÓN
"""

import os
import sys
import numpy as np
from PIL import Image

# =======================
# FORZAR PATH AL PROYECTO
# =======================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from utils.labels import Colores


# ============================================================
# RUTAS DEL PROYECTO : Fijamos los directorios más importantes
# ============================================================
MASKS_LABELS_DIR = os.path.join(BASE_DIR, "dataset", "masks_labels")
MASKS_COLOR_DIR  = os.path.join(BASE_DIR, "dataset", "masks_color")

os.makedirs(MASKS_COLOR_DIR, exist_ok=True)


# ===============================
# COLOREAR MÁSCARAS
# ===============================
for mask_file in os.listdir(MASKS_LABELS_DIR):

    if not mask_file.endswith(".png"): ### <- Aseguramos de trabajar solo con imágenes
        continue 

    mask_path = os.path.join(MASKS_LABELS_DIR, mask_file)

    # Leer máscara numérica
    mask = np.array(Image.open(mask_path))

    # Imagen RGB vacía
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8) 

    # Asignar colores por clase
    for class_id, color in Colores.items():
        color_mask[mask == class_id] = color

    # Guardar imagen coloreada
    out_path = os.path.join(MASKS_COLOR_DIR, mask_file)
    Image.fromarray(color_mask).save(out_path)

    print(f"máscara coloreada: {mask_file}")

print("Coloreado completado")

