"""
# rename.py
-> Script utilizado en main.py para estandarizar los nombres de las imágenes del dataset

## OBJETIVO DEL SCRIPT
Cambia el nombre de todas las imágenes del dataset para dejarlos en un formato estándar.
Formato: "Img_000X.formato" donde "formato in {png,jpg,jpeg}"
"""

import os
import sys

def rename_images(images_dir):
    # ===============================
    # RUTA A LAS IMÁGENES
    # ===============================
    folder = images_dir

    # ===============================
    # LISTAR ARCHIVOS
    # ===============================
    files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])


    # ===============================
    # RENOMBRAR
    # ===============================
    for i, filename in enumerate(files, start=1):
        old_path = os.path.join(folder, filename)
        new_name = f"img_{i:04d}.png"
        new_path = os.path.join(folder, new_name)

        os.rename(old_path, new_path)
        print(f"{filename} -> {new_name}")

    print("Renombrado completado")
