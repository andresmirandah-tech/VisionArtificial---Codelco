"""
# 00_use_labelme.py
-> Script utilizado en main.py para ejectuar labelme y armar los respectivos json por cada imágen.

## OBJETIVO DEL SCRIPT
Ejecutar labelme con las imagenes ya cargadas desde "raw_images" y guardar cada "Json" en "dataset/labelme_json".
"""
import os
import subprocess ### <- Permite ejecutar comandos de terminal

def tags_maker(images_dir,json_dir,mode):
    os.makedirs(json_dir, exist_ok=True)
    subprocess.run([ ### <- Permite ejecutar una linea del CMD. En este caso inicializa el labelme con todas las imagenes de "raw_images" ya cargadas
    "labelme",
    images_dir,
    "--output",
    json_dir
])
    

