"""
# 03_split_dataset.py
-> Script utilizado en main.py para dividir el dataset en entrenamiento y validación

## OBJETIVO DEL SCRIPT
Divide el conjunto de imagenes en entrenamiento y validación en una proporcion (80%/20%). 
Agrupa las imágenes desde "raw_images" y las respectivas mascaras que se encuentran en "masks_labels" 
asegurandose de que coincidan en nombre. 
NOTA: La estandarización inicial de los nombres de las imágenes es útil para este propósito.
"""

import os
import random
import shutil
import numpy as np
from PIL import Image


def copy_pair(filename, img_dst, mask_dst): ### <- Copia un par Imagen/Máscara en otro directorio

    #NO OLVIDAR QUE LA EL PAR (IMAGEN,MÁSCARA) DEBE COINCIDIR, ADEMÁS CADA ELEMENTO EN EL ENTRENAMIENTO DEBE SER EN PARES DE ESTA FORMA.
    shutil.copy( ### <- Este primer copy es para la imagen
        os.path.join(images_dir, filename), ### <- Ruta de origen
        os.path.join(img_dst, filename) ### <- Ruta de destino
    )
    shutil.copy( ### <- Este segundo copy es para la máscara
        os.path.join(mask_dir, filename), ### <- Ruta de origen
        os.path.join(mask_dst, filename) ### <- Ruta de destino
    )


# =======================
# FORZAR PATH AL PROYECTO
# =======================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# RUTAS DEL PROYECTO : Fijamos los directorios más importantes
# ============================================================
images_dir = os.path.join(BASE_DIR, "dataset", "raw_images") ### <- Directorio de las imágenes
mask_dir  = os.path.join(BASE_DIR, "dataset", "masks_labels") ### <- Directorio de las masks

SPLIT_DIR  = os.path.join(BASE_DIR, "dataset", "splits") ### <- Directorio de guardado para el split

TRAIN_IMG_DIR = os.path.join(SPLIT_DIR, "train", "images") ### <- Directorio de guardado de las imágenes de entrenamiento
TRAIN_MSK_DIR = os.path.join(SPLIT_DIR, "train", "masks") ### <- Directorio de guardado de las masks de entrenamiento
VAL_IMG_DIR   = os.path.join(SPLIT_DIR, "val", "images") ### <- Directorio de guardado de las imágenes de validación
VAL_MSK_DIR   = os.path.join(SPLIT_DIR, "val", "masks") ### <- Directorio de guardado de las masks de validación



SEED = 42
random.seed(SEED)


def split(images_dir, train_img_dir, train_mask_dir, val_img_dir, val_mask_dir, train_proportion):
    images = sorted([
        f for f in os.listdir(images_dir)
        if f.endswith(".png")
    ])  

#El método del bloque anterior permite construir la lista a partir del for sin necesidad de hacer algo como
#for x in archivo:
#lista.append(x) / lista[i] = x / etc...

#La sintaxis es: 
# [f for f in origen] ; Si queremos añadimos alguna condición como en el ejemplo de arriba, 
# en lugar de hacer algo como:

# for x in archivo
# if (condicion): lista.append(x)


    random.shuffle(images) ### <- Reordenamos los elementos de forma aleatoria. En este caso particular el random con la semilla fijada arriba


    n_train = int(len(images) * train_proportion)


    # En este bloque se aplica slicing sobre la lista que contiene los nombres de las imágenes del dataset
    # Para el entrenamiento se toman los elementos desde el indice 0 hasta el (n_train)-1
    # Para la validación se toman los elementos restantes desde el indice n_train
    train_images = images[:n_train]
    val_images   = images[n_train:]


    #Ambos ciclos for sirven para copiar el par (imagen,máscara) en las carpetas correspondientes.
    for f in train_images:
        copy_pair(f, train_img_dir, train_mask_dir)

    for f in val_images:
        copy_pair(f, val_img_dir, val_mask_dir)


    #Chequeamos como quedó la división del dataset
    print(f"Train: {len(train_images)} imágenes")
    print(f"Val:   {len(val_images)} imágenes")



    # ===============================
    # CHECKS ADICIONALES 
    # ===============================
    #Todos estos bloques son únicamente para verificar que los pares fueron copiados correctamente.
    #Es importante para el entrenamiento que cada imagen tenga su respectivo máscara en la carpeta
    print("train images:", len(os.listdir(f"{BASE_DIR}/dataset/splits/train/images")))
    print("train masks :", len(os.listdir(f"{BASE_DIR}/dataset/splits/train/masks")))
    print("val images  :", len(os.listdir(f"{BASE_DIR}/dataset/splits/val/images")))
    print("val masks   :", len(os.listdir(f"{BASE_DIR}/dataset/splits/val/masks")))

    train_imgs = sorted(os.listdir(f"{BASE_DIR}/dataset/splits/train/images"))
    train_msks = sorted(os.listdir(f"{BASE_DIR}/dataset/splits/train/images"))
    assert train_imgs == train_msks, "nombres no calzan en train"

    val_imgs = sorted(os.listdir(f"{BASE_DIR}/dataset/splits/train/images"))
    val_msks = sorted(os.listdir(f"{BASE_DIR}/dataset/splits/train/images"))
    assert val_imgs == val_msks, "nombres no calzan en val"

    print("Todos los nombres calzan")

    m = np.array(Image.open(os.path.join(f"{BASE_DIR}/dataset/splits/train/masks", train_msks[1])))
    print("unique:", np.unique(m))
