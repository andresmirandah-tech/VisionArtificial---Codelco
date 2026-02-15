"""
# main.py
-> Script principal de todo el programa

## OBJETIVO DEL SCRIPT
Contiene todos los pasos a ejecutar. 
Procesamiento de imágenes/Etiquetado/Split/Entrenamiento/Inferencia/Métricas y Porcentajes
El objetivo del script es mantener el orden de todo el programa.

## OBSERVACIÓN
El cambio que tiene respecto al main.py es el modo en que se hace el split.
En este script se hace split de las imágenes de entrenamiento, validación y las que usaremos en inferencia.
"""

import os
import sys
import yaml
import shutil
import subprocess
import random as rd
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image


# Función Extra 1
def pedir_modo():
    while True:
        modo = input("¿Entrenamiento o Inferencia?: ").strip()
        modo = modo.lower()
        if (modo in ["entrenamiento","inferencia"]):
            return modo
        print("Ingrese un modo válido")

# Función Extra 2
def clean_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir, ignore_errors = True) ### <- Borra la carpeta
    os.makedirs(dir,  exist_ok=True) ### <- Crea la carpeta vacía

# Función Extra 3
def load_config(path="config/config.yaml"):
    with open(path,"r") as f:
        config = yaml.safe_load(f)
    return config
      

### Ruta base del programa
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) ### <- Fija la carpeta base del programa. 
sys.path.append(BASE_DIR)   

### Rutas adicionales 
images_dir = os.path.join(BASE_DIR, "dataset", "raw_images") ### <- Directorio de las imágenes
train_img_dir = os.path.join(BASE_DIR, "dataset", "splits", "train", "images")
train_mask_dir = os.path.join(BASE_DIR, "dataset", "splits", "train", "masks")
val_img_dir = os.path.join(BASE_DIR, "dataset", "splits", "val", "images")
val_mask_dir = os.path.join(BASE_DIR, "dataset", "splits", "val", "masks")
inference_img_dir = os.path.join(BASE_DIR, "inference", "images")
inference_real_masks = os.path.join(BASE_DIR, "inference", "real_masks", "masks_labels")
inference_pred_masks = os.path.join(BASE_DIR, "inference", "outputs", "masks" )


### Limpiamos los directorios
if __name__ == "__main__":
    clean_dir(train_img_dir)
    clean_dir(train_mask_dir)
    clean_dir(val_img_dir)
    clean_dir(val_mask_dir)
    clean_dir(inference_img_dir)
    clean_dir(inference_real_masks)

### Importamos las etiquetas de labels.py
from utils.labels import Etiquetas

### Importamos las funciones de los otros scripts
from utils.rename import rename_images
from scripts.use_labelme import tags_maker
from scripts.json_to_masks import convert_json
from scripts.split_train_inference import split_train_inference
from training.train_deeplab import train_deeplab
from scripts.calculate_proportions import calculate_proportions
from inference.predict import run_inference
from metrics.segmentation_metrics import metrics
from metrics.plot_loss import plot_loss
from metrics.segmentation_global_metrics import global_metrics
from metrics.statistics_metrics import evaluate_with_statistics


flag = True
m = 0

# ==========================================================
# -1) Renombrar imágenes
# ==========================================================
while True:
    rename_option = input("¿Desea renombrar las imágenes? (s/n): ")
    rename_option = rename_option.strip()
    if (rename_option.lower() == "s"):
        rename_images(images_dir)
        break
    else:
        break


# ==========================================================
# 0) Pedimos el modo (Inferencia unicamente o entrenamiento)
# ==========================================================

while flag:
    mode = pedir_modo() ### <- Esto se hace para ver si ejecutamos los scripts de entrenamiento o los de inferencia

    if (mode == "entrenamiento"):
        print("Modo entrenamiento")
        m = 1
        break
    elif (mode == "inferencia"):
        print("Modo inferencia")
        m = 2
        break
    
# =========================================
# 1) Procesamiento de imágenes con Labelme
# =========================================

if (m == 1):
    print("Inicializando Labelme de Entrenamiento")
    images_dir = os.path.join(BASE_DIR, "dataset", "raw_images")
    json_dir = os.path.join(BASE_DIR, "dataset", "labelme_json")
    tags_maker(images_dir, json_dir, m)
    print("Etiquetado Finalizado")
    print("="*20)


# =========================================
# 2) Conversión JSON -> MASKS
# ==========================================

if (m == 1):
    json_dir = os.path.join(BASE_DIR, "dataset", "labelme_json")
    save_masks_dir = os.path.join(BASE_DIR, "dataset", "masks_labels")
    print("Conversión Json -> Masks de Entrenamiento")
    convert_json(json_dir, save_masks_dir, m)
    print("Conversión Finalizada")
    print("="*20)
    

# ===================================================
# 3) División de los datos (Entrenamiento/Validación)
# ===================================================
print("Split del Dataset")
inference_proportion = 0.1
train_proportion = 0.8
SEED = 42
rd.seed(SEED)


if (m == 1):
    for d in [train_img_dir, train_mask_dir, val_img_dir, val_mask_dir, inference_img_dir, inference_real_masks]: ### <- En caso de que no existan las 4 carpetas de la lista se crean. E.o.c se ignora la linea
        os.makedirs(d, exist_ok=True)

    split_train_inference(images_dir, train_img_dir, train_mask_dir, val_img_dir, val_mask_dir, inference_img_dir, inference_real_masks, train_proportion, inference_proportion)
    print("Split completado")
    print("="*20)


# =========================================
# 4) Entrenamiento del modelo
# ==========================================
config = load_config()
if (m == 1):
    train_deeplab(config)


# =========================================
# 5) Inferencia (Aplicación del Modelo)
# ==========================================

if (m == 1):
    MODEL_DIR = os.path.join(BASE_DIR, "models", "deeplab_model.pth")
    run_inference(config, MODEL_DIR)

elif (m == 2):
    print("Solo Inferencia")
    MODEL_DIR = os.path.join(BASE_DIR, "models", "deeplab_model.pth")
    run_inference(config, MODEL_DIR)


# =========================================
# 6) Cálculo de porcentajes de detección
# ==========================================
if (m == 1 or m == 2):
    calculate_proportions(os.path.join(BASE_DIR, "inference", "outputs", "masks"))


# =========================================
# 7) Grafico de la función de pérdida
# ==========================================

if (m == 1):
    plot_loss(os.path.join(BASE_DIR, "models", "training_loss.csv"  ))


# ================================================
# 8) Métricas (Individuales/Globales/Estadisticas)
# ================================================

if (m == 1 or m == 2):
    metrics()
    global_metrics(inference_real_masks,inference_pred_masks, config)
    evaluate_with_statistics(inference_real_masks, inference_pred_masks)
