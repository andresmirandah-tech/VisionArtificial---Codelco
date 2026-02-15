"""
# mean_global_metrics.py
-> Script para calcular métricas de desempeño del modelo de segmentación.

## OBJETIVO DEL SCRIPT
Compara máscaras anotadas manualmente con máscaras predichas por el modelo 

Métricas:
- IoU por clase (Intersección sobre Unión)
- Dice Similarity Coefficient (DSC) por clase
- Porcentaje de burbujas
"""
import os
import numpy as np
from PIL import Image
 
# ===============================
# CONFIGURACIÓN
# ===============================
        


def global_metrics(real_mask_dir, pred_mask_dir, config):

    NUM_CLASSES = config["training"]["num_classes"]

    # ===============================
    # ACUMULADORES GLOBALES
    # ===============================
    total_intersection = 0
    total_union = 0
    total_pred_pixels = 0
    total_real_pixels = 0
    
    class_intersections = np.zeros(NUM_CLASSES)
    class_unions = np.zeros(NUM_CLASSES)
    class_pred_pixels = np.zeros(NUM_CLASSES)
    class_real_pixels = np.zeros(NUM_CLASSES)
        

    # ===============================
    # LOOP SOBRE IMÁGENES
    # ===============================
    mask_files = os.listdir(pred_mask_dir)
    
    for mask_file in mask_files:
    
        if not mask_file.endswith(".png"):
            continue
    
        pred_path = os.path.join(pred_mask_dir, mask_file)
        real_path = os.path.join(real_mask_dir, mask_file)
    
        if not os.path.exists(real_path):
            print(f"[WARNING] No existe máscara real para {mask_file}")
            continue
    
        pred_mask = np.array(Image.open(pred_path))
        real_mask = np.array(Image.open(real_path))
    
        for class_id in range(NUM_CLASSES):
    
            pred_class = (pred_mask == class_id)
            real_class = (real_mask == class_id)
    
            intersection = np.logical_and(pred_class, real_class).sum()
            union        = np.logical_or(pred_class, real_class).sum()
    
            pred_pixels = pred_class.sum()
            real_pixels = real_class.sum()
    
            # Acumular por clase
            class_intersections[class_id] += intersection
            class_unions[class_id] += union
            class_pred_pixels[class_id] += pred_pixels
            class_real_pixels[class_id] += real_pixels
    
            # Acumular global
            total_intersection += intersection
            total_union += union
            total_pred_pixels += pred_pixels
            total_real_pixels += real_pixels
    
        
    # ===============================
    # MÉTRICAS POR CLASE
    # ===============================
    
    print("\n==============================")
    print("MÉTRICAS POR CLASE")
    print("==============================")
    
    for class_id in range(NUM_CLASSES):
    
        iou = (
            class_intersections[class_id] / class_unions[class_id]
            if class_unions[class_id] != 0 else 0
        )
    
        dsc = (
            (2 * class_intersections[class_id]) /
            (class_pred_pixels[class_id] + class_real_pixels[class_id])
            if (class_pred_pixels[class_id] + class_real_pixels[class_id]) != 0 else 0
        )
    
        print(f"\nClase {class_id}")
        print(f"IoU : {iou:.4f}")
        print(f"DSC : {dsc:.4f}")
    
    
    # ===============================
    # MÉTRICAS GLOBALES
    # ===============================
    
    global_iou = total_intersection / total_union if total_union != 0 else 0
    global_dsc = (
        (2 * total_intersection) /
        (total_pred_pixels + total_real_pixels)
        if (total_pred_pixels + total_real_pixels) != 0 else 0
    )
    
    print("\n==============================")
    print("MÉTRICAS GLOBALES")
    print("==============================")
    print(f"IoU Global : {global_iou:.4f}")
    print(f"DSC Global : {global_dsc:.4f}")
    print("==============================")

