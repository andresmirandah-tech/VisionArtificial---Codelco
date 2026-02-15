"""
# statistics_metrics.py
-> Script utilizado en main.py para calcular métricas estadísticas

## OBJETIVO DEL SCRIPT
Entrena el modelo utilizando el dataset armado anteriormente.
Por ahora no se ha prestado tanta atención a los hiperparametros}

Desviación estándar
"""

import os 
import cv2
import sys
import numpy as np

from utils.labels import Etiquetas

def compute_statistics(values):
    values = np.array(values)
    values = values[~np.isnan(values)]

    if (len(values) == 0):
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    return {
        "mean" : np.mean(values),
        "std" : np.std(values),
        "min" : np.min(values),
        "max" : np.max(values)
    }

def evaluate_with_statistics(real_dir, pred_dir):
    results_iou = {cname: [] for cname in Etiquetas.keys()}
    results_dice = {cname: [] for cname in Etiquetas.keys()}
    results_bubble = []

    for fname in os.listdir(real_dir):
        real_path = os.path.join(real_dir, fname)
        pred_path = os.path.join(real_dir, fname)

        if not os.path.exists(pred_path):
            continue
        real = cv2.imread(real_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        if real is None or pred is None:
            continue

        for cname, cid in Etiquetas.items():

            verdadero = (real == cid)
            predicho = (pred == cid)

            inter = np.logical_and(verdadero, predicho).sum()
            union = np.logical_or(verdadero, predicho).sum()

            if (union == 0):
                iou = np.nan
            else:
                iou = inter/union

            total = verdadero.sum() + predicho.sum()
            if (total == 0):
                dice = np.nan
            else: 
                dice = (2*inter)/total
            
            results_iou[cname].append(iou)
            results_dice[cname].append(dice)

        pixel_burbuja = (pred == Etiquetas["burbuja"]).sum()
        total = pred.size
        results_bubble.append(100*pixel_burbuja/total)
    print("\n===== ESTADÍSTICAS GLOBALES ====\n")

    for cname in Etiquetas.keys():
        stats_iou = compute_statistics(results_iou[cname])
        stats_dice = compute_statistics(results_dice[cname])

        print(f"Clase: {cname}")
        print(f"  IoU  -> mean: {stats_iou['mean']:.4f} | std: {stats_iou['std']:.4f}")
        print(f"  Dice -> mean: {stats_dice['mean']:.4f} | std: {stats_dice['std']:.4f}")
 
    stats_bubble = compute_statistics(results_bubble)
    print("Porcentaje de burbujas (predicción)")
    print(f"  mean: {stats_bubble['mean']:.2f}% | std: {stats_bubble['std']:.2f}%")


