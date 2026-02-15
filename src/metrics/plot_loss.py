import os
import sys
import pandas as pd
import matplotlib.pyplot as plt


# =======================
# FORZAR PATH AL PROYECTO
# =======================


def plot_loss(csv_dir):
    df = pd.read_csv(csv_dir)

    plt.plot(df["epoch"], df["loss"]  )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.show()