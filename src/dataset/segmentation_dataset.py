"""
# segmentation_dataset.py
-> Script utilizado en main.py enfocado al entrenamiento del modelo

## OBJETIVO DEL SCRIPT
En la implementación del modelo es necesario por ejemplo aplicar transformaciones a las imágenes y las masks de las imágenes
Esta clase permite procesar estos elementos tratandolos como objetos de la clase "SegmentationDataset". (Esto último permite asignarle estos cambios como si fuesen atributos de la clase)
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class SegmentationDataset(Dataset): ### <- A modo sencillo, esta clase le "enseña" a PyTorch como leer las imágenes del dataset
    def __init__(self, root_dir, split="train",
                 image_transform=None,
                 mask_transform=None):

        self.image_dir = os.path.join(root_dir, split, "images") ### <- Parametro definido internamente a partir de otro del constructor.
        self.mask_dir  = os.path.join(root_dir, split, "masks") ### <- Parametro definido internamente a partir de otro del constructor.

        self.files = sorted(os.listdir(self.image_dir)) ### <- Parametro definido internamente a partir de otro del constructor.

        self.image_transform = image_transform
        self.mask_transform  = mask_transform

    def __len__(self):
        return len(self.files) ### <- Entrega el tamaño del dataset

    def __getitem__(self, idx): ### <- De manera sencilla, este método se ejecuta cuando el modelo "necesita una muestra".
        fname = self.files[idx] ### <- Obtiene la imagen en la posición idx

        image = Image.open(os.path.join(self.image_dir, fname)).convert("RGB") ### <- Se inicializa con los canales "RGB" porque la imagen es a color
        mask  = Image.open(os.path.join(self.mask_dir, fname)) ### <- La máscara no es necesario que pase por el mismo tratamiento dado que solo interesa la información que contenga al verla como array

        if self.image_transform:
            image = self.image_transform(image) 

        if self.mask_transform:
            mask = self.mask_transform(mask)

        mask = torch.from_numpy(
            __import__("numpy").array(mask)
        ).long()

        return image, mask


### Los dos métodos implementados en la clase son necesarios por "Dataset"
### Ejemplos de transformaciones pueden ser el "Resize", "ToTensor"
### El segundo método retorna el par (imagen,máscara) necesarios para el entrenamiento del modelo.
