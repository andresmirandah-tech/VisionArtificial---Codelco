"""
# training_deeplab.py
-> Script utilizado en main.py para entrenar el modelo

## OBJETIVO DEL SCRIPT
Entrena el modelo utilizando el dataset armado anteriormente.
Por ahora no se ha prestado tanta atención a los hiperparametros}

## OBSERVACIÓN
Los hiperparametros del modelo deberian estar en config/config.yaml  !!!
"""
import sys
import os
import csv
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50


# =======================
# FORZAR PATH AL PROYECTO
# =======================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)


from dataset.segmentation_dataset import SegmentationDataset

def train_deeplab(config):
 
   # ================================
   # PARAMETROS PARA EL ENTRENAMIENTO
   # ================================
   ROOT_DIR = os.path.join(BASE_DIR, "dataset/splits")
   BATCH_SIZE = config["training"]["batch_size"]
   NUM_CLASSES = config["training"]["num_classes"]
   EPOCHS = config["training"]["epochs"]
   LR = config["training"]["learning_rate"]
   DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


   # ================================================
   # TRANSFORMACIONES A LAS IMÁGENES Y A LAS MÁSCARAS
   # ================================================
   # En este bloque se definen las transformaciones tanto a las imágenes como a las máscaras.
   image_transform = T.Compose([
       T.Resize((512, 512)), ### <- Todas las imágenes las coloca de ese tamaño. (Se puede perder información pero se gana eficiencia computacional)
       T.ToTensor(), 
       T.Normalize(  ### <- Estandariza para cada canal con el promedio mu y la desviación estándar sigma. Este paso ya sea reescalar o normalizar el dataset es parte del procesamiento de datos. Así evitamos el sesgo.
           mean=[0.485, 0.456, 0.406],
           std=[0.229, 0.224, 0.225]
       )
   ])

   mask_transform = T.Compose([
       T.Resize((512, 512), interpolation=T.InterpolationMode.NEAREST)
   ])


   # ========================
   # PREPARACIÓN DEL DATASET
   # ========================
   # Utilizamos la clase definida en "segmentation_dataset.py". Se inicializa el dataset de entrenamiento como un objeto de la clase construida.
   # Se define el directorio de las imágenes que se utilizarán y se añaden las transformaciones definidas anteriormente como atributos de este objeto.
   train_dataset = SegmentationDataset(
       root_dir=ROOT_DIR,
       split="train",
       image_transform=image_transform,
       mask_transform=mask_transform
   )


   train_loader = DataLoader(  ### <- Actua como intermediario entre el dataset y el modelo. Simplifica el proceso de cargado y creación de los "batches" para entrenamiento y evaluación.
       train_dataset,
       batch_size=BATCH_SIZE,
       shuffle=True,
       drop_last = True
   )


   # =====================
   # MODELO PRE-ENTRENADO
   # =====================
   model = deeplabv3_resnet50(  ### <- Tambien existe por ejemplo el resnet101 
       weights=None, ### <- Tambien se pueden definir pesos predefinidos. Por ejemplo: weights = DeepLabV3_ResNet50_Weights.DEFAULT, pesos entrenados con el repositorio de imágenes de COCO
       num_classes=NUM_CLASSES
   ).to(DEVICE) ### <- Mueve el modelo al dispositivo donde queremos que se ejecute


   # =====================
   # OPTIMIZACIÓN
   # =====================
   criterion = torch.nn.CrossEntropyLoss()
   optimizer = torch.optim.Adam(model.parameters(), lr=LR)


   # =====================
   # ENTRENAMIENTO
   # =====================
   model.train()
   loss_history = []

   for epoch in range(EPOCHS):
       epoch_loss = 0.0

       for images, masks in train_loader:
           images = images.to(DEVICE)
           masks  = masks.to(DEVICE)

           optimizer.zero_grad()
           outputs = model(images)["out"]
           loss = criterion(outputs, masks)
           loss.backward()
           optimizer.step()

           epoch_loss += loss.item()
       avg_loss = epoch_loss / len(train_loader)
       loss_history.append((epoch+1, avg_loss))

       print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss/len(train_loader):.4f}")

   loss_path = os.path.join(BASE_DIR, "models", "training_loss.csv")
   os.makedirs(os.path.dirname(loss_path), exist_ok = True)

   with open(loss_path, mode="w", newline="") as f:
       writer = csv.writer(f)
       writer.writerow(["epoch","loss"])
       writer.writerows(loss_history)
    
   print("Loss guardada en traininig_loss.csv")
   
   # =====================
   # GUARDADO
   # =====================
   save_dir = os.path.join(BASE_DIR,"models","deeplab_model.pth")
   torch.save(model.state_dict(), save_dir)
   print("Modelo guardado como deeplab_model.pth")

   return model
