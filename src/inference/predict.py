import os
import sys
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50

# =======================
# FORZAR PATH AL PROYECTO
# =======================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)





def run_inference(config,model_dir):

    # ============================================================
    # RUTAS DEL PROYECTO : Fijamos los directorios más importantes
    # ============================================================
    IMAGE_DIR = os.path.join(BASE_DIR, "inference", "images")
    OUT_MASK_DIR = os.path.join(BASE_DIR, "inference", "outputs", "masks")
    os.makedirs(OUT_MASK_DIR, exist_ok=True)


    NUM_CLASSES = config["training"]["num_classes"]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =====================
    # TRANSFORMACIONES
    # =====================
    image_transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


    # =====================
    # CARGAR EL MODELO
    # =====================
    model = deeplabv3_resnet50(
        weights=None,
        num_classes=NUM_CLASSES
    )

    model.load_state_dict(torch.load(model_dir, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()


    # ================================================
    # CICLO DE INFERENCIA (RECORRE TODAS LAS IMÁGENES)
    # ================================================
    for fname in os.listdir(IMAGE_DIR):

        if not fname.lower().endswith((".png", ".jpg", ".jpeg")): ### <- Asegura que solo trabajemos con imágenes
            continue

        image_path = os.path.join(IMAGE_DIR, fname)
        image = Image.open(image_path).convert("RGB")
        orig_size = image.size  # (W, H)

        x = image_transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = model(x)["out"]
            pred = torch.argmax(out, dim=1).squeeze(0)

        mask = pred.cpu().numpy().astype(np.uint8)

        mask_img = Image.fromarray(mask)
        mask_img = mask_img.resize(orig_size, Image.NEAREST)

        out_path = os.path.join(OUT_MASK_DIR, fname)
        mask_img.save(out_path)

        print(f"Máscara generada: {out_path}")

    print("Inferencia completada.")
