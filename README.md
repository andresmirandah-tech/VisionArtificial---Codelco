# ArtificialVision

Proyecto de segmentación semántica basado en DeepLabV3+ para la detección y cuantificación de burbujas en celdas de flotación.

## Descripción

Este proyecto implementa un pipeline completo de:

- Procesamiento de imágenes
- Generación de máscaras
- Entrenamiento con DeepLabV3+
- Inferencia
- Cálculo de métricas (IoU, Dice)
- Cálculo de indicador de proceso (% de burbujas)

## Estructura del proyecto

- `src/` → Código principal
- `report/` → Informe técnico
- `presentation/` → Presentaciones
- `metrics/` → Scripts de evaluación

## Cómo ejecutar

```bash
git clone https://github.com/andresmirandah-tech/VisionArtificial---Codelco.git
cd VisionArtificial---Codelco
pip install -r requirements.txt
python src/main.py
```


