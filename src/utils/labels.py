"""
# labels.py
-> Script utilizado en el resto de scripts implementado.

## OBJETIVO DEL SCRIPT
Definir correcta y de forma única las clases a utilizar durante todo el proceso de Etiquetado/Entrenamiento/Inferencia
Definimos las etiquetas (labelme) y los colores
"""

# Etiquetas explícitas (las que se usan en Labelme)
Etiquetas = {
    "borde": 0,
    "exterior": 1,
    "espuma": 2,
    "burbuja":3,
}



# Diccionario inverso (opcional, debug)
ID_To_Etiqueta = {v: k for k, v in Etiquetas.items()}


# Colores SOLO para la posterior visualización
Colores = {
    0: (0, 0, 255),      # borde
    1: (150, 150, 150),  # exterior
    2: (0, 255, 0),      # espuma
    3: (255, 0, 0)       # burbujas (complemento)
}
