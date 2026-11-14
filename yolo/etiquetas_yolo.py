"""Metodología propuesta (no entrenada en este trabajo): generación automática de
etiquetas de segmentación YOLO a partir de una matriz de temperaturas (Ec. 9.23).

Clases: 0 = baja, 1 = media, 2 = alta. La matriz T puede venir del plan A
(plan_a_radiometrico.py) o del plan B (plan_b_mapeo_color.py).
"""
from pathlib import Path
import cv2
import numpy as np

UMBRALES = (13.0, 17.5)   # T1, T2 en °C (Ec. 9.14)
AREA_MIN_PX = 150         # descarta regiones menores (ruido)


def mascaras_termicas(T, t1=UMBRALES[0], t2=UMBRALES[1]):
    return {0: T < t1, 1: (T >= t1) & (T < t2), 2: T >= t2}


def etiquetas_yolo(T, ruta_txt):
    """Escribe un polígono normalizado por región: 'c x1/W y1/H ... xn/W yn/H'."""
    H, W = T.shape
    lineas = []
    for clase, m in mascaras_termicas(T).items():
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contornos, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contornos:
            if cv2.contourArea(c) < AREA_MIN_PX:
                continue
            c = cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2)
            if len(c) >= 3:
                lineas.append(f"{clase} " + " ".join(f"{x / W:.6f} {y / H:.6f}" for x, y in c))
    Path(ruta_txt).write_text("\n".join(lineas))

# Entrenamiento propuesto (requiere: pip install ultralytics)
#   from ultralytics import YOLO
#   modelo = YOLO("yolov8n-seg.pt")
#   modelo.train(data="dataset_termico/termico.yaml", epochs=100, imgsz=640,
#                batch=8, fliplr=0.5, degrees=10, patience=20)
# Separar entrenamiento y validación por punto de inspección (p. ej., validar con el Punto 2).
