"""Error del plan B (color) frente al plan A (radiométrico) en las mismas imágenes.

Usa la escala (t_max, t_min) leída en cada imagen (datos/escalas_flir.csv) y
compara, sobre la misma zona de análisis, la temperatura por píxel y la
fracción asignada a cada régimen térmico.

Uso:
    python comparar_planes.py ../Fotos_Geotermia_salida_1 --escalas ../datos/escalas_flir.csv
"""
import argparse, csv, glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from plan_a_radiometrico import leer, a_celsius, zonas
from plan_b_mapeo_color import plan_b_mapeo_color

ap = argparse.ArgumentParser()
ap.add_argument('carpeta'); ap.add_argument('--escalas', required=True)
ap.add_argument('--t1', type=float, default=13.0); ap.add_argument('--t2', type=float, default=17.5)
ap.add_argument('--csv', default='comparacion_planes.csv')
a = ap.parse_args()
esc = {r['imagen']: (float(r['t_max']), float(r['t_min'])) for r in csv.DictReader(open(a.escalas, encoding='utf-8'))}
filas = []
for ruta in sorted(glob.glob(os.path.join(a.carpeta, '*', '*.jpg'))):
    n = os.path.basename(ruta)
    if n not in esc: continue
    raw, p = leer(ruta); TA = a_celsius(raw, p)
    TB = plan_b_mapeo_color(ruta, *esc[n])
    H, W = TA.shape
    TA = TA[int(H * 0.12):int(H * 0.90), int(W * 0.02):int(W * 0.93)]      # misma zona que el plan B
    d = TB - TA
    zA, zB = np.array(zonas(TA, a.t1, a.t2)), np.array(zonas(TB, a.t1, a.t2))
    filas.append(dict(imagen=n, error_abs_medio=round(float(np.abs(d).mean()), 2), sesgo=round(float(d.mean()), 2),
                      dif_baja=round(float(zB[0] - zA[0]), 1), dif_media=round(float(zB[1] - zA[1]), 1),
                      dif_alta=round(float(zB[2] - zA[2]), 1)))
    print(filas[-1])
with open(a.csv, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(filas[0])); w.writeheader(); w.writerows(filas)
e = np.array([r['error_abs_medio'] for r in filas]); s = np.array([r['sesgo'] for r in filas])
print(f'Error absoluto medio: {e.mean():.2f} °C | sesgo medio: {s.mean():+.2f} °C | n = {len(filas)}')
