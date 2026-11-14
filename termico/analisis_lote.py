"""Reproduce la Tabla 9.6 y la Figura 9.11 del capítulo con el plan A.

Recorre una carpeta con subcarpetas por punto (Punto 2/, Punto 3/, ...),
calcula la temperatura real de cada imagen, la fracción de cada régimen
térmico y la estadística por punto, y dibuja la distribución de temperaturas.

Uso:
    python analisis_lote.py ../Fotos_Geotermia_salida_1 --salida resultados/
"""
import argparse, csv, glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plan_a_radiometrico import leer, a_celsius, zonas

ap = argparse.ArgumentParser()
ap.add_argument('carpeta')
ap.add_argument('--salida', default='resultados')
ap.add_argument('--t1', type=float, default=13.0)
ap.add_argument('--t2', type=float, default=17.5)
a = ap.parse_args()
os.makedirs(a.salida, exist_ok=True)

filas = []
for ruta in sorted(glob.glob(os.path.join(a.carpeta, '*', '*.jpg'))):
    try:
        raw, p = leer(ruta)
    except Exception:
        continue                                   # JPG sin datos radiométricos
    T = a_celsius(raw, p)
    b, m, al = zonas(T, a.t1, a.t2)
    filas.append(dict(punto=os.path.basename(os.path.dirname(ruta)), imagen=os.path.basename(ruta),
                      p01=float(np.percentile(T, 1)), t_media=float(T.mean()),
                      p99=float(np.percentile(T, 99)), t_max=float(T.max()),
                      baja=b, media=m, alta=al))

with open(os.path.join(a.salida, 'zonas_por_imagen.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(filas[0])); w.writeheader()
    w.writerows([{k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()} for r in filas])

puntos = sorted({r['punto'] for r in filas})
with open(os.path.join(a.salida, 'tabla_9_6.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f); w.writerow(['punto', 'n', 'zona', 'media_%', 'desv_est', 'CV_%', 'min_%', 'max_%'])
    for pt in puntos:
        rs = [r for r in filas if r['punto'] == pt]
        for z in ('alta', 'media', 'baja'):
            v = np.array([r[z] for r in rs]); s = v.std(ddof=1) if len(v) > 1 else 0.0
            cv = 100 * s / v.mean() if v.mean() > 0 else float('nan')
            w.writerow([pt, len(v), z, round(v.mean(), 2), round(s, 2), round(cv, 1), round(v.min(), 1), round(v.max(), 1)])

fig, ax = plt.subplots(figsize=(8, 4.2)); x = 0; ticks = []
for pt in puntos:
    xs = []
    for r in [r for r in filas if r['punto'] == pt]:
        ax.vlines(x, r['p01'], r['p99'], lw=2.2); ax.plot(x, r['t_media'], 'o', ms=3.5, mfc='white')
        ax.plot(x, r['t_max'], '_', ms=7, color='k'); xs.append(x); x += 1
    ticks.append((np.mean(xs), pt)); x += 3
for t in (a.t1, a.t2):
    ax.axhline(t, ls='--', lw=.9, color='0.3')
ax.set_xticks([t for t, _ in ticks]); ax.set_xticklabels([l for _, l in ticks])
ax.set_ylabel('Temperatura de superficie (°C)'); ax.grid(axis='y', ls=':')
fig.tight_layout(); fig.savefig(os.path.join(a.salida, 'figura_9_11.png'), dpi=300)
print('Listo:', a.salida)
