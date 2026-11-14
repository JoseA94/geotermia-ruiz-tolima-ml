"""Plan B: temperatura aproximada a partir del color de un JPG térmico NO radiométrico.

Compara cada píxel con el gradiente de la barra de colores (distancia euclidiana
en RGB, Ec. 9.11–9.12) y convierte el índice del color más cercano en temperatura
por interpolación lineal entre los extremos de la barra (Ec. 9.13).

IMPORTANTE: t_max y t_min son los valores impresos en la barra de CADA imagen.
Las cámaras FLIR ajustan la escala automáticamente, de modo que estos límites
cambian de una captura a otra. Para procesar un lote, registre los límites de
cada imagen en datos/escalas_flir.csv.

Uso:
    python plan_b_mapeo_color.py FLIR3864.jpg --tmax 20.5 --tmin 9.1
    python plan_b_mapeo_color.py carpeta/ --escalas ../datos/escalas_flir.csv --csv zonas_planB.csv
"""
import argparse, csv, glob, os
import cv2
import numpy as np


def plan_b_mapeo_color(ruta_imagen, t_max, t_min):
    img = cv2.imread(ruta_imagen)
    if img is None:
        raise FileNotFoundError(ruta_imagen)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)          # OpenCV lee en BGR
    h, w, _ = img_rgb.shape
    # Barra de colores (margen derecho, sin rótulos): un color por fila, cálido -> frío
    barra = img_rgb[int(h * 0.12):int(h * 0.88), int(w * 0.96):int(w * 0.98)]
    gradiente = np.mean(barra, axis=1).astype(np.float32)
    # Zona de análisis sin textos, indicadores ni barra
    zona = img_rgb[int(h * 0.12):int(h * 0.90), int(w * 0.02):int(w * 0.93)]
    filas, columnas, _ = zona.shape
    pix = zona.reshape(-1, 3).astype(np.float32)
    k = np.empty(len(pix), np.int32)
    for i in range(0, len(pix), 20000):                      # por bloques: memoria acotada
        d = np.linalg.norm(pix[i:i + 20000, None, :] - gradiente[None, :, :], axis=2)
        k[i:i + 20000] = np.argmin(d, axis=1)               # Ec. 9.11 y 9.12
    T = t_max - k / (len(gradiente) - 1) * (t_max - t_min)  # Ec. 9.13
    return T.reshape(filas, columnas)


def zonas(T, t1=13.0, t2=17.5):
    return (100 * (T < t1).mean(), 100 * ((T >= t1) & (T < t2)).mean(), 100 * (T >= t2).mean())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('entrada')
    ap.add_argument('--tmax', type=float); ap.add_argument('--tmin', type=float)
    ap.add_argument('--escalas', help='CSV con columnas imagen,t_max,t_min')
    ap.add_argument('--t1', type=float, default=13.0); ap.add_argument('--t2', type=float, default=17.5)
    ap.add_argument('--csv')
    a = ap.parse_args()
    if os.path.isdir(a.entrada):
        esc = {r['imagen']: (float(r['t_max']), float(r['t_min']))
               for r in csv.DictReader(open(a.escalas, encoding='utf-8'))}
        rutas = sorted(glob.glob(os.path.join(a.entrada, '*.jp*g')))
    else:
        esc = {os.path.basename(a.entrada): (a.tmax, a.tmin)}
        rutas = [a.entrada]
    salida = []
    for r in rutas:
        n = os.path.basename(r)
        if n not in esc:
            print(f'{n}: sin escala registrada, se omite'); continue
        T = plan_b_mapeo_color(r, *esc[n])
        b, m, al = zonas(T, a.t1, a.t2)
        fila = dict(imagen=n, t_max=esc[n][0], t_min=esc[n][1],
                    baja=round(float(b), 2), media=round(float(m), 2), alta=round(float(al), 2))
        salida.append(fila); print(fila)
    if a.csv and salida:
        with open(a.csv, 'w', newline='', encoding='utf-8') as f:
            wr = csv.DictWriter(f, fieldnames=list(salida[0])); wr.writeheader(); wr.writerows(salida)
