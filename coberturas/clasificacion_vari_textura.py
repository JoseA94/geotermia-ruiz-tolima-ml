"""
Clasificación inicial de coberturas en imágenes RGB de dron
usando el índice VARI (verdor visible) + textura local (desviación estándar).

Proyecto: Geotermia – Actividades 1.8 / 1.9
Entrada : ortomosaico GeoTIFF (recomendado) o una foto JPG.
Salidas : <nombre>_clases.tif  (raster de clases; georreferenciado si la entrada es GeoTIFF)
          <nombre>_vari.tif    (índice VARI)
          <nombre>_resumen.csv (área y % por clase)
          <nombre>_figura.png  (RGB | VARI | clases, para el informe)

Los GeoTIFF se procesan por bloques (con solapamiento), de modo que ortomosaicos
de miles de millones de píxeles caben en memoria sin perder resolución.

Uso:
    python clasificacion_vari_textura.py ortomosaico.tif
    python clasificacion_vari_textura.py foto.JPG --gsd 0.058
    python clasificacion_vari_textura.py ortomosaico.tif --calibrar     (solo muestra histogramas para elegir umbrales)
    python clasificacion_vari_textura.py ortomosaico.tif --umbral-vari 0.06 --umbral-textura 22

Requisitos:
    pip install numpy scipy matplotlib pillow
    pip install rasterio          (para leer/escribir GeoTIFF con coordenadas)
"""
import argparse
import csv
import os
import time

import numpy as np
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

try:
    import rasterio
    from rasterio.windows import Window
    from rasterio.enums import Resampling
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

# ----------------------------------------------------------------------------
# PARÁMETROS (ajústelos con --calibrar; los valores por defecto funcionaron
# con las fotos DJI 0700–0703 de Murillo, 219 m AGL)
# ----------------------------------------------------------------------------
UMBRAL_VARI = 0.05         # VARI > umbral  -> vegetación
UMBRAL_TEXTURA = None      # desv. estándar (0-255) que separa dosel (rugoso) de pasto (liso).
                           # DEPENDE DEL GSD: a más resolución, más textura.
                           # None = valor según GSD calibrado en Murillo (≈25 a 5.8 cm; ≈16 a 17 cm).
                           # También se puede pasar con --umbral-textura.
UMBRAL_BRILLO = 0.33       # en no-vegetación: brillo >= umbral -> suelo/vía; menor -> zona oscura
VENTANA_TEXTURA_M = 2.5    # tamaño de la ventana de textura en metros (≈ diámetro de una copa)
VENTANA_SUAVIZADO_M = 1.5  # suavizado previo de VARI y brillo, en metros
FILTRO_MAYORIA_M = 1.2    # limpieza final ("sal y pimienta"), en metros
BLOQUE_PX = 4096           # tamaño del bloque de procesamiento (px); bájelo si falta memoria

CLASES = {
    1: ("Dosel arbóreo", "#1b7a3a"),
    2: ("Herbáceo / pastos", "#a6d96a"),
    3: ("Suelo desnudo / vías", "#d8b37a"),
    4: ("Zonas oscuras (sombra / suelo húmedo)", "#4a3b5c"),

}


class Fuente:
    """Acceso por ventanas a un GeoTIFF (rasterio) o a una imagen en memoria (JPG)."""

    def __init__(self, ruta, gsd_manual=None):
        self.src = None
        self.escala = 1.0
        es_tif = ruta.lower().endswith((".tif", ".tiff"))
        if es_tif and HAS_RASTERIO:
            src = rasterio.open(ruta)
            if src.crs is not None and src.crs.is_geographic and gsd_manual is None:
                raise SystemExit("El GeoTIFF está en grados (WGS84). Reproyecte a MAGNA-SIRGAS "
                                 "Origen Nacional (EPSG:9377) o pase --gsd en metros.")
            self.src = src
            self.alto, self.ancho = src.height, src.width
            self.gsd = abs(src.transform.a) if gsd_manual is None else gsd_manual
            self.perfil = src.profile.copy()
            if src.dtypes[0] != "uint8":  # p. ej. 16 bits: escala global a 0-255
                h = max(1, self.alto // 2000)
                muestra = src.read([1, 2, 3], out_shape=(3, self.alto // h, self.ancho // h),
                                   resampling=Resampling.nearest)
                self.escala = 255.0 / max(1, muestra.max())
            return
        from PIL import Image, ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        if gsd_manual is None:
            raise SystemExit("Para JPG indique el tamaño de píxel con --gsd (m). "
                             "Ej.: Mavic 3 Pro a 219 m -> --gsd 0.058")
        self.rgb = np.asarray(Image.open(ruta).convert("RGB"))
        self.alto, self.ancho = self.rgb.shape[:2]
        self.gsd = gsd_manual
        self.perfil = None

    def leer(self, r0, r1, c0, c1):
        """Devuelve (rgb uint8 HxWx3, máscara válida) de la ventana [r0:r1, c0:c1]."""
        if self.src is None:
            rgb = self.rgb[r0:r1, c0:c1]
            return rgb, rgb.sum(2) > 15
        arr = self.src.read(window=Window(c0, r0, c1 - c0, r1 - r0))
        rgb = np.transpose(arr[:3], (1, 2, 0))
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb * self.escala, 0, 255).astype(np.uint8)
        if arr.shape[0] >= 4:      # banda alfa del ortomosaico
            valido = arr[3] > 0
        else:
            valido = rgb.astype(np.uint16).sum(2) > 0
        return rgb, valido


def a_pixeles(metros, gsd):
    return max(3, int(round(metros / gsd)) | 1)  # entero impar >= 3


def calcular_indices(rgb, gsd):
    A = rgb.astype(np.float32) / 255.0
    R, G, B = A[..., 0], A[..., 1], A[..., 2]

    # VARI = (G - R) / (G + R - B)   (Gitelson et al., 2002)
    den = G + R - B
    with np.errstate(divide="ignore", invalid="ignore"):
        vari = np.where(np.abs(den) > 1e-3, (G - R) / den, 0).astype(np.float32)
    vari = np.clip(vari, -1, 1)

    brillo = A.mean(axis=2)
    gris = (0.299 * R + 0.587 * G + 0.114 * B) * 255
    del A, R, G, B, den

    # Textura = desviación estándar local:  sqrt(E[x²] - E[x]²)
    w = a_pixeles(VENTANA_TEXTURA_M, gsd)
    media = ndi.uniform_filter(gris, w)
    media2 = ndi.uniform_filter(gris ** 2, w)
    textura = np.sqrt(np.clip(media2 - media ** 2, 0, None))
    del media, media2, gris

    s = a_pixeles(VENTANA_SUAVIZADO_M, gsd)
    vari_s = ndi.uniform_filter(vari, s)
    brillo_s = ndi.uniform_filter(brillo, s)
    return vari, vari_s, brillo_s, textura


def otsu(valores, bins=256):
    """Umbral de Otsu: maximiza la varianza entre dos grupos."""
    h, bordes = np.histogram(valores, bins=bins)
    centros = (bordes[:-1] + bordes[1:]) / 2
    w0 = np.cumsum(h); w1 = w0[-1] - w0
    m0 = np.cumsum(h * centros) / np.maximum(w0, 1)
    m1 = (np.sum(h * centros) - np.cumsum(h * centros)) / np.maximum(w1, 1)
    return centros[np.argmax(w0 * w1 * (m0 - m1) ** 2)]


def umbral_textura_por_gsd(gsd):
    """Interpolación (log del GSD) entre dos calibraciones hechas con zonas de muestra
    de pasto y bosque en las fotos DJI 0700-0703 (Murillo, sep-2026)."""
    g1, u1 = 0.058, 25.0
    g2, u2 = 0.167, 16.0
    x = (np.log(gsd) - np.log(g1)) / (np.log(g2) - np.log(g1))
    return float(np.clip(u1 + x * (u2 - u1), 8, 40))


def filtro_mayoria(clases, tam):
    """Asigna a cada píxel la clase más frecuente en una ventana tam x tam."""
    salida = clases.copy()
    mejor = np.zeros(clases.shape, np.float32)
    for k in CLASES:
        frec = ndi.uniform_filter((clases == k).astype(np.float32), tam)
        m = frec > mejor
        salida[m] = k
        mejor[m] = frec[m]
    return salida


def clasificar(vari_s, brillo_s, textura, valido, gsd, u_tex):
    clases = np.zeros(vari_s.shape, np.uint8)  # 0 = sin dato
    veg = vari_s > UMBRAL_VARI
    clases[valido & veg & (textura >= u_tex)] = 1
    clases[valido & veg & (textura < u_tex)] = 2
    clases[valido & ~veg & (brillo_s >= UMBRAL_BRILLO)] = 3
    clases[valido & ~veg & (brillo_s < UMBRAL_BRILLO)] = 4
    clases = filtro_mayoria(clases, a_pixeles(FILTRO_MAYORIA_M, gsd))
    clases[~valido] = 0
    return clases


def calibrar(vari_s, brillo_s, textura, base, u_tex):
    """Recibe muestras (1D) de píxeles válidos."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, v, nombre, u in [
        (ax[0], vari_s, "VARI", UMBRAL_VARI),
        (ax[1], textura, "Textura (desv. est.)", round(u_tex, 1)),
        (ax[2], brillo_s, "Brillo", UMBRAL_BRILLO),
    ]:
        a.hist(v, bins=200, color="#555")
        a.axvline(u, color="red", label=f"umbral actual = {u}")
        a.set_title(nombre); a.legend()
        print(f"{nombre}: percentiles 5/25/50/75/95 =",
              np.round(np.percentile(v, [5, 25, 50, 75, 95]), 3))
    plt.tight_layout(); plt.savefig(base + "_histogramas.png", dpi=100)
    print("Histogramas guardados en", base + "_histogramas.png")
    print("Busque el 'valle' entre los dos picos de cada histograma y úselo como umbral.")


def abrir_salida(ruta, perfil, dtype, nodata):
    if perfil is None or not HAS_RASTERIO:
        return None
    p = perfil.copy()
    for clave in ("photometric", "blockxsize", "blockysize", "interleave", "jpeg_quality"):
        p.pop(clave, None)
    p.update(driver="GTiff", count=1, dtype=dtype, nodata=nodata, compress="lzw",
             tiled=True, blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")
    return rasterio.open(ruta, "w", **p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entrada")
    ap.add_argument("--gsd", type=float, default=None, help="tamaño de píxel en metros (obligatorio para JPG)")
    ap.add_argument("--umbral-vari", type=float, default=None)
    ap.add_argument("--umbral-textura", type=float, default=None)
    ap.add_argument("--umbral-brillo", type=float, default=None)
    ap.add_argument("--calibrar", action="store_true", help="solo genera histogramas para elegir umbrales")
    args = ap.parse_args()

    global UMBRAL_VARI, UMBRAL_TEXTURA, UMBRAL_BRILLO
    if args.umbral_vari is not None: UMBRAL_VARI = args.umbral_vari
    if args.umbral_textura is not None: UMBRAL_TEXTURA = args.umbral_textura
    if args.umbral_brillo is not None: UMBRAL_BRILLO = args.umbral_brillo
    base = os.path.splitext(args.entrada)[0]

    env = rasterio.Env(GDAL_CACHEMAX=2048) if HAS_RASTERIO else None
    if env is not None:
        env.__enter__()

    fuente = Fuente(args.entrada, args.gsd)
    gsd, alto, ancho = fuente.gsd, fuente.alto, fuente.ancho
    print(f"Imagen {ancho} x {alto} px, GSD = {gsd*100:.1f} cm")
    u_tex = UMBRAL_TEXTURA if UMBRAL_TEXTURA is not None else umbral_textura_por_gsd(gsd)
    print(f"Umbral de textura usado = {u_tex:.1f}")

    # Solapamiento suficiente para que los filtros den el mismo resultado que sin bloques
    halo = (a_pixeles(VENTANA_TEXTURA_M, gsd) + a_pixeles(VENTANA_SUAVIZADO_M, gsd)
            + a_pixeles(FILTRO_MAYORIA_M, gsd))

    # Imágenes reducidas para la figura
    paso = max(1, max(alto, ancho) // 2000)
    fh, fw = -(-alto // paso), -(-ancho // paso)
    fig_rgb = np.full((fh, fw, 3), 255, np.uint8)
    fig_vari = np.full((fh, fw), np.nan, np.float32)
    fig_cls = np.zeros((fh, fw), np.uint8)

    salto_muestra = max(1, (alto * ancho) // 4_000_000)  # para --calibrar
    muestras = ([], [], [])
    conteo = np.zeros(len(CLASES) + 1, np.int64)

    if not args.calibrar:
        dst_cls = abrir_salida(base + "_clases.tif", fuente.perfil, "uint8", 0)
        dst_vari = abrir_salida(base + "_vari.tif", fuente.perfil, "float32", -9999)
        if dst_cls is None:  # JPG: se guarda en memoria y luego como .npy
            cls_mem = np.zeros((alto, ancho), np.uint8)
            vari_mem = np.full((alto, ancho), -9999, np.float32)

    bloques = [(r0, c0) for r0 in range(0, alto, BLOQUE_PX) for c0 in range(0, ancho, BLOQUE_PX)]
    t0 = time.time()
    for i, (r0, c0) in enumerate(bloques, 1):
        r1, c1 = min(r0 + BLOQUE_PX, alto), min(c0 + BLOQUE_PX, ancho)
        R0, R1 = max(0, r0 - halo), min(alto, r1 + halo)
        C0, C1 = max(0, c0 - halo), min(ancho, c1 + halo)
        rgb, valido = fuente.leer(R0, R1, C0, C1)
        nucleo = (slice(r0 - R0, r1 - R0), slice(c0 - C0, c1 - C0))
        if not valido[nucleo].any():
            continue  # bloque sin datos (fuera del área volada)

        vari, vari_s, brillo_s, textura = calcular_indices(rgb, gsd)
        v_n = valido[nucleo]

        if args.calibrar:
            for lista, datos in zip(muestras, (vari_s, textura, brillo_s)):
                lista.append(datos[nucleo][v_n][::salto_muestra])
        else:
            clases = clasificar(vari_s, brillo_s, textura, valido, gsd, u_tex)[nucleo]
            vari_n = np.where(v_n, vari[nucleo], -9999).astype(np.float32)
            conteo += np.bincount(clases.ravel(), minlength=len(CLASES) + 1)
            if dst_cls is not None:
                ventana = Window(c0, r0, c1 - c0, r1 - r0)
                dst_cls.write(clases, 1, window=ventana)
                dst_vari.write(vari_n, 1, window=ventana)
            else:
                cls_mem[r0:r1, c0:c1] = clases
                vari_mem[r0:r1, c0:c1] = vari_n
            # Píxeles de este bloque que caen en la rejilla de la figura
            fr, fc = (-r0) % paso, (-c0) % paso
            sub = (slice(fr, None, paso), slice(fc, None, paso))
            gr, gc = (r0 + fr) // paso, (c0 + fc) // paso
            cs = clases[sub]
            fig_cls[gr:gr + cs.shape[0], gc:gc + cs.shape[1]] = cs
            fig_vari[gr:gr + cs.shape[0], gc:gc + cs.shape[1]] = np.where(
                v_n[sub], vari[nucleo][sub], np.nan)
            fig_rgb[gr:gr + cs.shape[0], gc:gc + cs.shape[1]] = np.where(
                v_n[sub][..., None], rgb[nucleo][sub], 255)

        print(f"  bloque {i}/{len(bloques)}  ({time.time() - t0:.0f} s)", flush=True)

    if args.calibrar:
        calibrar(*(np.concatenate(m) for m in muestras), base, u_tex)
        return

    if dst_cls is not None:
        dst_cls.close(); dst_vari.close()
    else:
        np.save(base + "_clases.npy", cls_mem)
        np.save(base + "_vari.npy", vari_mem)

    # --- Resumen de áreas
    area_px = gsd ** 2
    total = conteo[1:].sum()
    filas = []
    for k, (nombre, _) in CLASES.items():
        n = int(conteo[k])
        filas.append([k, nombre, round(n * area_px, 1), round(n * area_px / 10000, 3),
                      round(100 * n / max(total, 1), 2)])
    with open(base + "_resumen.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clase", "cobertura", "area_m2", "area_ha", "porcentaje"])
        w.writerows(filas)
    for fila in filas:
        print(f"  {fila[1]:<40} {fila[3]:>8} ha  {fila[4]:>6} %")

    # --- Figura para el informe
    cm = ListedColormap(["white"] + [c for _, c in CLASES.values()])
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    ax[0].imshow(fig_rgb); ax[0].set_title("RGB")
    m = ax[1].imshow(fig_vari, cmap="RdYlGn", vmin=-0.2, vmax=0.4)
    ax[1].set_title("VARI"); plt.colorbar(m, ax=ax[1], fraction=0.035)
    ax[2].imshow(fig_cls, cmap=cm, vmin=0, vmax=4, interpolation="nearest")
    ax[2].set_title("Clasificación VARI + textura")
    ax[2].legend(handles=[Patch(color=c, label=f"{n}: {f[4]} %")
                          for (n, c), f in zip(CLASES.values(), filas)],
                 loc="lower right", fontsize=9, framealpha=0.9)
    for a in ax:
        a.axis("off")
    plt.tight_layout(); plt.savefig(base + "_figura.png", dpi=110)
    print("Listo:", base + "_clases.tif |", base + "_resumen.csv |", base + "_figura.png")


if __name__ == "__main__":
    main()
