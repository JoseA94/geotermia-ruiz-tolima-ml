# geotermia-ruiz-tolima-ml

Código del capítulo 9, *Machine Learning con geodatos: procesamiento de imágenes térmicas y RGB para la exploración geotérmica del Complejo Ruiz–Tolima*.
Proyecto *Generación de conocimiento geocientífico y desarrollo tecnológico enfocado en el aprovechamiento de energía geotérmica en el Complejo Ruiz–Tolima* (Código 110992), Universidad Nacional de Colombia, sede Manizales.

El repositorio contiene dos flujos independientes:

1. **Imágenes térmicas (cámara FLIR T450sc).** Lectura de la temperatura de cada píxel y segmentación en regímenes térmicos.
2. **Ortomosaicos RGB de dron.** Clasificación de coberturas con el índice de verdor VARI y la textura local.

## Estructura

```
geotermia-ruiz-tolima-ml/
├── README.md
├── LICENSE                         Licencia MIT
├── CITATION.cff                    Cómo citar el repositorio
├── requirements.txt                Dependencias de Python
├── .gitignore
├── termico/
│   ├── plan_a_radiometrico.py      Temperatura exacta desde JPG radiométricos FLIR
│   ├── plan_b_mapeo_color.py       Temperatura aproximada a partir del color
│   ├── analisis_lote.py            Reproduce la Tabla 9.6 y la Figura 9.11
│   └── comparar_planes.py          Error del plan B frente al plan A
├── coberturas/
│   └── clasificacion_vari_textura.py   Coberturas en ortomosaicos RGB (VARI + textura)
├── yolo/
│   └── etiquetas_yolo.py           Metodología propuesta: etiquetas de segmentación YOLO
└── datos/
    ├── escalas_flir.csv            Límites de la barra de colores de cada imagen FLIR
    ├── resultados_planA.csv        Temperaturas y fracciones por imagen (plan A)
    └── comparacion_planA_planB.csv Diferencias entre ambos métodos por imagen
```

Las fotografías originales no se incluyen en el repositorio por su tamaño (unos 125 MB). Se pueden solicitar a los autores o depositar por separado en Zenodo.

## Instalación

```bash
git clone https://github.com/JoseA94/geotermia-ruiz-tolima-ml.git
cd geotermia-ruiz-tolima-ml
pip install -r requirements.txt
```

## Imágenes térmicas

### Plan A: JPG radiométricos (recomendado)

Las cámaras FLIR de la serie T guardan, dentro del JPG, la matriz de valores del sensor y sus constantes de calibración (Planck R1, R2, B, F, O), además de la emisividad, la distancia y las temperaturas reflejada y atmosférica. `plan_a_radiometrico.py` une los segmentos APP1 del archivo, lee esos registros y convierte cada valor del sensor a grados Celsius. Solo requiere `numpy` y `pillow`.

```bash
# una imagen
python termico/plan_a_radiometrico.py FLIR3864.jpg

# una carpeta completa, con umbrales y emisividad propios
python termico/plan_a_radiometrico.py fotos/Punto\ 2/ --t1 13 --t2 17.5 --emisividad 0.95 --csv punto2.csv

# Tabla 9.6 y Figura 9.11 (una subcarpeta por punto de inspección)
python termico/analisis_lote.py fotos/ --salida resultados/
```

Validación: en 38 imágenes, la temperatura calculada en el centro de la imagen difiere en promedio −0,11 °C de la lectura del puntero que la cámara imprime en pantalla (mediana de la diferencia absoluta: 0,8 °C).

### Plan B: imágenes sin datos radiométricos

Si solo se dispone de la imagen coloreada (captura de pantalla, PNG o JPG exportado), la temperatura se estima comparando cada píxel con la barra de colores. **La cámara ajusta la escala automáticamente, por lo que `t_max` y `t_min` cambian en cada imagen** y deben leerse imagen por imagen (`datos/escalas_flir.csv`).

```bash
python termico/plan_b_mapeo_color.py FLIR3864.jpg --tmax 20.5 --tmin 9.1
python termico/plan_b_mapeo_color.py fotos/Punto\ 2/ --escalas datos/escalas_flir.csv --csv zonas_planB.csv
python termico/comparar_planes.py fotos/ --escalas datos/escalas_flir.csv
```

Frente al plan A, el plan B tiene un error absoluto medio de 1,2 °C por píxel con la paleta arcoíris de la T450sc (0,3–0,8 °C en escenas de rango estrecho y hasta 4 °C en escenas de rango amplio). Cuando muchas temperaturas caen cerca de un umbral, la fracción de cada régimen puede cambiar varias decenas de puntos porcentuales.

## Coberturas en ortomosaicos RGB

```bash
# 1. Calibrar: histogramas de VARI, textura y brillo para elegir los umbrales
python coberturas/clasificacion_vari_textura.py ortomosaico.tif --calibrar

# 2. Clasificar con el umbral de textura calibrado
python coberturas/clasificacion_vari_textura.py ortomosaico.tif --umbral-textura <valor>
```

El script lee el tamaño de píxel del GeoTIFF. La fórmula interna del umbral de textura (ecuación 9.18) se calibró entre 5,8 y 16,7 cm/px. Para ortomosaicos exportados con otro tamaño de píxel (por ejemplo, 2 cm/px), el umbral debe calibrarse con `--calibrar` y pasarse con `--umbral-textura`.

Salidas: `<nombre>_clases.tif`, `<nombre>_vari.tif`, `<nombre>_resumen.csv` y `<nombre>_figura.png`.

## Segmentación con YOLO (metodología propuesta)

`yolo/etiquetas_yolo.py` convierte las máscaras térmicas en etiquetas de segmentación en formato YOLO (ecuación 9.23). En este trabajo la segmentación se hizo por reglas; el entrenamiento de la red se deja como metodología propuesta y no se reportan métricas.

## Limitaciones

- La cámara se usó con sus parámetros de fábrica (temperatura atmosférica 24 °C, temperatura reflejada 22 °C, humedad relativa 60 %, emisividad 0,98). Si se conocen las condiciones de campo, conviene recalcular con `--emisividad` o editando los parámetros.
- El reloj de la cámara no estaba configurado; las fechas de los metadatos (año 2000) solo sirven para ordenar las capturas.
- La clasificación de coberturas es por reglas y no cuenta aún con validación de exactitud en campo.

## Licencia y cita

Código bajo licencia MIT (ver `LICENSE`). Para citar, use `CITATION.cff` o el DOI de la versión archivada en Zenodo.
