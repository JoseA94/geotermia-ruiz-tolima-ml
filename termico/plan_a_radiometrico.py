"""Plan A: temperatura exacta desde JPG radiométricos FLIR (p. ej., FLIR T450sc).

Une los segmentos APP1 'FLIR' del JPG, lee el registro RawData (valores crudos
del sensor) y el registro CameraInfo (constantes de Planck R1, R2, B, F, O,
emisividad, distancia, temperaturas reflejada y atmosférica, humedad) y
convierte cada valor crudo a °C con la ecuación radiométrica de FLIR.
Solo requiere numpy y pillow.

Uso:
    python plan_a_radiometrico.py FLIR3864.jpg                 # una imagen
    python plan_a_radiometrico.py carpeta/ --csv zonas.csv     # lote
    python plan_a_radiometrico.py carpeta/ --t1 13 --t2 17.5 --emisividad 0.95
"""
import struct, io, sys, numpy as np
from PIL import Image

def fff_de_jpg(ruta):
    b = open(ruta, 'rb').read()
    i, partes = 2, {}
    while i < len(b) - 4:
        if b[i] != 0xFF: break
        marca = b[i+1]
        if marca == 0xDA: break
        L = struct.unpack('>H', b[i+2:i+4])[0]
        seg = b[i+4:i+2+L]
        if marca == 0xE1 and seg[:5] == b'FLIR\x00':
            partes[seg[6]] = seg[8:]          # número de parte
        i += 2 + L
    return b''.join(partes[k] for k in sorted(partes))

def registros(fff):
    assert fff[:4] in (b'FFF\x00', b'AFF\x00')
    E = '>'
    ver = struct.unpack('>I', fff[0x14:0x18])[0]
    if not 100 <= ver < 200: E = '<'
    off, n = struct.unpack(E+'II', fff[0x18:0x20])
    regs = {}
    for k in range(n):
        e = fff[off + 0x20*k: off + 0x20*(k+1)]
        tipo = struct.unpack(E+'H', e[0:2])[0]
        roff, rlen = struct.unpack(E+'II', e[0x0c:0x14])
        if tipo: regs[tipo] = fff[roff:roff+rlen]
    return regs

def leer(ruta):
    r = registros(fff_de_jpg(ruta))
    raw, ci = r[1], r[0x20]
    e = '<' if struct.unpack('<H', raw[:2])[0] == 2 else '>'
    w, h = struct.unpack(e+'HH', raw[2:6])
    datos = raw[0x20:]
    if datos[:4] == b'\x89PNG':
        a = np.array(Image.open(io.BytesIO(datos))).astype(np.uint16)
        a = ((a >> 8) | (a << 8)) & 0xFFFF          # FLIR guarda el PNG con bytes invertidos
    elif datos[:2] in (b'II', b'MM'):
        a = np.array(Image.open(io.BytesIO(datos))).astype(np.uint16)
    else:
        a = np.frombuffer(datos[:w*h*2], dtype=e+'u2').reshape(h, w)
    e = '<' if struct.unpack('<H', ci[:2])[0] == 2 else '>'
    f = lambda o: struct.unpack(e+'f', ci[o:o+4])[0]
    p = dict(E=f(0x20), OD=f(0x24), RT=f(0x28)-273.15, AT=f(0x2c)-273.15,
             IWT=f(0x30)-273.15, IWTr=f(0x34), RH=f(0x3c), R1=f(0x58), B=f(0x5c), F=f(0x60),
             A1=f(0x70), A2=f(0x74), B1=f(0x78), B2=f(0x7c), X=f(0x80),
             O=struct.unpack(e+'i', ci[0x308:0x30c])[0], R2=f(0x30c),
             modelo=ci[0xd4:0xf4].split(b'\x00')[0].decode('latin1'))
    if p['RH'] > 2: p['RH'] /= 100.0
    return a.astype(float), p

def a_celsius(raw, p, E=None):
    E = p['E'] if E is None else E
    R1, R2, B, F, O = p['R1'], p['R2'], p['B'], p['F'], p['O']
    AT, OD, RH, IWTr = p['AT'], p['OD'], p['RH'], p['IWTr']
    h2o = RH*np.exp(1.5587 + 0.06939*AT - 0.00027816*AT**2 + 0.00000068455*AT**3)
    tau = p['X']*np.exp(-np.sqrt(OD/2)*(p['A1']+p['B1']*np.sqrt(h2o))) + \
          (1-p['X'])*np.exp(-np.sqrt(OD/2)*(p['A2']+p['B2']*np.sqrt(h2o)))
    pl = lambda T: R1/(R2*(np.exp(B/(T+273.15))-F)) - O
    r_refl = (1-E)/E*pl(p['RT'])
    r_atm1 = (1-tau)/E/tau*pl(AT)
    r_win = (1-IWTr)/E/tau/IWTr*pl(p['IWT'])
    r_atm2 = (1-tau)/E/tau/IWTr/tau*pl(AT)
    r_obj = raw/E/tau/IWTr/tau - r_atm1 - r_atm2 - r_win - r_refl
    return B/np.log(R1/(R2*(r_obj+O))+F) - 273.15

def zonas(T, t1, t2):
    """Fracción (%) de la escena en cada régimen térmico (Ec. 9.14 y 9.15)."""
    return (100*(T < t1).mean(), 100*((T >= t1) & (T < t2)).mean(), 100*(T >= t2).mean())

if __name__ == '__main__':
    import argparse, csv, glob, os
    ap = argparse.ArgumentParser()
    ap.add_argument('entrada', help='JPG radiométrico o carpeta con JPG')
    ap.add_argument('--t1', type=float, default=13.0)
    ap.add_argument('--t2', type=float, default=17.5)
    ap.add_argument('--emisividad', type=float, default=None,
                    help='sobrescribe la emisividad guardada por la cámara')
    ap.add_argument('--csv', default=None)
    ap.add_argument('--npy', action='store_true', help='guarda la matriz de °C de cada imagen')
    a = ap.parse_args()
    rutas = sorted(glob.glob(os.path.join(a.entrada, '*.jp*g'))) if os.path.isdir(a.entrada) else [a.entrada]
    filas = []
    for r in rutas:
        try:
            raw, p = leer(r)
        except Exception as e:
            print(f'{os.path.basename(r)}: sin datos radiométricos ({e})'); continue
        T = a_celsius(raw, p, a.emisividad)
        b, m, al = zonas(T, a.t1, a.t2)
        f = dict(imagen=os.path.basename(r), t_min=round(float(T.min()), 2), t_p01=round(float(np.percentile(T, 1)), 2),
                 t_media=round(float(T.mean()), 2), t_p99=round(float(np.percentile(T, 99)), 2), t_max=round(float(T.max()), 2),
                 baja=round(float(b), 2), media=round(float(m), 2), alta=round(float(al), 2), emisividad=round(a.emisividad or p['E'], 3))
        filas.append(f); print(f)
        if a.npy: np.save(os.path.splitext(r)[0] + '_T.npy', T)
    if a.csv and filas:
        with open(a.csv, 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=list(filas[0])); w.writeheader(); w.writerows(filas)
