import cc3d
import numpy as np
from mayavi import mlab
from scipy import ndimage
import time

def Aufdicken(sp):
    pointcloud = []

    # Für jeden Punkt (px, py, pz) in der Eingabewolke
    for [px, py, pz] in sp:
        # Füge den aktuellen Punkt hinzu
        pointcloud.append([px, py, pz])

        # Füge die Nachbarpunkte hinzu
        pointcloud.append([px + 1, py, pz])
        pointcloud.append([px, py + 1, pz])
        pointcloud.append([px + 1, py + 1, pz])
        pointcloud.append([px, py, pz + 1])
        pointcloud.append([px + 1, py, pz + 1])
        pointcloud.append([px, py + 1, pz + 1])
        pointcloud.append([px + 1, py + 1, pz + 1])

    return pointcloud

def Connector(array3D):
    # Umwandlung des 3D-Arrays in ganze Zahlen
    int_array3D = array3D.astype(int)

    # Einzelne Areale im 3D-Bild erkennen
    labels_out, N = cc3d.connected_components(int_array3D, return_N=True)

    # Listen für Areale und Schwerpunkte initialisieren
    arealist = []
    centroidlist = []
    variables = {}
    ConnectorRaum = np.zeros((30, 40, 41)).astype(int)

    # Einzelne erkannte Areale in einzelnen Feldern speichern
    for i in range(1, N+1):
        variables['area%s' % i] = np.where((labels_out >= i) & (labels_out <= i), labels_out, 0*labels_out)
        arealist.append(variables['area%s' % i])

    # Schwerpunkte für jedes Areal finden
    for i in range(0, N):
        variables['area%s' % i] = arealist[i]
        centroid = ndimage.measurements.center_of_mass(variables['area%s' % i])
        centroidlist.append(centroid)

    # Größtes Volumen finden, zu dem verbunden wird
    AnzahlElemente = [np.count_nonzero(area) for area in arealist]
    IndexGrVol = AnzahlElemente.index(max(AnzahlElemente))

    # 3D-Geradengleichung für Verbindungslinien zwischen Schwerpunkten
    PunktVec = []
    for i in range(0, N):
        variables['area%s' % i] = centroidlist[i]
        B = centroidlist[IndexGrVol]
        for t in np.arange(0.001, 1, 0.005):
            # Richtungsvektor
            AB = [b-a for a, b in zip(variables['area%s' % i], B)]
            # Skalarprodukt
            SkalarProd = [t*x for x in AB]
            # Geradengleichung
            vecX = [a+b for a, b in zip(variables['area%s' % i], SkalarProd)]
            PunktVec.append(vecX)

    # Tuple in Integer umwandeln
    PunktVec = tuple(tuple(map(int, tup)) for tup in PunktVec)

    # Aufdickung der Punkte für die Verbindungslinien
    for [x, y, z] in (Aufdicken(PunktVec)):
        ConnectorRaum[x, y, z] = 1

    # Ursprüngliche Punkte ohne Aufdickung hinzufügen
    for [x, y, z] in PunktVec:
        ConnectorRaum[x, y, z] = 1

    # Ursprüngliches 3D-Bild mit den verbundenen Punkten aktualisieren
    Conncected = ConnectorRaum + array3D

    return Conncected