import numpy as np
from mayavi import mlab
from Perlin_Noise_Algorithm import generatePerlinVoxelPositions
from Connector_line import Connector
import time
import os
from skimage import measure
import trimesh
import subprocess
from CFDAble import CFDAble

# Designraum-Dimensionen
X = 30
Y = 40
Z = 40

# Name und Anzahl der Designs
dataname = 'GenericDesign'
AnzahlDesigns = 10000

# Initialisiere den Zähler für Designs
i = 0

# Schleife zur Generierung von Designs
while i < AnzahlDesigns:
    i += 1

    # Pfad für STL-Export
    stl_path = os.path.join('Designs\{}No_{}.stl'.format(dataname, i))

    # Generiere 3D-Voxel-Design mit Perlin-Noise
    image3D = np.zeros((X, Y, Z)).astype(int)
    sp = ((generatePerlinVoxelPositions(X, Y, Z)))
    for [x, y, z] in sp:
        image3D[x, y, z] = 1

    # Füge Motoranbindung hinzu
    def unit_circle(r):
        d = 2 * r + 1
        rx, ry = 15, 26  # Mittelpunkt des Kreises
        x, y = np.indices((X, Y))
        return (np.abs(np.hypot(rx - x, ry - y) - r) < 0.5).astype(int)  # Kreisformel

    motor = np.zeros((X, Y))
    for r in range(0, 5):
        r = r + 1
        motor = motor + unit_circle(r)

    GenDesign = np.dstack((image3D, motor))

    # Schneide Schräge aus
    GenDesign[:, 0:13, 0] = 0
    z = 42
    for w in range(1, 14):
        z = z - 3
        GenDesign[:, 0:w, z] = 0
        GenDesign[:, 0:w, z - 1] = 0
        GenDesign[:, 0:w, z - 2] = 0

    # Sicherstellen der Verbindung zur Struktur
    GenDesign[0:1, 13:17, 0:2] = 1
    GenDesign[29:30, 13:17, 0:2] = 1
    GenDesign[29:30, 0:4, 38:40] = 1
    GenDesign[0:1, 0:4, 38:40] = 1

    # Verbinde die Struktur mit dem Connector-Algorithmus
    image3D_connected = Connector(GenDesign)

    # Bewertung der CFD-Fähigkeit
    d = CFDAble(image3D_connected)

    if d == True:
        # Wenn das Design CFD-fähig ist, speichere es und exportiere STL
        print('\n', 'Design Nr.', i, '/', AnzahlDesigns)
        np.save('Designs\{}No_{}'.format(dataname, i), GenDesign)
        np.save('Designs\{}No_{}_connected'.format(dataname, i), image3D_connected)

        # Glätte die Oberfläche und exportiere STL
        image3D_connected = np.pad(image3D_connected, 2)
        verts, faces, normals, _ = measure.marching_cubes(image3D_connected, level=0, step_size=1.9, method='lorensen')
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
        mesh.export(stl_path)
    else:
        # Wenn das Design nicht CFD-fähig ist, reduziere den Zähler
        i -= 1


