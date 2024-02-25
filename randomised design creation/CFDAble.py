import numpy as np
import dijkstra3d
import time

def CFDAble(array3D):
    # Einfügen der Schräge als nicht passierbarer Raum
    array3D[:, 0:13, 0] = 1
    z = 42
    for w in range(1, 14):
        z = z - 3
        array3D[:, 0:w, z] = 1
        array3D[:, 0:w, z - 1] = 1
        array3D[:, 0:w, z - 2] = 1

    # Suchraum in der Startebene um Offset erweitern
    offsetarray3D = np.zeros((30, 40, 1))
    newarray3D = np.append(offsetarray3D, array3D, axis=2)

    # Start- und Zielpunkt suchen
    seedarray = np.argwhere(newarray3D[:, :, 1] == 0)
    random = np.random.randint(seedarray.shape[0], size=1)
    x, y = np.transpose(seedarray[random])
    seed = (x, y, 1)

    targetarray = np.argwhere(newarray3D[:, :, 41] == 0)
    random = np.random.randint(targetarray.shape[0], size=1)
    x, y = np.transpose(targetarray[random])
    target = (x, y, 41)

    # Dijkstra-Algorithmus anwenden
    path = dijkstra3d.dijkstra(newarray3D, seed, target, connectivity=26)

    # Bewertung der Durchströmbarkeit, Visualisierung des Pfads
    booldstr = 1
    Paths = np.zeros((30, 40, 42)).astype(int)

    # Überprüfen der Durchströmbarkeit entlang des Pfads
    for i in range(len(path)):
        x, y, z = path[i]
        if newarray3D[x, y, z] == 1:
            booldstr = 0
            print('Nicht durchströmbar Check 1')
            break
        else:
            Paths[x, y, z] = 1

    for i in range(len(path)):
        x, y, z = path[i]
        if newarray3D[x, y, z] == 1:
            booldstr = 0
            print('Nicht durchströmbar Check 2')
            break
        else:
            Paths[x, y, z] = 1

    for i in range(len(path)):
        x, y, z = path[i]
        if newarray3D[x, y, z] == 1:
            booldstr = 0
            print('Nicht durchströmbar Check 3')
            break
        else:
            Paths[x, y, z] = 1

    # Rücksetzen der Modifikationen im 3D-Array
    array3D[:, 0:13, 0] = 0
    z = 42
    for w in range(1, 14):
        z = z - 3
        array3D[:, 0:w, z] = 0
        array3D[:, 0:w, z - 1] = 0
        array3D[:, 0:w, z - 2] = 0
    return bool(booldstr)
