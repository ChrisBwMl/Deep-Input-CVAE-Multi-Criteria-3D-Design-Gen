import numpy as np
import csv

# Pfad zu den Designs
path = 'Designs/'
# Basisname für die Design-Dateien
dataname = 'GenericDesignNo'
# Anzahl der Designs
AnzahlDesigns = 10000
# 3D-Array für die Summe aller Designs
image3Dall = np.zeros((30, 40, 41)).astype(int)
# Leeres Tupel für Einträge pro Ebene
EintraegeProEbene = ()
# Leeres Tupel für Einträge pro Ebene 2
EintraegeProEbene2 = ()
# Anzahl der Einträge im Schnitt
EintraegeImSchnitt = 0

# CSV-Datei zum Schreiben der Ergebnisse öffnen
with open('Ergebnis_Leichtbau.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    # Header-Zeile schreiben
    writer.writerow(["DesignNr", "Voxeleinträge"])

    # Schleife über alle Designs
    for i in range(1, AnzahlDesigns + 1):
        print(i, '/', AnzahlDesigns)
        # Pfad zur Design-Datei
        pathDesign = path + '{}_{}_connected.npy'.format(dataname, i)
        # Design-Array laden
        DesignArray = np.load(pathDesign)
        
        # Werte im Design-Array ersetzen (0 oder 1)
        x = DesignArray
        DesignArray = np.where(x == 0, x, x == 1)
        print(np.max(DesignArray))
        
        # Voxel-Einträge für jedes Design berechnen
        Mat = DesignArray.sum(axis=2).sum(axis=1).sum(axis=0) / 50400
        # Ergebnisse in die CSV-Datei schreiben
        writer.writerow([i, Mat])

