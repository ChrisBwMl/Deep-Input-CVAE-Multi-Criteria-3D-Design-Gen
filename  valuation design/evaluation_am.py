import numpy as np
import csv
import time
from mayavi import mlab

# Pfade und Variablen initialisieren
path = 'Designs/'
dataname = 'GenericDesignNo'
AnzahlDesigns = 10000
image3Dall = np.zeros((30, 40, 41)).astype(int)
EintraegeProEbene = ()
EintraegeProEbene2 = ()
EintraegeImSchnitt = 0

# Ergebnisdatei öffnen
with open('Ergebnis_Energie.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["DesignNr", "Lageneinträge", "Standardabweichung"])

    # Für jedes Design in der Anzahl der Designs
    for i in range(1, AnzahlDesigns + 1):
        print(i, '/', AnzahlDesigns)
        
        # Pfad zum aktuellen Design
        pathDesign = path + '{}_{}_connected.npy'.format(dataname, i)
        DesignArray = np.load(pathDesign)
        
        # DesignArray vorbereiten
        x = DesignArray
        DesignArray = np.where(x == 0, x, x == 1)

        # Iteration über Ebenen
        for j in range(0, 40):
            EintraegeImSchnitt = 0
            Ebene = DesignArray[:, :, j]
            EintraegeProEbene = np.count_nonzero(Ebene == 1)
            Ebene2 = DesignArray[:, :, j + 1]
            EintraegeProEbene2 = np.count_nonzero(Ebene2 == 1)
            Dif = EintraegeProEbene2 - EintraegeProEbene
            EintraegeImSchnitt += abs(Dif)
        
        # Durchschnittliche Energie im Schnitt berechnen
        EnergieImSchnitt = EintraegeImSchnitt / 40
        writer.writerow([i, EnergieImSchnitt])


