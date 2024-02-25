import numpy
import trimesh
import os
import numpy as np
import math
import csv
import collections

dataname = 'GenericDesign'
AnzahlDesigns = 10000

with open('Testueberhang.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Flächen über 43°", "Flächen Insgesamt", "Prozent an nicht druckbaren Flächen"])

        for i in range (1,AnzahlDesigns+1):
                print(i, '/', AnzahlDesigns)
                stl_path = os.path.join('E:\Designs\{}No_{}.stl'.format(dataname, i))
                mesh = trimesh.load_mesh(stl_path)
                faces = mesh.faces
                normal = mesh.face_normals
                angleslist = []
                rotationmatrix = np.array([[0,-1,0],
                                           [1,0,0],
                                           [0,0,1]])
                def angle(v1, v2):
                        '''Winkel zwischen Vektor und Ebene'''
                        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                        '''Rad to Grad'''
                        angle = 180 * angle / np.pi
                        angleslist.append(angle)

                for j in range (0,len(normal)):
                        oldvector1 = np.array(normal[j])
                        '''Normalenvektor der Facette um 90° drehen'''
                        vector1 = rotationmatrix.dot(oldvector1)
                        '''Ebenenvektor XY-Ebene'''
                        vector2 = np.array([0, 0, 1])
                        '''Winkel zwischen Ebene und Vektor bestimmen'''
                        angledegree = angle(vector2,vector1)

                '''Winkel unter 43° oder 90° zur Z-Ebene/Druckrichtung'''
                flaechen = (len([k for k in angleslist if k <= 43 or k ==0]))
                facetten = (len(angleslist))
                '''Prozent der Facetten die einen Winkel unter 43'''
                prozent =((flaechen/len(faces))*100)
                writer.writerow([facetten,flaechen,round(prozent,1)])


