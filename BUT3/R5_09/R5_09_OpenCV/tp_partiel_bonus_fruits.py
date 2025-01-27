# -*- coding: utf-8 -*-

"""

Nom du PC: Pc_Thales_Bruno
Auteur: Hanna Bruno
Date de création: Mon Dec 16 12:56:54 2024
Description: 

"""

import cv2 
import numpy as np  
from urllib.request import urlopen
from bruno import erosion_dilatation, detect_contours  # Importation des fonctions personnalisées

print(cv2.__version__)

# Lecture de l'image originale depuis un fichier local
image_original = "pomme-peches-abricot-citrons.jpg"
img = cv2.imread(image_original)

# Lire l'image depuis une URL
# req = urlopen("https://www.vgies.com/downloads/Images/pomme-peches-abricot-citrons.jpg")
# arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
# img = cv2.imdecode(arr, -1)

# Définition du pourcentage de réduction de l'image (car elle est trop grande pour mon pc :D)
scale_percent = 50
width = int(img.shape[1] * scale_percent / 100)  # Calcul de la nouvelle largeur
height = int(img.shape[0] * scale_percent / 100)  # Calcul de la nouvelle hauteur
dim = (width, height)  # Dimensions pour le redimensionnement

# Conversion de l'image de l'espace de couleur BGR à HSV
imagehsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Définition des plages de couleurs pour les nectarines
lower_nectarine1 = np.array([0, 25, 25])    
upper_nectarine1 = np.array([10, 255, 255])
lower_nectarine2 = np.array([170, 25, 25])  
upper_nectarine2 = np.array([180, 255, 255])

# Définition des plages de couleurs pour le jaune
lower_jaune = np.array([10, 100, 100])  
upper_jaune = np.array([23, 255, 255])

# Définition des plages de couleurs pour le vert
lower_vert = np.array([30, 25, 25])  
upper_vert = np.array([80, 255, 160])

# Création des masques pour chaque plage de couleur
maskjaune = cv2.inRange(imagehsv, lower_jaune, upper_jaune)
maskvert = cv2.inRange(imagehsv, lower_vert, upper_vert)
mask1 = cv2.inRange(imagehsv, lower_nectarine1, upper_nectarine1)
mask2 = cv2.inRange(imagehsv, lower_nectarine2, upper_nectarine2)
masknectarine = cv2.bitwise_or(mask1, mask2)  # Combinaison des deux masques de nectarines

# Création d'une image entièrement verte pour appliquer les masques
full_green = np.zeros_like(img)
full_green[:] = [0, 255, 0]
vert_img = cv2.bitwise_and(full_green, full_green, mask=maskvert)  # Application du masque vert
jaune_img = cv2.bitwise_and(full_green, full_green, mask=maskjaune)  # Application du masque jaune
nectarine_img = cv2.bitwise_and(full_green, full_green, mask=masknectarine)  # Application du masque nectarine

# Définition des noyaux et des itérations pour l'érosion et la dilatation des masques
kernel_nectarine1 = [3, 3]
kernel_nectarine2 = [3, 3]
iteration_nectarine = [1, 1]
mask_nectarine = erosion_dilatation(masknectarine, kernel_nectarine1, kernel_nectarine2, iteration_nectarine)

kernel_vert1 = [3, 3]
kernel_vert2 = [3, 3]
iteration_vert = [1, 1]
mask_vert = erosion_dilatation(maskvert, kernel_vert1, kernel_vert2, iteration_vert)

kernel_jaune1 = [9, 9]
kernel_jaune2 = [7, 9]
iteration_jaune = [5, 5]
mask_jaune = erosion_dilatation(maskjaune, kernel_jaune1, kernel_jaune2, iteration_jaune)

# Combinaison des masques vert et jaune
mask_jaunevert = cv2.bitwise_or(mask_vert, mask_jaune)
# Combinaison finale avec le masque nectarine
mask_final = cv2.bitwise_or(mask_jaunevert, mask_nectarine)

# Conversion du masque final en image binaire
_, img_gray = cv2.threshold(mask_final, 1, 255, cv2.THRESH_BINARY)
# Détection des contours dans l'image binaire
contours, _ = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Détection et annotation des contours sur l'image originale
image_final = detect_contours(img, contours)
# Redimensionnement de l'image finale selon les dimensions définies précédemment
img_final = cv2.resize(image_final, dim, interpolation=cv2.INTER_AREA)
# Affichage de l'image finale avec les annotations
cv2.imshow('img_final', img_final)
cv2.waitKey(0)  # Attente d'une touche pour fermer la fenêtre