# -*- coding: utf-8 -*-

"""

Nom du PC: Pc_Thales_Bruno
Auteur: Hanna Bruno
Date de création: Thu Sep 26 09:39:55 2024
Description: Filtrage_Image

"""

import cv2
print(cv2.__version__)

import numpy as np
from urllib.request import urlopen

# Lecture d'une image hébergée via URL
req = urlopen("http://www.vgies.com/downloads/robocup.png")
# req = urlopen("file:///C:/Users/Pc_Thales_Bruno/Documents/Electronique%20SPE/Gies/robocup.png")
arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
img = cv2.imdecode(arr, -1)  # Décodage en image OpenCV

# Séparation des canaux B, G, R
B, G, R = cv2.split(img)

# Conversion de l'image BGR en HSV
imagehsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Définition des intervalles de couleur pour le masque jaune
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

# Création d'une image entièrement jaune
full_yellow = np.zeros_like(img)
full_yellow[:] = [0, 255, 255]

# Création d'un masque pour ne garder que le jaune
imagemaskyellow = cv2.inRange(imagehsv, lower_yellow, upper_yellow)

# Définition des intervalles de couleur pour le masque vert
lower_green = np.array([45, 20, 20])
upper_green = np.array([75, 255, 220])

# Création d'une image entièrement verte
full_green = np.zeros_like(img)
full_green[:] = [0, 255, 0]

# Création d'un masque pour ne garder que le vert
imagemaskgreen = cv2.inRange(imagehsv, lower_green, upper_green)

# Définition des intervalles de couleur pour le masque noir (pour en faire du bleu)
lower_black = np.array([0, 0, 0])
upper_black = np.array([255, 255, 40])

# Création d'une image entièrement bleue
full_blue = np.zeros_like(img)
full_blue[:] = [255, 0, 0]

# Création d'un masque pour ne garder que la partie considérée comme "noire"
imagemaskblue = cv2.inRange(imagehsv, lower_black, upper_black)

# Application des masques sur les images de couleur unie
yellow_image = cv2.bitwise_and(full_yellow, full_yellow, mask=imagemaskyellow)
green_image = cv2.bitwise_and(full_green, full_green, mask=imagemaskgreen)
blue_image = cv2.bitwise_and(full_blue, full_blue, mask=imagemaskblue)

# Fusion des trois images colorées (jaune, vert, bleu)
merged_image = cv2.addWeighted(yellow_image, 1, green_image, 1, 0)
merged_image = cv2.addWeighted(merged_image, 1, blue_image, 1, 0)

# Récupération de la hauteur, largeur et nombre de canaux de l'image
height = img.shape[0]
width = img.shape[1]
channels = img.shape[2]

# Transformation manuelle : remplacement de la moitié gauche de l'image par la composante verte uniquement
imgTransformGreen = img.copy()
for x in range(0, (int)(width/2)):
    for y in range(0, (int)(height)):
        # On ne garde que le canal vert pour la partie gauche
        imgTransformGreen[y, x][0] = 0
        imgTransformGreen[y, x][1] = img[y, x][1]
        imgTransformGreen[y, x][2] = 0

# Transformation manuelle : création d'un dégradé de gris de gauche à droite
imgTransformGrey = img.copy()
for x in range(0, width):
    for y in range(0, height):
        # On applique une pondération selon x/width pour chaque canal
        imgTransformGrey[y, x][0] *= x/width
        imgTransformGrey[y, x][1] *= x/width
        imgTransformGrey[y, x][2] *= x/width

# Transformation manuelle : assombrir circulairement le centre en bleu/vert
imgTransform2 = img.copy()
center = (width // 2, height // 2)
radius = min(width, height) // 4

for x in range(width):
    for y in range(height):
        # Calcul de la distance au centre
        centre = ((x - center[0]) ** 2 + (y - center[1]) ** 2) ** 0.5
        if centre < radius:
            # On modifie l'intensité dans la zone circulaire
            imgTransform2[y, x][0] *= 0.5   # Bleu réduit de moitié
            imgTransform2[y, x][1] *= 0.5   # Vert réduit de moitié
            imgTransform2[y, x][2] *= 1     # Rouge inchangé

from matplotlib import pyplot as plt

# Conversion de l'image en niveaux de gris
imageGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Calcul de l'histogramme
hist, bins = np.histogram(imageGray.flatten(), 256, [0, 256])
# Calcul de l'histogramme cumulé
cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()

# Affichage de l'histogramme cumulé en bleu et de l'histogramme original en rouge
plt.plot(cdf_normalized, color='b')
plt.hist(imageGray.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('cdf', 'histogramme'), loc='upper left')
# plt.show()  # Masqué ici pour rester compact

# Égalisation de l'histogramme
imgEqu = cv2.equalizeHist(imageGray)
# Calcul de l'histogramme égalisé
histEq, binsEq = np.histogram(imgEqu.flatten(), 256, [0, 256])
# Calcul du cdf égalisé
cdfEq = histEq.cumsum()
cdfEq_normalized = cdfEq * float(histEq.max()) / cdfEq.max()

plt.clf()
plt.plot(cdfEq_normalized, color='g')
plt.hist(imgEqu.flatten(), 256, [0, 256], color='y')
plt.xlim([0, 256])
plt.legend(('cdf', 'histogramme Eq'), loc='upper left')
plt.show()

# Égalisation sur chaque canal (B, G, R) et fusion
imgb = cv2.equalizeHist(B)
imgg = cv2.equalizeHist(G)
imgr = cv2.equalizeHist(R)
imgbgr = cv2.merge((imgb, imgg, imgr))

# Application d'un flou gaussien
img_blur = cv2.GaussianBlur(img, (15, 15), 0)

# Détection de contours Sobel
sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)

# Définition de différents noyaux de convolution pour la détection de contours
horizoint = np.array([[1,  2,  1],
                      [0,  0,  0],
                      [-1, -2, -1]])

horizoext = np.array([[-1, -2, -1],
                      [ 0,  0,  0],
                      [ 1,  2,  1]])

latdroit = np.array([[-1,  0,  1],
                     [-2,  0,  2],
                     [-1,  0,  1]])

latgauche = np.array([[ 1,  0, -1],
                      [ 2,  0, -2],
                      [ 1,  0, -1]])

# Application des filtres de convolution pour extraire différents contours
imgSobelHSuphi = cv2.filter2D(src=img_blur, ddepth=-1, kernel=horizoint)
imgSobelHSuphe = cv2.filter2D(src=img_blur, ddepth=-1, kernel=horizoext)
imgSobelHSupld = cv2.filter2D(src=img_blur, ddepth=-1, kernel=latdroit)
imgSobelHSuplg = cv2.filter2D(src=img_blur, ddepth=-1, kernel=latgauche)

# Fusion des contours
contourfin1 = cv2.bitwise_or(imgSobelHSuphi, imgSobelHSuphe)
contourfin2 = cv2.bitwise_or(imgSobelHSupld, imgSobelHSuplg)
contourfin3 = cv2.bitwise_or(contourfin1, contourfin2)

# Définition de différents noyaux de convolution pour accentuation, lissage et outline
accentuation = np.array([[ 0, -1,  0],
                         [-1,  5, -1],
                         [ 0, -1,  0]])

lissage = np.array([[0.0625, 0.125,  0.0625],
                    [0.125,  0.25,   0.125 ],
                    [0.0625, 0.125,  0.0625]])

outline = np.array([[-1, -1, -1],
                    [-1,  8, -2],
                    [-1, -1, -1]])

# Application de ces filtres
imgAcc = cv2.filter2D(src=img, ddepth=-1, kernel=accentuation)
imgLiss = cv2.filter2D(src=img, ddepth=-1, kernel=lissage)
imgOut = cv2.filter2D(src=img, ddepth=-1, kernel=outline)

contourMain = cv2.bitwise_or(imgAcc, imgLiss)
contourMain2 = cv2.bitwise_or(contourMain, imgOut)

# Flous supplémentaires : medianBlur et filtrage bilatéral
img_medianBlur = cv2.medianBlur(img, 5)
img_bilateral = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

# Détection de contours par Canny
edges = cv2.Canny(image=img_blur, threshold1=50, threshold2=100)

# Opérations d'érosion et de dilation
kernel = np.ones((5, 5), np.uint8)
img_erosion = cv2.erode(img, kernel, iterations=5)
img_dilation = cv2.dilate(img, kernel, iterations=2)

# Affichage final
cv2.imshow('Input', img)
cv2.imshow('Erosion', img_erosion)
cv2.imshow('Dilation', img_dilation)
cv2.waitKey(0)
