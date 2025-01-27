# -*- coding: utf-8 -*-

"""

Nom du PC: Pc_Thales_Bruno
Auteur: Hanna Bruno
Date de création: Wed Dec 18 12:56:54 2024
Description: 

"""
import cv2
import numpy as np

def erosion_dilatation(img, kernel1, kernel2, iteration):
    """
    Applique une érosion suivie d'une dilatation sur une image binaire.

    Parameters:
    img (ndarray): Image binaire à traiter.
    kernel1 (list): Taille du noyau pour l'érosion [hauteur, largeur].
    kernel2 (list): Taille du noyau pour la dilatation [hauteur, largeur].
    iteration (list): Nombre d'itérations pour l'érosion et la dilatation [érosion, dilatation].

    Returns:
    img_dilation (ndarray): Image après érosion et dilatation.
    """
    # Création du noyau pour l'érosion
    kernel = np.ones((kernel1[0], kernel1[1]), np.uint8)
    img_erosion = cv2.erode(img, kernel, iterations=iteration[0])  # Application de l'érosion

    # Création du noyau pour la dilatation
    kernel = np.ones((kernel2[0], kernel2[1]), np.uint8)
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=iteration[1])  # Application de la dilatation

    return img_dilation  # Retour de l'image traitée

def type_fruits(area):
    """
    Détermine le type de fruit et les couleurs associées en fonction de l'aire.

    Parameters:
    area (float): Aire du contour détecté.

    Returns:
    dict or None: Dictionnaire contenant le nom du fruit et les couleurs pour l'annotation,
                  ou None si l'aire ne correspond à aucun fruit connu.
    """
    if 8000 <= area < 22000:
        return {
            "nom": "abricot",
            "text_color": (255, 0, 0),        # Bleu pour le texte
            "circle_color": (255, 0, 0)       # Bleu pour le cercle
        }
    elif 22000 <= area < 30000:
        return {
            "nom": "citron vert",
            "text_color": (255, 255, 255),    # Blanc pour le texte
            "circle_color": (255, 255, 255)   # Blanc pour le cercle
        }
    elif 30000 <= area < 36000:
        return {
            "nom": "citron",
            "text_color": (255, 255, 255),    # Blanc pour le texte
            "circle_color": (0, 255, 255)     # Jaune pour le cercle
        }
    elif 36000 <= area < 42000:
        return {
            "nom": "nectarine",
            "text_color": (255, 255, 255),    # Blanc pour le texte
            "circle_color": (0, 0, 255)       # Vert pour le cercle
        }
    elif 42000 <= area < 60000:
        return {
            "nom": "pomme",
            "text_color": (0, 255, 0),        # Vert pour le texte
            "circle_color": (0, 255, 0)       # Vert pour le cercle
        }
    else:
        return None

def detect_contours(input_img, contours):
    """
    Détecte et annote les contours des fruits sur l'image originale.

    Parameters:
    input_img (ndarray): Image originale sur laquelle annoter.
    contours (list): Liste des contours détectés.

    Returns:
    output_img (ndarray): Image annotée avec les contours et informations des fruits.
    """
    output_img = input_img.copy()  # Copie de l'image originale pour l'annotation

    # Initialisation des compteurs pour chaque type de fruit
    fruit_counts = {
        "abricot": 0,
        "citron vert": 0,
        "citron": 0,
        "pomme": 0,
        "nectarine": 0
    }

    smallest_area = float('inf')  # Initialisation de la plus petite aire trouvée
    smallest_contour = None       # Contour du plus petit fruit
    smallest_center = (0, 0)      # Coordonnées du centre du plus petit fruit

    for contour in contours:
        area = cv2.contourArea(contour)  # Calcul de l'aire du contour
        if 100 < area < 100000:  # Filtrage des contours par aire pour éliminer le bruit
            fruit_props = type_fruits(area)  # Détermination du type de fruit basé sur l'aire
            if fruit_props is not None:
                nom = fruit_props["nom"]
                text_color = fruit_props["text_color"]
                circle_color = fruit_props["circle_color"]

                # Mise à jour du compteur du fruit détecté
                fruit_counts[nom] += 1

                # Dessin des contours en gris épais
                cv2.drawContours(output_img, [contour], -1, (200, 200, 200), 3)
                
                # Approximation d'un cercle autour du fruit
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(output_img, center, radius, circle_color, 2)  # Dessin du cercle

                # Ajout du nom du fruit et de sa surface en texte sur l'image
                cv2.putText(output_img, f"{nom}", (center[0] - 50, center[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
                cv2.putText(output_img, f"{int(area)}px2", (center[0] - 60, center[1] + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
                
                # Mise à jour du plus petit fruit détecté
                if area < smallest_area:
                    smallest_area = area
                    smallest_contour = contour
                    smallest_center = center

    # Dessin du plus petit fruit en vert fluo et affichage de ses coordonnées
    if smallest_contour is not None:
        cv2.drawContours(output_img, [smallest_contour], -1, (0, 255, 0), 3)  # Contour en vert fluo
        cv2.putText(output_img, f"Coordonnee petit fruit: Y={smallest_center[0]}, X={-smallest_center[1]}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Initialisation du décalage vertical pour l'affichage des compteurs
    decal_bas = 50
    for fruit, count in fruit_counts.items():
        if count > 0:  # Affichage uniquement des fruits détectés
            text = f"{fruit}(s) detecte(s) : {count}"
            cv2.putText(output_img, text, (500, output_img.shape[0] - (420 + decal_bas)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)  # Texte blanc
            decal_bas += 50  # Incrémentation du décalage pour le prochain texte

    # Affichage des résultats dans la console
    for fruit, count in fruit_counts.items():
        print(f"Nombre de {fruit}(s) detecte(s) : {count}")
    if smallest_contour is not None:
        print(f"Coordonnees du plus petit fruit : Y = {smallest_center[0]}, X = {-smallest_center[1]}")

    return output_img