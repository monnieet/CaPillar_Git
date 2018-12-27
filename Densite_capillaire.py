# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 08:19:23 2018

@author: Etienne Monnier
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

def seuillage(image):
    ''' On crée un seuil pour essayer de mieux faire ressortir les capillaires  sur 
        une image dont on a gardé une seule composante de couleur
    
    entrée: image (matrice n*m)
    sortie: resultat (matrice n*m)
    
    '''
    resultat = image.copy()
    resultat[resultat>140] = 255
    resultat[resultat<=140] = 130
    return resultat

def vecteurImageSelonDroite(imageFiltree,a,b,x):
    ''' On récupère un vecteur de pixels selon une droite  
    
    entrées: imageFiltree (Matrice n*m), a (float), b (float), x (int array)
    sortie: vecteur (int array)
    
    '''
    y = np.round(x*a + b)
    vecteur = np.zeros(y.shape)
    for k  in range(len(x)):
        vecteur[k] = imageFiltree[int(y[k]),x[k]]
    return vecteur

def suppressionBruitVecteur(vecteur):
    ''' On retire les points ponctuels qui fausseraien le calcul de densité 
        Pour ce faire on parcourt le vecteur et les groupes de taille inférieur 
        à 5 pixels sont remplacer 
    
    entrée: vecteur (int array)
    sortie: vect (int array)
    
    '''
    vect = np.copy(vecteur)
    n = len(vect)
    k = 0
    bruit = 0
    while k<n-1:
        i = k+1
        if vect[k] == 255:
            while vect[i] != 255:
                bruit+=1
                if bruit==5 or i == n-1:
                    break
                i+=1
            if bruit<5:
                vect[k+1:i] = 255
        else:
            while vect[i] == 255:
                bruit+=1
                if bruit==5 or i == n-1:
                    break
                i+=1
            if bruit<5:
                vect[k+1:i] = 130
        k = i
        bruit = 0
    return vect                              

def densiteVecteurSelonDroite(vect,grossissement,A,B): # 1pixel est de longueur variant entre 0.21 et 0.31 mm
    ''' On essaye de calculer la densité sur le vecteur couleur ayant subit un seuillage
        Ainsi qu'une suppression de bruit et une fonction laissant que deux valeurs 
        On prendra comme longueur absolue d'un pixel 0.26mm 
    
    entrées: vect (int array), grossissement (int), A (int tuple), B (int tuple)
    sortie: densite (float)
    
    '''
    compteur = 0
    switch = 1
    if vect[0] != 255:
        compteur = 1
        switch = 0
    for k in range(1,len(vect)-2):
        if vect[k] == 255 and switch != 1:
            switch = 1
        if vect[k] == 130 and switch != 0:
            switch = 0
            compteur += 1
    longueur = np.sqrt((B[0]-A[0])**2+(B[1]-A[1])**2)*0.26
    return compteur*grossissement/longueur

def calculMoyenneDensiteSelonDroites(img,grossissement,A,B,c,x):
    ''' Fonction reprenant les fonctions précédentes pour retourner la densité
        capillaire moyenne selon plusieurs droites parallèles en capillaire/mm 
    
    entrées: img (matrice n*m*3 pour RVB), grossissement (int), A (int tuple), B (int tuple)
                n (int), x (int array)
    sortie: densiteMoyenne (float)
        
    '''
    a = (B[1]-A[1])/(B[0]-A[0])
    b = B[1]-a*B[0]
    L=20*np.arange(0,c)
    s = 0
    imageApresSeuillage = seuillage(img[:,:,1])
    for i in L:
        vect2 = suppressionBruitVecteur(vecteurImageSelonDroite(imageApresSeuillage,a,b-i,x))
        densite = densiteVecteurSelonDroite(vect2,grossissement,A,B)
        s+=densite
        print(densite)
    return s/c

def affichage(img,A,B,c):
    ''' Fonction permettant d'afficher l'image des capillaires selon la composante verte,
        l'image après seuillage ainsi que les différentes droites grâce auxquelles on calcule
        la densité moyenne.
        
    entrées: img (matrice n*m*3 pour RVB), A (int tuple), B (int tuple), c (int)
    sortie: None
    
    '''
    vert = img[:,:,1] #Etonnamment le filtre vert fait bien ressortir les capillaires
    a = (B[1]-A[1])/(B[0]-A[0])
    b = B[1]-a*B[0] 
    
    plt.figure()
    plt.imshow(seuillage(vert),cmap = 'gray') #Image après avoir effectué un seuillage
    plt.colorbar()
    plt.plot(x,a*x+b) #première droite
    for i in 20*np.arange(1,c):
        plt.plot(x,a*x+b-i,'red')  #Droites sur lesquelles on calcule les capillaires
    plt.figure()
    plt.imshow(vert,cmap = 'gray') #Image après filtre vert
    plt.colorbar()
    plt.plot(x,a*x+b)
    for i in 20*np.arange(1,c):
        plt.plot(x,a*x+b-i,'red')
    plt.show()
    
    

if __name__ == '__main__':
    
    print("Entrez les coordonnées du premier point séparées par une virgule:") # (200,640)
    A = tuple(int(x.strip()) for x in input().split(','))
    print("Entrez les coordonnées du deuxième point séparées par une virgule:") # (1270,230)
    B = tuple(int(x.strip()) for x in input().split(','))
    print("Combien de droites pour calculer la moyenne ? (>0)")
    c = int(input())
    
    img = mpimg.imread("4D.jpg")
    x = np.arange(200,1300) #On réduit la zone où calculer les capillaires (On évite les bords)
    densite = calculMoyenneDensiteSelonDroites(img,100,A,B,c,x)
    print("La densité est de",densite," capillaire(s) par mm")
    affichage(img,A,B,c)
           
    
    #mpimg.imsave("resultat.png", img) Pous sauvegarder une image



''' Améliorations:
    
-> Programme effectué pour l'image 4D, "seuillage" devra s'adapter selon l'éclairage de la photo
et ne pourra donc se faire sur une constante (paramètre seuil a rajouter dans la fonction)

-> L'équation de la droite pourra être obtenue en demandant à l'utilisateur de placer la première
droite sur le bout de l'ongle

-> Une fois la droite positionnnée on pourra calculer une densité pour chaque droite parallèle à
celle positionnée pour ensuite effectuer une moyenne sur les densités trouvées en introduisant différentes
pondérations

-> Temps de calcul beaucoup trop élévé à cause de nombreuses boucles "for, essayer de jouer sur le fait qu'on
manipule des tableaux numpy pour réduire la complexité temporelle

'''
