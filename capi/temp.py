import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import sys
import scipy.ndimage


img = cv.imread(sys.argv[1])  #image brute


#---------------------------
# OPENCV: FILTRES ET FOURIER
#---------------------------

def gris(img):          #img en niveaux de gris
	return img[:,:,1]

	
def eq(img):        #img apres egalisation d'histo
	return cv.equalizeHist(img)

	
def eq_clahe(img, a=50.0, b=20):   #img apres egalisation d'histo CLAHE
	clahe = cv.createCLAHE(clipLimit=a, tileGridSize=(b,b))
	return clahe.apply(img)

	
def otsu(img, dim=15):       #img lissee et binarisee avec OTSU
	blur = cv.GaussianBlur(img,(dim,dim),0)
	reta2,img_otsu = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
	return img_otsu

	
def fourier(img):        #TF en amplitude de img
	f = np.fft.fft2(img)
	fshift = np.fft.fftshift(f)
	mag = 20*np.log(np.abs(fshift))
	return mag


#---------------
# AXE DE L'IMAGE
#---------------

def spectre_centre_bin(mag, p=11, seuil=280):     #centre, lisse et binarise le spectre
	centre = blur = cv.GaussianBlur(mag[350:450,656:844],(p,p),0)
	ret2,centre_bin = cv.threshold(centre,seuil,255,cv.THRESH_BINARY)
	return centre_bin

	
def axe(centre_bin):        #renvoie le vecteur (X,Y) de l'axe du spectre binarise
	aux = np.uint8(centre_bin)
	_, contours, _ = cv.findContours(aux, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

	# Construct a buffer used by the pca analysis
	print("nombre de formes: ", len(contours))
	if len(contours) == 3:   #si 3 formes, on prend la derniere (la plus grosse)
		pts = contours[2]
	else:
		pts = contours[0]
		
	sz = len(pts)
	data_pts = np.empty((sz, 2), dtype=np.float64)
	for i in range(data_pts.shape[0]):
		data_pts[i,0] = pts[i,0,0]
		data_pts[i,1] = pts[i,0,1]

	# Perform PCA analysis
	mean = np.empty((0))
	mean, eigenvectors = cv.PCACompute(data_pts,mean)
	x1, y1, x2, y2 = eigenvectors.flatten()
	print("vecteur de la mire:", x1, y1)
	
	return x1, y1

	
#---------------------------
# COMPTAGE DES CAPILLAIRES
#---------------------------
	
def lisser(f):    #lisse une fonction sous forme de liste
	n = len(f)
	g = n*[0]
	for i in range(1,len(f)-1):
		g[i] = (f[i-1] + 2*f[i] + f[i+1])/4
	g[0] = (3*f[0] + f[1])/4
	g[n-1] = (f[n-2] + 3*f[n-1])/4
	return g


def lisser_n(f,n):      #lisse n fois
	lisse = f
	for i in range(n):
		lisse = lisser(lisse)
	return lisse


def deriver(values):    #derive une fonction sous forme de liste
	dx = 1
	derivees = [(values[i+1] - values[i])/dx for i in range(len(values)-1)]
	return derivees


def cht(f):        #
	compteur = 0
	signe = 1
	graphe = len(f)*[0]
	for i in range(len(f)):
		if f[i] == 1 and signe == -1:
			compteur += 1
			signe = 1
			graphe[i] = 1
		if f[i] == -1 and signe == 1:
			signe = -1
	print("nombre de minima:", compteur)
	return compteur, graphe

	
def comptage(img, x0, y0, x1, y1, num=150, nb_liss=5):
	x, y = np.linspace(x0, x1, num), np.linspace(y0, y1, num)

	# Extract the values along the line, using cubic interpolation
	zi = scipy.ndimage.map_coordinates(img, np.vstack((y,x)))

	# Extract the values along the line, nearest neighbour sampling
	zn = img[y.astype(np.int), x.astype(np.int)]
	
	return cht(np.sign(deriver(lisser_n(zi,nb_liss))))  #PARAM: zi ou zn
	
	
