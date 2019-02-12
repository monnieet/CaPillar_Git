import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import sys

img = cv.imread(sys.argv[1])

#img = cv.imread("2D.jpg")
img = img[:,:,1]

clahe = cv.createCLAHE(clipLimit=50.0, tileGridSize=(20,20))
img_clahe = clahe.apply(img)

blur = cv.GaussianBlur(img_clahe,(15,15),0)
reta2,img_otsu = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

hough_otsu = cv.HoughLines(img_otsu,30,np.pi/180,100)    #applique hough à otsu
						#img,rho,theta,threshold
						



f = np.fft.fft2(img_otsu)
fshift = np.fft.fftshift(f)
mag = 20*np.log(np.abs(fshift))

ret2,fou_ot = cv.threshold(mag,280,255,cv.THRESH_BINARY)

print(type(fou_ot))
print(fou_ot.shape[0])

coords = []
for i in range(fou_ot.shape[0]):
	for j in range(fou_ot.shape[1]):
		if fou_ot[i][j] == 255:
			coords.append([i,j])
print(len(coords))
print(coords[0])
print(coords[1])
print(coords)

plt.subplot(2,1,1),plt.imshow(mag,'gray')
plt.title("clahe+otsu")
plt.subplot(2,1,2),plt.imshow(fou_ot,'gray')
plt.title("fourier")
plt.show()

