import numpy as np
import cv2
from numpy.linalg import inv
import math

def transform_image(image, homo_array):
	inv_homo_array = np.linalg.inv(homo_array)
	rows,columns,ch = np.shape(image)
	new_image = np.zeros([rows,columns,ch])
	image = image.astype(int)
	new_image = new_image.astype(int)
	 
	#rotation
	pixel=1
	for c in range(1):
		for px in range(rows):
			for py in range(columns):
				ip_vector = np.array([px,py,1])
				ip_vector.shape=(3,1)
				op_vector = np.dot(inv_homo_array,ip_vector)
				[px_dash,py_dash,w]=op_vector
				px_dash=px_dash/w
				py_dash=py_dash/w
				#print()
				#print(op_vector)
				if np.logical_and(np.logical_and(np.logical_and(px_dash >= 0, px_dash < rows),py_dash >= 0 ),py_dash<columns):
					new_image[int(px_dash),int(py_dash),c]=image[px,py,c]
					#pixel=pixel+1
					#print(pixel)
	cv2.imwrite('1.png',new_image)

def foo_bar():
	image = cv2.imread('image.png')
	#print(image)
	#b,g,r = cv2.split(image)
	#print(rows,columns)
	#angle = 30
	#theta = np.radians(angle)
	#c, s = np.cos(theta), np.sin(theta)
	#translation works fine 1 0 15 0 1 15
	#shear works fine 1 0.2 0 0 1 0 shear in x,
	#scale 0.5 0 0 0 1 0 halves in x dir 
	a=1
	b=0
	c=0
	d=0
	e=1
	f=0
	g=0.001
	h=0.001
	i=1
	homo_array = np.array(((a,b,c), (d, e,f),(g,h,i)))
	transform_image(image, homo_array)

foo_bar()
