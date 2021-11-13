import numpy as np
import cv2
from numpy.linalg import inv
import math


image = cv2.imread('image.png')
#print(image)
#b,g,r = cv2.split(image)
rows,columns,ch = np.shape(image)
#print(rows,columns)
#angle = 30
#theta = np.radians(angle)
#c, s = np.cos(theta), np.sin(theta)
#translation works fine 1 0 15 0 1 15
#shear works fine 1 0.2 0 0 1 0 shear in x,
#scale  0.5 0 0 0 1 0 halves in x dir
a=1
b=0.2
t_x=0
c=0.2
d=2
t_y=0
affine_array = np.array(((a,b,t_x), (c, d,t_y), (0,0,1)))
inv_rotation_array = np.linalg.inv(affine_array)

new_image = np.zeros([rows*3,columns*3,ch])
image = image.astype(int)
new_image = new_image.astype(int)
 
#rotation
pixel=1
for c in range(1):
	for px in range(rows):
		for py in range(columns):
			ip_vector = np.array([px,py,1])
			ip_vector.shape=(3,1)
			op_vector = np.dot(affine_array,ip_vector)
			[px_dash,py_dash,w]=op_vector
			px_dash=px_dash/w
			py_dash=py_dash/w
			# if np.logical_and(np.logical_and(np.logical_and(px_dash >= 0, px_dash < rows),py_dash >= 0 ),py_dash<columns):
			new_image[int(px_dash),int(py_dash),c]=image[px,py,c]
				
cv2.imwrite('fwd_map_affine_1_0.2_0_0.2_2_0.png',new_image)
