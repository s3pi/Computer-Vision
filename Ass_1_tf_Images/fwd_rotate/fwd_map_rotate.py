import numpy as np
import cv2
from numpy.linalg import inv
import math


image = cv2.imread('image.png')
image = image.astype(int)
#print(image)
#b,g,r = cv2.split(image)
rows,columns,ch = np.shape(image)
#print(rows,columns)
angle = 45
theta = np.radians(angle)
c, s = np.cos(theta), np.sin(theta)
rotation_array = np.array(((c,-s), (s, c)))
print(rotation_array)
inv_rotation_array = np.linalg.inv(rotation_array)

new_image = np.zeros([rows*2,columns*2,ch])
image = image.astype(int)
new_image = new_image.astype(int)
 
#rotation

for c in range(1):
	for px in range(rows):
		for py in range(columns):
			ip_vector = np.array([px,py])
			ip_vector.shape=(2,1)
			op_vector = np.dot(rotation_array,ip_vector)
			[px_dash,py_dash]=op_vector
			# if np.logical_and(np.logical_and(np.logical_and(px_dash >= 0, px_dash < rows),py_dash >= 0 ),py_dash<columns):
			new_image[int(px_dash),int(py_dash),c]=image[px,py,c]
#print(new_image)
cv2.imwrite('fwd_map_rotate_45_full.png',new_image)
