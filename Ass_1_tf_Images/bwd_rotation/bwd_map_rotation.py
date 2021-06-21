import numpy as np
import cv2
from numpy.linalg import inv
import math


image = cv2.imread('image.png')
#b,g,r = cv2.split(image)
rows,columns,ch = np.shape(image)
#print(rows,columns)
angle = 30
theta = np.radians(angle)
c, s = np.cos(theta), np.sin(theta)
rotation_array = np.array(((c,-s), (s, c)))
inv_rotation_array = np.linalg.inv(rotation_array)

new_image = np.zeros([rows,columns,ch])
image = image.astype(int)
new_image = new_image.astype(int)
 
#rotation

for c in range(ch):
	for px in range(rows):
		for py in range(columns):
			op_vector = np.array([px,py])
			op_vector.shape=(2,1)
			ip_vector = np.dot(inv_rotation_array,op_vector)
			[px_dash,py_dash]=ip_vector

			#bilinear interpolation
			x = math.floor(px_dash)
			y = math.floor(py_dash)
			if x >= 0 and x <rows and y >= 0 and y < columns:
				x1 =  x+1
				y1 = y+1
				if x1 >= 0 and x1 <rows and y1 >= 0 and y1 < columns:
					alpha = y - py_dash
					beta = px_dash - x
					value = ((1-alpha)*(1-beta)*image[x,y,c]) + ((1-alpha)*(beta)*image[x1,y,c]) + ((alpha)*(1-beta)*image[x,y1,c]) + ((alpha)*(beta)*image[x1,y1,c])
					new_image[px,py,c] = value
				elif x1 >= rows:
					alpha = y - py_dash
					beta = px_dash - x
					value = ((1-alpha)*(1-beta)*image[x,y,c]) + ((alpha)*(1-beta)*image[x,y1,c])
					new_image[px,py,c] = value
				elif y1 >= columns:
					alpha = y - py_dash
					beta = px_dash - x
					value = ((1-alpha)*(1-beta)*image[x,y,c]) + ((1-alpha)*(beta)*image[x1,y,c])
					new_image[px,py,c] = value
			else:
				new_image[px,py,c] = 0	

cv2.imwrite('bwd_map_rotate_30_.png',new_image)

