import numpy as np
import cv2
from numpy.linalg import inv
import math

def do_affine_tfm_on_image(image, affine_array):
	rows,columns,ch = np.shape(image)
	inv_affine_array = np.linalg.inv(affine_array)

	new_image = np.zeros([rows,columns,ch])
	image = image.astype(int)
	new_image = new_image.astype(int)
	 
	#rotation

	for c in range(1):
		for px in range(rows):
			for py in range(columns):
				op_vector = np.array([px,py,1])
				op_vector.shape=(3,1)
				ip_vector = np.dot(inv_affine_array,op_vector)
				[px_dash,py_dash,w]=ip_vector
				px_dash=px_dash/w
				py_dash=py_dash/w
				#bilinear interpolation
				x = math.floor(px_dash)
				y = math.floor(py_dash)
				if x >= 0 and x < (rows-1) and y >= 0 and y < (columns-1):
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
						# print(x, y, y1, c)
						value = ((1-alpha)*(1-beta)*image[x,y,c]) + ((alpha)*(1-beta)*image[x,y1,c])
						new_image[px,py,c] = value
					elif y1 >= columns:
						alpha = y - py_dash
						beta = px_dash - x
						value = ((1-alpha)*(1-beta)*image[x,y,c]) + ((1-alpha)*(beta)*image[x1,y,c])
						new_image[px,py,c] = value
				else:
					new_image[px,py,c] = 0

	cv2.imwrite('5_exact+20_points_pinv.png',new_image)

def foo_bar():
	image = cv2.imread('image.png')
	#b,g,r = cv2.split(image)
	affine_array = np.array(((1, 0.2, 0), (0, 1, 0), (0, 0, 1)))
	do_affine_tfm_on_image(image, affine_array)
