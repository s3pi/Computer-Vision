import numpy as np 
import cv2
from scipy.ndimage import imread
from numpy.linalg import inv
import math

def transform(b, g, r, H):
	b_out = np.zeros((b.shape))
	for row_num in range(b.shape[0]):
		for col_num in range(b.shape[1]):
			b_out_idx = np.matmul([row_num, col_num], H)
			x = int(round(b_out_idx[0]))
			y = int(round(b_out_idx[1]))
			try:
				b_out[x, y] = b[row_num, col_num]
			except IndexError:
				continue

	cv2.imwrite('img_out.jpg', b_out)

def rotation(b, g, r):
	theta_degrees = 30
	theta_rad = theta_degrees * 2 * math.pi / 360
	H = [[np.cos(theta_rad), np.sin(theta_rad)], [-np.sin(theta_rad), np.cos(theta_rad)]]
	transform(b, g, r, H)

img = cv2.imread("a4-checkerboard.png")
b,g,r = cv2.split(img)
rotation(b, g, r)







