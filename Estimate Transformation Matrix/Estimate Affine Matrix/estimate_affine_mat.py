import numpy as np
import cv2
from numpy.linalg import inv
import math
import scipy.linalg as la
import bwd_map_affine

def find_corres_points(inv_affine_array, points):
	corres_points = []
	for px, py in points:
		op_vector = np.array([px,py,1])
		op_vector.shape=(3,1)
		ip_vector = np.dot(inv_affine_array,op_vector)
		[px_dash,py_dash,w]=ip_vector
		px_dash=px_dash/w
		py_dash=py_dash/w
		#bilinear interpolation
		x = math.floor(px_dash)
		y = math.floor(py_dash)
		corres_points += [x, y]
	# print(points)
	# print(corres_points)
	corres_points = np.asarray(corres_points)
	corres_points += 20
	corres_points.reshape((len(points) * 2, 1))
	# print(corres_points.shape)
	return corres_points

def form_equations(points):
	list_1 = [x + [ 1, 0, 0, 0] for x in points]
	list_2 = [[0, 0, 0] + x + [1] for x in points]
	A = [None] * (len(list_1) + len(list_2))
	A[::2] = list_1
	A[1::2] = list_2
	A = np.asarray(A)
	A.reshape((len(points) * 2, 6))
	# print(A)
	return A

def main():
	img = cv2.imread('image.png')
	transformed_img = cv2.imread('bwd_map_affine.png')

	#Find 3 points
	affine_array = np.array(((1, 0.2, 0), (0, 1, 0), (0, 0, 1)))
	inv_affine_array = inv(affine_array)
	# img = img.astype(int)
	points = [[1, 2], [3, 4], [5, 6], [20, 30], [50, 50]]
	corres_points = find_corres_points(inv_affine_array, points)

	#Solve equations to get the transformation matrix (p = inv(A) * b). Find psuedo inverse in case of m > n.
	A = form_equations(points)
	# p = np.matmul(inv(A),corres_points)
	# p = la.solve(A, corres_points)
	A_pinv = np.linalg.pinv(A)
	p = np.matmul(A_pinv, corres_points)
	p = np.append(p, [0, 0, 1])
	flatten_points = np.asarray([item for sublist in points for item in sublist])
	affine_array = p.reshape(3, -1)
	print(affine_array)
	exit()
	bwd_map_affine.do_affine_tfm_on_image(img, affine_array)

if __name__ == '__main__':
	main()

