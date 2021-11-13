import numpy as np
import cv2
from numpy.linalg import inv
import math
import scipy.linalg as la
import homo

def find_corres_points(inv_homo_array, points):
	corres_points = []
	for px, py in points:
		ip_vector = np.array([px,py,1])
		ip_vector.shape=(3,1)
		op_vector = np.dot(inv_homo_array,ip_vector)
		[px_dash,py_dash,w]=op_vector
		px_dash=px_dash/w
		py_dash=py_dash/w
		x = math.floor(px_dash)
		y = math.floor(py_dash)
		corres_points += [[x, y]]

	corres_points = np.asarray(corres_points)
	corres_points += 20
	# print(corres_points)
	return corres_points

def form_equations(points, corres_points):
	list_1 = []
	for i in range(len(points)):
		temp = []
		temp += [0, 0, 0]
		temp += [-points[i][0], -points[i][1], -1]
		temp += [points[i][0] * corres_points[i][1], points[i][1] * corres_points[i][1], corres_points[i][1]]
		list_1.append(temp)

	list_2 = []
	for j in range(len(points)):
		temp = []
		temp += [points[j][0], points[j][1], 1]
		temp += [0, 0, 0]
		temp += [-1 * corres_points[j][0] * points[j][0], -1 * corres_points[j][0] * points[j][1], -1 * corres_points[j][0]]
		list_2.append(temp)

	A = [None] * (len(list_1) + len(list_2))
	A[::2] = list_1
	A[1::2] = list_2
	A = np.asarray(A)
	A.reshape((len(points) * 2, 9))
	# print(A)
	return A
def using_svd(A, img):
	U, s, V_T = la.svd(A)
	V = np.transpose(V_T)
	estimated_p = V[:, -1].reshape((3, 3))
	print(estimated_p)
	exit()
	homo.transform_image(img, estimated_p)

# def using_pinv(A):
# 	A_pinv = np.linalg.pinv(A)
# 	p = np.matmul(A_pinv, corres_points)
# 	print(p)

def main():
	img = cv2.imread('image.png')
	transformed_img = cv2.imread('bwd_map_affine.png')
	points = [[1, 10], [3, 40], [53, 6], [23, 30], [50, 50]]
	a=1
	b=0
	c=0
	d=0
	e=1
	f=0
	g=0.001
	h=0.001
	i=2
	homo_array = np.array(((a,b,c), (d, e,f),(g,h,i)))
	inv_homo_array = np.linalg.inv(homo_array)
	corres_points = find_corres_points(inv_homo_array, points)
	A = form_equations(points, corres_points)
	corres_points.reshape((len(points) * 2, 1))
	using_svd(A, img)

if __name__ == '__main__':
	main()