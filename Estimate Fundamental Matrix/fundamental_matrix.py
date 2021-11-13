
import numpy as np
import cv2
from numpy.linalg import inv
import math
import scipy.linalg as la
from skimage.color import rgb2gray
from skimage.feature import match_descriptors, ORB, plot_matches
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
import matplotlib.pyplot as plt
from math import sqrt


def norm_mat(p1):
	p1 = np.transpose(p1)
	centroid = np.mean(p1,axis=1)
	centroid = np.reshape(centroid,(3))
	dist = [ np.sqrt( np.sum( np.square( v - centroid ) ) ) for v in np.transpose(p1)]
	mean_dist = np.mean(dist)
	normmat = np.array([[sqrt(2)/mean_dist,0,-sqrt(2)/mean_dist*centroid[0]],[0,sqrt(2)/mean_dist,-sqrt(2)/mean_dist*centroid[1]],[0, 0, 1]])
	return normmat


def norm_points():
	#number of points = n
	n=8
	
	#dummy
	x1 = np.array([186, 178, 149, 51, 113, 110, 184, 229])
	y1 =  np.array([31, 60, 93, 139, 149, 184, 214, 242])
	x2 =  np.array([188, 178, 148, 72, 181, 104, 159, 173])
	y2 =  np.array([34, 58, 91, 121, 192, 176, 213, 191])

	first=np.stack((x1,y1),axis = 1)
	ones=np.ones((8,1))
	first = np.concatenate((first,ones),axis = 1)
	second=np.stack((x2,y2),axis = 1)
	second = np.concatenate((second,ones),axis = 1)

	norm_mat_1 = norm_mat(first)
	norm_mat_2 = norm_mat(second)
	first_norm = np.dot(norm_mat_1,np.transpose(first))
	second_norm = np.dot(norm_mat_2,np.transpose(second))
	return(first_norm,second_norm,norm_mat_1,norm_mat_2)


def make_equations():
	a1,a2,norm_mat_1,norm_mat_2 = norm_points()
	
	x1=a1[1,:]
	y1=a1[2,:]
	x2=a2[1,:]
	y2=a2[2,:]
	x1=np.reshape(x1,[1,8])
	y1=np.reshape(y1,[1,8])
	x2=np.reshape(x2,[1,8])
	y2=np.reshape(y2,[1,8])
	A = np.zeros((8,9))
	for i in range(8):
		A[i] = [x1[0,i]*x2[0,i], x2[0,i]*y1[0,i], x2[0,i],y2[0,i]*x1[0,i],y2[0,i]*y1[0,i], y2[0,i],x1[0,i], y1[0,i],1]
	return(A,norm_mat_1,norm_mat_2)

def main():
	A,T,T_dash = make_equations()
	#equation to solve:
	#Af=0 where
	#f = [f1 f2 f3 f4 f5 f6 f7 f8 f9]
	U, s, V_T = la.svd(A)
	V = np.transpose(V_T)
	rank3_F_dash = V[:, -1].reshape((3, 3))
	#now we must make it rank 2
	U_dash, s_dash, V_T_dash = la.svd(rank3_F_dash)
	new_s_dash = s_dash
	new_s_dash[2,]=0
	#final required rank2 F is:
	F_dash = U_dash * new_s_dash * V_T_dash 
	#denormalise it
	F = np.transpose(T_dash)*F_dash*T
	print(F)


main()
