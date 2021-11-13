import cv2
import numpy as np

#camera parameters got from website
focal_length = 3740 #pixels
Baseline = 160 #mm
const1 = Baseline * focal_length
#disparity d from disp map (in mm hopefully)
#if intensity = 0, disp not known

#other parameters #rho*r_o*v_o = k
k = 30
u_o = 20

#let's take view 1 as reference, hoping disp1 is for such case
disp = cv2.imread('disp1.png')
[m,n,c] = disp.shape
disp = cv2.cvtColor(disp, cv2.COLOR_BGR2GRAY)


#get depth map,blur map from disparity,depth respectively
Z = np.zeros((m,n))
sigma = np.zeros((m,n))

for i in range(0,m):
	for j in range(0,n):
		if(disp[i,j] != 0):
			Z[i,j] = const1 / disp[i,j]
			sigma[i,j] = k * ((1/u_o)-(1/Z[i,j]))


img1 = cv2.imread('view1.png')
blurred_image1 = np.zeros((m,n,c))
img1 = cv2.copyMakeBorder(img1,1,1,1,1,cv2.BORDER_CONSTANT,value=0)
b,g,r = cv2.split(img1)
blurred_image1_b,blurred_image1_g,blurred_image1_r = cv2.split(blurred_image1)
print(m,n,c)
#if disp=0,means not known,sigma also remains 0
for c in range(0,c):
	for i in range(0,m):
		for j in range(0,n):
			#apply spatially variant blur
			if c == 0:
				img = b
			elif c == 1:
				img = g
			elif c == 2:
				img = r
			#try:
			kernel = cv2.getGaussianKernel(3,sigma[i,j])
			kernel = np.outer(kernel,kernel)

			if c == 0:
				blurred_image1_b[i,j] = sum(sum(np.dot(kernel,img[i:i+3,j:j+3])))
			elif c == 1:
				blurred_image1_g[i,j] = sum(sum(np.dot(kernel,img[i:i+3,j:j+3])))
			elif c == 2:
				blurred_image1_r[i,j,c] = sum(sum(np.dot(kernel,img[i:i+3,j:j+3])))
			
			#except:
			#	print("Error occurred at i = %d and j = %d",i,j)
blurred_image1 = cv2.merge(blurred_image1_r,blurred_image1_g,blurred_image1_b)
cv2.imwrite('blurred_image1.png',blurred_image1)




