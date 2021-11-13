import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread("/Users/3pi/Documents/Computer_Vision/Ass_3_Estimate_Fundamental_Matrix/image_pair_1/house1.jpg",0) # queryImage
img2 = cv2.imread("/Users/3pi/Documents/Computer_Vision/Ass_3_Estimate_Fundamental_Matrix/image_pair_1/house2.jpg",0) # trainImage

# Initiate SIFT detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:20],None, flags=2)


plt.imshow(img3),plt.show()

'''
descriptor_extractor = ORB()

descriptor_extractor.detect_and_extract(img_left)
keypoints_left = descriptor_extractor.keypoints
descriptors_left = descriptor_extractor.descriptors
#print("left keypoints : \n",keypoints_left)

descriptor_extractor.detect_and_extract(img_right)
keypoints_right = descriptor_extractor.keypoints
descriptors_right = descriptor_extractor.descriptors
#print("right keypoints : \n",keypoints_right)
matches = match_descriptors(descriptors_left, descriptors_right,
                            cross_check=True)
model, inliers = ransac((keypoints_left[matches[:, 0]],
                         keypoints_right[matches[:, 1]]),
                        FundamentalMatrixTransform, min_samples=8,
                        residual_threshold=1, max_trials=5000)
matches = matches[inliers]


#tried plotting circles over what I think are matching points but it's probably wrong

for i in range(0,9):
	left_image_match = int(matches[i,0])
	right_image_match = int(matches[i,1])
	print ("%d th Keypoint in left image is %f %f and in right is %f %f"%(i,keypoints_left[left_image_match,0],keypoints_left[left_image_match,1],keypoints_right[right_image_match,0],keypoints_right[right_image_match,1]))
	x1=int(keypoints_left[left_image_match,0])
	y1=int(keypoints_left[left_image_match,1])
	x2=int(keypoints_right[right_image_match,0])
	y2=int(keypoints_right[right_image_match,1])
	newleft = cv2.circle(img_left,(x1,y1), 10, (0,255,0))
	newright = cv2.circle(img_right,(x2,y2), 10, (0,255,0))
	cv2.imwrite(("img_left_"+str(i)+".png"),newleft*255)
	cv2.imwrite(("img_right_"+str(i)+".png"),newright*255)

newleft = cv2.circle(img_left,(181,32), 10, (0,255,0))
newright = cv2.circle(img_right,(181,34), 10, (0,255,0))
cv2.imwrite(("img_left_1.png"),newleft*255)
cv2.imwrite(("img_right_1.png"),newright*255)
newleft = cv2.circle(img_left,(260,101), 10, (0,255,0))
newright = cv2.circle(img_right,(236,103), 10, (0,255,0))
cv2.imwrite(("img_left_2.png"),newleft*255)
cv2.imwrite(("img_right_2.png"),newright*255)


#This part of code gives nice matches, 
#so it's probably giving good results and I'm using it the wrong way
#check link https://scikit-image.org/docs/dev/auto_examples/transform/plot_fundamental_matrix.html


inlier_keypoints_left = keypoints_left[matches[inliers, 0]]
inlier_keypoints_right = keypoints_right[matches[inliers, 1]]

print("Number of matches:", matches.shape[0])
print("Number of inliers:", inliers.sum())
print("coord of matches from left image")
print(keypoints_left)
print(matches)
fig, ax = plt.subplots(nrows=2, ncols=1)

plt.gray()

plot_matches(ax[0], img_left, img_right, keypoints_left, keypoints_right,
             matches[inliers], only_matches=True)
ax[0].axis("off")
ax[0].set_title("Inlier correspondences")
plt.show()

'''