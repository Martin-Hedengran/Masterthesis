import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

MIN_MATCH_COUNT = 10

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
    
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    
    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

img1 = cv2.imread("input2/DJI_0001.JPG", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("input2/DJI_0002.JPG", cv2.IMREAD_GRAYSCALE)
counter = 0


K = np.array([[3666.666504,   0.,         2736.0000002 ],
                [  0.,         3666.666504, 1824.000000 ],
                [0., 0., 1.]])



###############################
#1----SIFT feature matching---#
###############################

#detect sift features for both images
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

#use flann to perform feature matching
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

if len(good)>MIN_MATCH_COUNT:
    p1 = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    p2 = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    flags = 2)

img_siftmatch = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
cv2.imwrite('sift_match_' + str(counter) + '.png',img_siftmatch)

#########################
#2----essential matrix--#
#########################
E, mask = cv2.findEssentialMat(p1, p2, K, cv2.RANSAC, 0.999, 1.0);

matchesMask = mask.ravel().tolist()

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)

img_inliermatch = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
cv2.imwrite('inlier_match_' + str(counter) + '.png',img_inliermatch)
print("Essential matrix:")
print(E)

####################
#3----recoverpose--#
####################

points, R, t, mask = cv2.recoverPose(E, p1, p2)
print("Rotation:")
print(R)
print("Rotation in radians:")
print(rotationMatrixToEulerAngles(R))
print("Translation:")
print(t)
# p1_tmp = np.expand_dims(np.squeeze(p1), 0)
#p1_tmp = np.ones([3, p1.shape[0]])
#p1_tmp[:2,:] = np.squeeze(p1).T
#p2_tmp = np.ones([3, p2.shape[0]])
#p2_tmp[:2,:] = np.squeeze(p2).T
#print((np.dot(R, p2_tmp) + t) - p1_tmp)
#img3 = ResizeWithAspectRatio(img3, width=1280)
#cv2.imshow("img3", img3)
#cv2.waitKey(100000)