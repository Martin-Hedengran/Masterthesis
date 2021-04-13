import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import symmetric_transfer_error

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

def rescale(img):
    
    scale_percent = 20 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

#Gazebo pinhole camera formula: focal_length = image_width / (2*tan(hfov_radian / 2)
def camera_extrinsics(img):
    hfov = 1.047 # 60 degrees
    height, width = img.shape
    focal_length = width/(2*math.tan(hfov/2))
    cx = width/2
    cy = height/2
    
    return focal_length, cx, cy

img1 = cv2.imread("input2/DJI_0001.JPG", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("input2/DJI_0002.JPG", cv2.IMREAD_GRAYSCALE)
counter = 0
img1 = rescale(img1)
img2 = rescale(img2)
focal_lenght, cx, cy = camera_extrinsics(img1)

K = np.array([[focal_lenght,   0.,         cx ],
                [  0.,         focal_lenght, cy ],
                [0., 0., 1.]])

###############################
#----Fast feature matching----#
###############################
#!Note: fast doesnt have descriptors, only detection hence the use of BEBLID descriptors.

fast = cv2.FastFeatureDetector_create()
#! The input to belid is feature scale and depends on the detector used. fast=5, sift = 6,75, orb = 1
descriptor = cv2.xfeatures2d.BEBLID_create(5)

kp1 = fast.detect(img1, None)
kp2 = fast.detect(img2, None)
kp1, des1 = descriptor.compute(img1, kp1)
kp2, des2 = descriptor.compute(img2, kp2)
#Bruteforce hamming distance
#matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
#matches = matcher.knnMatch(des1, des2, 2)

#Flann matching using LSH(Close to hamming distance)
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)
###############################
#1----SIFT feature matching---#
###############################
'''
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
'''
# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)
print(len(good))
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
E, mask = cv2.findEssentialMat(p1, p2, K, cv2.RANSAC, 0.999, 1.0)
F, F_mask = cv2.findFundamentalMat(p1, p2, cv2.FM_LMEDS)
M, H_mask = cv2.findHomography(p1, p2, cv2.RANSAC,5.0)
E_score = symmetric_transfer_error.checkEssentialScore(E, K, p1, p2)
H_score = symmetric_transfer_error.checkHomographyScore(M, p1, p2)

#print("F matrix")
#print(F)
print(E_score)
print(H_score)
#RH = H_score / (H_score + E_score)

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

#######################
#4----triangulation---#
#######################

#calculate projection matrix for both camera
M_r = np.hstack((R, t))
M_l = np.hstack((np.eye(3, 3), np.zeros((3, 1))))

P_l = np.dot(K,  M_l)
P_r = np.dot(K,  M_r)

# undistort points
p1 = p1[np.asarray(matchesMask)==1,:,:]
p2 = p2[np.asarray(matchesMask)==1,:,:]
p1_un = cv2.undistortPoints(p1,K,None)
p2_un = cv2.undistortPoints(p2,K,None)
p1_un = np.squeeze(p1_un)
p2_un = np.squeeze(p2_un)

#triangulate points this requires points in normalized coordinate
point_4d_hom = cv2.triangulatePoints(P_l, P_r, p1_un.T, p2_un.T)
point_3d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
point_3d = point_3d[:3, :].T

#############################
#5----output 3D pointcloud--#
#############################
'''
#TODO: Display 3D points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

for x, y, z in point_3d:
    ax.scatter(x, y, z, c="r", marker="o")

plt.show()
fig.savefig('3-D_' + str(counter) + '.jpg')
'''