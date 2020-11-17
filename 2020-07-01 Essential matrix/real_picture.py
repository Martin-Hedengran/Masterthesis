#!/usr/bin/env python
import cv2
import numpy as np
import math
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from matplotlib import pyplot as plt

# Code borrowed from the following sources
# https://docs.opencv.org/master/d1/d89/tutorial_py_orb.html
# https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
# https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
# https://www.learnopencv.com/rotation-matrix-to-euler-angles/
# https://www.morethantechnical.com/2016/10/17/structure-from-motion-toy-lib-upgrades-to-opencv-3/
MIN_MATCH_COUNT = 10

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),15,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),15,color,-1)
    return img1,img2


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :

    assert(isRotationMatrix(R))
    
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

#Resize image while keeping aspect ratio.
def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

# Initiate ORB detector
#orb = cv2.ORB_create()
sift = cv2.SIFT_create()
# Load images
img1 = cv2.imread("input2/DJI_0001.JPG", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("input2/DJI_0002.JPG", cv2.IMREAD_GRAYSCALE)
#img1 = rescale(img1, 0.25, anti_aliasing=False)
#img2 = rescale(img2, 0.25, anti_aliasing=False)
#img3 = cv2.imread("input2/DJI_0003.JPG", cv2.IMREAD_GRAYSCALE)
print(1)
# find the keypoints and descriptors with ORB
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
#kp3, des3 = orb.detectAndCompute(img3, None)
print(2)
if ((len(des1) < 500) and (len(des2) < 500)):
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
else:
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(des1,des2,k=2)
print(matches)
# Need to draw only good matches, so create a mask
'''
matchesMask = [[0,0] for i in range(len(matches))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

'''

# Sort them in the order of their distance.
#matches = sorted(matches, key = lambda x:x.distance)
#matches2 = sorted(matches2, key = lambda x:x.distance)
print(3)
# Draw first 10 matches.
#img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#img5 = cv2.drawMatches(img2,kp2,img3,kp3,matches2[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

if len(good)>MIN_MATCH_COUNT:
    pts1 = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    pts2 = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

# ratio test as per Lowe's paper
#for m in matchesMask:
#    pts2.append(kp2[m.trainIdx].pt)
#    pts1.append(kp1[m.queryIdx].pt)
#    pts3.append(kp3[m.trainIdx].pt)
#    pts4.append(kp2[m.queryIdx].pt)   

#print(matchesMask)
#pts1, pts2 = np.hsplit(matchesMask, 2)
print(pts1, pts2)
print(4)
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
#pts3 = np.int32(pts3)
#pts4 = np.int32(pts4)

F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
#F, mask = cv2.findFundamentalMat(pts2,pts3,cv2.FM_LMEDS)

# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]
#pts3 = pts3[mask.ravel()==1]
#pts4 = pts4[mask.ravel()==1]
print(5)
cameraMatrix = np.array([[3666.666504,   0.,         2736.0000002 ],
                        [  0.,         3666.666504, 1824.000000 ],
                        [0., 0., 1.]])
essential_matrix, mask = cv2.findEssentialMat(pts1, pts2, cameraMatrix)
#essential_matrix2, mask2 = cv2.findEssentialMat(pts3, pts4, cameraMatrix)
#print(essential_matrix)
print(6)
retval, R, t, mask = cv2.recoverPose(essential_matrix, pts1, pts2, cameraMatrix)
#retval2, R2, t2, mask2 = cv2.recoverPose(essential_matrix2, pts3, pts4, cameraMatrix)
print(7)
#rtMatrix = np.concatenate((R, t), axis=1)
#rtMatrix2 = np.concatenate((R2, t2), axis=1)
#projectionMatrix = np.matmul(cameraMatrix, rtMatrix)
#projectionMatrix2 = np.matmul(cameraMatrix, rtMatrix2)

#cv2.triangulatePoints()

print(t)
#print(R)
#print(rtMatrix)
print(rotationMatrixToEulerAngles(R))
#print(projectionMatrix2)




# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
#lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
#lines1 = lines1.reshape(-1,3)
#img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
#lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
#lines2 = lines2.reshape(-1,3)
#img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)



#cv2.imshow("img1", img1)
#cv2.imshow("img2", img2)
img3 = ResizeWithAspectRatio(img3, width=1280)
#img5 = ResizeWithAspectRatio(img5, width=1280)
cv2.imshow("img3", img3)
#cv2.imshow("img4", img4)
#cv2.imshow("img5", img5)
cv2.waitKey(100000)

