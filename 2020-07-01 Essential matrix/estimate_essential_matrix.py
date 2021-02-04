#!/usr/bin/env python
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Code borrowed from the following sources
# https://docs.opencv.org/master/d1/d89/tutorial_py_orb.html
# https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
# https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
# https://www.learnopencv.com/rotation-matrix-to-euler-angles/
# https://www.morethantechnical.com/2016/10/17/structure-from-motion-toy-lib-upgrades-to-opencv-3/


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
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
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

def generate_path(translations):
    path = []
    current_point = np.array([0, 0, 0, 0])

    for t in zip(translations):
        path.append(current_point)
        # don't care about rotation of a single point
        current_point = current_point + np.reshape(t, 4)

    return np.array(path)
    #return np.array(path)



# Initiate ORB detector
orb = cv2.ORB_create()
rotation = []
translation = []
position = []
currPos = np.array([[0],[0],[0],[1]])
for i in range(50):
    # Load images
    n = "input2/DSC00" + str(203+i) + ".jpg"
    k = "input2/DSC00" + str(203+i+1) + ".jpg"
    img1 = cv2.imread(n, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(k, cv2.IMREAD_GRAYSCALE)

    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for m in matches:
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)

    # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]


    cameraMatrix = np.array([[277.191356,   0.,         320.5 ],
                            [  0.,         277.191356, 240.5 ],
                            [0., 0., 1.]])
    essential_matrix, mask = cv2.findEssentialMat(pts1, pts2, cameraMatrix)
    #print(essential_matrix)
    i+=1

    retval, R, t, mask = cv2.recoverPose(essential_matrix, pts1, pts2, cameraMatrix)
    x, y ,z = R
    rotation.append(R) 
    translation.append(t)

    H = np.concatenate((R,t), axis=1)
    H = np.concatenate((H,np.array([[0,0,0,1]])), axis=0)
    print("R: "+ str(R) +"\t t: " + str(t) +"\t H: "+ str(H))
    print(currPos)
    position.append(np.matmul(H,currPos))
    currPos = np.matmul(H,currPos)
    print("done with img "+str(i))

path = generate_path(position)
print(path)
print(translation)
print(rotation)
#print(rotationMatrixToEulerAngles(R))

# plt.plot(path[:,1], path[:,0])
# plt.axis('equal')
# plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')

X = path[:,0]
Y = path[:,1]
Z = path[:,2]

scat = ax.scatter(X, Y, Z)

# Create cubic bounding box to simulate equal aspect ratio
max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
# Comment or uncomment following both lines to test the fake bounding box:
for xb, yb, zb in zip(Xb, Yb, Zb):
   ax.plot([xb], [yb], [zb], 'w')

ax.set_xlabel("x axis")
ax.set_ylabel("y axis")
ax.set_zlabel("z axis")

plt.grid()
plt.show()

# # Find epilines corresponding to points in right image (second image) and
# # drawing its lines on left image
# lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
# lines1 = lines1.reshape(-1,3)
# img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
# # Find epilines corresponding to points in left image (first image) and
# # drawing its lines on right image
# lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
# lines2 = lines2.reshape(-1,3)
# img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)


# #cv2.imshow("img1", img1)
# #cv2.imshow("img2", img2)
# cv2.imshow("img3", img3)
# #cv2.imshow("img4", img4)
# cv2.imshow("img5", img5)
# cv2.waitKey(100000)

