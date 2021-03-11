import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from GPSPhoto import gpsphoto
from mpl_toolkits.mplot3d import Axes3D
import bundleadjust_camera_coords

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

def rescale(img):
    
    scale_percent = 20 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

def camera_extrinsics(img):
    hfov = 1.047 # 60 degrees
    height, width = img.shape
    focal_length = width/(2*math.tan(hfov/2))
    cx = width/2
    cy = height/2
    
    return focal_length, cx, cy

def triangulation(R_first, T_first, R_second, T_second, cameraMatrix, p1, p2, matchesMask):
    #calculate projection matrix for both camera
    M_r = np.hstack((R_first, T_first))
    M_l = np.hstack((R_second,T_second))

    P_l = np.dot(cameraMatrix,  M_l)
    P_r = np.dot(cameraMatrix,  M_r)

    # undistort points
    p1 = p1[np.asarray(matchesMask)==1,:,:]
    p2 = p2[np.asarray(matchesMask)==1,:,:]
    p1_un = cv2.undistortPoints(p1, cameraMatrix, None)
    p2_un = cv2.undistortPoints(p2, cameraMatrix, None)
    p1_un = np.squeeze(p1_un)
    p2_un = np.squeeze(p2_un)

    #triangulate points this requires points in normalized coordinate
    point_4d_hom = cv2.triangulatePoints(P_l, P_r, p1_un.T, p2_un.T)
    point_3d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
    point_3d = point_3d[:3, :].T
    return point_3d


# Initiate detector
#orb = cv2.ORB_create()
sift = cv2.SIFT_create()

rotation = []
translation = []
position = []
posLat = []
posLon = []
altitude = []
triang_points = []
window_size = 5
count = 0

GPSx = []
GPSy = []

GPSData = gpsphoto.getGPSData("input2/DSC00" + str(203) + ".jpg")

#currPos = np.array([[111.320*math.cos(math.radians(GPSData['Latitude']))*GPSData['Longitude']*1000],[110.574*GPSData['Latitude']*1000],[GPSData['Altitude']],[1]])
currPos = np.array([[0],[0],[0],[1]])

for i in range(54):
    # Load images
    n = "input2/DSC00" + str(203+i) + ".jpg"
    k = "input2/DSC00" + str(203+i+1) + ".jpg"
    img1 = cv2.imread(n, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(k, cv2.IMREAD_GRAYSCALE)

    img1 = rescale(img1)
    img2 = rescale(img2)

    # -------------------- ORB TEST ---------------------------------- #
    # # find the keypoints and descriptors with ORB
    # kp1, des1 = orb.detectAndCompute(img1, None)
    # kp2, des2 = orb.detectAndCompute(img2, None)

    # # create BFMatcher object
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # # Match descriptors.
    # matches = bf.match(des1,des2)

    # # Sort them in the order of their distance.
    # matches = sorted(matches, key = lambda x:x.distance)

    # # Draw first 10 matches.
    # img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # pts1 = []
    # pts2 = []
    # # ratio test as per Lowe's paper
    # for m in matches:
    #     pts2.append(kp2[m.trainIdx].pt)
    #     pts1.append(kp1[m.queryIdx].pt)

    # pts1 = np.int32(pts1)
    # pts2 = np.int32(pts2)
    # F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)

    # # We select only inlier points
    # pts1 = pts1[mask.ravel()==1]
    # pts2 = pts2[mask.ravel()==1]

    # -------------------- SIFT TEST --------------------------------- #

    MIN_MATCH_COUNT = 10

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
    print(len(good))
    if len(good)>MIN_MATCH_COUNT:
        pts1 = np.float64([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        pts2 = np.float64([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    # ---------------------------------------------------------------- #

    focal_lenght, cx, cy = camera_extrinsics(img1)

    cameraMatrix = np.array([[focal_lenght,   0.,         cx ],
                [  0.,         focal_lenght, cy ],
                [0., 0., 1.]])

    essential_matrix, mask = cv2.findEssentialMat(pts1, pts2, cameraMatrix, cv2.RANSAC, 0.999, 1.0)
    #print(essential_matrix)
    matchesMask = mask.ravel().tolist()
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
    if i == 0:
        triang_points = (triangulation(np.eye(3, 3), np.zeros((3, 1)), R, t, cameraMatrix, pts1, pts2, matchesMask))
    elif triang_points == []:
        triang_points = (triangulation(rotation[i-1], translation[i-1], rotation[i], translation[i], cameraMatrix, pts1, pts2, matchesMask))
    else:
        temp = triang_points
        triang_points = (triangulation(rotation[i-1], translation[i-1], rotation[i], translation[i], cameraMatrix, pts1, pts2, matchesMask))
        triang_points = np.concatenate((temp, triang_points), axis=0)
    if count == window_size:
        bundleadjust_camera_coords.bundle_adjustment(focal_lenght, cx, cy, triang_points, rotation, translation, 10, 0)
        count = 0
        triang_points = []
    count+=1
    print("done with img "+str(i))


    GPSData = gpsphoto.getGPSData(k)

    posLat.append(GPSData['Latitude'])
    posLon.append(GPSData['Longitude'])

    altitude.append(GPSData['Altitude'])
    print(GPSData)

    GPSy.append(110.574*GPSData['Latitude']*1000)                           # Convert Latitude to meters
    GPSx.append(111.320*math.cos(math.radians(GPSData['Latitude']))*GPSData['Longitude']*1000) # Convert Longitude to meters
    
path = generate_path(position)
#print(path)
#print(translation)
#print(rotation)
#print(rotationMatrixToEulerAngles(R))

# plt.plot(posLon, posLat)
# plt.axis('equal')

fig, axs = plt.subplots(2)
fig.suptitle('Vertically stacked subplots')
axs[0].plot(GPSx, GPSy)
axs[1].plot(path[:,1], -path[:,0])

plt.axis('equal')


# ----------------------- Camera Estimate ------------------- #

fig = plt.figure()
ax = fig.gca(projection='3d')

X = path[:,0]
Y = path[:,1]
Z = path[:,2]

ax.scatter(X, Y, Z)

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

# ------------------ GPS DATA ------------------------------------ #

fig2 = plt.figure()
ax2 = fig2.gca(projection='3d')

X2 = np.array(GPSx)
Y2 = np.array(GPSy)
Z2 = np.array(altitude)

ax2.scatter(X2, Y2, Z2)

# Create cubic bounding box to simulate equal aspect ratio
max_range2 = np.array([X2.max()-X2.min(), Y2.max()-Y2.min(), Z2.max()-Z2.min()]).max()
Xb2 = 0.5*max_range2*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X2.max()+X2.min())
Yb2 = 0.5*max_range2*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y2.max()+Y2.min())
Zb2 = 0.5*max_range2*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z2.max()+Z2.min())
# Comment or uncomment following both lines to test the fake bounding box:
for xb2, yb2, zb2 in zip(Xb2, Yb2, Zb2):
   ax2.plot([xb2], [yb2], [zb2], 'w')

ax2.set_xlabel("Longitude")
ax2.set_ylabel("Latitude")
ax2.set_zlabel("Altitude")

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
