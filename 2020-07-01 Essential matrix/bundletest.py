import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from bundleclass import BundleAdjustment
from camera_model import camera
import g2o

MIN_MATCH_COUNT = 10
Window_Size = 2
matched_points = []
Rotations = []
Translations = []
points_2d = []
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    
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

def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q.x()
    q1 = Q.y()
    q2 = Q.z()
    q3 = Q.w()
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix

def rescale(img):
    
    scale_percent = 20 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

img1 = cv2.imread("/home/kubuntu/Masterthesis/2020-07-01 Essential matrix/input2/DJI_0001.JPG", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("/home/kubuntu/Masterthesis/2020-07-01 Essential matrix/input2/DJI_0002.JPG", cv2.IMREAD_GRAYSCALE)
counter = 0
img1 = rescale(img1)
img2 = rescale(img2)
cam = camera(img1)

K = np.array([[cam.focal_length,   0.,         cam.cx ],
                [  0.,         cam.focal_length, cam.cy ],
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
print(len(good))
if len(good)>MIN_MATCH_COUNT:
    p1 = np.float64([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    p2 = np.float64([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

matched_points.append(len(p1))

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
#initial rot
Rotations.append(np.identity(3))
Rotations.append(R)
Translations.append([0, 0, 0])
Translations.append(t.T[0])
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
#TODO: Display 3D points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

for x, y, z in point_3d:
    ax.scatter(x, y, z, c="r", marker="o")

#plt.show()
#fig.savefig('3-D_' + str(counter) + '.png')

#############################
#6-----Bundleadjustment-----#
#############################
#TODO: add winodwed bundle adjustment
#! Only 2 picture comparison
#Initiate class
bundleAdjust = BundleAdjustment()
#@param(focal length, principal point, baseline)
bundleAdjust.camera(cam, 0) 
cam = g2o.CameraParameters(cam.focal_length, np.array([cam.cx, cam.cy], dtype=np.float64), 0)
z = cam.cam_map(g2o.SE3Quat(Rotations[0], Translations[0]) * point_3d[0])
print(z)
'''
#Pose 1 set to 0, pose 2 derived from calculated rotation and translation
#@param (Rotation[3x3], Translation[3,])
pose_camera_1 = g2o.SE3Quat(np.identity(3), [0, 0, 0])
pose_camera_2 = g2o.SE3Quat(R, t.T[0])
#@param (pose id, pose, camera parameters) 
bundleAdjust.add_pose(1, pose_camera_1)
bundleAdjust.add_pose(2, pose_camera_2)
'''
new_p1 = p1.reshape(p1.shape[0], (p1.shape[1] * p1.shape[2]))
new_p2 = p2.reshape(p2.shape[0], (p2.shape[1] * p2.shape[2]))

#Add poses
for j in range(Window_Size):
    #Remove old poses
    if (len(Rotations) or len(Translations)) > Window_Size:
        Rotations.pop(0)
        Translations.pop(0)
    bundleAdjust.add_pose(j+1, g2o.SE3Quat(Rotations[j], Translations[j]))

for i in range(len(point_3d)):
    #Add 3d points
    #@param (point id, point[3,])
    bundleAdjust.add_point(i+1, point_3d[i])
    #Add edges from first camera
    bundleAdjust.add_edge(i+1, 1, new_p2[i])
    #@param (point id, pose id, point2d[2,])
    #Add edges from second camera
    bundleAdjust.add_edge(i+1, 2, new_p1[i])

bundleAdjust.optimizer()
bundle_T = bundleAdjust.get_pose(2).translation()
bundle_Q = bundleAdjust.get_pose(2).rotation()
bundle_rot = quaternion_rotation_matrix(bundle_Q)
print(bundle_rot)
print(bundle_T)
