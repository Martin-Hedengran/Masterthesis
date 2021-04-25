
import cv2
import numpy as np
import symmetric_transfer_error
from skimage.measure import ransac
from skimage.transform import EssentialMatrixTransform

def Triangulation(K, skew, pose1, pose2, matchesMask, p1, p2):
    #calculate projection matrix for both camera
    P_l = np.dot(K,  np.delete(pose1, 3, 0))
    P_r = np.dot(K,  np.delete(pose2, 3, 0))
    
    #print(matchesMask)
    # undistort points
    p1 = p1[np.asarray(matchesMask)==1,:]
    p2 = p2[np.asarray(matchesMask)==1,:]

    p1_un = cv2.undistortPoints(p1,K,skew)
    p2_un = cv2.undistortPoints(p2,K,skew)
    p1_un = np.squeeze(p1_un)
    p2_un = np.squeeze(p2_un)

    #triangulate points this requires points in normalized coordinate
    point_4d_hom = cv2.triangulatePoints(P_l, P_r, p1_un.T, p2_un.T)
    point_3d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
    point_3d = point_3d[:3, :].T
    #Reject points behind camera
    point_3d = (point_3d[:, 2] > 0)
    
    return point_3d

def EstimateRt(p1, p2, K):
    #Calculate lowest reprojection error between E and H, choose lowest
    E, mask = cv2.findEssentialMat(p1, p2, K, cv2.RANSAC, 0.999, 1.0)
    M, H_mask = cv2.findHomography(p1, p2, cv2.RANSAC,5.0)
    E_score = symmetric_transfer_error.checkEssentialScore(E, K, p1, p2)
    H_score = symmetric_transfer_error.checkHomographyScore(M, p1, p2)
    
    if E_score >= H_score:
        points, R, t, mask_temp = cv2.recoverPose(E, p1, p2)
    else:
        #Figure out correct pose from homography using SVD
        H = np.transpose(M)
        h1 = H[0]
        h2 = H[1]
        h3 = H[2]

        Kinv = np.linalg.inv(K)

        L = 1 / np.linalg.norm(np.dot(Kinv, h1))

        r1 = L * np.dot(Kinv, h1)
        r2 = L * np.dot(Kinv, h2)
        r3 = np.cross(r1, r2)

        t = L * np.dot(Kinv, h3)

        R = np.array([[r1], [r2], [r3]])
        R = np.reshape(R, (3, 3))
        U, S, V = np.linalg.svd(R, full_matrices=True)

        U = np.matrix(U)
        V = np.matrix(V)
        R = U * V
        mask = H_mask
    matchesMask = mask.ravel().tolist()
    return R, t, matchesMask

def extract(img):
    #Use Fast for feature detection, BEBLID for description.
    fast = cv2.FastFeatureDetector_create()
    #The input to beblid is feature scale and depends on the detector used. fast=5, sift = 6,75, orb = 1
    descriptor = cv2.xfeatures2d.BEBLID_create(5)

    #Feature extracting
    kps = fast.detect(img, None)
    kps, des = descriptor.compute(img, kps)

    return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des
    
def Match_features(f1, f2, K):
    good, idx1, idx2 = [], [], []
    #Define matcher parameters
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    bfmatch = cv2.BFMatcher()
    #Flann matching using LSH(Close to hamming distance)
    FLANN_INDEX_LSH = 6
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                    table_number = 6, # 12
                    key_size = 12,     # 20
                    multi_probe_level = 1) #2
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    #If less than 500 keypoints use bf matcher
    if (len(f1.descriptors) < 500) or  (len(f2.descriptors) < 500):
        matches = matcher.knnMatch(f1.descriptors, f2.descriptors, 2)
    else:
        matches = flann.knnMatch(f1.descriptors, f2.descriptors,k=2)
    
    # store all the good matches as per Lowe's ratio test.
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            idx1.append(m.queryIdx)
            idx2.append(m.trainIdx)

            p1 = f1.points[m.queryIdx]
            p2 = f2.points[m.trainIdx]
            good.append((p1, p2))

    #Essential matrix filter
    if len(good) > 0:
        good = np.array(good)
        model, inliers = ransac((good[:, 0], good[:, 1]), 
                                EssentialMatrixTransform,
                                min_samples = 8,
                                residual_threshold = 0.005,
                                max_trials = 100)
        good = good[inliers]
        
        #Estimate pose
        R, t, mask = EstimateRt(good[:, 0], good[:, 1], K)
        current_pose = np.eye(4)
        current_pose[:3, :3] = R
        current_pose[:3, 3] = t.T
        
        idx1 = np.array(idx1)
        idx2 = np.array(idx2)

    return idx1[inliers], idx2[inliers], current_pose, mask

class Frame(object):
    
    #Class / structure for saving information about a single frame.
    def __init__(self, img, map):
        self.id = len(map.frames)
        self.points, self.descriptors = extract(img)
        self.pose = np.eye(4)
        map.frames.append(self)