import cv2
import numpy as np
np.set_printoptions(suppress=True)
import symmetric_transfer_error
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
from skimage.transform import EssentialMatrixTransform

# turn [[x,y]] -> [[x,y,1]]
def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

IRt = np.eye(4)

# pose
def extractRt(E):
    W = np.mat([[0,-1,0],[1,0,0],[0,0,1]],dtype=float)
    U,d,Vt = np.linalg.svd(E)
    assert np.linalg.det(U) > 0
    if np.linalg.det(Vt) < 0:
        Vt *= -1.0
    R = np.dot(np.dot(U, W), Vt)
    if np.sum(R.diagonal()) < 0:
        R = np.dot(np.dot(U, W.T), Vt)
    t = U[:, 2]
    ret = np.eye(4)
    ret[:3, :3] = R
    ret[:3, 3] = t
    return ret
'''
def extract(img):
    orb = cv2.ORB_create()
    # detection
    pts = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=3)

    # extraction
    kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in pts]
    kps, des = orb.compute(img, kps)

    # return pts and des
    return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des
'''
def extract(img):
    #Use Fast for feature detection, BEBLID for description.
    fast = cv2.FastFeatureDetector_create()
    #The input to beblid is feature scale and depends on the detector used. fast=5, sift = 6,75, orb = 1
    descriptor = cv2.xfeatures2d.BEBLID_create(5)

    #Feature extracting
    kps = fast.detect(img, None)
    kps, des = descriptor.compute(img, kps)

    return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des
    


def normalize(Kinv, pts):
    return np.dot(Kinv, add_ones(pts).T).T[:, 0:2]

def denormalize(K, pt):
    ret = np.dot(K, np.array([pt[0], pt[1], 1.0]))
    ret /= ret[2]
    return int(round(ret[0])), int(round(ret[1]))

'''
def match_frames(f1, f2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(f1.des, f2.des, k=2)

    # Lowe's ratio test
    ret = []
    idx1,idx2 = [], []

    for m,n in matches:
        if m.distance < 0.75*n.distance:
            # keep around indices
            idx1.append(m.queryIdx)
            idx2.append(m.trainIdx)

            p1 = f1.pts[m.queryIdx]
            p2 = f2.pts[m.trainIdx]
            ret.append((p1, p2))
    assert len(ret) >= 8
    ret = np.array(ret)
    idx1 = np.array(idx1)
    idx2 = np.array(idx2)

    # fit matrix
    model, inliers = ransac((ret[:, 0], ret[:, 1]),
                            EssentialMatrixTransform,
                            #FundamentalMatrixTransform,
                            min_samples=8,
                            #residual_threshold=1,
                            residual_threshold=0.005,
                            max_trials=200)
    #print(sum(inliers), len(inliers))

    # ignore outliers
    Rt = extractRt(model.params)

    # return
    return idx1[inliers], idx2[inliers], Rt
'''
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

def Match_features(f1, f2, K):
    lowe, good, idx1, idx2 = [], [], [], []
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
    if (len(f1.des) < 500) or  (len(f2.des) < 500):
        matches = matcher.knnMatch(f1.des, f2.des, 2)
    else:
        matches = flann.knnMatch(f1.des, f2.des,k=2)
    
    # store all the good matches as per Lowe's ratio test.
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            #good.append(m)
            idx1.append(m.queryIdx)
            idx2.append(m.trainIdx)

            p1 = f1.pts[m.queryIdx]
            p2 = f2.pts[m.trainIdx]
            lowe.append((p1, p2))


    pts1 = np.float64([ f1.pts[m.queryIdx] for m in good ]).reshape(-1,1,2)
    pts2 = np.float64([ f2.pts[m.trainIdx] for m in good ]).reshape(-1,1,2)
    
    #Minimum 8 matches
    lowe = np.array(lowe)
    assert len(lowe) >= 8
    #Essential matrix filter
    model, inliers = ransac((lowe[:, 0], lowe[:, 1]), 
                            EssentialMatrixTransform,
                            min_samples = 8,
                            residual_threshold = 0.005,
                            max_trials = 100)

    Rt = extractRt(model.params)

    #!TODO fix this?
    #pts1 = np.float64([ f1.pts[m.queryIdx] for m in good ]).reshape(-1,1,2)
    #pts2 = np.float64([ f2.pts[m.trainIdx] for m in good ]).reshape(-1,1,2)

    #R, t, mask = EstimateRt(pts1, pts2, K)
    #ret = np.eye(4)
    #ret[:3, :3] = R
    #ret[:3, 3] = t.T[0]

    idx1 = np.array(idx1)
    idx2 = np.array(idx2)

    return idx1[inliers], idx2[inliers], Rt

class Frame(object):
    def __init__(self, mapp, img, K):
        self.K = K
        self.Kinv = np.linalg.inv(self.K)
        self.pose = IRt

        pts, self.des = extract(img)
        self.pts = normalize(self.Kinv, pts)
        self.img = img
        self.id = len(mapp.frames)
        mapp.frames.append(self)