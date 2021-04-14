#!/usr/bin/env python3
import cv2
import numpy as np
from codetiming import Timer
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
import symmetric_transfer_error

#Img scaled to 40 percent for 
Display_scale = 25
Display_height = 2160 * Display_scale / 100
Display_width= 3840 * Display_scale / 100
Display_dim = (int(Display_width), int(Display_height))


class PoseEstimation():
    def EstimateRt(self, p1, p2, K):
        E, mask = cv2.findEssentialMat(p1, p2, K, cv2.RANSAC, 0.999, 1.0)
        M, H_mask = cv2.findHomography(p1, p2, cv2.RANSAC,5.0)
        E_score = symmetric_transfer_error.checkEssentialScore(E, K, p1, p2)
        H_score = symmetric_transfer_error.checkHomographyScore(M, p1, p2)

        if E_score >= H_score:
            points, R, t, mask = cv2.recoverPose(E, p1, p2)
        
        else:
            H = np.transpose(H)
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
    
    def Triangulation(self, R, t, matches):
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

class CameraSettings():
    def __init__(self):
        # DJI Phantom 4 pro
        # Calibration from Metashape 
        #! The values from metashape are offset from optical center
        self.cameraMatrix = np.array([[3657,   0.,         5472 / 2 - 33 ],
                                [  0.,         3657, 3648 / 2 - 3.5 ],
                                [0., 0., 1.]])
        self.distCoeffs = np.array(
                [[ 0.0029, 0.0155, 0.014, 0.0022, 0.00025 ]])
    
    def MatrixScaling(self):
        self.scale_factor = 0.20
        #self.scale_factor = 1
        self.cameraMatrix *= self.scale_factor
        self.cameraMatrix[2, 2] = 1
        
        return self.cameraMatrix, self.distCoeffs 

    def ImageScaling(self, img):
        width = int(img.shape[1] * self.scale_factor)
        height = int(img.shape[0] * self.scale_factor)
        dim = (width, height)
        img = cv2.resize(img, self.dim, interpolation = cv2.INTER_AREA)
        return img


class FeatureExtractor(object):
    def __init__(self):
        self.last_frame = None
        self.last_frame_sift = None
        self.last_frame_orb = None
        self.last_frame_beblid = None
        #Hardcode image dimensions 768x432
        self.scale_percent = 20
        self.height = 2160 * self.scale_percent / 100
        self.width= 3840 * self.scale_percent / 100
        self.dim = (int(self.width), int(self.height))

        #Initialize detector and descriptor
        #!Note: fast doesnt have descriptors, only detection hence the use of BEBLID descriptors.
        self.fast = cv2.FastFeatureDetector_create()
        #! The input to belid is feature scale and depends on the detector used. fast=5, sift = 6,75, orb = 1
        self.descriptor = cv2.xfeatures2d.BEBLID_create(5)
        self.descriptor_orb = cv2.xfeatures2d.BEBLID_create(1)

        self.sift = cv2.SIFT_create()
        self.orb = cv2.ORB_create()
        #Bruteforce hamming distance
        self.matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
        self.bfmatch = cv2.BFMatcher()
        #Flann matching using LSH(Close to hamming distance)
        FLANN_INDEX_LSH = 6
        self.index_params= dict(algorithm = FLANN_INDEX_LSH,
                        table_number = 6, # 12
                        key_size = 12,     # 20
                        multi_probe_level = 1) #2
        self.search_params = dict(checks = 50)
        self.flann = cv2.FlannBasedMatcher(self.index_params, self.search_params)

        FLANN_INDEX_KDTREE = 0
        self.index_params_sift = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        self.search_params_sift = dict(checks = 50)

        self.flann_sift = cv2.FlannBasedMatcher(self.index_params_sift, self.search_params_sift)
    
    @Timer(text="fast: {:.4f}")
    def process_frame(self, img):
        #Img resizing
        img = cv2.resize(img, self.dim, interpolation = cv2.INTER_AREA)
        #Feature extracting
        kp = self.fast.detect(img, None)
        kp, des = self.descriptor.compute(img, kp)
        #Feature matching
        good = []
        if len(kp) == 0:
            return
        if self.last_frame is not None:
            #If less than 500 keypoints use bf matcher
            if len(kp) < 500:
                matches = self.matcher.knnMatch(des, self.last_frame['des'], 2)
            else:
                matches = self.flann.knnMatch(des,self.last_frame['des'],k=2)
            
            # store all the good matches as per Lowe's ratio test.
            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    kp1 = kp[m.queryIdx].pt
                    kp2 = self.last_frame['kp'][m.trainIdx].pt
                    good.append((kp1, kp2))
            '''
        #Fundamental matrix filter
        if len(good) > 0:
            good = np.array(good)
            model, inliers = ransac((good[:, 0], good[:, 1]), 
                                    FundamentalMatrixTransform,
                                    min_samples = 8,
                                    residual_threshold = 1,
                                    max_trials = 100)
            good = good[inliers]
            '''
            #Draw matches and points
            for pt1, pt2 in good:
                u1,v1 = map(lambda x: int(round(x)), pt1)
                u2,v2 = map(lambda x: int(round(x)), pt2)
                cv2.circle(img, (u1,v1), color=(0,255,0), radius=3)
                cv2.line(img, (u1,v1), (u2,v2), (255,0,0))
            cv2.putText(img, 'fast + beblid', (50,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
            cv2.putText(img, str(len(good)), (50,75), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
            cv2.putText(img, str(len(kp)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        self.last_frame = {'kp': kp, 'des': des}
        return img, good
        
    @Timer(text="sift: {:.4f}")
    def sift_frame(self, img):
        #Img resizing
        img = cv2.resize(img, self.dim, interpolation = cv2.INTER_AREA)
        #Feature extracting
        kp, des = self.sift.detectAndCompute(img, None)
        #Feature matching
        good = []
        if len(kp) == 0:
            return
        if self.last_frame_sift is not None:
            #If less than 500 keypoints use bf matcher
            if len(kp) < 500:
                matches = self.bfmatch.knnMatch(des, self.last_frame_sift['des'], k=2)
            else:
                matches = self.flann_sift.knnMatch(des,self.last_frame_sift['des'],k=2)
            
            # store all the good matches as per Lowe's ratio test.
            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    good.append((kp[m.queryIdx], self.last_frame_sift['kp'][m.trainIdx]))
            
            #Draw matches and points
            for pt1, pt2 in good:
                u1,v1 = map(lambda x: int(round(x)), pt1.pt)
                u2,v2 = map(lambda x: int(round(x)), pt2.pt)
                cv2.circle(img, (u1,v1), color=(0,255,0), radius=3)
                cv2.line(img, (u1,v1), (u2,v2), (255,0,0))
            cv2.putText(img, 'SIFT', (50,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
            cv2.putText(img, str(len(good)), (50,75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
            cv2.putText(img, str(len(kp)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        self.last_frame_sift = {'kp': kp, 'des': des}
        return img

    @Timer(text="orb: {:.4f}")
    def orb_frame(self, img):
        #Img resizing
        img = cv2.resize(img, self.dim, interpolation = cv2.INTER_AREA)
        #Feature extracting
        kp = self.orb.detect(img, None)
        kp, des = self.orb.compute(img, kp)
        #Feature matching
        if len(kp) == 0:
            return
        good = []
        if self.last_frame_orb is not None:
            #If less than 500 keypoints use bf matcher
            if len(kp) < 500:
                matches = self.matcher.knnMatch(des, self.last_frame_orb['des'], 2)
            else:
                matches = self.flann.knnMatch(des,self.last_frame_orb['des'],k=2)
            
            # store all the good matches as per Lowe's ratio test.
            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    good.append((kp[m.queryIdx], self.last_frame_orb['kp'][m.trainIdx]))
            
            #Draw matches and points
            for pt1, pt2 in good:
                u1,v1 = map(lambda x: int(round(x)), pt1.pt)
                u2,v2 = map(lambda x: int(round(x)), pt2.pt)
                cv2.circle(img, (u1,v1), color=(0,255,0), radius=3)
                cv2.line(img, (u1,v1), (u2,v2), (255,0,0))
            cv2.putText(img, 'ORB', (50,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
            cv2.putText(img, str(len(good)), (50,75), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
            cv2.putText(img, str(len(kp)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        self.last_frame_orb = {'kp': kp, 'des': des}
        return img

    @Timer(text="orb + beblid: {:.4f}")
    def beblid_frame(self, img):
        #Img resizing
        img = cv2.resize(img, self.dim, interpolation = cv2.INTER_AREA)
        #Feature extracting
        kp = self.orb.detect(img, None)
        kp, des = self.descriptor_orb.compute(img, kp)
        #Feature matching
        good = []
        if len(kp) == 0:
            return
        if self.last_frame_beblid is not None:
            #If less than 500 keypoints use bf matcher
            if len(kp) < 500:
                matches = self.matcher.knnMatch(des, self.last_frame_beblid['des'], 2)
            else:
                matches = self.flann.knnMatch(des,self.last_frame_beblid['des'],k=2)
            
            # store all the good matches as per Lowe's ratio test.
            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    good.append((kp[m.queryIdx], self.last_frame_beblid['kp'][m.trainIdx]))
            
            #Draw matches and points
            for pt1, pt2 in good:
                u1,v1 = map(lambda x: int(round(x)), pt1.pt)
                u2,v2 = map(lambda x: int(round(x)), pt2.pt)
                cv2.circle(img, (u1,v1), color=(0,255,0), radius=3)
                cv2.line(img, (u1,v1), (u2,v2), (255,0,0))
            cv2.putText(img, 'ORB + beblid', (50,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
            cv2.putText(img, str(len(good)), (50,75), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
            cv2.putText(img, str(len(kp)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        self.last_frame_beblid = {'kp': kp, 'des': des}
        return img

fe = FeatureExtractor()

if __name__=="__main__":
    cap = cv2.VideoCapture('/home/kubuntu/Downloads/DJI_0199.MOV')
    while cap.isOpened():
        ret,frame = cap.read()
        if ret == True:
            img, matches = fe.process_frame(frame)
            img_sift = fe.sift_frame(frame)
            img_orb = fe.orb_frame(frame)
            img_beblid = fe.beblid_frame(frame)
            resize = cv2.resize(img, Display_dim, interpolation = cv2.INTER_AREA)
            resize_sift = cv2.resize(img_sift, Display_dim, interpolation = cv2.INTER_AREA)
            resize_orb = cv2.resize(img_orb, Display_dim, interpolation = cv2.INTER_AREA)
            resize_beblid = cv2.resize(img_beblid, Display_dim, interpolation = cv2.INTER_AREA)
            resize = np.concatenate((resize, resize_sift), axis=1)
            resize_orb = np.concatenate((resize_orb, resize_beblid), axis=1)
            final_img = np.concatenate((resize, resize_orb), axis=0)
            cv2.imshow('fast output', final_img)
            print("--------")
            #cv2.imshow('sift output', resize_sift)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()