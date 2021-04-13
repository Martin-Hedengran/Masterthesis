#!/usr/bin/env python3
import cv2
import numpy as np

#Img scaled to 40 percent for viewer
Display_height = 2160 * 30 / 100
Display_width= 3840 * 30 / 100
Display_dim = (int(Display_width), int(Display_height))

class FeatureExtractor(object):
    def __init__(self):
        self.last_frame = None
        self.last_frame_sift = None
        #Hardcode image dimensions
        self.scale_percent = 20
        self.height = 2160 * self.scale_percent / 100
        self.width= 3840 * self.scale_percent / 100
        self.dim = (int(self.width), int(self.height))

        #Initialize detector and descriptor
        #!Note: fast doesnt have descriptors, only detection hence the use of BEBLID descriptors.
        self.fast = cv2.FastFeatureDetector_create()
        #! The input to belid is feature scale and depends on the detector used. fast=5, sift = 6,75, orb = 1
        self.descriptor = cv2.xfeatures2d.BEBLID_create(5)

        self.sift = cv2.SIFT_create()

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

    def process_frame(self, img):
        #Img resizing
        img = cv2.resize(img, self.dim, interpolation = cv2.INTER_AREA)
        #Feature extracting
        kp = self.fast.detect(img, None)
        kp, des = self.descriptor.compute(img, kp)
        #Feature matching
        good = []
        if self.last_frame is not None:
            #If less than 500 keypoints use bf matcher
            if len(kp) < 500:
                matches = self.matcher.knnMatch(des, self.last_frame['des'], 2)
            else:
                matches = self.flann.knnMatch(des,self.last_frame['des'],k=2)
            
            # store all the good matches as per Lowe's ratio test.
            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    good.append((kp[m.queryIdx], self.last_frame['kp'][m.trainIdx]))
            
            #Draw matches and points
            for pt1, pt2 in good:
                u1,v1 = map(lambda x: int(round(x)), pt1.pt)
                u2,v2 = map(lambda x: int(round(x)), pt2.pt)
                cv2.circle(img, (u1,v1), color=(0,255,0), radius=3)
                cv2.line(img, (u1,v1), (u2,v2), (255,0,0))
            cv2.putText(img, str(len(good)), (50,75), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
            cv2.putText(img, str(len(kp)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        self.last_frame = {'kp': kp, 'des': des}
        return img
    
    def sift_frame(self, img):
        #Img resizing
        img = cv2.resize(img, self.dim, interpolation = cv2.INTER_AREA)
        #Feature extracting
        kp, des = self.sift.detectAndCompute(img, None)
        #Feature matching
        good = []
        if self.last_frame_sift is not None:
            #If less than 500 keypoints use bf matcher
            if len(kp) < 500:
                matches = self.bfmatch.knnMatch(des, self.last_frame_sift['des'], k=2)
            else:
                matches = self.flann.knnMatch(des,self.last_frame_sift['des'],k=2)
            
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
            cv2.putText(img, str(len(good)), (50,75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
            cv2.putText(img, str(len(kp)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        self.last_frame_sift = {'kp': kp, 'des': des}
        return img
fe = FeatureExtractor()

if __name__=="__main__":
    cap = cv2.VideoCapture('/home/kubuntu/Downloads/DJI_0199.MOV')
    while cap.isOpened():
        ret,frame = cap.read()
        if ret == True:
            img = fe.process_frame(frame)
            img_sift = fe.sift_frame(frame)
            resize = cv2.resize(img, Display_dim, interpolation = cv2.INTER_AREA)
            resize_sift = cv2.resize(img_sift, Display_dim, interpolation = cv2.INTER_AREA)
            resize = np.concatenate((resize, resize_sift), axis=1)
            cv2.imshow('fast output', resize)
            #cv2.imshow('sift output', resize_sift)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()