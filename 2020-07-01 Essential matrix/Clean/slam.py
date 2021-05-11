#!/usr/bin/env python3
import os
import time
import cv2
from display import Display
from frame import Frame, denormalize, Match_features
import numpy as np
from pointmap import Map, Point
import math

# camera intrinsics for car driving scaled to half
#W, H = 1920//2, 1080//2
#Cx, Cy = 1920//4, 1080//4
#F = 270

# camera intrinsics for kitti test
#W, H = 1242, 375
#Cx, Cy = 1242//2, 375//2
#F = 984

# DJI Phantom 4 pro
# Calibration from Metashape 
#! The values from metashape are offset from optical center
#Scaled to 20%
W, H = int(3840/5), int(2160/5)
Cx, Cy = (3840 / 2 - 35.24)//5, (2160 / 2 - 279)//5
F = 2676/5
K = np.array([[F,0,Cx],
            [0,F,Cy],
            [0,0,1]])

Kinv = np.linalg.inv(K)

# main classes
mapp = Map()
disp = Display(W, H) if os.getenv("display") is not None else None

def triangulate(pose1, pose2, pts1, pts2):
    #Direct linear triangulation from orbslam
    ret = np.zeros((pts1.shape[0], 4))
    for i, p in enumerate(zip(pts1, pts2)):
        A = np.zeros((4,4))
        A[0] = p[0][0] * pose1[2] - pose1[0]
        A[1] = p[0][1] * pose1[2] - pose1[1]
        A[2] = p[1][0] * pose2[2] - pose2[0]
        A[3] = p[1][1] * pose2[2] - pose2[1]
        _, _, vt = np.linalg.svd(A)
        ret[i] = vt[3]
    return ret

def process_frame(img):
    img = cv2.resize(img, (W,H))
    frame = Frame(mapp, img, K)
    if frame.id == 0:
        return
    print("\n*** frame %d ***" % (frame.id,))

    f1 = mapp.frames[-1]
    f2 = mapp.frames[-2]

    idx1, idx2, Rt = Match_features(f1, f2, K)
    f1.pose = np.dot(Rt, f2.pose)


    #Filter out reoccuring points
    for i,idx in enumerate(idx2):
        if f2.pts[idx] is not None:
            f2.pts[idx].add_observation(f1, idx1[i])
    
    good_pts4d = np.array([f1.pts[i] is None for i in idx1])

    #Local triangulation to reject points behind camera
    pts_tri_local = triangulate(Rt, np.eye(4), f1.kps[idx1], f2.kps[idx2])
    pts_tri_local /= pts_tri_local[:, 3:]
    good_pts4d &= pts_tri_local[:, 2] > 0

    pts4d = triangulate(f1.pose, f2.pose, f1.kps[idx1], f2.kps[idx2])

    #Reject points with too low parallax
    good_pts4d &= np.abs(pts4d[:, 3]) > 0.005

    # homogeneous 3-D coords
    pts4d /= pts4d[:, 3:]

    print("Adding:   %d points" % np.sum(good_pts4d))

    for i,p in enumerate(pts4d):
        if not good_pts4d[i]:
            continue
        u,v = int(round(f1.ukps[idx1[i],0])), int(round(f1.ukps[idx1[i],1]))
        pt = Point(mapp, p)
        pt.add_observation(f1, idx1[i])
        pt.add_observation(f2, idx2[i])

    for pt1, pt2 in zip(f1.kps[idx1], f2.kps[idx2]):
        u1, v1 = denormalize(K, pt1)
        u2, v2 = denormalize(K, pt2)
        cv2.circle(img, (u1, v1), color=(0,255,0), radius=3)
        cv2.line(img, (u1, v1), (u2, v2), color=(255,0,0))

    # 2-D display
    if disp is not None:
        disp.paint(img)

    # optimize the map
    if frame.id >= 4:
        err = mapp.optimize()
        print("Optimize: %f units of error" % err)

    # 3-D display
    mapp.display()
c=1
if __name__ == "__main__":
    skip = int(os.getenv("skip", "1"))
    #Video files available
    #DJI_0199.MOV, DJI_0199_turn.mp4, test_drive.mp4, test_kitti984.mp4
    cap = cv2.VideoCapture("/home/kubuntu/Downloads/DJI_0199_turn.mp4")
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            if c%skip==0:
                process_frame(frame)
            c+=1
        else:
            break
