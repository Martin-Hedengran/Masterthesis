#!/usr/bin/env python3
import os
import time
import cv2
from display import Display
from frame import Frame, denormalize, Match_features, IRt
import numpy as np
import g2o
from pointmap import Map, Point

# camera intrinsics for car driving
#W, H = 1920//2, 1080//2
#Cx, Cy = 1920//4, 1080//4
#F = 270

# DJI Phantom 4 pro
# Calibration from Metashape 
#! The values from metashape are offset from optical center
W, H = int(3840/5), int(2160/5)
Cx, Cy = (3840 / 2 - 35.24)//5, (2160 / 2 - 279)//5
F = 2676
K = np.array([[F,0,Cx],
            [0,F,Cy],
            [0,0,1]])

#K = np.array([[3657,   0.,         5472 / 2 - 33 ],
#                            [  0.,         3657, 3648 / 2 - 3.5 ],
#                            [0., 0., 1.]])
Kinv = np.linalg.inv(K)

# main classes
mapp = Map()
disp = Display(W, H) if os.getenv("D2D") is not None else None

def triangulate(pose1, pose2, pts1, pts2):
    #Direct linear triangulation from orbslam
    ret = np.zeros((pts1.shape[0], 4))
    pose1 = np.linalg.inv(pose1)
    pose2 = np.linalg.inv(pose2)
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

    f1 = mapp.frames[-1]
    f2 = mapp.frames[-2]

    idx1, idx2, Rt = Match_features(f1, f2, K)
    f1.pose = np.dot(Rt, f2.pose)
    
    # homogeneous 3-D coords
    pts4d = triangulate(f1.pose, f2.pose, f1.pts[idx1], f2.pts[idx2])
    pts4d /= pts4d[:, 3:]

    # reject pts without enough "parallax"
    #reject points behind the camera
    good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0)

    for i,p in enumerate(pts4d):
        if not good_pts4d[i]:
            continue
        pt = Point(mapp, p)
        pt.add_observation(f1, idx1[i])
        pt.add_observation(f2, idx2[i])


    for pt1, pt2 in zip(f1.pts[idx1], f2.pts[idx2]):
        u1, v1 = denormalize(K, pt1)
        u2, v2 = denormalize(K, pt2)
        cv2.circle(img, (u1, v1), color=(0,255,0), radius=3)
        cv2.line(img, (u1, v1), (u2, v2), color=(255,0,0))

    # 2-D display
    if disp is not None:
        disp.paint(img)
    
    # 3-D display
    mapp.display()

if __name__ == "__main__":
    #Video files available
    #DJI_0199.MOV, DJI_0199_turn.mp4, test_drive.mp4
    cap = cv2.VideoCapture("/home/kubuntu/Downloads/DJI_0199_turn.mp4")
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
        else:
            break
