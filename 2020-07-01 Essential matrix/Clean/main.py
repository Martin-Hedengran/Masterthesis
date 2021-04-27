#!/usr/bin/env python3
import numpy as np
import cv2
from frame import Frame, Match_features, Triangulation
from pangomap import Map, Point


def ImageScaling(img):
    #Scale the image and camera matrix. Values 0-1 represent 0-100% image scale
    scale_factor = 0.20
    width = int(img.shape[1] * scale_factor)
    height = int(img.shape[0] * scale_factor)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    # DJI Phantom 4 pro
    # Calibration from Metashape 
    #! The values from metashape are offset from optical center
    cameraMatrix = np.array([[3657,   0.,         5472 / 2 - 33 ],
                            [  0.,         3657, 3648 / 2 - 3.5 ],
                            [0., 0., 1.]])
    distCoeffs = np.array(
            [[ 0.0029, 0.0155, 0.014, 0.0022, 0.00025 ]])

    #self.scale_factor = 1
    cameraMatrix *= scale_factor
    cameraMatrix[2, 2] = 1
    return img, cameraMatrix, distCoeffs


def process_frame(img):
    img, K, skew = ImageScaling(img)
    frame = Frame(img, mapp)

    #Takes 2 frames to match
    if frame.id == 0:
        return img
    
    f1 = mapp.frames[-1]
    f2 = mapp.frames[-2]

    idx1, idx2, pose, mask, flag = Match_features(f1, f2, K)
    f1.pose = np.dot(pose, f2.pose)

    pts3d, good_pts3d = Triangulation(K, skew, f1.pose, f2.pose, mask, f1.points[idx1], f2.points[idx2])

    for i, p in enumerate(pts3d):
        if not good_pts3d[i]:
            continue
        pt = Point(mapp, p)
        pt.add_observation(f1, idx1[i])
        pt.add_observation(f2, idx2[i])

    for pt1, pt2 in zip(f1.points[idx1], f2.points[idx2]):
        u1,v1 = map(lambda x: int(round(x)), pt1)
        u2,v2 = map(lambda x: int(round(x)), pt2)
        cv2.circle(img, (u1, v1), color=(0,255,0), radius=3)
        cv2.line(img, (u1, v1), (u2, v2), color=(255,0,0))
    cv2.putText(img, str(len(f1.points[idx1])), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
    cv2.putText(img, str(flag), (50,75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
    mapp.display()
    return img


mapp = Map()

if __name__ == "__main__":
    cap = cv2.VideoCapture('/home/kubuntu/Downloads/DJI_0199.MOV')

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            img = process_frame(frame)
            cv2.imshow('fast output', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()