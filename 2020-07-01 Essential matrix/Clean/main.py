#!/usr/bin/env python3
import numpy as np
import cv2
from frame import Frame, Match_features, Triangulation
from multiprocessing import Process, Queue
import OpenGL.GL as gl
import pangolin


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

class mapp_class(object):
    #A class that holds information about points and poses for building a mapp
    def __init__(self):
        self.frames = []
        self.points = []
        self.viewer_init()

    def viewer_init(self):
        import OpenGL.GL as gl
        import pangolin

        pangolin.CreateWindowAndBind('Main', 640, 480)
        gl.glEnable(gl.GL_DEPTH_TEST)

        self.scam = pangolin.OpenGlRenderState(
        pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
        pangolin.ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin.AxisDirection.AxisY))
        self.handler = pangolin.Handler3D(self.scam)

        # Create Interactive View in window
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
        self.dcam.SetHandler(self.handler)


    def viewer_refresh(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        self.dcam.Activate(self.scam)

        gl.glPointSize(10)
        gl.glColor3f(0.0, 1.0, 0.0)
        pangolin.DrawPoints(d[:3, 3] for d in self.state[0])

        gl.glPointSize(2)
        gl.glColor3f(0.0, 1.0, 0.0)
        pangolin.DrawPoints(d for d in self.state[1])

        pangolin.FinishFrame()

    def display(self):
        poses, pts = [], []
        for f in self.frames:
            poses.append(f.pose)
        for p in self.points:
            pts.append(p.pt)
            self.state = poses, pts
            self.viewer_refresh()

class Point(object):
    #An class that holds information about 3d points
    def __init__(self, mapp, loc):
        self.pt = loc
        self.frames = []
        self.idxs = []

        self.id = len(mapp.points)
        mapp.points.append(self)
    
    def add_observation(self, frame, idx):
        #adds the observation of a point in a given frame
        self.frames.append(frame)
        self.idxs.append(idx)

def process_frame(img):
    img, K, skew = ImageScaling(img)
    frame = Frame(img, mapp)

    #Takes 2 frames to match
    if frame.id == 0:
        return img
    
    f1 = mapp.frames[-1]
    f2 = mapp.frames[-2]

    idx1, idx2, pose, mask = Match_features(f1, f2, K)
    f1.pose = np.dot(pose, f2.pose)

    pts3d = Triangulation(K, skew, f1.pose, f2.pose, mask, f1.points[idx1], f2.points[idx2])
    
    for i, p in enumerate(pts3d):
        pt = Point(mapp, p)
        pt.add_observation(f1, idx1[i])
        pt.add_observation(f2, idx2[i])
        
    for pt1, pt2 in zip(f1.points[idx1], f2.points[idx2]):
        u1,v1 = map(lambda x: int(round(x)), pt1)
        u2,v2 = map(lambda x: int(round(x)), pt2)
        cv2.circle(img, (u1, v1), color=(0,255,0), radius=3)
        cv2.line(img, (u1, v1), (u2, v2), color=(255,0,0))
    cv2.putText(img, str(len(f1.points[idx1])), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
    
    return img



mapp = mapp_class()
mapp.display()

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