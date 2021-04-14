#!/usr/bin/env python
import cv2
import numpy as np
import math
import argparse
import matplotlib.pyplot as plt
import time
import plotly.graph_objects as go
from codetiming import Timer
import g2o 


# Code borrowed from the following sources
# https://docs.opencv.org/master/d1/d89/tutorial_py_orb.html
# https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
# https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
# https://www.learnopencv.com/rotation-matrix-to-euler-angles/
# https://www.morethantechnical.com/2016/10/17/structure-from-motion-toy-lib-upgrades-to-opencv-3/


def drawlines(img1,img2,lines,points1,points2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,points1,points2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2


def isRotationMatrix(R) :
    # Checks if a matrix is a valid rotation matrix.
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotationMatrixToEulerAngles(R) :
    """Calculates rotation matrix to euler angles

    The result is the same as MATLAB except the order
    of the euler angles ( x and z are swapped ).
    """

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


class ImageAndKeypoints():
    def __init__(self):
        if False:
            # Use ORB features
            self.detector = cv2.ORB_create()
            self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            # Use SIFT features
            self.detector = cv2.SIFT_create()
            self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        # Camera parameters are taken from a manual calibration 
        # of the used camera.
        # See the file input/camera_calibration_extended.txt
        self.cameraMatrix = np.array([[704.48172143,   0.,         637.4243092 ],
                                [  0.,         704.01349597, 375.7176407 ],
                                [0., 0., 1.]])
        self.distCoeffs = np.array(
                [[ 8.93382520e-02, -1.57262105e-01, -9.82974653e-05,
                    5.65668273e-04, 7.19784192e-02]])

        # DJI Phantom 4 pro
        # Calibration from Metashape 
        #! The values from metashape are offset from optical center
        self.cameraMatrix = np.array([[3657,   0.,         5472 / 2 - 33 ],
                                [  0.,         3657, 3648 / 2 - 3.5 ],
                                [0., 0., 1.]])
        #TODO check distortion coefficient method
        self.distCoeffs = np.array(
                [[ 0.0029, 0.0155, 0.014, 0.0022, 0.00025 ]])
        
        self.scale_factor = 0.25
        #self.scale_factor = 1
        self.cameraMatrix *= self.scale_factor
        self.cameraMatrix[2, 2] = 1


    def set_image(self, image):
        width = int(image.shape[1] * self.scale_factor)
        height = int(image.shape[0] * self.scale_factor)
        dim = (width, height)
          
        self.image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)


    def detect_keypoints(self):
        # find keypoints and descriptors with the selected feature detector
        self.keypoints, self.descriptors = self.detector.detectAndCompute(self.image, None)


class ImagePair():
    def __init__(self):
        if False:
            # Use ORB features
            self.detector = cv2.ORB_create()
            self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            # Use SIFT features
            self.detector = cv2.SIFT_create()
            self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        # Camera parameters are taken from a manual calibration 
        # of the used camera.
        # See the file input/camera_calibration_extended.txt
        self.cameraMatrix = np.array([[704.48172143,   0.,         637.4243092 ],
                                [  0.,         704.01349597, 375.7176407 ],
                                [0., 0., 1.]])
        self.distCoeffs = np.array(
                [[ 8.93382520e-02, -1.57262105e-01, -9.82974653e-05,
                    5.65668273e-04, 7.19784192e-02]])

        # DJI Phantom 4 pro
        # Calibration from Metashape 
        self.cameraMatrix = np.array([[3657,   0.,         5472 / 2 - 33 ],
                                [  0.,         3657, 3648 / 2 - 3.5 ],
                                [0., 0., 1.]])
        self.distCoeffs = np.array(
                [[ 0.0029, 0.0155, 0.014, 0.0022, 0.00025 ]])

        self.scale_factor = 0.25
        #self.scale_factor = 1
        self.cameraMatrix *= self.scale_factor
        self.cameraMatrix[2, 2] = 1


    def set_images(self, img1, img2):
        width = int(img1.shape[1] * self.scale_factor)
        height = int(img1.shape[0] * self.scale_factor)
        dim = (width, height)
          
        self.img1 = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)
        self.img2 = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)


    def set_images_new(self, img1, img2):
        self.image1 = img1
        self.image2 = img2


    @Timer(text="detect_keypoints {:.4f}")
    def detect_keypoints(self):
        # find the keypoints and descriptors with ORB
        self.kp1, self.des_img1 = self.detector.detectAndCompute(self.img1, None)
        self.kp2, self.des_img2 = self.detector.detectAndCompute(self.img2, None)


    @Timer(text="match_detected_keypoints {:.4f}")
    def match_detected_keypoints(self):
        # Match descriptors.
        self.matches = self.bf.match(self.image1.descriptors, self.image2.descriptors)

        points1_temp = []
        points2_temp = []
        match_indices_temp = []
        # ratio test as per Lowe's paper
        for idx, m in enumerate(self.matches):
            points2_temp.append(self.image2.keypoints[m.trainIdx].pt)
            points1_temp.append(self.image1.keypoints[m.queryIdx].pt)
            match_indices_temp.append(idx)

        self.points1_1to2 = np.float32(points1_temp)
        self.points2_1to2 = np.float32(points2_temp)
        self.match_indices_1to2 = np.int32(match_indices_temp)
        #print("self.matches.shape")
        #print(len(self.matches))
        #print(self.points1_1to2.shape)


    @Timer(text="determine_fundamental_matrix{:.4f}")
    def determine_fundamental_matrix(self):
        ransacReprojecThreshold = 1
        confidence = 0.99
        self.fundamentalMatrix_1to2, mask = cv2.findFundamentalMat(self.points1_1to2, 
                self.points2_1to2, 
                cv2.FM_RANSAC, 
                ransacReprojecThreshold, 
                confidence)

        # We select only inlier points
        self.points1_1to2 = self.points1_1to2[mask.ravel()==1]
        self.points2_1to2 = self.points2_1to2[mask.ravel()==1]
        self.match_indices_1to2 = self.match_indices_1to2[mask.ravel()==1]
        #print(self.points1_1to2.shape)

    @Timer(text="determine_essential_matrix {:.4f}")
    def determine_essential_matrix(self):
        confidence = 0.99
        ransacReprojecThreshold = 1
        self.essential_matrix_1to2, mask = cv2.findEssentialMat(
                self.points1_1to2, 
                self.points2_1to2, 
                self.cameraMatrix, 
                cv2.FM_RANSAC, 
                confidence,
                ransacReprojecThreshold)

        # We select only inlier points
        self.points1_1to2 = self.points1_1to2[mask.ravel()==1]
        self.points2_1to2 = self.points2_1to2[mask.ravel()==1]
        self.match_indices_1to2 = self.match_indices_1to2[mask.ravel()==1]
        #print(self.points1_1to2.shape)

    @Timer(text="estimate_camera_movement {:.4f}")
    def estimate_camera_movement(self):
        retval, self.R, self.t, mask = cv2.recoverPose(self.essential_matrix_1to2, self.points1_1to2, self.points2_1to2, self.cameraMatrix)


    @Timer(text="reconstruct_3d_points {:.4f}")
    def reconstruct_3d_points(self):
        self.null_projection_matrix = self.cameraMatrix @ np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        self.projection_matrix = self.cameraMatrix @ np.hstack((self.R.T, self.t))

        self.points3d_reconstr = cv2.triangulatePoints(
                self.projection_matrix, self.null_projection_matrix,
                self.points1_1to2.T, self.points2_1to2.T) 
        # Convert back to unit value in the homogeneous part.
        self.points3d_reconstr /= self.points3d_reconstr[3, :]

    def get_validated_matches(self):
        # Construct list with all points from matches.
        self.points1_temp = []
        self.points2_temp = []

        self.filtered_matches_1to2 = []
        for index in self.match_indices_1to2:
            m = self.matches[index]
            self.points2_temp.append(self.image2.keypoints[m.trainIdx].pt)
            self.points1_temp.append(self.image1.keypoints[m.queryIdx].pt)
            self.filtered_matches_1to2.append(m)

        self.points1_filtered = np.float32(self.points1_temp)
        self.points2_filtered = np.float32(self.points2_temp)
        #print("self.points1_filtered.shape")
        #print(self.points1_filtered.shape)


    def show_estimated_camera_motion(self):
        print("Estimated direction of translation between images")
        print(self.t)
        print("Estimated rotation of camera between images in degrees")
        print(rotationMatrixToEulerAngles(self.R) * 180 / math.pi)


    def visualize_filtered_matches(self):
        visualization = cv2.drawMatches(self.image2.image,
            self.image1.keypoints,
            self.image2.image,
            self.image2.keypoints,
            self.filtered_matches_1to2,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return visualization


    def visualise_points_in_3d_with_plotly(self):
       # The reconstructed points appear to be mirrored.
        # This is changed here prior to visualization
        # TODO: Figure out why this is the case?
        pitch = 19.7 / 180 * np.pi
        transform = np.array(
                [[1, 0, 0, 0],
                 [0, np.cos(pitch), np.sin(pitch), 0], 
                 [0, -np.sin(pitch), np.cos(pitch), 0], 
                 [0, 0, 0, 1]])

        print("self.points3d_reconstr.shape")
        print(self.points3d_reconstr.shape)
        point3dtemp = transform @ self.points3d_reconstr

        xs = point3dtemp[0]
        ys = point3dtemp[1]
        zs = point3dtemp[2]

        print(self.t)
        plotlyfig = go.Figure(data=[go.Scatter3d(
            y=xs, z=-ys, x=zs, mode='markers', 
            marker=dict(
                    size=2,
                    color=ys,                # set color to an array/list of desired values
                    colorscale='Viridis',   # choose a colorscale
                    opacity=0.8
                )
            )],
            layout={
                "title": "3D points from triangulation"}
            )

        # Show estimated camera positions
        camera2x = self.t[0, 0]
        camera2y = self.t[1, 0]
        camera2z = self.t[2, 0]
        plotlyfig.add_trace(
            go.Scatter3d(
                y=[0, camera2x], 
                z=[0, -camera2y], 
                x=[0, camera2z], mode='markers', 
                marker=dict(
                        size=10,
                        color=['red', 'green']
                    )
                )
            )

        xptp = np.hstack((xs, 0, camera2x)).ptp()
        yptp = np.hstack((-ys, 0, -camera2y)).ptp()
        zptp = np.hstack((zs, 0, camera2z)).ptp()
        plotlyfig.update_layout(scene=dict(
            aspectmode='data',
            aspectratio=go.layout.scene.Aspectratio(
               x=xptp, y=xptp/yptp, z=zptp/xptp)
            ))

        plotlyfig.show()

    @Timer(text="build_bundle_adjustment_graph {:.4f}")
    def build_bundle_adjustment_graph(self):
        optimizer = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        optimizer.set_algorithm(solver)

        # Define camera parameters
        print(self.cameraMatrix)
        #focal_length = 1000
        focal_length = self.cameraMatrix[0, 0]
        #principal_point = (320, 240)
        principal_point = (self.cameraMatrix[0, 2], self.cameraMatrix[1, 2])
        baseline = 0
        cam = g2o.CameraParameters(focal_length, principal_point, baseline)
        cam.set_id(0)
        optimizer.add_parameter(cam)

        ## Add the two camera views to the graph
        # pose here means transform points from world coordinates to camera coordinates
        pose_camera_1 = g2o.SE3Quat(np.identity(3), [0, 0, 0])

        # Use the estimated pose of the second camera based on the 
        # essential matrix.
        #pose_camera_2 = g2o.SE3Quat(np.identity(3), [-0.57, -0.65, -0.5])
        print("self.R")
        print(self.R)
        print("self.t.T[0]")
        print(self.t.T[0])
        pose_camera_2 = g2o.SE3Quat(self.R, self.t.T[0])
        print(pose_camera_2.rotation().x())
        print(pose_camera_2.rotation().y())
        print(pose_camera_2.rotation().z())
        print(pose_camera_2.rotation().w())


        # Set the poses that should be optimized.
        # Define their initial value to be the true pose
        # keep in mind that there is added noise to the observations afterwards.
        v_se3 = g2o.VertexSE3Expmap()
        v_se3.set_id(0)
        v_se3.set_estimate(pose_camera_1)
        v_se3.set_fixed(True)
        optimizer.add_vertex(v_se3)

        v_se3 = g2o.VertexSE3Expmap()
        v_se3.set_id(1)
        v_se3.set_estimate(pose_camera_2)
        #v_se3.set_fixed(True)
        optimizer.add_vertex(v_se3)

        cx = self.cameraMatrix[0, 2]
        cy = self.cameraMatrix[1, 2]
        f = self.cameraMatrix[0, 0]

        point_id = 2 # Because we already have added two cameras

        edges = []
        for cnt, index in enumerate(self.match_indices_1to2):
            match = self.matches[index]
            point_one = self.image2.keypoints[match.trainIdx];
            point_two = self.image1.keypoints[match.queryIdx];
            print("point one", point_one.pt)

            # Add 3d location of point to the graph
            vp = g2o.VertexSBAPointXYZ()
            vp.set_id(point_id)
            vp.set_marginalized(True)
            # Place points on an image plane in front of the camera
            #z = 1
            #x = (point_one.pt[0] - cx) * z / f
            #y = (point_one.pt[1] - cy) * z / f
            # Use positions of 3D points from the triangulation
            x = self.points3d_reconstr[0, cnt]
            y = self.points3d_reconstr[1, cnt]
            z = self.points3d_reconstr[2, cnt]
            point = np.array([x, y, z], dtype=np.float64)
            vp.set_estimate(point)
            optimizer.add_vertex(vp)

            # Add edge from first camera to the point
            edge = g2o.EdgeProjectXYZ2UV()
            edge.set_vertex(0, vp) # 3D point - newly created
            edge.set_vertex(1, optimizer.vertex(0)) # Pose of first camera
            edge.set_measurement(point_one.pt)
            edge.set_information(np.identity(2))
            edge.set_robust_kernel(g2o.RobustKernelHuber())

            edge.set_parameter_id(0, 0)
            optimizer.add_edge(edge)
            edges.append(edge)


            # Add edge from second camera to the point
            edge = g2o.EdgeProjectXYZ2UV()
            edge.set_vertex(0, vp) # 3D point - newly created
            edge.set_vertex(1, optimizer.vertex(1)) # Pose of second camera
            edge.set_measurement(point_two.pt)
            edge.set_information(np.identity(2))
            edge.set_robust_kernel(g2o.RobustKernelHuber())

            edge.set_parameter_id(0, 0)
            optimizer.add_edge(edge)
            edges.append(edge)

            point_id += 1

        print('num vertices:', len(optimizer.vertices()))
        print('num edges:', len(optimizer.edges()))

        print('Performing full BA:')
        optimizer.initialize_optimization()
        optimizer.set_verbose(True)
        optimizer.optimize(150)

        optimizer.save("ba.g2o");

        point_coordinates = []
        for vertex_idx in optimizer.vertices():
            vertex = optimizer.vertex(vertex_idx)
            vertex_coord = vertex.estimate()
            if type(vertex_coord) is np.ndarray:
                point_coordinates.append(vertex_coord)

        temp = np.vstack(point_coordinates)
        print("Points from bundle adjustment")
        print(temp)
        print("Points from triangulation")
        print(self.points3d_reconstr.T[:, 0:3])

        print("Camera positions")
        print(optimizer.vertex(1).estimate().translation())
        q = optimizer.vertex(1).estimate().rotation()
        print(q.x())
        print(q.y())
        print(q.z())
        print(q.w())


        # Adjust for camera pitch
        pitch = 19.7 / 180 * np.pi
        transform_adjust_pitch = np.array(
                [[1, 0, 0],
                 [0, np.cos(pitch), np.sin(pitch)], 
                 [0, -np.sin(pitch), np.cos(pitch)]])

        temp = transform_adjust_pitch @ temp.T

        xs = temp[0]
        ys = temp[1]
        zs = temp[2]

        plotlyfig = go.Figure(data=[go.Scatter3d(
            x=xs, y=ys, z=zs, mode='markers', 
            marker=dict(
                    size=2,
#                    color=ys,                # set color to an array/list of desired values
#                    colorscale='Viridis',   # choose a colorscale
                    opacity=0.8
                )
            )],
            layout={
                "title": "3D points from bundle adjustment"}
            )


        # Show estimated camera positions
        cam2position = optimizer.vertex(1).estimate().translation()
        cam2position = transform_adjust_pitch @ cam2position;
        camera2x = cam2position[0]
        camera2y = cam2position[1]
        camera2z = cam2position[2]
        plotlyfig.add_trace(
            go.Scatter3d(
                x=[0, camera2x], 
                y=[0, camera2y], 
                z=[0, camera2z], mode='markers', 
                marker=dict(
                        size=10,
                        color=['red', 'green']
                    )
                )
            )

        xptp = np.hstack((xs, 0, camera2x)).ptp()
        yptp = np.hstack((ys, 0, camera2y)).ptp()
        zptp = np.hstack((zs, 0, camera2z)).ptp()
        print("xptp, yptp, zptp")
        print(xptp, yptp, zptp)
        ptp = max(xptp, yptp, zptp)
        plotlyfig.update_layout(scene=dict(
            aspectmode='data',
            aspectratio=go.layout.scene.Aspectratio(
               x=xptp, y=xptp / yptp, z=xptp / zptp)
            ))

        plotlyfig.show()


        return 
      



    @Timer(text="standard_pipeline {:.4f}")
    def standard_pipeline(self):
        #self.detect_keypoints()
        self.match_detected_keypoints()
        self.determine_fundamental_matrix()
        self.determine_essential_matrix()
        self.estimate_camera_movement()
        self.get_validated_matches()
        self.reconstruct_3d_points()

class TriangulatePointsFromTwoImages():
    def __init__(self):
        # Initiate ORB detector
        self.orb = cv2.ORB_create()
        self.sift = cv2.SIFT_create()

        # create BFMatcher object
        #self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)


        # Camera parameters are taken from a manual calibration 
        # of the used camera.
        # See the file input/camera_calibration_extended.txt
        self.cameraMatrix = np.array([[704.48172143,   0.,         637.4243092 ],
                                [  0.,         704.01349597, 375.7176407 ],
                                [0., 0., 1.]])
        self.distCoeffs = np.array(
                [[ 8.93382520e-02, -1.57262105e-01, -9.82974653e-05,
                    5.65668273e-04, 7.19784192e-02]])


    @Timer(text="load_images {:.4f}")
    def load_images(self, filename_one, filename_two, filename_three):
        # Load images
        self.img1 = cv2.imread(filename_one, cv2.IMREAD_GRAYSCALE)
        self.img2 = cv2.imread(filename_two, cv2.IMREAD_GRAYSCALE)
        self.img3 = cv2.imread(filename_three, cv2.IMREAD_GRAYSCALE)


    @Timer(text="detect_keypoints {:.4f}")
    def detect_keypoints(self):
        # find the keypoints and descriptors with ORB
        # TODO: Switch to sift features
        self.kp1, self.des_img1 = self.orb.detectAndCompute(self.img1, None)
        self.kp2, self.des_img2 = self.orb.detectAndCompute(self.img2, None)
        self.kp3, self.des_img3 = self.orb.detectAndCompute(self.img3, None)


    @Timer(text="match_detected_keypoints_between_first_and_second_image {:.4f}")
    def match_detected_keypoints_between_first_and_second_image(self):
        # Match descriptors.
        self.matches_1to2 = self.bf.match(self.des_img1, self.des_img2)

        points1_temp = []
        points2_temp = []
        match_indices_temp = []
        # ratio test as per Lowe's paper
        for idx, m in enumerate(self.matches_1to2):
            points2_temp.append(self.kp2[m.trainIdx].pt)
            points1_temp.append(self.kp1[m.queryIdx].pt)
            match_indices_temp.append(idx)

        self.points1_1to2 = np.float32(points1_temp)
        self.points2_1to2 = np.float32(points2_temp)
        self.match_indices_1to2 = np.int32(match_indices_temp)
        print("self.matches_1to2.shape")
        print(len(self.matches_1to2))
        print(self.points1_1to2.shape)


    @Timer(text="determine_fundamental_matrix_1to2 {:.4f}")
    def determine_fundamental_matrix_1to2(self):
        ransacReprojecThreshold = 2
        confidence = 0.99
        self.fundamentalMatrix_1to2, mask = cv2.findFundamentalMat(self.points1_1to2, 
                self.points2_1to2, 
                cv2.FM_RANSAC, 
                ransacReprojecThreshold, 
                confidence)

        # We select only inlier points
        self.points1_1to2 = self.points1_1to2[mask.ravel()==1]
        self.points2_1to2 = self.points2_1to2[mask.ravel()==1]
        self.match_indices_1to2 = self.match_indices_1to2[mask.ravel()==1]
        print(self.points1_1to2.shape)

    @Timer(text="determine_essential_matrix {:.4f}")
    def determine_essential_matrix(self):
        confidence = 0.99
        ransacReprojecThreshold = 2
        self.essential_matrix_1to2, mask = cv2.findEssentialMat(
                self.points1_1to2, 
                self.points2_1to2, 
                self.cameraMatrix, 
                cv2.FM_RANSAC, 
                confidence,
                ransacReprojecThreshold)

        # We select only inlier points
        self.points1_1to2 = self.points1_1to2[mask.ravel()==1]
        self.points2_1to2 = self.points2_1to2[mask.ravel()==1]
        self.match_indices_1to2 = self.match_indices_1to2[mask.ravel()==1]
        print(self.points1_1to2.shape)

    @Timer(text="estimate_camera_movement {:.4f}")
    def estimate_camera_movement(self):
        retval, self.R, self.t, mask = cv2.recoverPose(self.essential_matrix_1to2, self.points1_1to2, self.points2_1to2, self.cameraMatrix)


    def show_estimated_camera_motion(self):
        print("Estimated direction of translation between images")
        print(self.t)
        print("Estimated rotation of camera between images in degrees")
        print(rotationMatrixToEulerAngles(self.R) * 180 / math.pi)


    @Timer(text="reconstruct_3d_points {:.4f}")
    def reconstruct_3d_points(self):
        self.null_projection_matrix = self.cameraMatrix @ np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        self.projection_matrix = self.cameraMatrix @ np.hstack((self.R.T, self.t))

        self.points3d_reconstr = cv2.triangulatePoints(
                self.projection_matrix, self.null_projection_matrix,
                self.points1_1to2.T, self.points2_1to2.T) 
        # Convert back to unit value in the homogeneous part.
        self.points3d_reconstr /= self.points3d_reconstr[3, :]

    def get_validated_matches(self):
        # Construct list with all points from matches.
        self.points1_temp = []
        self.points2_temp = []

        self.filtered_matches_1to2 = []
        for index in self.match_indices_1to2:
            m = self.matches_1to2[index]
            self.points2_temp.append(self.kp2[m.trainIdx].pt)
            self.points1_temp.append(self.kp1[m.queryIdx].pt)
            self.filtered_matches_1to2.append(m)

        self.points1_filtered = np.float32(self.points1_temp)
        self.points2_filtered = np.float32(self.points2_temp)
        print("self.points1_filtered.shape")
        print(self.points1_filtered.shape)



    def visualise_points_in_3d(self):
        # The reconstructed points appear to be mirrored.
        # This is changed here prior to visualization
        xs = -self.points3d_reconstr[0]
        ys = self.points3d_reconstr[1]
        zs = self.points3d_reconstr[2]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xs, ys, zs)
        ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))  # aspect ratio is 1:1:1 in data space


    def visualise_points_in_3d_with_plotly(self):
        # The reconstructed points appear to be mirrored.
        # This is changed here prior to visualization
        xs = -self.points3d_reconstr[0]
        ys = self.points3d_reconstr[1]
        zs = self.points3d_reconstr[2]

        print(self.t)
        plotlyfig = go.Figure(data=[go.Scatter3d(
            y=xs, z=ys, x=zs, mode='markers', 
            marker=dict(
                    size=2,
                    color=zs,                # set color to an array/list of desired values
                    colorscale='Viridis',   # choose a colorscale
                    opacity=0.8
                )
            )],
            )

        # Show estimated camera positions
        camera2x = self.t[0, 0]
        camera2y = self.t[1, 0]
        camera2z = self.t[2, 0]
        camera3x = self.estimated_position_of_camera_3[0, 0]
        camera3y = self.estimated_position_of_camera_3[1, 0]
        camera3z = self.estimated_position_of_camera_3[2, 0]
        plotlyfig.add_trace(
            go.Scatter3d(
                y=[0, camera2x, camera3x], 
                z=[0, camera2y, camera3y], 
                x=[0, camera2z, camera3z], mode='markers', 
                marker=dict(
                        size=10,
                        color=['red', 'green']
                    )
                )
            )

        plotlyfig.update_layout(scene=dict(
            aspectmode='manual',
            aspectratio=go.layout.scene.Aspectratio(
               x=xs.ptp(), y=xs.ptp()/ys.ptp(), z=zs.ptp()/xs.ptp())
            ))

        plotlyfig.show()



    def draw_epilines_on_images(self):
        # Find epilines corresponding to points in right image (second image) and
        # drawing its lines on left image
        lines1 = cv2.computeCorrespondEpilines(self.points2_1to2.reshape(-1,1,2), 2,self.fundamentalMatrix_1to2)
        lines1 = lines1.reshape(-1,3)
        self.img5,self.img6 = drawlines(self.img1,self.img2,lines1,self.points1_1to2,self.points2_1to2)
        # Find epilines corresponding to points in left image (first image) and
        # drawing its lines on right image
        lines2 = cv2.computeCorrespondEpilines(self.points1_1to2.reshape(-1,1,2), 1,self.fundamentalMatrix_1to2)
        lines2 = lines2.reshape(-1,3)
        self.img3,self.img4 = drawlines(self.img2,self.img1,lines2,self.points2_1to2,self.points1_1to2)


    def draw_matched_points_on_first_image(self):
        self.img7 = cv2.cvtColor(self.img1, cv2.COLOR_GRAY2BGR)
        for point in self.points1_1to2:
            x, y = point
            cv2.circle(self.img7, (x, y), 2, (0, 0, 255), -1)

    def locate_matched_points_in_third_image(self):
        # Make list of matched descriptors between image one and 
        # image two
        self.matched_key_points_in_image_2 = []
        self.matched_descriptors_in_image_2 = []
        for idx in self.match_indices_1to2:
            m = self.matches_1to2[idx]
            self.matched_key_points_in_image_2.append(self.kp2[m.queryIdx])
            self.matched_descriptors_in_image_2.append(self.des_img2[m.queryIdx])

        self.matched_descriptors_in_image_2 = np.uint8(self.matched_descriptors_in_image_2)

        print("len(self.matches_1to2)")
        print(len(self.matches_1to2))

        print("type(self.des_img3)")
        print(self.des_img3.shape)
        print("type(self.matched_descriptors_in_image_2)")
        print(self.matched_descriptors_in_image_2.shape)

        # Match descriptors.
        self.matches_2_to_3 = self.bf.match(
                self.matched_descriptors_in_image_2, 
                self.des_img3)
        print("len(self.matches_2_to_3)")
        print(len(self.matches_2_to_3))
        
        points1_temp = []
        points2_temp = []
        match_indices_temp = []
        # ratio test as per Lowe's paper
        for idx, m in enumerate(self.matches_2_to_3):
            points2_temp.append(self.matched_key_points_in_image_2[m.queryIdx].pt)
            points1_temp.append(self.kp3[m.trainIdx].pt)
            match_indices_temp.append(idx)

        self.points1_2to3 = np.float32(points1_temp)
        self.points2_2to3 = np.float32(points2_temp)
        self.match_indices_2to3 = np.int32(match_indices_temp)
        print("self.matches_1to2.shape")
        print(len(self.matches_2_to_3))
        print(self.points1_2to3.shape)


        ransacReprojecThreshold = 2
        confidence = 0.99
        self.fundamentalMatrix_2to3, mask = cv2.findFundamentalMat(
                self.points1_2to3, 
                self.points2_2to3, 
                cv2.FM_RANSAC, 
                ransacReprojecThreshold, 
                confidence)
        print("self.fundamentalMatrix_2to3")
        print(self.fundamentalMatrix_2to3)

        # We select only inlier points
        self.points1_2to3 = self.points1_2to3[mask.ravel()==1]
        self.points2_2to3_filtered = self.points2_2to3[mask.ravel()==1]
        print("self.points1_2to3.shape")
        print(self.points1_2to3.shape)

        # Get the matching 3D points (from img1 and img2) and determine the
        # camera position in img3.
        point_coords_3d = self.points3d_reconstr[:, mask.ravel() == 1]
        print("point_coords_3d.T.shape")
        print(point_coords_3d.T[:, 0:3].shape)

        print("self.points2_2to3_filtered.shape") 
        print(self.points2_2to3_filtered.shape) 

        temp_3d_points = np.ascontiguousarray(point_coords_3d.T[:, 0:3]).reshape((-1, 1, 3))
        print("temp_3d_points.shape")
        print(temp_3d_points.shape)

        assert temp_3d_points.shape[0] == self.points2_2to3_filtered.shape[0], "3D points and 2D points should be of same number."
        retval, rvec, tvec, inliners = cv2.solvePnPRansac(
            temp_3d_points, 
            self.points2_2to3_filtered, 
            self.cameraMatrix, 
            self.distCoeffs, 
            reprojectionError = 1)
        print(inliners)

        print("tvec")
        print(tvec)
        self.estimated_position_of_camera_3 = tvec

    def main(self, filename_one, filename_two, filename_three):
        self.load_images(filename_one, filename_two, filename_three)

        image1 = ImageAndKeypoints()
        image1.set_image(self.img1)
        image1.detect_keypoints()

        image2 = ImageAndKeypoints()
        image2.set_image(self.img2)
        image2.detect_keypoints()

        image3 = ImageAndKeypoints()
        image3.set_image(self.img3)
        image3.detect_keypoints()

        pair12 = ImagePair()
        pair12.set_images_new(image1, image3)
        pair12.standard_pipeline()
        pair12.show_estimated_camera_motion()
        pair12.visualise_points_in_3d_with_plotly()
        pair12.build_bundle_adjustment_graph()

        visualization12 = pair12.visualize_filtered_matches()
        cv2.imshow("12 image pair", visualization12)

        key = cv2.waitKey(10)
        while key is not ord('q'):
            key = cv2.waitKey(10)

        return 

        pair12 = ImagePair()
        pair12.set_images(self.img1, self.img2)
        pair12.detect_keypoints()
        pair12.match_detected_keypoints()
        pair12.standard_pipeline()
        pair12.show_estimated_camera_motion()
        pair12.visualise_points_in_3d_with_plotly()
        pair12.build_bundle_adjustment_graph()

        visualization12 = pair12.visualize_filtered_matches()
        cv2.imshow("12 image pair", visualization12)

        key = cv2.waitKey(10)
        while key is not ord('q'):
            key = cv2.waitKey(10)

        return

        pair13 = ImagePair()
        pair13.set_images(self.img1, self.img3)
        pair13.standard_pipeline()
        pair13.show_estimated_camera_motion()
        pair13.visualise_points_in_3d_with_plotly()

        pair23 = ImagePair()
        pair23.set_images(self.img2, self.img3)
        pair23.standard_pipeline()
        pair23.show_estimated_camera_motion()
        pair23.visualise_points_in_3d_with_plotly()

        
        visualization13 = pair12.visualize_filtered_matches()
        visualization23 = pair23.visualize_filtered_matches()

        cv2.imshow("13 image pair", visualization13)
        cv2.imshow("23 image pair", visualization23)

        key = cv2.waitKey(10)
        while key is not ord('q'):
            key = cv2.waitKey(10)

        return

        self.detect_keypoints()
        self.match_detected_keypoints_between_first_and_second_image()
        self.determine_fundamental_matrix_1to2()
        self.determine_essential_matrix()
        self.estimate_camera_movement()
        #self.show_estimated_camera_motion()
        self.reconstruct_3d_points()
        self.get_validated_matches()
        #self.visualise_points_in_3d()
        # self.draw_epilines_on_images()
        self.draw_matched_points_on_first_image()
        self.locate_matched_points_in_third_image()
        self.visualise_points_in_3d_with_plotly()

        self.img4 = cv2.drawMatches(self.img1,
                self.kp1,
                self.img2,
                self.kp2,
                self.filtered_matches_1to2,
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        temp_matches_2_to_3 = self.bf.match(
                self.des_img2, 
                self.des_img3)
        self.img3 = cv2.drawMatches(self.img2,
                self.kp2,
                self.img3,
                self.kp3,
                self.matches_2_to_3,
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        #cv2.imshow("img1", self.img1)
        #cv2.imshow("img2", self.img2)
        cv2.imshow("img3", self.img3)
        cv2.imshow("img4", self.img4)
        #cv2.imshow("img5", self.img5)
        #cv2.imshow("img7", self.img7)
        key = cv2.waitKey(10)
        while key is not ord('q'):
            key = cv2.waitKey(10)

        #plt.show()
        return


# Parse command line args
parser = argparse.ArgumentParser()
parser.add_argument("image1", type=str,
        help="path to the first image")
parser.add_argument("image2", type=str,
        help="path to the second image")
parser.add_argument("image3", type=str,
        help="path to the third image")
args = parser.parse_args()
TPFTI = TriangulatePointsFromTwoImages()
TPFTI.main(args.image1, args.image2, args.image3)


# Notes to my self
# The order of images seems to be important for the reconstruction. This should be investigated.
