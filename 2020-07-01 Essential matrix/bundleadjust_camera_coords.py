import numpy as np
import g2o 
from collections import defaultdict


def bundle_adjustment(focal_length, cx, cy, points_3d, rotation, translation, iterations, outlier_ratio):
    optimizer = g2o.SparseOptimizer()
    solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
    solver = g2o.OptimizationAlgorithmLevenberg(solver)
    optimizer.set_algorithm(solver)

    cam = g2o.CameraParameters(focal_length, np.array([cx, cy], dtype=np.float64), 0)
    cam.set_id(0)
    optimizer.add_parameter(cam)
    '''
        true_points = np.hstack([
            np.random.random((500, 1)) * 3 - 1.5,
            np.random.random((500, 1)) - 0.5,
            np.random.random((500, 1)) + 3])
    '''
    poses = []
    for i in range(len(rotation)):
        # pose here means transform points from world coordinates to camera coordinates
        v_se3 = g2o.VertexSE3Expmap()
        v_se3.set_id(i)
        pose = g2o.SE3Quat(rotation[i], translation[i].T[0])
        poses.append(pose)
        v_se3.set_estimate(pose)
        if i < 2:
            v_se3.set_fixed(True)
        optimizer.add_vertex(v_se3)


    point_id = len(rotation)

    for i, point in enumerate(points_3d):
        visible = []
        for j, pose in enumerate(poses):
            #Project point to camera coordinates
            #print(point)
            z = cam.cam_map(pose * point)
            #only add the point if its visible in this frame
            if 0 <= z[0] < cx*2 and 0 <= z[1] < cy*2:
                visible.append((j, z))
        if len(visible) < 2:
            continue

        vp = g2o.VertexSBAPointXYZ()
        vp.set_id(point_id)
        vp.set_marginalized(True)
        vp.set_estimate(point)
        optimizer.add_vertex(vp)

        for j, z in visible:
            if np.random.random() < outlier_ratio:
                z = np.random.random(2) * [cx*2, cy*2]

            edge = g2o.EdgeProjectXYZ2UV()
            edge.set_vertex(0, vp)
            edge.set_vertex(1, optimizer.vertex(j))
            edge.set_measurement(z)
            edge.set_information(np.identity(2))
            edge.set_robust_kernel(g2o.RobustKernelHuber())

            edge.set_parameter_id(0, 0)
            optimizer.add_edge(edge)

        point_id += 1

    print('num vertices:', len(optimizer.vertices()))
    print('num edges:', len(optimizer.edges()))

    print('Performing full BA:')
    optimizer.initialize_optimization()
    optimizer.set_verbose(True)
    optimizer.optimize(iterations)


    adjusted_T = []
    adjusted_R = []
    for i in range(len(poses)):
        adjusted_T.append(optimizer.vertex(i).estimate().translation())
        adjusted_R.append(optimizer.vertex(i).estimate().rotation())
    
    return adjusted_R, adjusted_T

def PoseOptimizer(focal_length, cx, cy, points_3d, rotation, translation, iterations, outlier_ratio):
    optimizer = g2o.SparseOptimizer()
    solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
    solver = g2o.OptimizationAlgorithmLevenberg(solver)
    optimizer.set_algorithm(solver)

    cam = g2o.CameraParameters(focal_length, np.array([cx, cy], dtype=np.float64), 0)
    cam.set_id(0)
    optimizer.add_parameter(cam)

    v_se3 = g2o.VertexSE3Expmap()
    v_se3.set_id(0)
    pose = g2o.SE3Quat(rotation, translation.T[0])
    v_se3.set_estimate(pose)
    #v_se3.set_fixed(True)
    optimizer.add_vertex(v_se3)

    point_id = 1

    for i, point in enumerate(points_3d):
        vp = g2o.VertexSBAPointXYZ()
        vp.set_id(point_id)
        vp.set_marginalized(True)
        vp.set_estimate(point)
        optimizer.add_vertex(vp)

        edge = g2o.EdgeProjectXYZ2UV()
        edge.set_vertex(0, vp)
        edge.set_vertex(1, optimizer.vertex(0))
        edge.set_measurement(cam.cam_map(pose * point))
        edge.set_information(np.identity(2))
        edge.set_robust_kernel(g2o.RobustKernelHuber())

        edge.set_parameter_id(0, 0)
        optimizer.add_edge(edge)

        point_id += 1

    print('num vertices:', len(optimizer.vertices()))
    print('num edges:', len(optimizer.edges()))

    print('Performing PGO:')
    optimizer.initialize_optimization()
    optimizer.set_verbose(True)
    optimizer.optimize(iterations)

    adjusted_T = optimizer.vertex(0).estimate().translation()
    adjusted_R = optimizer.vertex(0).estimate().rotation()
    
    return adjusted_R, adjusted_T

