#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import ATE.transformations as tfs


def get_best_yaw(C):
    '''
    maximize trace(Rz(theta) * C)
    '''
    assert C.shape == (3, 3)

    A = C[0, 1] - C[1, 0]
    B = C[0, 0] + C[1, 1]
    theta = np.pi / 2 - np.arctan2(B, A)

    return theta


def rot_z(theta):
    R = tfs.rotation_matrix(theta, [0, 0, 1])
    R = R[0:3, 0:3]

    return R


def vis_traj(x, y):
    import open3d as o3d
    pts_x = o3d.geometry.PointCloud()
    pts_x.points = o3d.utility.Vector3dVector(x)
    pts_x.paint_uniform_color([1, 0, 0])
    pts_y = o3d.geometry.PointCloud()
    pts_y.points = o3d.utility.Vector3dVector(y)
    pts_y.paint_uniform_color([0, 1, 0])
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0,0,0])
    o3d.visualization.draw_geometries([pts_x, pts_y, coord])

def align_umeyama(model, data, known_scale=False, yaw_only=False):
    """Implementation of the paper: S. Umeyama, Least-Squares Estimation
    of Transformation Parameters Between Two Point Patterns,
    IEEE Trans. Pattern Anal. Mach. Intell., vol. 13, no. 4, 1991.

    model = s * R * data + t

    Input:
    model -- first trajectory (nx3), numpy array type
    data -- second trajectory (nx3), numpy array type

    Output:
    s -- scale factor (scalar)
    R -- rotation matrix (3x3)
    t -- translation vector (3x1)
    t_error -- translational error per point (1xn)

    """

    # subtract mean
    mu_M = model.mean(0)
    mu_D = data.mean(0)
    # vis_traj(model, data)
    model_zerocentered = model - mu_M
    data_zerocentered = data - mu_D
    # vis_traj(model_zerocentered, data_zerocentered)
    n = np.shape(model)[0]

    # correlation
    C = 1.0/n*np.dot(model_zerocentered.transpose(), data_zerocentered)
    sigma2 = 1.0/n*np.multiply(data_zerocentered, data_zerocentered).sum()
    U_svd, D_svd, V_svd = np.linalg.linalg.svd(C)

    D_svd = np.diag(D_svd)
    V_svd = np.transpose(V_svd)

    S = np.eye(3)
    if(np.linalg.det(U_svd)*np.linalg.det(V_svd) < 0):
        S[2, 2] = -1

    if yaw_only:
        rot_C = np.dot(data_zerocentered.transpose(), model_zerocentered)
        theta = get_best_yaw(rot_C)
        R = rot_z(theta)
    else:
        R = np.dot(U_svd, np.dot(S, np.transpose(V_svd)))

    if known_scale:
        s = 1
    else:
        s = 1.0/sigma2*np.trace(np.dot(D_svd, S))

    t = mu_M-s*np.dot(R, mu_D)
    pred = np.zeros(data.shape)
    for i in range(pred.shape[0]):
        pred[i] = s * (R @ data[i,:] + t)
    # vis_traj(model, pred)
    # print(s)
    # print(R)
    # print(t)
    return s, R, t
