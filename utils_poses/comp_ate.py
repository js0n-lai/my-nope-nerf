
import numpy as np

import ATE.trajectory_utils as tu
import ATE.transformations as tf
from scipy.spatial.transform import Rotation as R

def rotation_error(pose_error):
    """Compute rotation error
    Args:
        pose_error (4x4 array): relative pose error
    Returns:
        rot_error (float): rotation error
    """
    a = pose_error[0, 0]
    b = pose_error[1, 1]
    c = pose_error[2, 2]
    d = 0.5*(a+b+c-1.0)
    rot_error = np.arccos(max(min(d, 1.0), -1.0))
    return rot_error

def translation_error(pose_error):
    """Compute translation error
    Args:
        pose_error (4x4 array): relative pose error
    Returns:
        trans_error (float): translation error
    """
    dx = pose_error[0, 3]
    dy = pose_error[1, 3]
    dz = pose_error[2, 3]
    trans_error = np.sqrt(dx**2+dy**2+dz**2)
    return trans_error

def compute_rpe(gt, pred):
    trans_errors = []
    rot_errors = []
    for i in range(len(gt)-1):
        gt1 = gt[i]
        gt2 = gt[i+1]
        gt_rel = np.linalg.inv(gt1) @ gt2

        pred1 = pred[i]
        pred2 = pred[i+1]
        pred_rel = np.linalg.inv(pred1) @ pred2
        rel_err = np.linalg.inv(gt_rel) @ pred_rel
        
        trans_errors.append(translation_error(rel_err))
        rot_errors.append(rotation_error(rel_err))
    rpe_trans = np.sqrt(np.mean(np.asarray(trans_errors) ** 2))
    rpe_rot = np.sqrt(np.mean(np.asarray(rot_errors) ** 2))
    return rpe_trans, rpe_rot

def compute_ATE(gt, pred):
    """Compute RMSE of ATE
    Args:
        gt: ground-truth poses
        pred: predicted poses
    """
    errors = []

    for i in range(len(pred)):
        # cur_gt = np.linalg.inv(gt[0]) @ gt[i]
        cur_gt = gt[i]
        gt_xyz = cur_gt[:3, 3] 

        # cur_pred = np.linalg.inv(pred[0]) @ pred[i]
        cur_pred = pred[i]
        pred_xyz = cur_pred[:3, 3]

        align_err = gt_xyz - pred_xyz

        errors.append(np.sqrt(np.sum(align_err ** 2)))
    ate = np.sqrt(np.mean(np.asarray(errors) ** 2)) 
    return ate

def compute_ATE_v2(gt, pred):
    xyz_err = []
    rot_err = []

    for i in range(len(pred)):
        cur_gt = gt[i]
        gt_xyz = cur_gt[:3, 3]
        gt_rot = cur_gt[:3, :3]

        cur_pred = pred[i]
        pred_xyz = cur_pred[:3, 3]
        pred_rot = cur_pred[:3, :3]

        err_r = gt_rot @ (pred_rot.T)
        err_t = gt_xyz - err_r @ pred_xyz

        xyz_err.append(np.sqrt(np.sum(err_t ** 2)))
        # print(err_t, end=' ')
        # print(xyz_err[-1])
        r = R.from_matrix(err_r).as_rotvec()

        rot_err.append(np.linalg.norm(r))
    
    ate_t = np.sqrt(np.mean(np.asarray(xyz_err) ** 2))
    ate_r = np.sqrt(np.mean(np.asarray(rot_err) ** 2))

    return (ate_t, ate_r)

