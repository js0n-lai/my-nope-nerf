import os
import shutil
import argparse
import numpy as np
import yaml
import pdb
import pandas as pd
import cv2
from scipy.spatial.transform import Rotation as R
from utils_poses.vis_cam_traj import draw_camera_frustum_geometry

# intrinsics
K = np.array([[725, 0, 620.5],
                  [0, 725, 187.0],
                  [0, 0, 1]])

def has_met_movement_thresholds(x, y, thresh_rot, thresh_translate):
    # no thresholds provided so accept
    if thresh_rot is None and thresh_translate is None:
        return True
    
    r_x = R.from_matrix(x[:,:-1])
    r_y = R.from_matrix(y[:,:-1])
    d_rot = np.abs(r_x.magnitude() - r_y.magnitude())
    d_translate = np.linalg.norm(x[:,3] - y[:,3])
    # print(d_rot, d_translate)

    if thresh_rot is not None and d_rot >= thresh_rot:
        return True
    if thresh_translate is not None and d_translate >= thresh_translate:
        return True
    return False

def get_num_frames(args):
    img_src = os.path.join(args.root, f"vkitti_{args.version}_rgb", args.id, args.variation)
    return len([name for name in os.listdir(img_src)])

def get_filtered_ids(args):
    print('Getting frame IDs...', end=' ')

    # determine last frame ID if no end arg is given
    end = get_num_frames(args) if args.end is None else args.end

    # no filtering - return the given range
    if args.method == "interval":
        return list(range(args.start, end, args.interval))
    
    # read pose file
    pose_src = os.path.join(args.root, f"vkitti_{args.version}_extrinsicsgt", f"{args.id}_{args.variation}.txt")
    poses = pd.read_csv(pose_src, sep=" ", index_col=False)
    ids = []
    step = 1

    for i in range(args.start, end):
        current = np.reshape(poses.iloc[i][1:], [4,4])[:3,:]
        if len(ids) == 0:
            previous = current
            ids.append(i)
        elif has_met_movement_thresholds(current, previous, args.thresh_rot, args.thresh_translate):
            if step == args.interval:
                previous = current
                ids.append(i)
            else:
                step += 1

    return ids

def make_calib(dest):
    # generate intrinsics.npz
    np.savez(os.path.join(dest, "intrinsics.npz"), K=K)

def make_img(src, dest, frames, skip_copy=False):
    print('Making images directory...', end=' ')
    if not skip_copy:
        img_dest = os.path.join(dest, "images")

        # create folder structure
        if not os.path.exists(img_dest):
            os.mkdir(img_dest)
        
        img_src = os.path.join(src, f"vkitti_{args.version}_rgb", args.id, args.variation)
        
        # copy frames over
        for fr in frames:
            shutil.copy(os.path.join(img_src, f"{str(fr).zfill(5)}.png"), img_dest)

    print('done!')

def make_depth(src, dest, frames, skip_copy=False):
    print('Making depth directory...', end=' ')
    
    if not skip_copy:
        depth_dest = os.path.join(dest, "depth")
        os.makedirs(depth_dest, exist_ok=True)

        depth_src = os.path.join(src, f"vkitti_{args.version}_depthgt", args.id, args.variation)

        # copy depth frames
        for fr in frames:
            shutil.copy(os.path.join(depth_src, f"{str(fr).zfill(5)}.png"), depth_dest)
    
    print('done!')

def make_poses(src, dest, frames, make_gt=False, make_colmap=False):
    poses_src = os.path.join(src, f"vkitti_{args.version}_extrinsicsgt", f"{args.id}_{args.variation}.txt")
    poses = pd.read_csv(poses_src, sep=" ", index_col=False)
    depth_src = os.path.join(src, f"vkitti_{args.version}_depthgt", args.id, args.variation)

    # create gt_poses.npz or poses_bounds.npy file
    gt_pose_dest = os.path.join(dest, "gt_poses.npz")
    poses_bounds_dest = os.path.join(dest, "poses_bounds.npy")
    gt_llff_dest = os.path.join(dest, "poses_gt.npy")
    
    # make camera-to-world coordinates (right, up, backward) instead of (forward, down, left) 
    P_gl = np.array([[0, 0, -1],
                     [0, -1, 0],
                     [-1, 0, 0]])

    result = np.zeros((len(frames), 17))
    result_gl = np.zeros((len(frames), 4, 4))
    result_llff = np.zeros((len(frames), 4, 4))
    result_orig = np.zeros((len(frames), 4, 4))

    for i, f in enumerate(frames):

        # change to camera-to-world
        x = np.linalg.inv(np.reshape(poses.iloc[f][1:], [4,4]))
        result_orig[i] = x
        r = x[:3,:3]
        t = x[:3,-1]

        # correct camera-coordinates: originally (right, down, forward) but needs to be (right, up, backward)
        # by rotating each camera about its local x axis
        x_local = r[:,0]
        rot = R.from_rotvec(x_local*np.pi).as_matrix()
        r = np.matmul(rot, r)

        # make world coordinate system (right, up, backward) instead of (forward, down, left) 
        r_gl = np.matmul(np.linalg.inv(P_gl), r)
        t_gl = np.matmul(np.linalg.inv(P_gl), t) 
        x[:3,:3] = r_gl
        x[:3,-1] = t_gl
        result_gl[i] = x

        # do the inverse transformation of rotations to (y, -x, z) done when preprocessing poses_bounds.npy
        r_llff = np.hstack([-r_gl[:,1].reshape([-1, 1]), r_gl[:,0].reshape([-1, 1]), r_gl[:,2:]])
        x[:3,:3] = r_llff
        x[:3,-1] = t_gl
        result_llff[i] = x

        # get depth range - given in cm
        depth = cv2.imread(os.path.join(depth_src, f"{str(f).zfill(5)}.png"), cv2.IMREAD_UNCHANGED)
        height, width = depth.shape        

        depth_min = depth[depth > 0].min() / 100
        depth_max = depth.max() / 100
        llff_35 = np.hstack((x[:3,:], np.array([width, height, K[0,0]]).reshape([3, 1])))
        llff_flattened = np.hstack((llff_35.flatten(), depth_min, depth_max))
        result[i] = llff_flattened

    # pdb.set_trace()
    # draw_camera_frustum_geometry(result_llff, height, width, K[0,0], K[1,1], frustum_length=0.5, coord='opengl',draw_now=True)
    # draw_camera_frustum_geometry(result_gl, height, width, K[0,0], K[1,1], coord='opengl', draw_now=True)
    # draw_camera_frustum_geometry(result_orig, height, width, K[0,0], K[1,1], coord='opengl', draw_now=True)

    if make_gt:
        np.savez(gt_pose_dest, poses=result_gl)
    
    np.save(gt_llff_dest, result)
    if make_colmap:
        np.save(poses_bounds_dest, result)

    print("done!")
    
    return [height, width]

def make_yaml(dest, args, resolution):
    print("Making config files...", end=' ')
    # create preprocess yaml
    with open(os.path.join("configs", "preprocess.yaml"), "r") as f:
        preprocess = yaml.safe_load(f)

    path = os.path.normpath(os.path.join(dest, ".."))
    scene = os.path.basename(os.path.normpath(dest))
    preprocess['dataloading']['path'] = path
    preprocess['dataloading']['scene'] = [scene]
    preprocess['dataloading']['resize_factor'] = args.resize_factor
    preprocess['dataloading']['customized_poses'] = args.customised_poses
    preprocess['dataloading']['customized_focal'] = args.customised_focal
    preprocess['dataloading']['load_colmap_poses'] = args.load_colmap_poses

    config_dest = os.path.join(os.getcwd(), "configs", "V_KITTI")
    os.makedirs(config_dest, exist_ok=True)

    preprocess_yaml = os.path.join(config_dest, f"preprocess_{scene}.yaml")
    with open(preprocess_yaml, "w") as f:
        yaml.dump(preprocess, f)
    
    # create train yaml
    with open(os.path.join("configs", "Tanks", "Ballroom_default.yaml"), "r") as f:
        train = yaml.safe_load(f)

    train["dataloading"]["path"] = path
    train["dataloading"]["scene"] = [scene]
    train["dataloading"]["customized_poses"] = args.customised_poses
    train["dataloading"]["customized_focal"] = args.customised_focal
    train["dataloading"]["resize_factor"] = args.resize_factor
    train["dataloading"]["load_colmap_poses"] = args.load_colmap_poses
    train["dataloading"]["with_depth"] = args.with_depth
    train["pose"]["learn_pose"] = args.learn_pose
    train["pose"]["learn_R"] = args.learn_pose
    train["pose"]["learn_t"] = args.learn_pose
    train["pose"]["init_pose"] = args.init_pose
    if args.load_colmap_poses:
        train["pose"]["init_pose_type"] = "colmap"
    train["pose"]["init_R_only"] = False
    train["pose"]["learn_focal"] = args.learn_focal
    train["pose"]["update_focal"] = args.update_focal
    train["distortion"] = dict()
    train["distortion"]["learn_distortion"] = args.learn_distortion
    train["training"]["out_dir"] = os.path.join("out", os.path.relpath(dest, "data"))
    train["training"]["with_ssim"] = args.with_ssim

    if args.match_method != 'dense':
        print(f"Warning: {args.match_method} is not implemented")
    train["training"]["match_method"] = args.match_method
    train["extract_images"]["resolution"] = [int(np.ceil(x / args.resize_factor)) for x in resolution]
    train["extract_images"]["eval_depth"] = True
    train["extract_images"]["traj_option"] = args.traj_option
    train["extract_images"]["bspline_degree"] = args.bspline_degree
    train["depth"]["depth_loss_type"] = args.depth_loss_type

    # override settings for Vanilla NeRF simulation
    if args.simulate_vanilla:
        train["pose"]["init_pose"] = True
        train["pose"]["learn_R"] = False
        train["pose"]["learn_t"] = False
        train["pose"]["learn_focal"] = False
        train["training"]["auto_scheduler"] = False
        train["training"]["scheduling_start"] = 0
        train["training"]["annealing_epochs"] = 0

    train_yaml = os.path.join(config_dest, f"{scene}.yaml")
    with open(train_yaml, "w") as f:
        yaml.dump(train, f)

    print(f"Wrote configs to {preprocess_yaml} and {train_yaml}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Creates directory of frames where both RGB images and pose information was available")
    setup = parser.add_argument_group("Setup")
    setup.add_argument("root", type=str, help="Root directory of the Virtual KITTI dataset")
    setup.add_argument("version", type=str, help="Version of the Virtual KITTI dataset")
    setup.add_argument("id", type=str, help="Drive ID xxxx")
    setup.add_argument("--variation", type=str, choices=['15-deg-left', '15-deg-right', '30-deg-left', '30-deg-right', 'clone', 'fog', 'morning', 'overcast', 'rain', 'sunset'], default='clone',\
                       help="Rendering variation (default: clone)")
    setup.add_argument("dest", type=str, help="Name of the new dataset")
    setup.add_argument("--skip-copy", action="store_true", default=False, help="Skip copying data over (default: False)")
    method = parser.add_argument_group("Method")
    method.add_argument("method", choices=["interval", "threshold"], action="store", help="""Keyframe selection method.
                        'interval' to select all frames with IDs in [start, stop] over fixed intervals. 'threshold' to select frames where motion
                        relative to the previous frame exceeds rotation or translation thresholds (defaults are None). Only frames with pose information are considered.
                        May also be combined with start, end and interval arguments.""")
    method.add_argument("-s", "--start", type=int, action="store", default=0, help="Frame ID to start from (default: 0)")
    method.add_argument("-e", "--end", type=int, action="store", default=None, help="Frame ID to end at (default: end)")
    method.add_argument("-i", "--interval", type=int, action="store", default=1, help="Step between selected frames (default: 1)")
    method.add_argument("-r", type=float, action="store", dest="thresh_rot", default=None, help="Minimum rotation threshold between consecutive frames (deg). If not provided, rotation will not be considered.")
    method.add_argument("-t", type=float, action="store", dest="thresh_translate", default=None, help="Minimum translation threshold between consecutive frames (m). If not provided, translation will not be considered.")
    config = parser.add_argument_group("Config", description="Parameters and hyperparameters for the NoPe-NeRF model. Default values are based on configs/default.yaml")
    config.add_argument('--resize-factor', action="store", type=int, default=1, help="Factor to downscale each input image dimension (default: 1)")
    config.add_argument('--init-pose', action="store_true", default=False, help="Provide initial pose (default: False)")
    config.add_argument("--learn-pose", action="store", type=bool, default=True, help="Enable NoPe-NeRF to optimise camera poses (default: True)")
    config.add_argument("--learn-focal", action="store", type=bool, default=False, help="Enable NoPe-NeRF to optimise camera intrinsics (default: False)")
    config.add_argument("--learn-distortion", action="store", type=bool, default=True, help="Enable NoPe-NeRF to optimise DPT depth frame distortion coefficients (default: True)")
    config.add_argument("--load-colmap-poses", action="store_true", default=False, help="Use COLMAP poses (default: False)")
    config.add_argument("--mock-colmap-poses", action="store_true", default=False, help="Generate mock COLMAP output using other poses (default: False)")
    config.add_argument("--customised-poses", action="store_true", default=False, help="Use poses other than those from COLMAP (default: False)")
    config.add_argument("--customised-focal", action="store_true", default=False, help="Use intrinsics other than those from COLMAP (default: False)")
    config.add_argument("--update-focal", action="store", default=True, help="Enable NoPe-NeRF to update camera intrinsics (default: True)")
    config.add_argument("--match-method", action="store", choices=["dense", "sparse"], default="dense", help="Method to compute point cloud loss. sparse is unimplemented (default: dense)")
    config.add_argument("--with-ssim", action="store_true", default=False, help="Use SSIM loss when computing DPT reprojection loss (default: False)")
    config.add_argument("--with-depth", action="store_true", default=False, help="Use GT depths (default: False)")
    config.add_argument("--traj-option", choices=["sprial", "interp", "bspline"], default="bspline", help="Camera trajectory option for rendering (default: bspline)")
    config.add_argument("--bspline-degree", type=int, default=100, help="Basis function degree for BSpline trajectory (default: 100)")
    config.add_argument("--depth-loss-type", choices=["l1", "invariant"], default="l1", help="Type of depth loss (default: invariant)")
    config.add_argument("--simulate-vanilla", action="store_true", default=False, help="Configure settings to simulate Vanilla NeRF. Overrides most other config settings (default: False)")
    args = parser.parse_args()

    # get frame IDs based on filter parameters
    frames = get_filtered_ids(args)
    print(f"Got {len(frames)} frames")
    
    # create directory structure
    out_dir = os.path.join("data", f"V_KITTI", args.dest)
    os.makedirs(out_dir, exist_ok=True)
    
    make_calib(out_dir)
    make_img(args.root, out_dir, frames, args.skip_copy)
    make_depth(args.root, out_dir, frames, args.skip_copy)
    resolution = make_poses(args.root, out_dir, frames, args.customised_poses, args.mock_colmap_poses)
    make_yaml(out_dir, args, resolution)
