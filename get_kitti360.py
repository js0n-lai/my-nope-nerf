import os
import shutil
import argparse
import subprocess
import numpy as np
import yaml
import pdb
from scipy.spatial.transform import Rotation as R
from utils_poses.vis_cam_traj import draw_camera_frustum_geometry

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
    img_src = os.path.join(args.root, "data_2d_raw", f"2013_05_28_drive_{args.id}_sync", "image_00", "data_rect")
    return len([name for name in os.listdir(img_src)])

def get_filtered_ids(args):
    print('Getting frame IDs...', end=' ')

    # determine last frame ID if no end arg is given
    end = get_num_frames(args) if args.end is None else args.end

    # no filtering - return the given range
    if args.method == "interval":
        return list(range(args.start, end, args.interval))
    
    # read pose file
    pose_src = os.path.join(args.root, "data_poses", f"2013_05_28_drive_{args.id}_sync", "poses.txt")
    ids = []
    first = True

    with open(pose_src, 'r') as f:
        step = 1

        while True:
            line = f.readline()
            if line == '':
                break
            
            # check id is in range
            id = int(line.split()[0])
            if id > end:
                break
            elif id < args.start:
                continue

            current = np.reshape(np.array(line.split()[1:], dtype=float), (3, 4))
            if first:
                first = False
                previous = current
                ids.append(id)
            elif has_met_movement_thresholds(current, previous, args.thresh_rot, args.thresh_translate):
                if step == args.interval:
                    previous = current
                    ids.append(id)
                    step = 1
                else:
                    step += 1

    return ids

def make_calib(src, dest, frames=None, skip_copy=False):
    print('Making calibration directory...', end=' ')

    if not skip_copy:
        calib_dest = os.path.join(dest, "calibration")
        if not os.path.exists(calib_dest):
            os.mkdir(calib_dest)

        calib_src = os.path.join(src, "calibration")
        files = ["calib_cam_to_pose.txt", "calib_cam_to_velo.txt", "perspective.txt"]

        for f in files:
            shutil.copy(os.path.join(calib_src, f), calib_dest)

    # generate intrinsics.npz
    intrinsics = { line.split(': ')[0] : line.split(': ')[1] for line in open(os.path.join(dest, "calibration", "perspective.txt"), "r") }
    K = np.reshape(np.array(intrinsics["K_00"].split(), dtype=float), [3, 3])
    np.savez(os.path.join(dest, "intrinsics.npz"), K=K)

def make_img(src, dest, frames, skip_copy=False):
    print('Making images directory...', end=' ')
    
    if not skip_copy:
        img_dest = os.path.join(dest, "images")
        folders = ["image_00"]

        # create folder structure
        if not os.path.exists(img_dest):
            os.mkdir(img_dest)
        
        seq = os.path.normpath(dest).split(os.sep)[1]
        img_src = os.path.join(src, "data_2d_raw", seq)
        
        # copy frames over
        for fr in frames:
            for f in folders:
                shutil.copy(os.path.join(img_src, f, "data_rect", f"{str(fr).zfill(10)}.png"), img_dest)

    print('done!')

def make_lidar(src, dest, frames, decode_root, num_frames, skip_copy=False):
    print('Making lidar directory...', end=' ')
    
    if not skip_copy:
        lidar_dest = os.path.join(dest, "lidar")
        os.makedirs(os.path.join(lidar_dest, "raw"), exist_ok=True)

        seq = os.path.normpath(dest).split(os.sep)[1]
        lidar_src = os.path.join(src, "data_3d_raw", seq)

        # copy scan timestamps
        shutil.copy(os.path.join(lidar_src, "velodyne_points", "timestamps.txt"), lidar_dest)

        # copy bin scans
        for fr in frames:
            shutil.copy(os.path.join(lidar_src, "velodyne_points", "data", f"{str(fr).zfill(10)}.bin"), os.path.join(lidar_dest, "raw"))

        # decode data - need to split this up to conserve RAM usage (~500 frames = 8 GB RAM)
        # while also allowing enough frames (> 200) for the script to run without errors
            
        # cache current directory
        cwd = os.getcwd()
        lidar_dest_abs = os.path.abspath(lidar_dest)
        os.chdir(decode_root)
        executable = os.path.join(decode_root, "run_accumulation.sh")
        start = frames[0]
        stop = frames[-1] if frames[0] != frames[-1] else frames[0] + 1

        for i in range(start, stop, 500):
            # last block has insufficient frames so move the bounds
            if stop - i < 200:
                lower = i
                upper = min(num_frames - 1, stop + 200 - (stop - lower))
                if upper - lower < 200:
                    lower -= 200 - (upper - lower)  
                command = f"{executable} {src} {lidar_dest_abs} {seq} {lower} {upper} 1".split()
            else:
                command = f"{executable} {src} {lidar_dest_abs} {seq} {i} {min(stop, i+499)} 1".split()
            print(command)
            subprocess.run(command)
        
        os.chdir(cwd)

        # rename generated directories to data_xxxx_yyyy
        folders = os.listdir(lidar_dest)
        for folder in folders:
            os.rename(os.path.join(lidar_dest, folder), os.path.join(lidar_dest, f"data_{'_'.join(folder.split('_')[-2:])}"))
    
    print('done!')

def make_poses(src, dest, frames, skip_copy=False, make_gt=False, make_colmap=False):
    print('Making poses directory...', end=' ')
    poses_dest = os.path.join(dest, "poses")
    if not os.path.exists(poses_dest):
        os.mkdir(poses_dest)

    seq = os.path.normpath(dest).split(os.sep)[1]
    poses_src = os.path.join(src, "data_poses", seq)

    files = ["cam0_to_world.txt", "poses.txt"]

    if not skip_copy:
        for f in files:
            shutil.copy(os.path.join(poses_src, f), poses_dest)

    # create gt_poses.npz or poses_bounds.npy file
    gt_pose_dest = os.path.join(dest, "gt_poses.npz")
    poses_bounds_dest = os.path.join(dest, "poses_bounds.npy")
    gt_llff_dest = os.path.join(dest, "poses_gt.npy")

    # get poses in camera-to-world format
    # originally in (forward, left, up) but needs to be in (down, right, backward)
    P = np.array([[0, 0, -1],
                  [0, -1, 0],
                  [-1, 0, 0]])
    
    # load original poses
    cam0_to_world_orig = np.loadtxt(os.path.join(poses_dest, "cam0_to_world.txt"))

    # load intrinsics
    intrinsics = { line.split(': ')[0] : line.split(': ')[1] for line in open(os.path.join(dest, "calibration", "perspective.txt"), "r") }

    result = np.zeros((len(frames), 17))
    result_gl = np.zeros((len(frames), 4, 4))
    result_colmap = np.zeros((len(frames), 4, 4))
    result_orig = np.zeros((len(frames), 4, 4))

    # originally in (forward, left, up), OpenGL uses (right, up, backward)
    P_gl = np.array([[0, 0, -1],
                  [-1, 0, 0],
                  [0, 1, 0]])
    # P_gl = np.eye(3)
    P_rotate_x = np.array([[1, 0, 0],
                           [0, -1, 0],
                           [0, 0, -1]])
    P_rotate_y = np.array([[-1, 0, 0],
                           [0, 1, 0],
                           [0, 0, -1]])
    P_rotate_z = np.array([[-1, 0, 0],
                           [0, -1, 0],
                           [0, 0, 1]])
    P_preprocess = np.eye(3)

    depth_range = np.loadtxt(os.path.join(src, "data_3d_raw", seq, "velodyne_points", "frames", "depth_range.txt"))
    from scipy.spatial.transform import Rotation as R
    j = 0
    for i in range(len(frames)):
        # print(f"{i}: Frame {frames[i]}")

        # change original transformation matrix from (forward, left, up) to (down, right, backward) convention
        while cam0_to_world_orig[j,0] != frames[i]:
            j += 1
        
        x = cam0_to_world_orig[j,1:].reshape([4, 4])
        r_new = x[:3,:3]
        # print()
        # print(r_new)

        # rotate about local z axis
        z_local = r_new[:,-1]
        rot = R.from_rotvec(z_local*np.pi).as_matrix()
        r_new = np.matmul(np.linalg.inv(rot), r_new)

        # rotate about local y axis
        y_local = r_new[:,1]
        rot = R.from_rotvec(y_local*np.pi).as_matrix()
        r_new = np.matmul(np.linalg.inv(rot), r_new)
        t_new = x[:3,3]

        # change from (forward, left, up) to (down, right, backward)
        # r_new = np.matmul(np.linalg.inv(P_gl), r_new)
        # t_new = np.matmul(np.linalg.inv(P_gl), t_new)

        r_new_gl = r_new
        t_new_gl = t_new

        # do the inverse transformation of rotations to (y, -x, z) done when preprocessing poses_bounds.npy
        r_new = np.hstack([-r_new[:,1].reshape([-1, 1]), r_new[:,0].reshape([-1, 1]), r_new[:,2:]])

        rt_34 = np.hstack((r_new, t_new.reshape(-1, 1)))
        result_colmap[i,:,:] = np.vstack((rt_34, np.array([[0,0,0,1]])))

        depth_min = depth_range[frames[i],1]
        depth_max = depth_range[frames[i],2]

        width = float(intrinsics['S_rect_00'].split()[0])
        height = float(intrinsics['S_rect_00'].split()[1])
        colmap_35 = np.hstack((rt_34, np.array([[width],[height],[float(intrinsics["K_00"].split()[0])]])))
        y = np.hstack((colmap_35.flatten(), depth_min, depth_max))

        result[i,:] = y

        # change to (right, up, backward)
        # r_new_gl = np.matmul(P_rotate_y, np.matmul(np.linalg.inv(P_gl), x[:3,:3]))
        # t_new_gl = np.matmul(np.linalg.inv(P_gl), x[:3,3])
        # pdb.set_trace()
        z = np.vstack((np.hstack((r_new_gl, t_new_gl.reshape(-1, 1))), np.array([[0,0,0,1]])))
        result_gl[i,:,:] = z

        r_orig = x[:3,:3]
        t_orig =x[:3,3]
        w = np.vstack((np.hstack((r_orig, t_orig.reshape(-1, 1))), np.array([[0,0,0,1]])))
        result_orig[i,:,:] = w

    # pdb.set_trace()
    # draw_camera_frustum_geometry(result_colmap, height, width, float(intrinsics["K_00"].split()[0]), float(intrinsics["K_00"].split()[4]), frustum_length=0.5, coord='opengl',draw_now=True)
    gl = draw_camera_frustum_geometry(result_gl, height, width, float(intrinsics["K_00"].split()[0]), float(intrinsics["K_00"].split()[4]), coord='opengl', draw_now=True)
    draw_camera_frustum_geometry(result_orig, height, width, float(intrinsics["K_00"].split()[0]), float(intrinsics["K_00"].split()[4]), coord='opengl', draw_now=True)

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

    config_dest = os.path.join(os.getcwd(), "configs", "KITTI")
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
    train["training"]["out_dir"] = os.path.join("out", "kitti360", os.path.relpath(dest, "data"))
    train["training"]["with_ssim"] = args.with_ssim
    train["training"]["use_gt_depth"] = args.use_gt_depth

    if args.match_method != 'dense':
        print(f"Warning: {args.match_method} is not implemented")
    train["training"]["match_method"] = args.match_method
    train["extract_images"]["resolution"] = [int(x / args.resize_factor) for x in resolution]
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
    setup.add_argument("root", type=str, help="Root directory of the KITTI dataset")
    setup.add_argument("id", type=str, help="Drive ID xxxx as it appears in 2013_05_28_xxxx_sync in the KITTI dataset")
    setup.add_argument("dest", type=str, help="Name of the new dataset")
    setup.add_argument("--skip-copy", action="store_true", default=False, help="Skip copying data over (default: False)")
    setup.add_argument("decoder", type=str, action="store", help="Path to folder containing run_accumulation.sh script from kitti360Scripts repository")
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
    config.add_argument("--use-gt-depth", action="store_true", default=False, help="Use GT depths - not implemented (default: False)")
    config.add_argument("--traj-option", choices=["sprial", "interp", "bspline"], default="bspline", help="Camera trajectory option for rendering (default: bspline)")
    config.add_argument("--bspline-degree", type=int, default=100, help="Basis function degree for BSpline trajectory (default: 100)")
    config.add_argument("--depth-loss-type", choices=["l1", "invariant"], default="l1", help="Type of depth loss (default: invariant)")
    config.add_argument("--simulate-vanilla", action="store_true", default=False, help="Configure settings to simulate Vanilla NeRF. Overrides most other config settings (default: False)")
    args = parser.parse_args()

    # get frame IDs based on filter parameters
    frames = get_filtered_ids(args)
    print(f"Got {len(frames)} frames")
    
    # create directory structure
    out_dir = os.path.join("data", f"2013_05_28_drive_{args.id}_sync", args.dest)
    os.makedirs(out_dir, exist_ok=True)
    
    make_calib(args.root, out_dir, frames, args.skip_copy)
    make_img(args.root, out_dir, frames, args.skip_copy)
    # make_lidar(args.root, out_dir, frames, args.decoder, get_num_frames(args), args.skip_copy)
    resolution = make_poses(args.root, out_dir, frames, args.skip_copy, args.customised_poses, args.mock_colmap_poses)
    make_yaml(out_dir, args, resolution)
