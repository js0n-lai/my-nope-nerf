import os
import shutil
import argparse
import subprocess
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

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
            subprocess.run(command)
        
        os.chdir(cwd)

        # rename generated directories to data_xxxx_yyyy
        folders = os.listdir(lidar_dest)
        for folder in folders:
            os.rename(os.path.join(lidar_dest, folder), os.path.join(lidar_dest, f"data_{'_'.join(folder.split('_')[-2:])}"))
    
    print('done!')

def get_depth_frame(root, id):
    
    # find which directory it's in
    lidar_root = os.path.join(root, "lidar")
    dirs = os.listdir(lidar_root)

    for d in dirs:
        frames = d.split("_")

        # found directory containing that frame
        if len(frames) == 3 and id >= int(frames[1]) and id <= int(frames[2]):
            points = open(os.path.join(lidar_root, d, "lidar_points_velodyne.dat"), "r")
            timestamps = open(os.path.join(lidar_root, d, "lidar_timestamp_velodyne.dat"), "r")
            found = False
            frame = None

            while True:
                t = timestamps.readline()
                p = points.readline()

                # reached EOF
                if t == '' or p == '':
                    break
                
                # found first point in frame
                if int(t) == id and not found:
                    found = True
                    frame = np.array(p.split()[:3], dtype=float)
                
                # found subsequent points in frame
                elif int(t) == id:
                    frame = np.vstack((frame, np.array(p.split()[:3], dtype=float)))

                # found all points in frame
                elif int(t) == id and found:
                    break
            
            points.close()
            timestamps.close()
            return frame

def make_poses(src, dest, frames, skip_copy=False):
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
    for i in range(len(frames)):
        # print(f"{i}: Frame {frames[i]}")

        # change original transformation matrix to new (down, right, backward) convention
        x = cam0_to_world_orig[i,1:].reshape([4, 4])
        r_new = np.matmul(np.linalg.inv(P), x[:3,:3])
        t_new = np.matmul(np.linalg.inv(P), x[:3,3])

        # get near and far depths for the frame
        depths = get_depth_frame(dest, frames[i])
        width = float(intrinsics['S_rect_00'].split()[0])
        height = float(intrinsics['S_rect_00'].split()[1])
        y = np.hstack((r_new.flatten(), t_new.flatten(),
                       width, height,
                       float(intrinsics["K_00"].split()[0]),
                       np.min(depths[:,2]), np.max(depths[:,2])))

        result[i,:] = y

    np.savez(gt_pose_dest, poses=result)
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

    config_dest = os.path.join(os.getcwd(), "configs", "KITTI")
    preprocess_yaml = os.path.join(config_dest, f"preprocess_{scene}.yaml")
    with open(preprocess_yaml, "w") as f:
        yaml.dump(preprocess, f)
    
    # create train yaml
    with open(os.path.join("configs", "Tanks", "Ballroom.yaml"), "r") as f:
        train = yaml.safe_load(f)

    train["dataloading"]["path"] = path
    train["dataloading"]["scene"] = [scene]
    train["dataloading"]["customized_poses"] = args.customised_poses
    train["dataloading"]["customized_focal"] = args.customised_focal
    train["dataloading"]["resize_factor"] = args.resize_factor
    train["dataloading"]["load_colmap_poses"] = not args.customised_poses
    train["pose"]["learn_pose"] = args.learn_pose
    train["pose"]["learn_R"] = args.learn_pose
    train["pose"]["learn_t"] = args.learn_pose
    train["pose"]["init_pose"] = not args.learn_pose
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
    train["extract_images"]["resolution"] = [x // 2 for x in resolution]
    train["extract_images"]["eval_depth"] = True

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
    config.add_argument('--resize-factor', action="store", type=int, default=2, help="Factor to downscale each input image dimension (default: 2)")
    config.add_argument("--learn-pose", action="store", type=bool, default=True, help="Enable NoPe-NeRF to optimise camera poses (default: True)")
    config.add_argument("--learn-focal", action="store", type=bool, default=False, help="Enable NoPe-NeRF to optimise camera intrinsics (default: False)")
    config.add_argument("--learn-distortion", action="store", type=bool, default=True, help="Enable NoPe-NeRF to optimise DPT depth frame distortion coefficients (default: True)")
    config.add_argument("--customised-poses", action="store_true", default=False, help="Use poses other than those from COLMAP (default: False)")
    config.add_argument("--customised-focal", action="store_true", default=False, help="Use intrinsics other than those from COLMAP (default: False)")
    config.add_argument("--update-focal", action="store", default=True, help="Enable NoPe-NeRF to update camera intrinsics (default: True)")
    config.add_argument("--match-method", action="store", choices=["dense", "sparse"], default="dense", help="Method to compute point cloud loss. sparse is unimplemented (default: dense)")
    config.add_argument("--with-ssim", action="store_true", default=False, help="Use SSIM loss when computing DPT reprojection loss (default: False)")
    config.add_argument("--use-gt-depth", action="store_true", default=False, help="Use GT depths - not implemented (default: False)")
    
    args = parser.parse_args()

    # get frame IDs based on filter parameters
    frames = get_filtered_ids(args)
    print(f"Got {len(frames)} frames")
    
    # create directory structure
    out_dir = os.path.join("data", f"2013_05_28_drive_{args.id}_sync", args.dest)
    os.makedirs(out_dir, exist_ok=True)
    
    make_calib(args.root, out_dir, frames, args.skip_copy)
    make_img(args.root, out_dir, frames, args.skip_copy)
    make_lidar(args.root, out_dir, frames, args.decoder, get_num_frames(args), args.skip_copy)
    resolution = make_poses(args.root, out_dir, frames, args.skip_copy)
    make_yaml(out_dir, args, resolution)
