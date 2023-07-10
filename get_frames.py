import os
import shutil
import argparse
import numpy as np
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

def get_filtered_ids(args):
    print('Getting frame IDs...', end=' ')

    # determine last frame ID if no end arg is given
    if args.end is None:
        img_src = os.path.join(args.root, "data_2d_raw", f"2013_05_28_drive_{args.id}_sync", "image_00", "data_rect")
        end = len([name for name in os.listdir(img_src)])
    else:
        end = args.end

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

def make_calib(src, dest, frames=None):
    print('Making calibration directory...', end=' ')
    calib_dest = os.path.join(dest, "calibration")
    if not os.path.exists(calib_dest):
        os.mkdir(calib_dest)

    calib_src = os.path.join(src, "calibration")
    files = ["calib_cam_to_pose.txt", "calib_cam_to_velo.txt", "perspective.txt"]

    for f in files:
        shutil.copy(os.path.join(calib_src, f), calib_dest)

def make_img(src, dest, frames):
    print('Making images directory...', end=' ')
    img_dest = os.path.join(dest, "images")
    folders = ["image_00", "image_01"]

    # create folder structure
    if not os.path.exists(img_dest):
        os.mkdir(img_dest)
        for f in folders:
            os.mkdir(os.path.join(img_dest, f))
    
    seq = os.path.normpath(dest).split(os.sep)[1]
    img_src = os.path.join(src, "data_2d_raw", seq)
    
    # copy frames over
    for fr in frames:
       for f in folders:
        shutil.copy(os.path.join(img_src, f, "data_rect", f"{str(fr).zfill(10)}.png"), os.path.join(img_dest, f))

def make_lidar(src, dest, frames):
    print('Making lidar directory...', end=' ')
    lidar_dest = os.path.join(dest, "lidar")
    os.makedirs(os.path.join(lidar_dest, "data"), exist_ok=True)

    seq = os.path.normpath(dest).split(os.sep)[1]
    lidar_src = os.path.join(src, "data_3d_raw", seq)

    # copy scan timestamps
    shutil.copy(os.path.join(lidar_src, "velodyne_points", "timestamps.txt"), lidar_dest)

    # copy bin scans
    for fr in frames:
        shutil.copy(os.path.join(lidar_src, "velodyne_points", "data", f"{str(fr).zfill(10)}.bin"), os.path.join(lidar_dest, "data"))

def make_poses(src, dest, frames=None):
    print('Making poses directory...', end=' ')
    poses_dest = os.path.join(dest, "poses")
    if not os.path.exists(poses_dest):
        os.mkdir(poses_dest)

    seq = os.path.normpath(dest).split(os.sep)[1]
    poses_src = os.path.join(src, "data_poses", seq)

    files = ["cam0_to_world.txt", "poses.txt"]
    for f in files:
        shutil.copy(os.path.join(poses_src, f), poses_dest)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creates directory of frames where both RGB images and pose information was available")
    parser.add_argument("root", type=str, help="Root directory of the KITTI dataset")
    parser.add_argument("id", type=str, help="Drive ID xxxx as it appears in 2013_05_28_xxxx_sync in the KITTI dataset")
    parser.add_argument("dest", type=str, help="Name of the new dataset")
    parser.add_argument("method", choices=["interval", "threshold"], action="store", help="""Keyframe selection method.
                        'interval' to select all frames with IDs in [start, stop] over fixed intervals (default 1). 'threshold' to select frames where motion
                        relative to the previous frame exceeds rotation or translation thresholds (defaults are None). Only frames with pose information are considered.
                        May also be combined with start, end and interval arguments.""")
    parser.add_argument("-s", "--start", type=int, action="store", default=0, help="Frame ID to start from")
    parser.add_argument("-e", "--end", type=int, action="store", default=None, help="Frame ID to end at")
    parser.add_argument("-i", "--interval", type=int, action="store", default=1, help="Step between selected frames")
    parser.add_argument("-r", type=float, action="store", dest="thresh_rot", default=None, help="Minimum rotation threshold between consecutive frames (deg). If not provided, rotation will not be considered.")
    parser.add_argument("-t", type=float, action="store", dest="thresh_translate", default=None, help="Minimum translation threshold between consecutive frames (m). If not provided, translation will not be considered.")
    
    args = parser.parse_args()

    # get frame IDs based on filter parameters
    frames = get_filtered_ids(args)
    print(f"Got {len(frames)} frames")
    
    # create directory structure
    out_dir = os.path.join("data", f"2013_05_28_drive_{args.id}_sync", args.dest)
    os.makedirs(out_dir, exist_ok=True)
    
    callbacks = [make_calib, make_img, make_lidar, make_poses]
    for f in callbacks:
        f(args.root, out_dir, frames)
        print('Done!')
