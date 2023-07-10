import os
import shutil
import sys
import argparse

def get_filtered_ids(args):
    # determine last frame ID if no end arg is given
    if args.end is None:
        img_src = os.path.join(args.root, "data_2d_raw", f"2013_05_28_drive_{args.id}_sync", "image_00", "data_rect")
        end = len([name for name in os.listdir(img_src)])
    else:
        end = args.end

    # no filtering - return the given range
    if args.no_filter:
        return list(range(args.start, end, args.interval))
    
    # read pose file
    pose_src = os.path.join(args.root, "data_poses", f"2013_05_28_drive_{args.id}_sync", "poses.txt")
    ids = []
    with open(pose_src, 'r') as f:
        step = 1

        while True:
            line = f.readline()
            if line == '':
                break
            
            id = int(line.split()[0])
            if id > end:
                break
            elif id < args.start:
                continue

            if step == args.interval:
                ids.append(id)
                step = 1
            else:
                step += 1

    return ids

def make_calib(src, dest, frames=None):
    calib_dest = os.path.join(dest, "calibration")
    if not os.path.exists(calib_dest):
        os.mkdir(calib_dest)

    calib_src = os.path.join(src, "calibration")
    files = ["calib_cam_to_pose.txt", "calib_cam_to_velo.txt", "perspective.txt"]

    for f in files:
        shutil.copy(os.path.join(calib_src, f), calib_dest)

def make_img(src, dest, frames):
    img_dest = os.path.join(dest, "images")
    folders = ["image_00", "image_01"]
    if not os.path.exists(img_dest):
        os.mkdir(img_dest)
        for f in folders:
            os.mkdir(os.path.join(img_dest, f))
    
    seq = os.path.normpath(dest).split(os.sep)[1]
    img_src = os.path.join(src, "data_2d_raw", seq)
    
    for fr in frames:
       for f in folders:
        shutil.copy(os.path.join(img_src, f, "data_rect", f"{str(fr).zfill(10)}.png"), os.path.join(img_dest, f))

def make_lidar(src, dest, frames):
    lidar_dest = os.path.join(dest, "lidar")
    os.makedirs(os.path.join(lidar_dest, "data"), exist_ok=True)

    seq = os.path.normpath(dest).split(os.sep)[1]
    lidar_src = os.path.join(src, "data_3d_raw", seq)
    shutil.copy(os.path.join(lidar_src, "velodyne_points", "timestamps.txt"), lidar_dest)
    for fr in frames:
        shutil.copy(os.path.join(lidar_src, "velodyne_points", "data", f"{str(fr).zfill(10)}.bin"), os.path.join(lidar_dest, "data"))

def make_poses(src, dest, frames=None):
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
    parser.add_argument("-nf", "--no-filter", action="store_true", default=False, help="Disable data filtering based on pose availability")
    parser.add_argument("-s", "--start", type=int, action="store", default=0, help="Frame ID to start from")
    parser.add_argument("-e", "--end", type=int, action="store", default=None, help="Frame ID to end at")
    parser.add_argument("-i", "--interval", type=int, action="store", default=1, help="Step between selected frames")
    # parser.add_argument("-r", type=float, default=None, action="store", dest="thresh_rot", help="Minimum rotation threshold between consecutive frames (deg)")
    # parser.add_argument("-t", type=float, default=None, action="store", dest="thresh_translate", help="Minimum translation threshold between consecutive frames (m)")
    
    args = parser.parse_args()

    # get frame IDs based on filter parameters
    frames = get_filtered_ids(args)
    print(frames)

    # create directory structure
    out_dir = os.path.join("data", f"2013_05_28_drive_{args.id}_sync", args.dest)
    os.makedirs(out_dir, exist_ok=True)
    
    callbacks = [make_calib, make_img, make_lidar, make_poses]
    for f in callbacks:
        f(args.root, out_dir, frames)
        print("Done")
