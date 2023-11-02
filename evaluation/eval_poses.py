import os
import sys
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# to prevent Qt issues
fig = plt.figure()

import torch
sys.path.append(os.path.join(sys.path[0], '..'))
from dataloading import get_dataloader, load_config
import model as mdl
import numpy as np
import open3d as o3d

from utils_poses.vis_cam_traj import draw_camera_frustum_geometry, draw_camera_trajectory
from utils_poses.align_traj import align_ate_init_pose
from utils_poses.comp_ate import compute_rpe, compute_ATE, compute_ATE_v2
import ATE.transformations as tf

def revert_to_metric(poses, reverse):
    x = poses.clone().cpu().numpy()
    for i in range(x.shape[0]):
        
        # undo spherify
        if reverse.get('sc_spherify', None) is not None:
            x[i,:3,3] /= reverse['sc_spherify']
            x[i] = reverse['c2w_spherify'] @ x[i]

        # undo recentering
        x[i] = reverse['recenter'] @ x[i]

        # undo rescaling
        x[i,:3,3] /= reverse['sc']
    
    return torch.from_numpy(x).to(poses.device)

# rotates Open3D figure and saves frames
def custom_draw_geometry_with_camera_trajectory(geos):
    import matplotlib.pyplot as plt

    custom_draw_geometry_with_camera_trajectory.index = -1
    custom_draw_geometry_with_camera_trajectory.vis = o3d.visualization.Visualizer()

    def rotate(vis):
        global pose_folder
        # This function is called within the o3d.visualization.Visualizer::run() loop
        # The run loop calls the function, then re-render
        # So the sequence in this function is to:
        # 1. Capture frame
        # 2. index++, check ending criteria
        # 3. Set camera
        # 4. (Re-render)
        ctr = vis.get_view_control()
        glb = custom_draw_geometry_with_camera_trajectory
        if glb.index >= 0:
            print("Capture image {:05d}".format(glb.index))
            image = vis.capture_screen_float_buffer(False)
            plt.imsave(os.path.join(pose_folder, f"{glb.index:03d}.png"), np.asarray(image), dpi = 1)
            #vis.capture_depth_image("depth/{:05d}.png".format(glb.index), False)
            #vis.capture_screen_image("image/{:05d}.png".format(glb.index), False)
        glb.index = glb.index + 1
        factor = 5
        if glb.index < 360 // factor - 2:
            ctr.rotate(6.0 * factor, 0.0)
        else:
            custom_draw_geometry_with_camera_trajectory.vis.register_animation_callback(None)
        return False

    vis = custom_draw_geometry_with_camera_trajectory.vis
    vis.create_window()

    for g in geos:
        vis.add_geometry(g)
    vis.register_animation_callback(rotate)
    vis.run()
    vis.destroy_window()

torch.manual_seed(0)

# Config
parser = argparse.ArgumentParser(
    description='Eval Poses.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--vis',action='store_true')
args = parser.parse_args()
cfg = load_config(args.config, 'configs/default.yaml')

is_cuda = (torch.cuda.is_available())
device = torch.device("cuda" if is_cuda else "cpu")

out_dir = cfg['training']['out_dir']
global pose_folder
pose_folder = os.path.join(out_dir, "poses")
os.makedirs(pose_folder, exist_ok=True)

test_loader, field = get_dataloader(cfg, mode='train', shuffle=False)
N_imgs = field['img'].N_imgs

with torch.no_grad():
    if cfg['pose']['init_pose']:
        init_reverse = field['img'].reverse_init
        if cfg['pose']['init_pose_type']=='gt':
            init_pose = field['img'].c2ws # init with colmap
        elif cfg['pose']['init_pose_type']=='colmap':
            init_pose = field['img'].c2ws_colmap
    else:
        init_pose = None
        init_reverse = None
    pose_param_net = mdl.LearnPose(N_imgs, cfg['pose']['learn_R'], 
                            cfg['pose']['learn_t'], cfg=cfg, init_c2w=init_pose).to(device=device)
    checkpoint_io_pose = mdl.CheckpointIO(out_dir, model=pose_param_net)
    checkpoint_io_pose.load(cfg['extract_images']['model_file_pose'], device)
    learned_poses = torch.stack([pose_param_net(i) for i in range(N_imgs)])

    H = field['img'].H
    W = field['img'].W
    gt_poses = field['img'].c2ws_gt_llff
    gt_reverse = field['img'].reverse_gt
    if cfg['pose']['learn_focal']:
        focal_net = mdl.LearnFocal(cfg['pose']['learn_focal'], cfg['pose']['fx_only'], order=cfg['pose']['focal_order'])
        checkpoint_io_focal = mdl.CheckpointIO(out_dir, model=focal_net)
        checkpoint_io_focal.load(cfg['extract_images']['model_file_focal'], device)
        fxfy = focal_net(0)
        fx = fxfy[0] * W / 2
        fy = fxfy[1] * H / 2
    else:
        fx = field['img'].focal
        fy = field['img'].focal


'''Define camera frustums'''
frustum_length = 1
est_traj_color = np.array([39, 125, 161], dtype=np.float32) / 255
est_traj_color_2 = np.array([125, 161, 39], dtype=np.float32) / 255
cmp_traj_color = np.array([249, 65, 68], dtype=np.float32) / 255

# Recover metric scale by reverting LLFF preprocessing, then align the trajectories by setting the initial poses
# # of each to be identical
c2ws_est_to_draw_align2cmp = learned_poses.clone()
revert_LLFF = True

if revert_LLFF:

    # est pose = metric + aligned (for ATE)
    # learned pose = metric only (for RPE)
    c2ws_est_metric = revert_to_metric(c2ws_est_to_draw_align2cmp, gt_reverse)
    c2ws_learned_metric = revert_to_metric(learned_poses, gt_reverse)
    c2ws_gt_metric = revert_to_metric(gt_poses, gt_reverse)
    c2ws_est_aligned = align_ate_init_pose(c2ws_est_metric, c2ws_gt_metric)

    c2ws_est_to_draw_align2cmp = c2ws_est_aligned
    ate = compute_ATE(c2ws_gt_metric.cpu().numpy(), c2ws_est_aligned.cpu().numpy())
    ate_t, ate_r = compute_ATE_v2(c2ws_gt_metric.cpu().numpy(), c2ws_est_aligned.cpu().numpy())
    print(f"ATE_t (m) = {ate:.6f}, ATE_r (deg) = {(ate_r * 180/ np.pi):.6f}")
    # rpe_trans, rpe_rot = compute_rpe(gt_poses.cpu().numpy(), c2ws_est_aligned.cpu().numpy())
    rpe_trans, rpe_rot = compute_rpe(c2ws_gt_metric.cpu().numpy(), c2ws_learned_metric.cpu().numpy())
    print(f"RPE_t: {rpe_trans:.6f} RPE_r: {(rpe_rot * 180 / np.pi):.3f}")

    with open(os.path.join(out_dir, 'extraction', 'evaluation.txt'), 'a') as f:
        f.write(f"\nATE_t (m) = {ate:.6f}, ATE_r (deg) = {(ate_r * 180/ np.pi):.6f}\n")
        f.write(f"RPE_t: {rpe_trans:.6f} RPE_r: {(rpe_rot * 180 / np.pi):.3f}")

if args.vis:
    geometry_to_draw = []

    # draw learned poses w/o alignment in green (for debugging purposes)
    # frustum_est_2_list = draw_camera_frustum_geometry(learned_poses.cpu().numpy(), H, W,
    #                                                 fx, fy,
    #                                                 frustum_length, est_traj_color_2, draw_now=False)
    # geometry_to_draw.append(frustum_est_2_list)

    # draw refined poses in blue (equivalent to init pose if pose refinement is disabled)
    frustum_est_list = draw_camera_frustum_geometry(c2ws_est_to_draw_align2cmp.cpu().numpy(), H, W,
                                                    fx, fy,
                                                    frustum_length, est_traj_color, draw_now=False)
    geometry_to_draw.append(frustum_est_list)

    # draw GT/gold standard poses in red
    frustum_gt_list = draw_camera_frustum_geometry(c2ws_gt_metric.cpu().numpy(), H, W,
                                                        fx, fy,
                                                        frustum_length, cmp_traj_color, draw_now=False)
    geometry_to_draw.append(frustum_gt_list)

    # draw initial poses in cyan (useful if init_pose is set to True and pose refinement is enabled)
    if init_pose is not None:
        c2ws_init = align_ate_init_pose(revert_to_metric(init_pose, gt_reverse), c2ws_gt_metric)
        init_traj_color = np.array([29, 215, 158], dtype=np.float32) / 255
        frustum_init_list = draw_camera_frustum_geometry(c2ws_init.cpu().numpy(), H, W,
                                                            fx, fy,
                                                            frustum_length, init_traj_color, draw_now=False)                                                 
        geometry_to_draw.append(frustum_init_list)
    
    # o3d for line drawing
    t_est_list = c2ws_est_to_draw_align2cmp[:, :3, 3].cpu()
    t_cmp_list = c2ws_gt_metric[:, :3, 3].cpu()

    # line set to note pose correspondence between two trajs
    line_points = torch.cat([t_est_list, t_cmp_list], dim=0).cpu().numpy()  # (2N, 3)
    line_ends = [[i, i+N_imgs] for i in range(N_imgs)]  # (N, 2) connect two end points.

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector(line_ends)

    # uncomment below to show (can be messy)
    # geometry_to_draw.append(line_set)

    # line sets for camera trajectories
    est_points, est_lines, est_color = draw_camera_trajectory(c2ws_est_to_draw_align2cmp.cpu().numpy(), est_traj_color)
    gt_points, gt_lines, gt_color = draw_camera_trajectory(c2ws_gt_metric.cpu().numpy(), cmp_traj_color)

    # show coordinate system
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=gt_points[0])
    geometry_to_draw.append(coord)

    est_traj = o3d.geometry.LineSet()
    gt_traj = o3d.geometry.LineSet()
    est_traj.points = o3d.utility.Vector3dVector(est_points)
    gt_traj.points = o3d.utility.Vector3dVector(gt_points)
    est_traj.lines = o3d.utility.Vector2iVector(est_lines)
    gt_traj.lines = o3d.utility.Vector2iVector(gt_lines)
    est_traj.colors = o3d.utility.Vector3dVector(est_color)
    gt_traj.colors = o3d.utility.Vector3dVector(gt_color)

    geometry_to_draw.append(est_traj)
    geometry_to_draw.append(gt_traj)

    # Matplotlib 3D plots - set to True to enable
    use_matplotlib = False

    if use_matplotlib:
        ax = fig.add_subplot(111, projection='3d')

        gt_frustums = np.asarray(frustum_gt_list.points)
        est_frustums = np.asarray(frustum_est_list.points)

        for i in range(0, gt_frustums.shape[0], 5):
            frustum_vertices = gt_frustums[i:i+5]
            edges = [
            [frustum_vertices[4], frustum_vertices[1], frustum_vertices[2], frustum_vertices[3], frustum_vertices[4]],
            [frustum_vertices[0], frustum_vertices[1]],
            [frustum_vertices[0], frustum_vertices[2]],
            [frustum_vertices[0], frustum_vertices[3]],
            [frustum_vertices[0], frustum_vertices[4]]
            ]

            # Create a Poly3DCollection to represent the frustum
            poly3d_gt = Poly3DCollection(edges, linewidths=1, edgecolors=[1, 0, 0, 0.5], facecolors=[0, 0, 0, 0], label='Ground Truth')
            ax.add_collection(poly3d_gt)
        
        for i in range(0, est_frustums.shape[0], 5):
            frustum_vertices = est_frustums[i:i+5]
            edges = [
            [frustum_vertices[4], frustum_vertices[1], frustum_vertices[2], frustum_vertices[3], frustum_vertices[4]],
            [frustum_vertices[0], frustum_vertices[1]],
            [frustum_vertices[0], frustum_vertices[2]],
            [frustum_vertices[0], frustum_vertices[3]],
            [frustum_vertices[0], frustum_vertices[4]]
            ]

            # Create a Poly3DCollection to represent the frustum
            poly3d_est = Poly3DCollection(edges, linewidths=1, edgecolors=[0, 0, 1, 0.5], facecolors=[0, 0, 0, 0], label='Refined Poses')
            ax.add_collection(poly3d_est)

        # Set labels
        ax.set_xlabel('X (m)', labelpad=15)
        ax.set_ylabel('Y (m)', labelpad=15)
        ax.set_zlabel('Z (m)', labelpad=15)

        ax.set_xlim(-60, 10)
        ax.set_ylim(80, 150)
        ax.set_zlim(-15, 55)
        ax.set_box_aspect([1, 1, 1])
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Adjust the spacing of the tick labels
        ax.tick_params(axis='x', pad=10)
        ax.tick_params(axis='y', pad=10)
        ax.tick_params(axis='z', pad=10)
        
        ax.legend(handles=[poly3d_gt, poly3d_est], loc='upper right')
        ax.view_init(elev=0, azim=60, roll=90)
        plt.show()

    if init_pose is not None:
        init_points, init_lines, init_color = draw_camera_trajectory(c2ws_init.cpu().numpy(), cmp_traj_color)
        init_traj = o3d.geometry.LineSet()
        init_traj.points = o3d.utility.Vector3dVector(init_points)
        init_traj.lines = o3d.utility.Vector2iVector(init_lines)
        init_traj.colors = o3d.utility.Vector3dVector(init_color)
        geometry_to_draw.append(init_traj)

    # o3d.visualization.draw_geometries(geometry_to_draw)

    # uncomment below to rotate trajectories and save as gif
    # custom_draw_geometry_with_camera_trajectory(geometry_to_draw)
    # from PIL import Image
    # frames = [Image.open(os.path.join(pose_folder,x)) for x in sorted(os.listdir(pose_folder)) if '.png' in x]
    # frames[0].save(os.path.join(pose_folder, 'poses.gif'), save_all=True, append_images=frames[1:], loop=0)

    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(1.0, 0.0)
        return False

    # o3d.visualization.draw_geometries_with_animation_callback(geometry_to_draw, rotate_view)

    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    for p in geometry_to_draw:
        viewer.add_geometry(p)
    viewer.run()
