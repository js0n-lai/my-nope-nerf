import os
import sys
import argparse
import torch
sys.path.append(os.path.join(sys.path[0], '..'))
from dataloading import get_dataloader, load_config
import model as mdl
import numpy as np
import pdb

from utils_poses.vis_cam_traj import draw_camera_frustum_geometry, draw_camera_trajectory
from utils_poses.align_traj import align_ate_c2b_use_a2b, align_ate_init_pose
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
# pdb.set_trace()
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
    # pdb.set_trace()
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
frustum_length = 0.1
est_traj_color = np.array([39, 125, 161], dtype=np.float32) / 255
est_traj_color_2 = np.array([125, 161, 39], dtype=np.float32) / 255
cmp_traj_color = np.array([249, 65, 68], dtype=np.float32) / 255

''' Recover metric scale by reverting LLFF preprocessing and align poses'''
c2ws_est_to_draw_align2cmp = learned_poses.clone()
revert_LLFF = True

if revert_LLFF:
    c2ws_est_metric = revert_to_metric(c2ws_est_to_draw_align2cmp, gt_reverse)
    # c2ws_est_metric = c2ws_est_to_draw_align2cmp.clone()
    c2ws_learned_metric = revert_to_metric(learned_poses, gt_reverse)
    # c2ws_learned_metric = learned_poses.clone()
    c2ws_gt_metric = revert_to_metric(gt_poses, gt_reverse)
    # c2ws_gt_metric = gt_poses.clone()
    c2ws_est_aligned = align_ate_init_pose(c2ws_est_metric, c2ws_gt_metric)
    # c2ws_est_aligned = align_ate_c2b_use_a2b(c2ws_est_metric, c2ws_gt_metric)  # (N, 4, 4)
    
    # c2ws_est_aligned = align_ate_c2b_use_a2b(learned_poses, gt_poses)  # (N, 4, 4)
    # c2ws_est_aligned = align_ate_init_pose(learned_poses, gt_poses)
    # # pdb.set_trace()
    c2ws_est_to_draw_align2cmp = c2ws_est_aligned
    # # compute ate
    ate = compute_ATE(c2ws_gt_metric.cpu().numpy(), c2ws_est_aligned.cpu().numpy())
    ate_t, ate_r = compute_ATE_v2(c2ws_gt_metric.cpu().numpy(), c2ws_est_aligned.cpu().numpy())
    print(f"V1: ate (m) = {ate:.6f}, V2: ate_t (m) = {ate_t:.6f}, ate_r (deg) = {(ate_r * 180/ np.pi):.6f}")
    # rpe_trans, rpe_rot = compute_rpe(gt_poses.cpu().numpy(), c2ws_est_aligned.cpu().numpy())
    rpe_trans, rpe_rot = compute_rpe(c2ws_gt_metric.cpu().numpy(), c2ws_learned_metric.cpu().numpy())
    print(f"RPE_t: {rpe_trans:.6f} RPE_r: {(rpe_rot * 180 / np.pi):.3f}")

    with open(os.path.join(out_dir, 'extraction', 'evaluation.txt'), 'a') as f:
        f.write(f"\nV1: ate (m) = {ate:.6f}, V2: ate_t (m) = {ate_t:.6f}, ate_r (deg) = {(ate_r * 180/ np.pi):.6f}\n")
        f.write(f"RPE_t: {rpe_trans:.6f} RPE_r: {(rpe_rot * 180 / np.pi):.3f}")
    # # save poses to file
    # pose_dest = os.path.join(out_dir, 'poses.npz')
    # np.savez(pose_dest, ids=field['img'].i_train, gt=gt_poses.cpu().numpy(), est=c2ws_est_aligned.cpu().numpy(), est_no_align=learned_poses.cpu().numpy())
    # # pdb.set_trace()

if args.vis:
    import open3d as o3d
    geometry_to_draw = []

    # blue
    # frustum_est_2_list = draw_camera_frustum_geometry(learned_poses.cpu().numpy(), H, W,
    #                                                 fx, fy,
    #                                                 frustum_length, est_traj_color_2, draw_now=False)
    # geometry_to_draw.append(frustum_est_2_list)

    # blue
    frustum_est_list = draw_camera_frustum_geometry(c2ws_est_to_draw_align2cmp.cpu().numpy(), H, W,
                                                    fx, fy,
                                                    frustum_length, est_traj_color, draw_now=False)
    geometry_to_draw.append(frustum_est_list)

    # red
    frustum_gt_list = draw_camera_frustum_geometry(c2ws_gt_metric.cpu().numpy(), H, W,
                                                        fx, fy,
                                                        frustum_length, cmp_traj_color, draw_now=False)
    geometry_to_draw.append(frustum_gt_list)

    # teal
    if init_pose is not None:
        c2ws_init = align_ate_init_pose(revert_to_metric(init_pose, gt_reverse), c2ws_gt_metric)
        # c2ws_init = init_pose.clone()
        init_traj_color = np.array([29, 215, 158], dtype=np.float32) / 255
        frustum_init_list = draw_camera_frustum_geometry(c2ws_init.cpu().numpy(), H, W,
                                                            fx, fy,
                                                            frustum_length, init_traj_color, draw_now=False)                                                 
        geometry_to_draw.append(frustum_init_list)
    
    '''o3d for line drawing'''
    t_est_list = c2ws_est_to_draw_align2cmp[:, :3, 3].cpu()
    t_cmp_list = c2ws_gt_metric[:, :3, 3].cpu()

    '''line set to note pose correspondence between two trajs'''
    line_points = torch.cat([t_est_list, t_cmp_list], dim=0).cpu().numpy()  # (2N, 3)
    line_ends = [[i, i+N_imgs] for i in range(N_imgs)]  # (N, 2) connect two end points.

    # unit_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=2)
    # unit_sphere = o3d.geometry.LineSet.create_from_triangle_mesh(unit_sphere)
    # unit_sphere.paint_uniform_color((0, 1, 0))

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector(line_ends)

    # geometry_to_draw.append(line_set)

    # line sets for camera trajectories
    est_points, est_lines, est_color = draw_camera_trajectory(c2ws_est_to_draw_align2cmp.cpu().numpy(), est_traj_color)
    gt_points, gt_lines, gt_color = draw_camera_trajectory(c2ws_gt_metric.cpu().numpy(), cmp_traj_color)
    
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=gt_points[0])

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

    if init_pose is not None:
        init_points, init_lines, init_color = draw_camera_trajectory(c2ws_init.cpu().numpy(), cmp_traj_color)
        init_traj = o3d.geometry.LineSet()
        init_traj.points = o3d.utility.Vector3dVector(init_points)
        init_traj.lines = o3d.utility.Vector2iVector(init_lines)
        init_traj.colors = o3d.utility.Vector3dVector(init_color)
        geometry_to_draw.append(init_traj)

    # geometry_to_draw.append(coord)
    # o3d.visualization.draw_geometries(geometry_to_draw)
    # custom_draw_geometry_with_camera_trajectory(geometry_to_draw)
    # from PIL import Image
    # frames = [Image.open(os.path.join(pose_folder,x)) for x in sorted(os.listdir(pose_folder)) if '.png' in x]
    # frames[0].save(os.path.join(pose_folder, 'poses.gif'), save_all=True, append_images=frames[1:], loop=0)

    # def rotate_view(vis):
    #     ctr = vis.get_view_control()
    #     ctr.rotate(1.0, 0.0)
    #     return False

    # o3d.visualization.draw_geometries_with_animation_callback(geometry_to_draw, rotate_view)

    viewer = o3d.visualization.Visualizer()
    viewer.create_window()

    for p in geometry_to_draw:
        viewer.add_geometry(p)
    opt = viewer.get_render_option()
    opt.show_coordinate_frame = True

    # # o3d.visualization.draw_geometries([frustums_geometry], )
    viewer.run()
    # o3d.visualization.draw_geometries(geometry_to_draw)

