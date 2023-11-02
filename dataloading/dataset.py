import os
import glob
import random
import logging
import torch
from PIL import Image
import numpy as np
import imageio
import cv2
import pdb
from dataloading.common import _load_data, recenter_poses, spherify_poses, load_depths_npz, load_gt_depths, vis_poses
logger = logging.getLogger(__name__)

class DataField(object):
    def __init__(self, model_path,
                 transform=None, 
                 with_camera=False, 
                with_depth=False,
                 use_DPT=False, scene_name=[' '], mode='train', spherify=False, 
                 load_ref_img=False,customized_poses=False,
                 customized_focal=False,resize_factor=2, depth_net='dpt',crop_size=0, 
                 random_ref=False,norm_depth=False,load_colmap_poses=True, sample_rate=8,
                 bd_factor=0.75, sparsify_depth=False, sparsify_depth_pattern=[1, 0, 1, 0],
                 out_dir=None, show_pose_only=False, **kwargs):
        """load images, depth maps, etc.
        Args:
            model_path (str): path of dataset
            transform (class, optional):  transform made to the image. Defaults to None.
            with_camera (bool, optional): load camera intrinsics. Defaults to False.
            with_depth (bool, optional): load gt depth maps (if available). Defaults to False.
            DPT (bool, optional): run DPT model. Defaults to False.
            scene_name (list, optional): scene folder name. Defaults to [' '].
            mode (str, optional): train/eval/all/render. Defaults to 'train'.
            spherify (bool, optional): spherify colmap poses (no effect to training). Defaults to False.
            load_ref_img (bool, optional): load reference image. Defaults to False.
            customized_poses (bool, optional): use GT pose if available. Defaults to False.
            customized_focal (bool, optional): use GT focal if provided. Defaults to False.
            resize_factor (int, optional): image downsample factor. Defaults to 2.
            depth_net (str, optional): which depth estimator use. Defaults to 'dpt'.
            crop_size (int, optional): crop if images have black border. Defaults to 0.
            random_ref (bool/int, optional): if use a random reference image/number of neaest images. Defaults to False.
            norm_depth (bool, optional): normalise depth maps. Defaults to False.
            load_colmap_poses (bool, optional): load colmap poses. Defaults to True.
            sample_rate (int, optional): 1 in 'sample_rate' images as test set. Defaults to 8.
        """
        self.transform = transform
        self.with_camera = with_camera
        self.with_depth = with_depth
        self.use_DPT = use_DPT
        self.mode = mode
        self.ref_img = load_ref_img
        self.random_ref = random_ref
        self.sample_rate = sample_rate
        
        load_dir = os.path.join(model_path, scene_name[0])
        if crop_size!=0:
            depth_net = depth_net + '_' + str(crop_size)
        poses, bds, imgs, img_names, crop_ratio, focal_crop_factor = _load_data(load_dir, factor=resize_factor, crop_size=crop_size, load_colmap_poses=load_colmap_poses, load_gt_llff=False)
        if load_colmap_poses:
            c2ws_colmap, H, W, focal, reverse_init = self.make_c2ws_from_llff(poses, bds, spherify, True, False, bd_factor)
            self.reverse_init = reverse_init

        imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
        imgs = np.transpose(imgs, (0, 3, 1, 2))
        _, _, h, w = imgs.shape
        if customized_focal:
            print(os.path.join(load_dir, 'intrinsics.npz'))
            focal_gt = np.load(os.path.join(load_dir, 'intrinsics.npz'))['K'].astype(np.float32)
            if resize_factor is None:
                resize_factor = 1
            fx = focal_gt[0, 0] / resize_factor
            fy = focal_gt[1, 1] / resize_factor
        else:
            if load_colmap_poses:
                fx, fy = focal, focal
            else:
                print('No focal provided, use image size as default')
                fx, fy = w, h
        fx = fx / focal_crop_factor
        fy = fy / focal_crop_factor
        
        self.H, self.W, self.focal = h, w, fx
        self.K = np.array([[2*fx/w, 0, 0, 0], 
            [0, -2*fy/h, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]]).astype(np.float32)
        ids = np.arange(imgs.shape[0])
        i_test = ids[int(sample_rate/2)::sample_rate]
        i_train = np.array([i for i in ids if i not in i_test])
        self.i_train = i_train
        self.i_test = i_test
        image_list_train = [img_names[i] for i in i_train]
        image_list_test = [img_names[i] for i in i_test]
        print('test set: ', image_list_test)

        if customized_poses:
            c2ws_gt = np.load(os.path.join(load_dir, 'gt_poses.npz'))['poses'].astype(np.float32)
            # T = torch.tensor(np.array([[1, 0, 0, 0],[0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float32)) # ScanNet coordinate
            c2ws_gt = torch.from_numpy(c2ws_gt)
            c2ws = c2ws_gt
            # c2ws = c2ws_gt @ T
        else:
            if load_colmap_poses:
                c2ws = c2ws_colmap
            else:
                c2ws = None
        
        # load preprocessed KITTI GT poses in LLFF format
        poses_gt, bds_gt, *other = _load_data(load_dir, factor=resize_factor, crop_size=crop_size, load_colmap_poses=False, load_gt_llff=True)
        c2ws_gt_llff, H_gt, W_gt, focal_gt, reverse_gt = self.make_c2ws_from_llff(poses_gt, bds_gt, spherify, False, False, bd_factor)
        self.c2ws_gt_llff = c2ws_gt_llff
        self.reverse_gt = reverse_gt

        if show_pose_only and c2ws is not None:
            import open3d as o3d
            from utils_poses.vis_cam_traj import draw_camera_frustum_geometry
            init = draw_camera_frustum_geometry(c2ws_colmap.cpu().numpy(), H, W, fx, fy, 0.1, np.array([29, 215, 158], dtype=np.float32) / 255, coord='opengl')
            gt = draw_camera_frustum_geometry(c2ws_gt_llff.cpu().numpy(), H, W, fx, fy, 0.1, np.array([255, 0, 0], dtype=np.float32) / 255, coord='opengl')
            viewer = o3d.visualization.Visualizer()
            viewer.create_window()
            viewer.add_geometry(init)
            viewer.add_geometry(gt)
            opt = viewer.get_render_option()
            opt.show_coordinate_frame = True

            viewer.run()
            exit()
        
        self.N_imgs_train = len(i_train)
        self.N_imgs_test = len(i_test)
        
        pred_depth_path = os.path.join(load_dir, depth_net)
        self.dpt_depth = None
        if mode in ('train','eval_trained', 'render'):
            idx_list = i_train
            self.img_list = image_list_train
        elif mode=='eval':
            idx_list = i_test
            self.img_list = image_list_test
        elif mode=='all':
            idx_list = ids
            self.img_list = img_names

        self.imgs = imgs[idx_list]
        self.N_imgs = len(idx_list)
        self.c2ws_gt_llff = self.c2ws_gt_llff[idx_list]
        if c2ws is not None:
            self.c2ws = c2ws[idx_list]
        if load_colmap_poses:
            self.c2ws_colmap = c2ws_colmap[i_train]
        try:
            self.gt_depth = load_gt_depths(self.img_list, load_dir, crop_ratio=crop_ratio, H=self.imgs.shape[-2], W=self.imgs.shape[-1])
        except AttributeError:
            self.gt_depth = None

        if not use_DPT and not with_depth:
            self.dpt_depth = load_depths_npz(self.img_list, pred_depth_path, norm=norm_depth)
            self.depth_mask = np.ones(self.dpt_depth.shape, dtype=bool)
        elif with_depth:
            self.depth = load_gt_depths(self.img_list, load_dir, crop_ratio=crop_ratio, H=self.imgs.shape[-2], W=self.imgs.shape[-1], reverse=reverse_gt)

            # black out depth pixels according to pattern [x_retain, x_skip, y_retain, y_skip]
            if sparsify_depth:
                self.depth_mask = self.sparsify_depths(sparsify_depth_pattern, os.path.join(load_dir, f'sparse_{out_dir}'), self.img_list)
                self.sparsify_depths(sparsify_depth_pattern, os.path.join(load_dir, f'sparse_{out_dir}'), self.img_list)
            else:
                self.depth_mask = np.ones(self.depth.shape, dtype=bool)

    def sparsify_depths(self, pattern, sparse_dir, img_list):
        N, H, W = self.depth.shape
        mask = np.ones(self.depth.shape, dtype=bool)
        x_mask = [True] * pattern[0] + [False] * pattern[1]
        y_mask = [True] * pattern[2] + [False] * pattern[3]
        y = 0

        for h in range(H):
            x = 0
            for w in range(W):
                self.depth[:,h,w] *= x_mask[x] and y_mask[y]
                mask[:,h,w] = x_mask[x] and y_mask[y]
                x = (x + 1) % len(x_mask)
            y = (y + 1) % len(y_mask)
        
        os.makedirs(sparse_dir, exist_ok=True)
        for i in range(N):
            depth_img = (np.clip(255.0 / self.depth[i].max() * (self.depth[i] - self.depth[i].min()), 0, 255)).astype(np.uint8)
            imageio.imwrite(os.path.join(sparse_dir, img_list[i]), depth_img)
        
        return mask

    def make_c2ws_from_llff(self, poses, bds, spherify, overwrite_hwf=False, visualise=False, bd_factor=0.75):
        if visualise:
            vis_poses(poses.transpose([2, 0, 1]), 'just loaded', 1)

        poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
        if visualise:
            vis_poses(poses.transpose([2, 0, 1]), 'transform from (x,y,z) --> (y, -x, z)', 1)

        poses = np.moveaxis(poses, -1, 0).astype(np.float32)
        if visualise:
            vis_poses(poses, 'move axis', 1)

        bds = np.moveaxis(bds, -1, 0).astype(np.float32)

        # Rescale if bd_factor is provided
        sc = 1. if bd_factor == 'None' else 1./(bds.min() * bd_factor)
        poses[:,:3,3] *= sc
        if visualise:
            vis_poses(poses, f'rescaled t by {sc}', 1)

        bds *= sc
        poses, poses_avg = recenter_poses(poses)
        if visualise:
            vis_poses(poses, 'recentered', 1)

        reverse = dict()
        reverse['sc'] = sc
        reverse['recenter'] = poses_avg

        if spherify:
            poses, render_poses, bds, sc_spherify, c2w_spherify = spherify_poses(poses, bds)
            reverse['sc_spherify'] = sc_spherify
            reverse['c2w_spherify'] = c2w_spherify
            if visualise:
                vis_poses(poses, 'spherify')
            
        input_poses = poses.astype(np.float32)
        hwf = input_poses[0,:3,-1]
        if overwrite_hwf:
            self.hwf = input_poses[:,:3,:]
        input_poses = input_poses[:,:3,:4]
        H, W, focal = hwf
        H, W = int(H), int(W)
        poses_tensor = torch.from_numpy(input_poses)
        bottom = torch.FloatTensor([0, 0, 0, 1]).unsqueeze(0)
        bottom = bottom.repeat(poses_tensor.shape[0], 1, 1)
        c2ws = torch.cat([poses_tensor, bottom], 1)

        return c2ws, H, W, focal, reverse
       

    def load(self, input_idx_img=None):
        ''' Loads the field.
        '''
        return self.load_field(input_idx_img)

    def load_image(self, idx, data={}):
        image = self.imgs[idx]
        data[None] = image
        if self.use_DPT:
            data_in = {"image": np.transpose(image, (1, 2, 0))}
            data_in = self.transform(data_in)
            data['normalised_img'] = data_in['image']
        data['idx'] = idx

    def load_ref_img(self, idx, data={}):
        if self.random_ref:
            if idx==self.N_imgs-1:
                ref_idx = idx-1
            else:
                ran_idx = random.randint(1, min(self.random_ref, self.N_imgs-idx-1))
                ref_idx = idx + ran_idx
        image = self.imgs[ref_idx]

        if self.dpt_depth is not None:
            dpt = self.dpt_depth[ref_idx]
            data['ref_dpts'] = dpt
            
        elif self.with_depth:
            depth = self.depth[ref_idx]
            data['ref_depths'] = depth
        
        if self.use_DPT:
            data_in = {"image": np.transpose(image, (1, 2, 0))}
            data_in = self.transform(data_in)
            normalised_ref_img = data_in['image']
            data['normalised_ref_img'] = normalised_ref_img
        
        data['ref_imgs'] = image
        data['ref_idxs'] = ref_idx
        data['ref_pose_gt'] = self.c2ws_gt_llff[ref_idx]


    def load_depth(self, idx, data={}):
        depth = self.depth[idx]
        data['depth'] = depth
        data['depth_mask'] = self.depth_mask[idx]
    
    def load_gt_depth(self, idx, data={}):
        data['gt_depths'] = self.gt_depth[idx]

    def load_DPT_depth(self, idx, data={}):
        depth_dpt = self.dpt_depth[idx]
        data['dpt'] = depth_dpt
        data['depth_mask'] = self.depth_mask[idx]

    def load_camera(self, idx, data={}):
        data['camera_mat'] = self.K
        data['scale_mat'] = np.array([[1, 0, 0, 0], [0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]]).astype(np.float32)
        data['idx'] = idx
    
    def load_gt_pose(self, idx, data={}):
        data['pose_gt'] = self.c2ws_gt_llff[idx]
        
    def load_field(self, input_idx_img=None):
        if input_idx_img is not None:
            idx_img = input_idx_img
        else:
            idx_img = 0
        # Load the data
        data = {}
        if not self.mode =='render':
            self.load_image(idx_img, data)
            self.load_gt_pose(idx_img, data)
            self.load_gt_depth(idx_img, data)
            if self.ref_img:
                self.load_ref_img(idx_img, data)
            if self.with_depth:
                self.load_depth(idx_img, data)
            elif self.dpt_depth is not None:
                self.load_DPT_depth(idx_img, data)
        if self.with_camera:
            self.load_camera(idx_img, data)
        
        return data





