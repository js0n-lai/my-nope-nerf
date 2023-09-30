import numpy as np
import imageio
import os
import cv2

root = 'out/V_KITTI'
paths = ["1_clone_turn_4_v7_2"]
# paths = ["1_clone_v2",
#     "1_clone_2_v4",
#     "1_clone_7_v2",
#     "1_clone_2_tc",
#     "1_clone_7_v2_sparse",
#     "1_clone_7_v2_sparse_2",
#     "1_clone_7_v2_sparse_3",
#     "1_clone_6",
#     "1_clone_3",
#     "1_clone_3_v2",
#     "1_clone_4_v3",
#     "1_clone_8_v2",
#     "1_clone_turn_2",
#     "1_clone_turn",
#     "1_clone_turn_v2",
#     "1_clone_turn_4_v8_2",
#     "1_clone_turn_4_v7_2",
#     "1_clone_turn_4_v8_3",
#     "1_clone_turn_4_v7_sparse",
#     "1_clone_turn_4_v7_sparse_2",
#     "1_clone_turn_4_v7_sparse_3",
#     "1_clone_turn_7",
#     "1_clone_turn_9",
#     "1_clone_turn_9_v2",
#     "1_clone_turn_8",
#     "1_clone_turn_6",
#     "6_clone_turn_3",
#     "6_clone_turn_4",
#     "6_clone_turn_5",
#     "6_clone_turn",
#     "6_clone_turn_2",
#     "6_clone_turn_tc"]
    # experiments below not run yet, above all done
    # "6_clone_turn_2_sparse",
    # "6_clone_turn_2_sparse_2",
    # "6_clone_turn_2_sparse_3",
    # "6_clone_turn_6",
    # "6_clone_turn_9",
    # "6_clone_turn_9_v2",
    # "6_clone_turn_8",
    # "6_clone_turn_7"]

for p in paths:
    print(f"--------------------------------------- {p} -------------------------")
    render_root = os.path.join(root, p, 'extraction', 'extracted_images', 'interp')
    depth_dir = os.path.join(render_root, 'depth_out')
    disp_dir = os.path.join(render_root, 'disp_out')
    num_paths = len([depth_dir for x in os.listdir(depth_dir) if '.npy' in x])
    depth_paths = [os.path.join(depth_dir, f'{x}.npy') for x in range(num_paths)]
    if not os.path.exists(disp_dir):
        os.mkdir(disp_dir)
    
    disps = []
    for d in depth_paths:
        base = os.path.splitext(os.path.basename(d))[0]
        depth = np.load(d)
        disp_out = 1 / depth
        disp_out = (np.clip(255.0 / disp_out.max() * (disp_out - disp_out.min()), 0, 255)).astype(np.uint8)
        # disp_out = (255 - disp_out).astype(np.uint8) # to reverse the colormap
        disp_out = cv2.applyColorMap(disp_out, cv2.COLORMAP_INFERNO)
        disp_out = disp_out[:,:,::-1]
        disps.append(disp_out)
        imageio.imwrite(os.path.join(disp_dir, f'{base}.png'), disp_out)
    disps = np.stack(disps, axis=0)
    imageio.mimwrite(os.path.join(render_root, 'video_out', 'disp.mp4'), disps, fps=30, quality=9)