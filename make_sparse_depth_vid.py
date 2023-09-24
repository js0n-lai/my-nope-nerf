import os
import imageio

root = 'data/V_KITTI/1_clone_turn'
paths = ['sparse_1_clone_turn_4_v7_sparse', 'sparse_1_clone_turn_4_v7_sparse_2', 'sparse_1_clone_turn_4_v7_sparse_3']

for p in paths:
    writer = imageio.get_writer(os.path.join(root, p, 'depth_in.mp4'), fps=6)
    frames = [os.path.join(root, p, x) for x in sorted(os.listdir(os.path.join(root, p))) if '.png' in x]
    for im in frames:
        writer.append_data(imageio.imread(im))
    writer.close()