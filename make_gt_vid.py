import os
import imageio
import subprocess

imgs = 'data/V_KITTI/1_clone_turn/images'
writer = imageio.get_writer(os.path.join(imgs, '..', 'img.mp4'), fps=6)
frames = [os.path.join(imgs, x) for x in sorted(os.listdir(imgs)) if '.png' in x]

for im in frames:
    writer.append_data(imageio.imread(im))
writer.close()

imgs = 'data/V_KITTI/1_clone_turn/depth'
writer = imageio.get_writer(os.path.join(imgs, '..', 'depth.mp4'), fps=6)
frames = [os.path.join(imgs, x) for x in sorted(os.listdir(imgs)) if '.png' in x]

for im in frames:
    writer.append_data(imageio.imread(im))
writer.close()

os.chdir(os.path.join(imgs, '..'))
command = "mpv --lavfi-complex=\"[vid1][vid2]vstack[vo]\" depth.mp4 --external-file=img.mp4 -o out.mp4"
os.system(command)

# frames[0].save(os.path.join(imgs, 'img_out.mp4'), save_all=True, append_images=frames[1:])

# imgs = 'data/V_KITTI/1_clone_turn/depths'
# frames = [Image.open(os.path.join(imgs,x)) for x in sorted(os.listdir(imgs)) if '.png' in x]
# frames[0].save(os.path.join(imgs, 'depth_out.mp4'), save_all=True, append_images=frames[1:], loop=0)
