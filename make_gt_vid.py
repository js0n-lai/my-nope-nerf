import os
import imageio
import cv2
import numpy as np

def make_disp(data):
    imgs = f'data/V_KITTI/{data}/depth'
    frames = [os.path.join(imgs, x) for x in sorted(os.listdir(imgs)) if '.png' in x]
    disp_path = f'data/V_KITTI/{data}/disp'
    if not os.path.exists(disp_path):
        os.mkdir(disp_path)

    writer = imageio.get_writer(os.path.join(imgs, '..', 'disp.mp4'), fps=6)
    for im in frames:
        depth = cv2.imread(im, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        disp = 1 / depth
        disp = (np.clip(255.0 / disp.max() * (disp - disp.min()), 0, 255)).astype(np.uint8)
        disp = cv2.applyColorMap(disp, cv2.COLORMAP_INFERNO)
        cv2.imwrite(os.path.join(disp_path, os.path.basename(im)), disp)
        writer.append_data(disp[:,:,::-1])
    writer.close()

if __name__ == "__main__":
    data = '6_clone_turn'
    imgs = f'data/V_KITTI/{data}/images'
    writer = imageio.get_writer(os.path.join(imgs, '..', 'img.mp4'), fps=6)
    frames = [os.path.join(imgs, x) for x in sorted(os.listdir(imgs)) if '.png' in x]

    for im in frames:
        writer.append_data(imageio.imread(im))
    writer.close()

    imgs = f'data/V_KITTI/{data}/depth'
    writer = imageio.get_writer(os.path.join(imgs, '..', 'depth.mp4'), fps=6)
    frames = [os.path.join(imgs, x) for x in sorted(os.listdir(imgs)) if '.png' in x]

    for im in frames:
        writer.append_data(imageio.imread(im))
    writer.close()

    make_disp(data)

    os.chdir(os.path.join(imgs, '..'))
    command = "mpv --lavfi-complex=\"[vid1][vid2]vstack[vo]\" depth.mp4 --external-file=img.mp4 -o out.mp4"
    os.system(command)

    command = "mpv --lavfi-complex=\"[vid1][vid2]vstack[vo]\" disp.mp4 --external-file=img.mp4 -o out1.mp4"
    os.system(command)

# frames[0].save(os.path.join(imgs, 'img_out.mp4'), save_all=True, append_images=frames[1:])

# imgs = 'data/V_KITTI/1_clone_turn/depths'
# frames = [Image.open(os.path.join(imgs,x)) for x in sorted(os.listdir(imgs)) if '.png' in x]
# frames[0].save(os.path.join(imgs, 'depth_out.mp4'), save_all=True, append_images=frames[1:], loop=0)
