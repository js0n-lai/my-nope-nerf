import os
import imageio
import argparse

def generate_video(path, dest):
    writer = imageio.get_writer(os.path.join(path, '..', dest), fps=6)
    frames = [os.path.join(path, x) for x in sorted(os.listdir(path)) if '.png' in x]

    for im in frames:
        writer.append_data(imageio.imread(im))
    writer.close()

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('path')
    args.add_argument('dest')

    args = args.parse_args()
    generate_video(args.path, args.dest)