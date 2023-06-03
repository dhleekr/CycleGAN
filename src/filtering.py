import os
from PIL import Image

"""
read jpg files from a directory and remove the ones that have 1 channel
"""
def filter_jpg_files(path):
    files = os.listdir(path)
    for f in files:
        if f.endswith('.jpg'):
            img = Image.open(os.path.join(path, f))
            if img.mode != 'RGB':
                print(f)
                os.remove(os.path.join(path, f))


if __name__ == '__main__':
    filter_jpg_files('datasets/horse2zebra/trainA')
    filter_jpg_files('datasets/horse2zebra/trainB')
    filter_jpg_files('datasets/horse2zebra/testA')
    filter_jpg_files('datasets/horse2zebra/testB')
