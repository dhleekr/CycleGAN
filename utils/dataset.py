import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, path, transforms_=None, unaligned=False, mode='train'):
        self.transform_ = transforms_
        self.unaligned = unaligned
        self.mode = mode

        self.files_A = sorted(glob.glob(os.path.join(path, '%sA' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(path, '%sB' % mode) + '/*.*'))

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

    def __getitem__(self, index):
        A = self.transform_(Image.open(self.files_A[index % len(self.files_A)]))
        
        if self.unaligned:
            B = self.transform_(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            B = self.transform_(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': A, 'B': B}
    
