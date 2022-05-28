from glob import glob
from math import ceil
import os

import cv2
import numpy as np
import torchvision.transforms as tf
from PIL import Image
from torch.utils import data


class ScannetDataset(data.Dataset):
    """ Images extracted from Scannet sens files. 
    
    Images are returned in a stable order, and will be the same for
    a given combination of `files` and `skip`. """
    image_size = (640, 478)  # (w, h)

    def __init__(self, config, files='data/scannet/images', skip=60):
        self.config = config
        self.files = files
        self.skip = skip
        self.transforms = tf.Compose([
            tf.ToTensor(),
            # TODO: correct values?
            tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.filenames = []
        # Each dir visited is a scan (e.g., scan0123_00).
        # Each contains a BUNCH of images that are consecutive
        # frames of video, 1.jpg 2.jpg ... 15.jpg ... 5432.jpg.
        for dirpath, dirnames, filenames in os.walk(files):
            dirnames.sort()
            nums = sorted([
                int(f.replace('.jpg', ''))
                for f in filenames
                if f.endswith('.jpg')
            ])
            for i in nums[::self.skip]:
                self.filenames.append(os.path.join(dirpath, f'{i}.jpg'))
        self.Ks = {}  # {scene_name: ndarray (3,3)}
        self.Kinvs = {}  # {scene_name: ndarray (3,3)}

    def padimage(self, image):
        outsize = [self.image_size[1], self.image_size[0], 3]
        h, w = image.shape[0], image.shape[1]
        padimage = np.zeros(outsize, dtype=np.uint8)
        padimage[:h, :w] = image
        return padimage, outsize[0], outsize[1]

    def __getitem__(self, index):
        filename = self.filenames[index]
        in_dir = os.path.dirname(filename)
        scan_name = os.path.split(in_dir)[-1]
        intrinsics_filename = os.path.join(in_dir, 'intrinsic_color.txt')

        if scan_name not in self.Ks:
            K = np.zeros((3, 3), dtype=np.float32)
            with open(intrinsics_filename, 'r') as f:
                for i, line in enumerate(f):
                    if i >= 3:
                        break
                    K[i] = [float(s) for s in line.split()[:3]]
            self.Ks[scan_name] = K
            self.Kinvs[scan_name] = np.linalg.inv(K).astype(np.float32)

        img = Image.open(filename)
        assert img.size == self.image_size
        img = np.array(img)[:, :, [0, 1, 2]]
        img, inh, inw = self.padimage(img)
        img = self.transforms(img)

        return {'img': img}

    def __len__(self):
        return len(self.filenames)
