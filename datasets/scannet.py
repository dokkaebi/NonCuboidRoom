from glob import glob
from math import ceil
import os

import cv2
import numpy as np
import torchvision.transforms as tf
from PIL import Image
from torch.utils import data


class ScannetDataset(data.Dataset):
    image_size = (640, 478)  # (w, h)

    def __init__(self, config, phase='test', files='data/scannet/images'):
        self.config = config
        self.phase = phase
        self.max_objs = config.max_objs
        self.transforms = tf.Compose([
            tf.ToTensor(),
            # TODO: correct values?
            tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.files = files
        self.filenames = glob(os.path.join(files, '**', '*.jpg'), recursive=True)
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
        ret = {'img': img}

        ret['intri'] = self.Ks[scan_name]
        ret['intri_inv'] = self.Kinvs[scan_name]

        oh, ow = ceil(inh / self.config.downsample), ceil(inw / self.config.downsample)
        x = np.arange(ow * 8)
        y = np.arange(oh * 8)
        xx, yy = np.meshgrid(x, y)
        xymap = np.stack([xx, yy], axis=2).astype(np.float32)
        oxymap = cv2.resize(xymap, (ow, oh), interpolation=cv2.INTER_LINEAR)
        oxy1map = np.concatenate([
            oxymap, np.ones_like(oxymap[:, :, :1])], axis=-1).astype(np.float32)
        ret['oxy1map'] = oxy1map

        ixymap = cv2.resize(xymap, (inw, inh), interpolation=cv2.INTER_LINEAR)
        ixy1map = np.concatenate([
            ixymap, np.ones_like(ixymap[:, :, :1])], axis=-1).astype(np.float32)
        ret['ixy1map'] = ixy1map
        ret['iseg'] = np.ones([inh, inw])
        ret['ilbox'] = np.zeros(20)

        return ret, filename

    def __len__(self):
        return len(self.filenames)
