
# Extract images from ScanNet .sens files
# Code mostly copied from SensReader in https://github.com/ScanNet/ScanNet/

import argparse
import os, struct, sys
import numpy as np
import shutil
from tqdm import tqdm
import zlib
import imageio.v2 as imageio
import cv2

COMPRESSION_TYPE_COLOR = {-1:'unknown', 0:'raw', 1:'png', 2:'jpeg'}
COMPRESSION_TYPE_DEPTH = {-1:'unknown', 0:'raw_ushort', 1:'zlib_ushort', 2:'occi_ushort'}

class RGBDFrame():

  def load(self, file_handle):
    self.camera_to_world = np.asarray(struct.unpack('f'*16, file_handle.read(16*4)), dtype=np.float32).reshape(4, 4)
    self.timestamp_color = struct.unpack('Q', file_handle.read(8))[0]
    self.timestamp_depth = struct.unpack('Q', file_handle.read(8))[0]
    self.color_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
    self.depth_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
    self.color_data = b''.join(struct.unpack('c'*self.color_size_bytes, file_handle.read(self.color_size_bytes)))
    self.depth_data = b''.join(struct.unpack('c'*self.depth_size_bytes, file_handle.read(self.depth_size_bytes)))

  def decompress_depth(self, compression_type):
    if compression_type == 'zlib_ushort':
       return self.decompress_depth_zlib()
    else:
       raise

  def decompress_depth_zlib(self):
    return zlib.decompress(self.depth_data)

  def decompress_color(self, compression_type):
    if compression_type == 'jpeg':
       return self.decompress_color_jpeg()
    else:
       raise

  def decompress_color_jpeg(self):
    return imageio.imread(self.color_data)


class SensorData:

    def __init__(self, filename):
        self.version = 4
        self.load(filename)

    def load(self, filename):
        with open(filename, 'rb') as f:
            version = struct.unpack('I', f.read(4))[0]
            assert self.version == version
            strlen = struct.unpack('Q', f.read(8))[0]
            self.sensor_name = b''.join(struct.unpack('c'*strlen, f.read(strlen)))
            self.intrinsic_color = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
            self.extrinsic_color = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
            self.intrinsic_depth = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
            self.extrinsic_depth = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
            self.color_compression_type = COMPRESSION_TYPE_COLOR[struct.unpack('i', f.read(4))[0]]
            self.depth_compression_type = COMPRESSION_TYPE_DEPTH[struct.unpack('i', f.read(4))[0]]
            self.color_width = struct.unpack('I', f.read(4))[0]
            self.color_height =  struct.unpack('I', f.read(4))[0]
            self.depth_width = struct.unpack('I', f.read(4))[0]
            self.depth_height =  struct.unpack('I', f.read(4))[0]
            self.depth_shift =  struct.unpack('f', f.read(4))[0]
            num_frames =  struct.unpack('Q', f.read(8))[0]
            self.frames = []
            print('loading frames...')
            for i in tqdm(range(num_frames)):
                frame = RGBDFrame()
                frame.load(f)
                self.frames.append(frame)

    def save_mat_to_file(self, matrix, filename):
        with open(filename, 'w') as f:
            for line in matrix:
                np.savetxt(f, line[np.newaxis], fmt='%f')

    def export(self, output_path, image_size=None, frame_skip=1):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print('exporting', len(self.frames)//frame_skip, 'color frames to', output_path)
        for f in tqdm(range(0, len(self.frames), frame_skip)):
            color = self.frames[f].decompress_color(self.color_compression_type)
            if image_size is not None:
                color = cv2.resize(color, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
            imageio.imwrite(os.path.join(output_path, str(f) + '.jpg'), color)

        print('exporting camera intrinsics to', output_path)
        mat = self.intrinsic_color.copy()
        if image_size is not None:
            sw = image_size[0] / self.color_width
            sh = image_size[1] / self.color_height
            mat[0][0] *= sw
            mat[1][1] *= sh
            mat[0][2] *= sw
            mat[1][2] *= sh
        self.save_mat_to_file(mat, os.path.join(output_path, 'intrinsic_color.txt'))


    def export_color_images(self, output_path, image_size=None, frame_skip=1):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print('exporting', len(self.frames)//frame_skip, 'color frames to', output_path)
        for f in tqdm(range(0, len(self.frames), frame_skip)):
            color = self.frames[f].decompress_color(self.color_compression_type)
            if image_size is not None:
                color = cv2.resize(color, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
            imageio.imwrite(os.path.join(output_path, str(f) + '.jpg'), color)

    def export_intrinsics(self, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print('exporting camera intrinsics to', output_path)
        self.save_mat_to_file(self.intrinsic_color, os.path.join(output_path, 'intrinsic_color.txt'))
        self.save_mat_to_file(self.extrinsic_color, os.path.join(output_path, 'extrinsic_color.txt'))


def main():
    # params
    parser = argparse.ArgumentParser()
    # data paths
    parser.add_argument('--filename', required=True, help='path to sens file to read')
    parser.add_argument('--out_dir', required=True, help='path to output folder')
    parser.add_argument('--tmp_dir', help='optional temp dir')

    opt = parser.parse_args()
    print(opt)

    out_path = opt.tmp_dir or opt.out_dir
    sens_name = os.path.basename(opt.filename).replace('.sens', '')
    out_path = os.path.join(out_path, sens_name)

    in_path = opt.filename
    if opt.tmp_dir:
        shutil.copy(in_path, opt.tmp_dir)
        in_path = os.path.join(opt.tmp_dir, os.path.basename(opt.filename))

    if not os.path.exists(opt.out_dir):
        os.makedirs(opt.out_dir)
    if opt.tmp_dir and not os.path.exists(opt.tmp_dir):
        os.makedirs(opt.tmp_dir)

    # load the data
    print(f'loading {in_path}...')
    sd = SensorData(in_path)
    print('loaded!')

    # 640x478 is the correct input size for the NonCuboidRoom model:
    # 640w matches other input specs, and 478 matches input aspect ratio
    sd.export(out_path, image_size=(478, 640))

    if opt.tmp_dir:
        shutil.move(out_path, opt.out_dir)

if __name__ == '__main__':
    main()
