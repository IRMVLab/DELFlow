import os
import cv2
import shutil
import logging
import argparse
import torch.utils.data
import numpy as np
from tqdm import tqdm
import re
import sys


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default = '/data/FlyingThings3D_subset', help='Path to the FlyingThings3D subset')
parser.add_argument('--output_dir', required=False, default='/data/processed_flyingthings3d')
parser.add_argument('--max_depth', required=False, default=35.0)
parser.add_argument('--remove_occluded_points', action='store_true')
args = parser.parse_args()

def init_logging(filename=None, debug=False):
    logging.root = logging.RootLogger('DEBUG' if debug else 'INFO')
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] - %(message)s')

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logging.root.addHandler(stream_handler)

    if filename is not None:
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)

def load_fpm(filename):
    with open(filename, 'rb') as f:
        header = f.readline().rstrip()
        if header.decode("ascii") == 'PF':
            color = True
        elif header.decode("ascii") == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', f.readline().decode("ascii"))
        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            raise Exception('Malformed PFM header.')

        scale = float(f.readline().decode("ascii").rstrip())
        if scale < 0:  # little-endian
            endian = '<'
        else:
            endian = '>'  # big-endian

        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)

    return data

def disp2pc(disp, baseline, f, cx, cy, flow=None):
    h, w = disp.shape
    depth = baseline * f / (disp + 1e-5)

    xx = np.tile(np.arange(w, dtype=np.float32)[None, :], (h, 1))
    yy = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))

    if flow is None:
        x = (xx - cx) * depth / f
        y = (yy - cy) * depth / f
    else:
        x = (xx - cx + flow[..., 0]) * depth / f
        y = (yy - cy + flow[..., 1]) * depth / f

    pc = np.concatenate([
        x[:, :, None],
        y[:, :, None],
        depth[:, :, None],
    ], axis=-1)

    return pc

def load_flow(filepath):
    with open(filepath, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert (202021.25 == magic), 'Invalid .flo file: incorrect magic number'
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        flow = np.fromfile(f, np.float32, count=2 * w * h).reshape([h, w, 2])

    return flow


def save_flow_png(filepath, flow, mask=None, scale=64.0):
    assert flow.shape[2] == 2
    assert np.abs(flow).max() < 32767.0 / scale
    flow = flow * scale
    flow = flow + 32768.0

    if mask is None:
        mask = np.ones_like(flow)[..., 0]
    else:
        mask = np.float32(mask > 0)

    flow_img = np.concatenate([
        mask[..., None],
        flow[..., 1:2],
        flow[..., 0:1]
    ], axis=-1).astype(np.uint16)

    cv2.imwrite(filepath, flow_img)




class Preprocessor(torch.utils.data.Dataset):
    def __init__(self, input_dir, output_dir, split, max_depth, remove_occluded_points):
        super(Preprocessor, self).__init__()

        self.input_dir = input_dir
        self.output_dir = output_dir
        self.split = split
        self.max_depth = max_depth
        self.remove_occluded_points = remove_occluded_points

        self.indices = []
        for filename in os.listdir(os.path.join(input_dir, split, 'flow', 'left', 'into_future')):
            index = int(filename.split('.')[0])
            self.indices.append(index)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        np.random.seed(0)

        index1 = self.indices[i]
        index2 = index1 + 1

        # camera intrinsics
        baseline, f, cx, cy = 1.0, 1050.0, 479.5, 269.5

        # load data
        disp1 = -load_fpm(os.path.join(
            self.input_dir, self.split, 'disparity', 'left', '%07d.pfm' % index1
        ))
        disp2 = -load_fpm(os.path.join(
            self.input_dir, self.split, 'disparity', 'left', '%07d.pfm' % index2
        ))
        disp1_change = -load_fpm(os.path.join(
            self.input_dir, self.split, 'disparity_change', 'left', 'into_future', '%07d.pfm' % index1
        ))
        flow_2d = load_flow(os.path.join(
            self.input_dir, self.split, 'flow', 'left', 'into_future', '%07d.flo' % index1
        ))
        occ_mask_2d = cv2.imread(os.path.join(
            self.input_dir, self.split, 'flow_occlusions', 'left', 'into_future', '%07d.png' % index1
        ))
        occ_mask_2d = occ_mask_2d[..., 0] > 1

        if self.remove_occluded_points:
            pc1 = disp2pc(disp1, baseline, f, cx, cy)
            pc2 = disp2pc(disp1 + disp1_change, baseline, f, cx, cy, flow_2d)

            # apply non-occlusion mask
            noc_mask_2d = np.logical_not(occ_mask_2d)
            pc1, pc2 = pc1[noc_mask_2d], pc2[noc_mask_2d]

            # apply depth mask
            mask = np.logical_and(pc1[..., -1] < self.max_depth, pc2[..., -1] < self.max_depth)
            pc1, pc2 = pc1[mask], pc2[mask]

            # NaN check
            mask = np.logical_not(np.isnan(np.sum(pc1, axis=-1) + np.sum(pc2, axis=-1)))
            pc1, pc2 = pc1[mask], pc2[mask]

            # compute scene flow
            flow_3d = pc2 - pc1
            occ_mask_3d = np.zeros(len(pc1), dtype=np.bool)
        else:
            pc1 = disp2pc(disp1, baseline, f, cx, cy)
            pc2 = disp2pc(disp2, baseline, f, cx, cy)
            flow_3d = disp2pc(disp1 + disp1_change, baseline, f, cx, cy, flow_2d) - pc1 # h x w x 3


            x, y = np.meshgrid(np.arange(pc1.shape[1]), np.arange(pc1.shape[0]))
            coords = np.concatenate([y[:, :, None], x[:, :, None]], axis= -1) # h x w x 2
            
            # apply depth mask and NaN check
            mask1 = np.logical_and((pc1[..., -1] < self.max_depth), np.logical_not(np.isnan(np.sum(pc1, axis=-1) + np.sum(flow_3d, axis=-1))))            
            mask2 = np.logical_and((pc2[..., -1] < self.max_depth), np.logical_not(np.isnan(np.sum(pc2, axis=-1))))
            

            
            pc1, pc2, flow_3d, occ_mask_3d = pc1[mask1], pc2[mask2], flow_3d[mask1], occ_mask_2d[mask1]


        # save point clouds and occ mask
        np.savez(
            os.path.join(self.output_dir, self.split, 'pc', '%07d.npz' % index1),
            pc1=pc1, pc2=pc2
        )
        np.save(
            os.path.join(self.output_dir, self.split, 'occ_mask_3d', '%07d.npy' % index1),
            np.packbits(occ_mask_3d)
        )
        
        np.save(
            os.path.join(self.output_dir, self.split, 'pc1_mask', '%07d.npy' % index1),
            np.packbits(mask1)
        )
        
        np.save(
            os.path.join(self.output_dir, self.split, 'pc2_mask', '%07d.npy' % index1),
            np.packbits(mask2)
        )

        # mask regions moving extremely fast
        flow_mask = np.logical_and(np.abs(flow_2d[..., 0]) < 500, np.abs(flow_2d[..., 1]) < 500)
        flow_2d[np.logical_not(flow_mask)] = 0.0

        # save ground-truth flow
        save_flow_png(
            os.path.join(self.output_dir, self.split, 'flow_2d', '%07d.png' % index1),
            flow_2d, flow_mask
        )
        np.save(
            os.path.join(self.output_dir, self.split, 'flow_3d', '%07d.npy' % index1),
            flow_3d
        )

        return 0


def main():
    for split_idx, split in enumerate(['train', 'val']):
        
        if not os.path.exists(os.path.join(args.input_dir, split)):
            print(os.path.join(args.input_dir, split))
            continue

        logging.info('Processing "%s" split...' % split)

        os.makedirs(os.path.join(args.output_dir, split, 'pc'), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, split, 'flow_2d'), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, split, 'flow_3d'), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, split, 'pc1_mask'), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, split, 'pc2_mask'), exist_ok=True)

        if not os.path.exists(os.path.join(args.output_dir, split, 'image')):
            logging.info('Copying images...')
            shutil.copytree(
                src=os.path.join(args.input_dir, split, 'image_clean', 'left'),
                dst=os.path.join(args.output_dir, split, 'image')
            )

        if not os.path.exists(os.path.join(args.output_dir, split, 'occ_mask_2d')):
            logging.info('Copying occ_mask_2d...')
            shutil.copytree(
                src=os.path.join(args.input_dir, split, 'flow_occlusions', 'left', 'into_future'),
                dst=os.path.join(args.output_dir, split, 'occ_mask_2d')
            )

        logging.info('Generating point clouds...')
        preprocessor = Preprocessor(
            args.input_dir,
            args.output_dir,
            split,
            args.max_depth,
            args.remove_occluded_points,
        )
        preprocessor = torch.utils.data.DataLoader(dataset=preprocessor, num_workers=4)

        for i in tqdm(preprocessor):
            # print(i)
            # if i > 5:
            #     break
            pass


if __name__ == '__main__':
    init_logging()
    main()
    logging.info('All done.')
