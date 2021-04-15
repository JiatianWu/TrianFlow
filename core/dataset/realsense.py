import os, sys
import numpy as np
import imageio
import cv2
import copy
import h5py
import scipy.io as sio
import torch
import torch.utils.data
import pdb
from tqdm import tqdm
import torch.multiprocessing as mp

def collect_image_list(path):
    # Get pgm images list of a folder.
    files = os.listdir(path)
    sorted_file = sorted([f for f in files])
    image_list = []
    for l in sorted_file:
        if l.split('.')[-1] == 'pgm':
            image_list.append(l)
    return image_list

def process_folder(q, data_dir, output_dir, stride):
    # Directly process the original nyu v2 depth dataset.
    while True:
        if q.empty():
            break
        folder = q.get()
        image_path = os.path.join(data_dir, folder)
        dump_image_path = os.path.join(output_dir, folder)
        if not os.path.isdir(dump_image_path):
            os.makedirs(dump_image_path)
        f = open(os.path.join(dump_image_path, 'train.txt'), 'w')
        
        # Note. the os.listdir method returns arbitary order of list. We need correct order.
        image_list = collect_image_list(image_path)
        numbers = len(image_list) - 1  # The last ppm file seems truncated.
        for n in range(numbers - stride):
            s_idx = n
            e_idx = s_idx + stride
            s_name = image_list[s_idx].strip()
            e_name = image_list[e_idx].strip()
            
            curr_image = imageio.imread(os.path.join(image_path, s_name))
            next_image = imageio.imread(os.path.join(image_path, e_name))
            seq_images = np.concatenate([curr_image, next_image], axis=0)
            imageio.imsave(os.path.join(dump_image_path,  os.path.splitext(s_name)[0]+'.png'), seq_images.astype('uint8'))

            # Write training files
            f.write('%s %s\n' % (os.path.join(folder, os.path.splitext(s_name)[0]+'.png'), 'calib_cam_to_cam.txt'))
        print(folder)

class RS_Prepare(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def __len__(self):
        raise NotImplementedError

    def prepare_data_mp(self, output_dir, stride=1):
        num_processes = 32
        processes = []
        q = mp.Queue()
        if not os.path.isfile(os.path.join(output_dir, 'train.txt')):
            os.makedirs(output_dir)
            print('Preparing sequence data....')
            if not os.path.isdir(self.data_dir):
                raise NotImplementedError
            dirlist = os.listdir(self.data_dir)
            total_dirlist = []
            # Get the different folders of images
            for d in dirlist:
                if not os.path.isdir(os.path.join(self.data_dir, d)):
                    continue
                seclist = os.listdir(os.path.join(self.data_dir, d))
                for s in seclist:
                    if os.path.isdir(os.path.join(self.data_dir, d, s)):
                        total_dirlist.append(os.path.join(d, s))
                        q.put(os.path.join(d, s))
            # Process every folder
            for rank in range(num_processes):
                p = mp.Process(target=process_folder, args=(q, self.data_dir, output_dir, stride))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
        
        # Collect the training frames.
        f = open(os.path.join(output_dir, 'train.txt'), 'w')
        for dirlist in os.listdir(output_dir):
            if os.path.isdir(os.path.join(output_dir, dirlist)):
                seclists = os.listdir(os.path.join(output_dir, dirlist))
                for s in seclists:
                    train_file = open(os.path.join(output_dir, dirlist, s, 'train.txt'), 'r')
                    for l in train_file.readlines():
                        f.write(l)
        f.close()
        
        f = open(os.path.join(output_dir, 'calib_cam_to_cam.txt'), 'w')
        f.write('P_rect: 615.67386784 0.0 336.54986191 0.0 0.0 615.12911225 241.57932892 0.0 0.0 0.0 1.0 0.0')
        f.close()
        print('Data Preparation Finished.')

    def __getitem__(self, idx):
        raise NotImplementedError



class RS(torch.utils.data.Dataset):
    def __init__(self, data_dir, num_scales=3, img_hw=(480, 640), num_iterations=None):
        super(RS, self).__init__()
        self.data_dir = data_dir
        self.num_scales = num_scales
        self.img_hw = img_hw
        self.num_iterations = num_iterations

        info_file = os.path.join(self.data_dir, 'train.txt')
        self.data_list = self.get_data_list(info_file)

    def get_data_list(self, info_file):
        with open(info_file, 'r') as f:
            lines = f.readlines()
        data_list = []
        for line in lines:
            k = line.strip('\n').split()
            data = {}
            data['image_file'] = os.path.join(self.data_dir, k[0])
            data['cam_intrinsic_file'] = os.path.join(self.data_dir, k[1])
            data_list.append(data)
        print('A total of {} image pairs found'.format(len(data_list)))
        return data_list

    def count(self):
        return len(self.data_list)

    def rand_num(self, idx):
        num_total = self.count()
        np.random.seed(idx)
        num = np.random.randint(num_total)
        return num

    def __len__(self):
        if self.num_iterations is None:
            return self.count()
        else:
            return self.num_iterations

    def resize_img(self, img, img_hw):
        '''
        Input size (N*H, W, 3)
        Output size (N*H', W', 3), where (H', W') == self.img_hw
        '''
        img_h, img_w = img.shape[0], img.shape[1]
        img_hw_orig = (int(img_h / 2), img_w) 
        img1, img2 = img[:img_hw_orig[0], :, :], img[img_hw_orig[0]:, :, :]
        img1_new = cv2.resize(img1, (img_hw[1], img_hw[0]))
        img2_new = cv2.resize(img2, (img_hw[1], img_hw[0]))
        img_new = np.concatenate([img1_new, img2_new], 0)
        return img_new

    def random_flip_img(self, img):
        is_flip = (np.random.rand() > 0.5)
        if is_flip:
            img = cv2.flip(img, 1)
        return img

    def preprocess_img(self, img, K, img_hw=None):
        if img_hw is None:
            img_hw = self.img_hw
            
        img = self.resize_img(img, img_hw)
        img = img / 255.0
        return img

    def read_cam_intrinsic(self, fname):
        with open(fname, 'r') as f:
            lines = f.readlines()
        data = lines[-1].strip('\n').split(' ')[1:]
        data = [float(k) for k in data]
        data = np.array(data).reshape(3,4)
        cam_intrinsics = data[:3,:3]
        return cam_intrinsics
    
    def rescale_intrinsics(self, K, img_hw_orig, img_hw_new):
        K_new = copy.deepcopy(K)
        K_new[0,:] = K_new[0,:] * img_hw_new[0] / img_hw_orig[0]
        K_new[1,:] = K_new[1,:] * img_hw_new[1] / img_hw_orig[1]
        return K_new

    def get_intrinsics_per_scale(self, K, scale):
        K_new = copy.deepcopy(K)
        K_new[0,:] = K_new[0,:] / (2**scale)
        K_new[1,:] = K_new[1,:] / (2**scale)
        K_new_inv = np.linalg.inv(K_new)
        return K_new, K_new_inv

    def get_multiscale_intrinsics(self, K, num_scales):
        K_ms, K_inv_ms = [], []
        for s in range(num_scales):
            K_new, K_new_inv = self.get_intrinsics_per_scale(K, s)
            K_ms.append(K_new[None,:,:])
            K_inv_ms.append(K_new_inv[None,:,:])
        K_ms = np.concatenate(K_ms, 0)
        K_inv_ms = np.concatenate(K_inv_ms, 0)
        return K_ms, K_inv_ms

    def __getitem__(self, idx):
        '''
        Returns:
        - img		torch.Tensor (N * H, W, 3)
        - K	torch.Tensor (num_scales, 3, 3)
        - K_inv	torch.Tensor (num_scales, 3, 3)
        '''
        if idx >= self.num_iterations:
            raise IndexError
        if self.num_iterations is not None:
            idx = self.rand_num(idx)
        data = self.data_list[idx]
        # load img
        img = cv2.imread(data['image_file'])
        img_hw_orig = (int(img.shape[0] / 2), img.shape[1])
        
        # load intrinsic
        cam_intrinsic_orig = self.read_cam_intrinsic(data['cam_intrinsic_file'])
        cam_intrinsic = self.rescale_intrinsics(cam_intrinsic_orig, img_hw_orig, self.img_hw)
        K_ms, K_inv_ms = self.get_multiscale_intrinsics(cam_intrinsic, self.num_scales) # (num_scales, 3, 3), (num_scales, 3, 3)
        
        # image preprocessing
        img = self.preprocess_img(img, cam_intrinsic_orig, self.img_hw) # (img_h * 2, img_w, 3)
        img = img.transpose(2,0,1)

        
        return torch.from_numpy(img).float(), torch.from_numpy(K_ms).float(), torch.from_numpy(K_inv_ms).float()

if __name__ == '__main__':
    data_dir = '/home4/zhaow/data/kitti'
    dirlist = os.listdir('/home4/zhaow/data/kitti')
    output_dir = '/home4/zhaow/data/kitti_seq/data_generated_s2'
    total_dirlist = []
    # Get the different folders of images
    for d in dirlist:
        seclist = os.listdir(os.path.join(data_dir, d))
        for s in seclist:
            if os.path.isdir(os.path.join(data_dir, d, s)):
                total_dirlist.append(os.path.join(d, s))
    
    F = open(os.path.join(output_dir, 'train.txt'), 'w')
    for p in total_dirlist:
        traintxt = os.path.join(os.path.join(output_dir, p), 'train.txt')
        f = open(traintxt, 'r')
        for line in f.readlines():
            F.write(line)
        print(traintxt)