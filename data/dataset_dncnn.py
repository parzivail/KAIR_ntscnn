import os.path
import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util


class DatasetDnCNN(data.Dataset):
    """
    # -----------------------------------------
    # Get L/H for denosing on AWGN with fixed sigma.
    # Only dataroot_H is needed.
    # -----------------------------------------
    # e.g., DnCNN
    # -----------------------------------------
    """

    def __init__(self, opt):
        super(DatasetDnCNN, self).__init__()
        print('Dataset: Denosing on AWGN with fixed sigma. Only dataroot_H is needed.')
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = opt['H_size'] if opt['H_size'] else 64
        self.sigma = opt['sigma'] if opt['sigma'] else 25
        self.sigma_test = opt['sigma_test'] if opt['sigma_test'] else self.sigma

        # ------------------------------------
        # get path of H
        # return None if input is None
        # ------------------------------------
        self.paths_H = util.get_image_paths(opt['dataroot_H'])
        self.paths_L = util.get_image_paths(opt['dataroot_L'])

    def __getitem__(self, index):

        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)
        L_path = self.paths_L[index]
        img_L = util.imread_uint(L_path, self.n_channels)

        if self.opt['phase'] == 'train':
            """
            # --------------------------------
            # get L/H patch pairs
            # --------------------------------
            """
            # H, W, _ = img_H.shape

            # Limit patch to within image area to prevent overfitting metadata border
            H = 414
            W = 583
            dX = 129
            dY = 32

            # --------------------------------
            # randomly crop the patch
            # --------------------------------
            rnd_h = dY + random.randint(0, max(0, H - self.patch_size))
            rnd_w = dX + random.randint(0, max(0, W - self.patch_size))

            # Only allow even Y offsets to allow learning interlacing
            # rnd_h = rnd_h - (rnd_h % 2)

            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            patch_L = img_L[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # --------------------------------
            # augmentation - flip, rotate
            # --------------------------------
            #mode = random.randint(0, 7)
            #patch_H = util.augment_img(patch_H, mode=mode)
            #patch_L = util.augment_img(patch_L, mode=mode)

            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_H = util.uint2tensor3(patch_H)
            img_L = util.uint2tensor3(patch_L)

        else:
            """
            # --------------------------------
            # get L/H image pairs
            # --------------------------------
            """
            img_H = util.uint2single(img_H)
            img_L = util.uint2single(img_L)

            # --------------------------------
            # HWC to CHW, numpy to tensor
            # --------------------------------
            img_L = util.single2tensor3(img_L)
            img_H = util.single2tensor3(img_H)

        return {'L': img_L, 'H': img_H, 'H_path': H_path, 'L_path': L_path}

    def __len__(self):
        return len(self.paths_H)
