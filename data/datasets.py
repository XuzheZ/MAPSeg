import torch.utils.data as data
import os
from .data_utils import *
import torchio as tio
import torch
import numpy as np


class mae_dataset(data.Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        # get all image paths (source and target)
        # folder should end with '_train'
        all_img = 0
        candidate_dir = list_mae_domains(cfg.data.mae_root)

        self.path_dic = {}
        for i in range(len(candidate_dir)):
            self.path_dic[str(i)] = sorted(
                list_scans(candidate_dir[i], self.cfg.data.extension))
            all_img += len(self.path_dic[str(i)])
        self.num_domain = len(candidate_dir)
        print('num of mae domains: '+str(self.num_domain))
        print('num of mae scans: ' + str(all_img))

        self.all_img = all_img

    def __getitem__(self, index):
        idx = int(np.random.random_sample() // (1 / self.num_domain))
        tmp_path = self.path_dic[str(idx)]
        index = np.random.randint(0, len(tmp_path))
        # index = index % len(tmp_path)
        # scans should be first warped to the RAS space
        # see preprocessing for more details
        tmp_scans = nib.load(tmp_path[index])
        tmp_scans = np.squeeze(tmp_scans.get_fdata())
        tmp_scans[tmp_scans < 0] = 0
        tmp_scans = random_flip(tmp_scans)

        # normalization
        # we do a little trick to normalize the scan at random percentile
        if self.cfg.data.normalize:
            if np.random.uniform() <= self.cfg.data.aug_prob:
                perc_dif = 100-self.cfg.data.norm_perc
                tmp_scans = norm_img(tmp_scans, np.random.uniform(
                    self.cfg.data.norm_perc-perc_dif, 100))
            else:
                tmp_scans = norm_img(tmp_scans, self.cfg.data.norm_perc)

        # whether to pad the image to match the patch size
        # and then cast to torch.tensor
        x, y, z = self.cfg.data.patch_size

        if min(tmp_scans.shape) < min(x, y, z):
            diff = min(x, y, z) - min(tmp_scans.shape)
            idx = tmp_scans.shape.index(min(tmp_scans.shape))
            if idx == 0:
                npad = ((int(np.round(diff / 2)), diff -
                        int(np.round(diff / 2))), (0, 0), (0, 0))
                tmp_scans = np.pad(tmp_scans, pad_width=npad,
                                   mode='constant', constant_values=0)
            elif idx == 1:
                npad = ((0, 0), (int(np.round(diff / 2)),
                        diff - int(np.round(diff / 2))), (0, 0))
                tmp_scans = np.pad(tmp_scans, pad_width=npad,
                                   mode='constant', constant_values=0)
            elif idx == 2:
                npad = ((0, 0), (0, 0), (int(np.round(diff / 2)),
                        diff - int(np.round(diff / 2))))
                tmp_scans = np.pad(tmp_scans, pad_width=npad,
                                   mode='constant', constant_values=0)
            tmp_scans = torch.unsqueeze(torch.from_numpy(tmp_scans), 0)
        else:
            tmp_scans = torch.unsqueeze(torch.from_numpy(tmp_scans), 0)
        _, x1, y1, z1 = tmp_scans.shape
        transforms = tio.RandomAffine(p=self.cfg.data.aug_prob, scales=(0.75, 1.5), degrees=40,
                                      isotropic=True,
                                      default_pad_value=0, image_interpolation='linear')
        tmp_scans = tio.ScalarImage(tensor=tmp_scans)
        tmp_scans = transforms(tmp_scans)

        # if remove_bg, the patch will only be sampled from the foreground (non-zero) region

        if self.cfg.data.remove_bg:
            bound = get_bounds(tmp_scans.data)
            if bound[1] - x > bound[0]:
                x_idx = np.random.randint(bound[0], bound[1] - x)
            else:
                if bound[1] - x >= 0:
                    x_idx = bound[1] - x
                else:
                    x_idx = int((x1-x)/2)
            if bound[3] - y > bound[2]:
                y_idx = np.random.randint(bound[2], bound[3] - y)
            else:
                if bound[3] - y >= 0:
                    y_idx = bound[3] - y
                else:
                    y_idx = int((y1-y)/2)

            if bound[5] - z > bound[4]:
                z_idx = np.random.randint(bound[4], bound[5] - z)
            else:
                if bound[5] - z >= 0:
                    z_idx = bound[5] - z
                else:
                    z_idx = int((z1-z)/2)
        else:
            # if not remove_bg, the patch will be sampled from the whole image
            bound = [0, x1, 0, y1, 0, z1]
            if x1 - x == 0:
                x_idx = 0
            else:
                x_idx = np.random.randint(0, x1 - x)
            if y1 - y == 0:
                y_idx = 0
            else:
                y_idx = np.random.randint(0, y1 - y)
            if z1 - z == 0:
                z_idx = 0
            else:
                z_idx = np.random.randint(0, z1 - z)
        # location indicates the sampled patch location
        location = torch.zeros_like(tmp_scans.data)
        location[:, x_idx:x_idx + x, y_idx:y_idx + y, z_idx:z_idx + z] = 1

        sbj = tio.Subject(one_image=tio.ScalarImage(tensor=tmp_scans.data[:, bound[0]:bound[1], bound[2]:bound[3],
                                                                          bound[4]:bound[5]]))
        transforms = tio.transforms.Resize(target_shape=(x, y, z))
        sbj = transforms(sbj)
        down_scan = sbj['one_image'].data

        input_dict = {'local_patch': tmp_scans.data[:, x_idx:x_idx + x, y_idx:y_idx + y, z_idx:z_idx + z],
                      'global_images': down_scan}

        return input_dict

    def __len__(self):
        # we used fixed 2000 as the number of samples in each epoch
        # one can choose max(2000, self.all_img)
        # return max(2000, self.all_img)
        return 2000


class mpl_dataset(data.Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        # get all image paths (source and target)
        # folder should end with '_train'

        # data from target domain, only img (folder name should end with '_train')

        tgt_dir, src_dir1 = list_finetune_domains(
            cfg.data.tgt_data, cfg.data.src_data)

        self.path_dic = {}
        for i in range(len(tgt_dir)):
            self.path_dic[str(i)] = sorted(
                list_scans(tgt_dir[i], self.cfg.data.extension))
        self.num_domain = len(tgt_dir)

        # data from source domain,  img + label (folder name should end with '_img' for img and '_label' for label)

        self.path_dic_B1 = {}
        self.path_dic_B2 = {}
        for i in range(len(src_dir1)):
            self.path_dic_B1[str(i)] = sorted(
                list_scans(src_dir1[i], self.cfg.data.extension))
            self.path_dic_B2[str(i)] = [i.replace(
                '_img', '_label') for i in self.path_dic_B1[str(i)]]

        self.num_domain_B = len(src_dir1)

        print('num of target domian: ' + str(self.num_domain))
        print('num of source domain: ' + str(self.num_domain_B))

    def __getitem__(self, index):
        idx = int(np.random.random_sample() // (1 / self.num_domain))
        tmp_path = self.path_dic[str(idx)]
        indexA = np.random.randint(0, len(tmp_path))

        idx = int(np.random.random_sample() // (1 / self.num_domain_B))
        tmp_path_B1 = self.path_dic_B1[str(idx)]
        tmp_path_B2 = self.path_dic_B2[str(idx)]

        indexB = np.random.randint(0, len(tmp_path_B1))
        x, y, z = self.cfg.data.patch_size
        '''
        getitem for training/validation
        '''

        '''
        load non-labeled data
        '''
        tmp_scansA = nib.load(tmp_path[indexA])

        tmp_scansA = np.squeeze(tmp_scansA.get_fdata())
        tmp_scansA[tmp_scansA < 0] = 0

        # normalization
        if self.cfg.data.normalize:
            if np.random.uniform() <= self.cfg.data.aug_prob:
                perc_dif = 100 - self.cfg.data.norm_perc
                tmp_scansA = norm_img(tmp_scansA, np.random.uniform(
                    self.cfg.data.norm_perc - perc_dif, 100))
            else:
                tmp_scansA = norm_img(tmp_scansA, self.cfg.data.norm_perc)
        # padding
        if min(tmp_scansA.shape) < min(x, y, z):
            diff = min(x, y, z) - min(tmp_scansA.shape)
            idx = tmp_scansA.shape.index(min(tmp_scansA.shape))
            if idx == 0:
                npad = ((int(np.round(diff / 2)), diff -
                        int(np.round(diff / 2))), (0, 0), (0, 0))
                tmp_scansA = np.pad(
                    tmp_scansA, pad_width=npad, mode='constant', constant_values=0)
            elif idx == 1:
                npad = ((0, 0), (int(np.round(diff / 2)),
                        diff - int(np.round(diff / 2))), (0, 0))
                tmp_scansA = np.pad(
                    tmp_scansA, pad_width=npad, mode='constant', constant_values=0)
            elif idx == 2:
                npad = ((0, 0), (0, 0), (int(np.round(diff / 2)),
                        diff - int(np.round(diff / 2))))
                tmp_scansA = np.pad(
                    tmp_scansA, pad_width=npad, mode='constant', constant_values=0)
            tmp_scansA = torch.unsqueeze(torch.from_numpy(tmp_scansA), 0)
        else:
            tmp_scansA = torch.unsqueeze(torch.from_numpy(tmp_scansA), 0)
        # augmentation
        _, x1, y1, z1 = tmp_scansA.shape
        if self.cfg.data.aug:
            transforms = tio.Compose([tio.RandomAffine(p=self.cfg.data.aug_prob, scales=(0.7, 1.3), degrees=30,
                                                       isotropic=True,
                                                       default_pad_value=0, image_interpolation='linear',
                                                       label_interpolation='nearest')

                                      ])
            tmp_scans = tio.ScalarImage(tensor=tmp_scansA)
            tmp_scans = transforms(tmp_scans)
        else:
            tmp_scans = tio.ScalarImage(tensor=tmp_scansA)
        # randomly select patch
        if self.cfg.data.remove_bg:
            bound = get_bounds(tmp_scans.data)
            if bound[1] - x > bound[0]:
                x_idx = np.random.randint(bound[0], bound[1] - x)
            else:
                if bound[1] - x >= 0:
                    x_idx = bound[1] - x
                else:
                    if bound[0] + x < x1:
                        x_idx = bound[0]
                    else:
                        x_idx = int((x1 - x) / 2)
            if bound[3] - y > bound[2]:
                y_idx = np.random.randint(bound[2], bound[3] - y)
            else:
                if bound[3] - y >= 0:
                    y_idx = bound[3] - y
                else:
                    if bound[2] + y < y1:
                        y_idx = bound[2]
                    else:
                        y_idx = int((y1 - y) / 2)

            if bound[5] - z > bound[4]:
                z_idx = np.random.randint(bound[4], bound[5] - z)
            else:
                if bound[5] - z >= 0:
                    z_idx = bound[5] - z
                else:
                    if bound[4] + z < z1:
                        z_idx = bound[4]
                    else:
                        z_idx = int((z1 - z) / 2)
        else:
            bound = [0, x1, 0, y1, 0, z1]
            if x1 - x == 0:
                x_idx = 0
            else:
                x_idx = np.random.randint(0, x1 - x)
            if y1 - y == 0:
                y_idx = 0
            else:
                y_idx = np.random.randint(0, y1 - y)
            if z1 - z == 0:
                z_idx = 0
            else:
                z_idx = np.random.randint(0, z1 - z)

        location = torch.zeros_like(tmp_scans.data).float()
        location[:, x_idx:x_idx + x, y_idx:y_idx + y, z_idx:z_idx + z] = 1

        sbj = tio.Subject(one_image=tio.ScalarImage(tensor=tmp_scans.data[:, bound[0]:bound[1], bound[2]:bound[3],
                                                                          bound[4]:bound[5]]), a_segmentation=tio.LabelMap(
            tensor=location[:, bound[0]:bound[1],
                            bound[2]:bound[3], bound[4]:bound[5]]
        ))
        transforms = tio.transforms.Resize(target_shape=(x, y, z))
        sbj = transforms(sbj)
        down_scan = sbj['one_image'].data
        locA = sbj['a_segmentation'].data

        tmp_coor = get_bounds(locA)
        coordinates_A = np.array([np.floor(tmp_coor[0] / 4),
                                  np.ceil(tmp_coor[1] / 4),
                                  np.floor(tmp_coor[2] / 4),
                                  np.ceil(tmp_coor[3] / 4),
                                  np.floor(tmp_coor[4] / 4),
                                  np.ceil(tmp_coor[5] / 4)
                                  ]).astype(int)

        patchA = tmp_scans.data[:, x_idx:x_idx + x,
                                y_idx:y_idx + y, z_idx:z_idx + z].float()
        downA = down_scan.float()

        '''
        load annotated data
        '''

        tmp_scans = nib.load(tmp_path_B1[indexB])
        tmp_scans = np.squeeze(tmp_scans.get_fdata())
        '''
        WARNING: HERE WE ONLY USE POSITIVE INTENSITY 
        FOR CT, USE PREPROCESSING TO turn negatives to positives 
        
        '''
        tmp_scans[tmp_scans < 0] = 0
        tmp_label = np.squeeze(
            np.round(nib.load(tmp_path_B2[indexB]).get_fdata()))
        assert tmp_scans.shape == tmp_label.shape, 'scan and label must have the same shape'

        if self.cfg.data.normalize:
            if np.random.uniform() <= self.cfg.data.aug_prob:
                perc_dif = 100 - self.cfg.data.norm_perc
                tmp_scans = norm_img(tmp_scans, np.random.uniform(
                    self.cfg.data.norm_perc - perc_dif, 100))
            else:
                tmp_scans = norm_img(tmp_scans, self.cfg.data.norm_perc)

        if min(tmp_scans.shape) < min(x, y, z):
            diff = min(x, y, z) - min(tmp_scans.shape)
            idx = tmp_scans.shape.index(min(tmp_scans.shape))
            if idx == 0:
                npad = ((int(np.round(diff / 2)), diff -
                        int(np.round(diff / 2))), (0, 0), (0, 0))
                tmp_scans = np.pad(
                    tmp_scans, pad_width=npad, mode='constant', constant_values=0)
                tmp_label = np.pad(
                    tmp_label, pad_width=npad, mode='constant', constant_values=0)

            elif idx == 1:
                npad = ((0, 0), (int(np.round(diff / 2)),
                        diff - int(np.round(diff / 2))), (0, 0))
                tmp_scans = np.pad(
                    tmp_scans, pad_width=npad, mode='constant', constant_values=0)
                tmp_label = np.pad(
                    tmp_label, pad_width=npad, mode='constant', constant_values=0)

            elif idx == 2:
                npad = ((0, 0), (0, 0), (int(np.round(diff / 2)),
                        diff - int(np.round(diff / 2))))
                tmp_scans = np.pad(
                    tmp_scans, pad_width=npad, mode='constant', constant_values=0)
                tmp_label = np.pad(
                    tmp_label, pad_width=npad, mode='constant', constant_values=0)

            tmp_scans = torch.unsqueeze(torch.from_numpy(tmp_scans), 0)
            tmp_label = torch.unsqueeze(torch.from_numpy(tmp_label), 0)

        else:
            tmp_scans = torch.unsqueeze(
                torch.from_numpy(tmp_scans.copy()), 0)
            tmp_label = torch.unsqueeze(
                torch.from_numpy(tmp_label.copy()), 0)

        _, x1, y1, z1 = tmp_scans.shape
        tmp_scans = tio.ScalarImage(tensor=tmp_scans)
        tmp_label = tio.LabelMap(tensor=tmp_label)
        sbj = tio.Subject(one_image=tmp_scans, a_segmentation=tmp_label)
        if self.cfg.data.aug:
            transforms = tio.Compose([tio.RandomAffine(p=self.cfg.data.aug_prob, scales=(0.7, 1.4), degrees=30,
                                                       isotropic=True,
                                                       default_pad_value=0, image_interpolation='linear',
                                                       label_interpolation='nearest'),
                                      tio.RandomBiasField(
                                      p=self.cfg.data.aug_prob),
                                      tio.RandomGamma(
                                      p=self.cfg.data.aug_prob, log_gamma=(-0.4, 0.4))
                                      ])
            sbj = transforms(sbj)
        tmp_scans = sbj['one_image'].data.float()
        tmp_label = sbj['a_segmentation'].data.float()

        if self.cfg.data.remove_bg:
            bound = get_bounds(tmp_scans.data)
            if bound[1] - x > bound[0]:
                x_idx = np.random.randint(bound[0], bound[1] - x)
            else:
                if bound[1] - x >= 0:
                    x_idx = bound[1] - x
                else:
                    if bound[0] + x < x1:
                        x_idx = bound[0]
                    else:
                        x_idx = int((x1 - x) / 2)
            if bound[3] - y > bound[2]:
                y_idx = np.random.randint(bound[2], bound[3] - y)
            else:
                if bound[3] - y >= 0:
                    y_idx = bound[3] - y
                else:
                    if bound[2] + y < y1:
                        y_idx = bound[2]
                    else:
                        y_idx = int((y1 - y) / 2)

            if bound[5] - z > bound[4]:
                z_idx = np.random.randint(bound[4], bound[5] - z)
            else:
                if bound[5] - z >= 0:
                    z_idx = bound[5] - z
                else:
                    if bound[4] + z < z1:
                        z_idx = bound[4]
                    else:
                        z_idx = int((z1 - z) / 2)
        else:
            bound = [0, x1, 0, y1, 0, z1]
            if x1 - x == 0:
                x_idx = 0
            else:
                x_idx = np.random.randint(0, x1 - x)
            if y1 - y == 0:
                y_idx = 0
            else:
                y_idx = np.random.randint(0, y1 - y)
            if z1 - z == 0:
                z_idx = 0
            else:
                z_idx = np.random.randint(0, z1 - z)

        location_B = torch.zeros_like(tmp_scans.data).float()
        location_B[:, x_idx:x_idx + x,
                   y_idx:y_idx + y, z_idx:z_idx + z] = 1

        sbj = tio.Subject(one_image=tio.ScalarImage(tensor=tmp_scans[:, bound[0]:bound[1], bound[2]:bound[3],
                                                                     bound[4]:bound[5]]),
                          a_segmentation=tio.LabelMap(
            tensor=location_B[:, bound[0]:bound[1], bound[2]:bound[3], bound[4]:bound[5]])
        )
        transforms = tio.transforms.Resize(target_shape=(x, y, z))
        sbj = transforms(sbj)
        down_scan = sbj['one_image'].data.float()
        locB = sbj['a_segmentation'].data

        tmp_coor = get_bounds(locB)
        sbj = tio.Subject(one_image=tio.ScalarImage(tensor=tmp_scans[:, bound[0]:bound[1], bound[2]:bound[3],
                                                                     bound[4]:bound[5]]),
                          a_segmentation=tio.LabelMap(
            tensor=tmp_label[:, bound[0]:bound[1], bound[2]:bound[3], bound[4]:bound[5]])
        )
        sbj = transforms(sbj)
        aux_label = sbj['a_segmentation'].data

        coordinates_B = np.array([np.floor(tmp_coor[0] / 4),
                                  np.ceil(tmp_coor[1] / 4),
                                  np.floor(tmp_coor[2] / 4),
                                  np.ceil(tmp_coor[3] / 4),
                                  np.floor(tmp_coor[4] / 4),
                                  np.ceil(tmp_coor[5] / 4)
                                  ]).astype(int)
        input_dict = {'imgB': tmp_scans[:, x_idx:x_idx + x, y_idx:y_idx + y, z_idx:z_idx + z],
                      'labelB': torch.squeeze(tmp_label[:, x_idx:x_idx + x, y_idx:y_idx + y, z_idx:z_idx + z]),
                      'label_B_aux': torch.squeeze(aux_label),
                      'downB': down_scan,
                      'cord_B': coordinates_B,
                      'imgA': patchA,
                      'downA': downA,
                      'cord_A': coordinates_A}

        return input_dict

    def __len__(self):

        # we used fixed 100 steps for each epoch in finetuning
        # THIS PARAM WAS NEVER TUNED
        return 100
