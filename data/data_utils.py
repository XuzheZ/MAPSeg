import pandas as pd
import numpy as np
import os
import nibabel as nib
import torch
import torchio as tio


def list_mae_domains(path_to_mae_root):
    assert os.path.isdir(
        path_to_mae_root), '%s is not a directory' % path_to_mae_root
    candidate_dir = [os.path.join(path_to_mae_root, i) for i in os.listdir(
        path_to_mae_root) if i.endswith('_train')]

    assert len(
        candidate_dir) > 0, 'no folder ends with _train found in %s' % path_to_mae_root
    return candidate_dir


def list_finetune_domains(tgt_path, src_path):
    assert os.path.isdir(
        tgt_path), '%s is not a directory' % tgt_path
    candidate_dir = [os.path.join(tgt_path, i) for i in os.listdir(
        tgt_path) if i.endswith('_train')]
    assert len(
        candidate_dir) > 0, 'no folder ends with _train found in %s' % tgt_path

    assert os.path.isdir(
        src_path), '%s is not a directory' % src_path
    candidate_dir2 = [os.path.join(src_path, i) for i in os.listdir(
        src_path) if i.endswith('_img')]
    assert len(
        candidate_dir2) > 0, 'no folder ends with _img found in %s' % src_path
    return candidate_dir, candidate_dir2


def list_scans(path_to_fld, ext):

    assert os.path.isdir(path_to_fld), '%s is not a directory' % path_to_fld
    scans = []
    for root, _, fnames in sorted(os.walk(path_to_fld)):
        for fname in fnames:
            if fname.endswith(ext) and not fname.startswith('.'):
                scan_path = os.path.join(root, fname)
                scans.append(scan_path)

    return scans


def random_flip(img):
    # img: numpy ndarray

    tmp_odd1 = np.random.random_sample()
    tmp_odd2 = np.random.random_sample()
    tmp_odd3 = np.random.random_sample()

    # flip at 50% chance
    if tmp_odd1 <= 0.5:
        img = np.flip(img, axis=0)

    if tmp_odd2 <= 0.5:
        img = np.flip(img, axis=1)

    if tmp_odd3 <= 0.5:
        img = np.flip(img, axis=2)

    return img


def norm_img(img, percentile=100):
    img = (img - np.min(img)) / (np.percentile(img, percentile) - np.min(img))
    return np.clip(img, 0, 1)


def get_bounds(img):
    # img: torchio.ScalarImage.data / torch.tensor
    # return: idx, a list containing [x_min, x_max, y_min, y_max, z_min, z_max)
    img = np.squeeze(img.numpy())
    nz_idx = np.nonzero(img)
    idx = []
    for i in nz_idx:
        idx.append(i.min())
        idx.append(i.max())

    return idx
