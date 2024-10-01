from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import tarfile
import torch.nn.functional as F
import nibabel as nib
from datetime import datetime
# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array


def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def softmax_helper(x): return F.softmax(x, 1)


def tensor2im(image_tensor, imtype=np.uint8, normalize=False):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:
        image_numpy = image_numpy[:, :, 0]
    return image_numpy.astype(imtype)

# Converts a one-hot tensor into a colorful label map


def tensor2label(label_tensor, n_label, imtype=np.uint8):
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    return label_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

###############################################################################
# Code from
# https://github.com/ycszen/pytorch-seg/blob/master/transform.py
# Modified so it complies with the Citscape label map colors
###############################################################################


def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])


def labelcolormap(N):
    if N == 35:  # cityscape
        cmap = np.array([(0,  0,  0), (0,  0,  0), (0,  0,  0), (0,  0,  0), (0,  0,  0), (111, 74,  0), (81,  0, 81),
                         (128, 64, 128), (244, 35, 232), (250, 170, 160), (230,
                                                                           150, 140), (70, 70, 70), (102, 102, 156), (190, 153, 153),
                         (180, 165, 180), (150, 100, 100), (150, 120, 90), (153,
                                                                            153, 153), (153, 153, 153), (250, 170, 30), (220, 220,  0),
                         (107, 142, 35), (152, 251, 152), (70, 130, 180), (220,
                                                                           20, 60), (255,  0,  0), (0,  0, 142), (0,  0, 70),
                         (0, 60, 100), (0,  0, 90), (0,  0, 110), (0, 80, 100), (0,  0, 230), (119, 11, 32), (0,  0, 142)],
                        dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7-j))
                g = g ^ (np.uint8(str_id[-2]) << (7-j))
                b = b ^ (np.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap


class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image


def cal_dice(pred, tar, tag):
    return (2 * np.sum(np.multiply(pred == tag, tar == tag), axis=None) + 1e-5) / (
        np.sum(pred == tag, axis=None) + np.sum(tar == tag, axis=None) + 1e-5)


def is_source_file(x):
    if x.isdir() or x.name.endswith(('.py', '.sh', '.yml', '.json', '.txt', '.yaml')) \
            and '.mim' not in x.name and 'jobs/' not in x.name:
        # print(x.name)
        return x
    else:
        return None


def gen_code_archive(out_dir):
    time_now = datetime.now()
    file_name = 'code_' + time_now.strftime('%Y%m%d_%H%M%S') + '.tar.gz'
    archive = os.path.join(out_dir, file_name)
    with tarfile.open(archive, mode='w:gz') as tar:
        tar.add('.', filter=is_source_file)
    return archive


def _params_equal(ema_model, model):
    for ema_param, param in zip(ema_model.named_parameters(),
                                model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            # print("Difference in", ema_param[0])
            return False
    return True


def save_nii(vis_dict, vis_dir, epoch):
    for i in vis_dict.keys():
        tmp_img = np.squeeze(vis_dict[i].cpu().numpy())
        if len(tmp_img.shape) > 3:
            tmp_img = tmp_img[0]
        tmp = nib.Nifti1Image(tmp_img, np.eye(4))
        nib.save(tmp, os.path.join(vis_dir, i + '_' + str(epoch) + '.nii.gz'))


def norm_img(img, percentile=100):
    img = (img - np.min(img)) / (np.percentile(img, percentile) - np.min(img))
    return np.clip(img, 0, 1)


def _gen_indices(i1, i2, k, s):
    assert i2 >= k, 'sample size has to be bigger than the patch size'
    for j in range(i1, i2 - k + 1, s):
        yield j
        if j + k < i2:
            yield i2 - k


def get_bounds(img):
    # img: torchio.ScalarImage.data
    # return: idx, a list containing [x_min, x_max, y_min, y_max, z_min, z_max)
    try:
        img = np.squeeze(img.numpy())
    except:
        img = np.squeeze(img)
    nz_idx = np.nonzero(img)
    idx = []
    for i in nz_idx:
        idx.append(i.min())
        idx.append(i.max())

    return idx


def patch_slicer(scan, mask, patch_size, stride, remove_bg=True, test=False, ori_path=None):
    x, y, z = scan.shape
    scan_patches = []
    mask_patches = []
    if test:
        file_path = []
        patch_idx = []
    if remove_bg:
        bound = get_bounds(torch.from_numpy(scan))

        x1 = bound[0]
        x2 = bound[1]
        y1 = bound[2]
        y2 = bound[3]
        z1 = bound[4]
        z2 = bound[5]
    else:
        x1 = 0
        x2 = x
        y1 = 0
        y2 = y
        z1 = 0
        z2 = z
    p1, p2, p3 = patch_size
    s1, s2, s3 = stride

    if x2 - x1 < p1:
        if x2-p1 > 0:
            x1 = x2-p1
        else:
            x1 = 0
            x2 = p1
    if y2 - y1 < p2:
        if y2-p2 > 0:
            y1 = y2-p2
        else:
            y1 = 0
            y2 = p2
    if z2 - z1 < p3:
        if z2-p3 > 0:
            z1 = z2-p3
        else:
            z1 = 0
            z2 = p3

    x_stpes = _gen_indices(x1, x2, p1, s1)
    for x_idx in x_stpes:
        y_steps = _gen_indices(y1, y2, p2, s2)
        for y_idx in y_steps:
            z_steps = _gen_indices(z1, z2, p3, s3)
            for z_idx in z_steps:
                tmp_scan = scan[x_idx:x_idx + p1,
                                y_idx:y_idx + p2, z_idx:z_idx + p3]
                tmp_label = mask[x_idx:x_idx + p1,
                                 y_idx:y_idx + p2, z_idx:z_idx + p3]
                scan_patches.append(tmp_scan)
                mask_patches.append(tmp_label)
                if test:
                    file_path.append(ori_path)
                    patch_idx.append(
                        [x_idx, x_idx + p1, y_idx, y_idx + p2, z_idx, z_idx + p3])
    if not test:
        return scan_patches, mask_patches
    else:
        return scan_patches, file_path, patch_idx
