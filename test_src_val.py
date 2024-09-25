import torch
import os
from cfg.default import get_cfg_defaults
from model.mpl_seg import EMA_MPL
from model.utils import util
import numpy as np
import pandas
import torchio as tio
import nibabel as nib
import medpy.metric.binary as mmb


def infer_single_scan(model, cfg, tmp_scans):
    pad_flag = False
    model.eval()
    x, y, z = cfg.data.patch_size
    if cfg.data.normalize:
        tmp_scans = util.norm_img(tmp_scans, cfg.data.norm_perc)
    if min(tmp_scans.shape) < min(x, y, z):
        x_ori_size, y_ori_size, z_ori_size = tmp_scans.shape
        pad_flag = True
        x_diff = x-x_ori_size
        y_diff = y-y_ori_size
        z_diff = z-z_ori_size
        tmp_scans = np.pad(tmp_scans, ((max(0, int(x_diff/2)), max(0, x_diff-int(x_diff/2))), (max(0, int(
            y_diff/2)), max(0, y_diff-int(y_diff/2))), (max(0, int(z_diff/2)), max(0, z_diff-int(z_diff/2)))), constant_values=1e-4)  # cant pad with 0s, otherwise the local and global patches wont be the same location

    pred = np.zeros((cfg.train.cls_num,) + tmp_scans.shape)
    tmp_norm = np.zeros((cfg.train.cls_num,) + tmp_scans.shape)

    scan_patches, _, tmp_idx = util.patch_slicer(tmp_scans, tmp_scans, cfg.data.patch_size,
                                                 (x - 16, y -
                                                     16, z - 16),
                                                 remove_bg=cfg.data.remove_bg, test=True, ori_path=None)
    bound = util.get_bounds(torch.from_numpy(tmp_scans))
    global_scan = torch.unsqueeze(torch.from_numpy(
        tmp_scans).to(dtype=torch.float), dim=0)

    '''
    Sliding window implementation to go through the whole scans
    '''
    for idx, patch in enumerate(scan_patches):
        ipt = torch.from_numpy(patch).to(dtype=torch.float).cuda()
        ipt = ipt.reshape((1, 1,) + ipt.shape)

        patch_idx = tmp_idx[idx]
        location = torch.zeros_like(
            torch.from_numpy(tmp_scans)).float()
        location = torch.unsqueeze(location, 0)
        location[:, patch_idx[0]:patch_idx[1], patch_idx[2]                 :patch_idx[3], patch_idx[4]:patch_idx[5]] = 1

        sbj = tio.Subject(one_image=tio.ScalarImage(
            tensor=global_scan[:, bound[0]:bound[1], bound[2]:bound[3], bound[4]:bound[5]]),
            a_segmentation=tio.LabelMap(
                tensor=location[:, bound[0]:bound[1], bound[2]:bound[3], bound[4]:bound[5]]))
        transforms = tio.transforms.Resize(target_shape=(x, y, z))
        sbj = transforms(sbj)
        down_scan = sbj['one_image'].data
        loc = sbj['a_segmentation'].data
        tmp_coor = util.get_bounds(loc)
        coordinates_A = np.array([np.floor(tmp_coor[0] / 4),
                                  np.ceil(tmp_coor[1] / 4),
                                  np.floor(tmp_coor[2] / 4),
                                  np.ceil(tmp_coor[3] / 4),
                                  np.floor(tmp_coor[4] / 4),
                                  np.ceil(tmp_coor[5] / 4)
                                  ]).astype(int)
        coordinates_A = torch.unsqueeze(
            torch.from_numpy(coordinates_A), 0)
        tmp_pred, _ = model(ipt, down_scan.cuda().reshape([1, 1, x, y, z]),
                            coordinates_A)

        patch_idx = (slice(0, cfg.train.cls_num),) + (
            slice(patch_idx[0], patch_idx[1]), slice(
                patch_idx[2], patch_idx[3]),
            slice(patch_idx[4], patch_idx[5]))
        pred[patch_idx] += torch.squeeze(
            tmp_pred).detach().cpu().numpy()
        tmp_norm[patch_idx] += 1

    pred[tmp_norm > 0] = (pred[tmp_norm > 0]) / \
        tmp_norm[tmp_norm > 0]
    sf = torch.nn.Softmax(dim=0)
    pred_vol = sf(torch.from_numpy(pred)).numpy()
    pred_vol = np.argmax(pred_vol, axis=0)
    if pad_flag:
        pred_vol = pred_vol[max(0, int(x_diff/2)): max(0, int(x_diff/2))+x_ori_size,
                            max(0, int(y_diff/2)): max(0, int(y_diff/2))+y_ori_size,
                            max(0, int(z_diff/2)): max(0, int(z_diff/2))+z_ori_size]
        assert pred_vol.shape == (
            x_ori_size, y_ori_size, z_ori_size), 'pred_vol shape must be the same as the original scan shape'
    return pred_vol


ckpt_dir = 'YOUR_CKPT_DIR'
proj_name = 'YOUR_PROJ_NAME'
exps_lst = [
    'YOUR_EXPS'
]

# if set as false, only save prediction (assuming no GT label existing)
is_test = True

if __name__ == '__main__':
    print('Start testing:')
    print('Ckpt dir: ', ckpt_dir)
    print('Project name: ', proj_name)
    print('Experiments: ', exps_lst)
    print('Is test: ', is_test)
    for i in exps_lst:
        print(i)
        exp_dir = os.path.join(ckpt_dir, proj_name, i)
        cfg = get_cfg_defaults()
        cfg.merge_from_file(os.path.join(exp_dir, 'train_cfg.yaml'))
        model = EMA_MPL(cfg)
        model.cuda()
        model.load_state_dict(torch.load(
            os.path.join(exp_dir, 'best_model.pth')), strict=True)

        data_dir = os.path.join(cfg.data.tgt_data, 'val')
        img_lst = [j for j in os.listdir(os.path.join(
            data_dir, 'img')) if j.endswith(cfg.data.extension)]
        if not os.path.exists(os.path.join(exp_dir, 'preds_src_val')):
            os.mkdir(os.path.join(exp_dir, 'preds_src_val'))
        for j in img_lst:
            print(j)
            test_img = nib.load(os.path.join(data_dir, 'img', j)).get_fdata()
            test_img = np.squeeze(test_img)
            test_data = test_img.copy()
            with torch.no_grad():
                pred_vol = infer_single_scan(model, cfg, test_img)
            pred_vol = pred_vol.astype(np.uint8)

            test_img = nib.load(os.path.join(data_dir, 'img', j))
            pred = nib.Nifti1Image(
                pred_vol, affine=test_img.affine, header=test_img.header)
            test_vol = nib.Nifti1Image(
                test_data, affine=test_img.affine, header=test_img.header)
            nib.save(pred, os.path.join(exp_dir, 'preds_src_val', j))
            nib.save(test_vol, os.path.join(exp_dir, 'preds_src_val',
                     j.replace(cfg.data.extension, '_IMG.nii.gz')))

        if is_test:
            dice_list = []
            for j in img_lst:

                pred = nib.load(os.path.join(
                    exp_dir, 'preds_src_val', j)).get_fdata()
                gt = nib.load(os.path.join(data_dir, 'label', j)).get_fdata()
                pred = np.round(np.squeeze(pred))
                gt = np.round(np.squeeze(gt))
                for idx in range(1, cfg.train.cls_num):
                    pred_test_data_tr = pred.copy()
                    pred_test_data_tr[pred_test_data_tr != idx] = 0

                    pred_gt_data_tr = gt.copy()
                    pred_gt_data_tr[pred_gt_data_tr != idx] = 0

                    dice_list.append(
                        mmb.dc(pred_test_data_tr, pred_gt_data_tr))
                gt = nib.load(os.path.join(data_dir, 'label', j))
                nib.save(gt, os.path.join(exp_dir, 'preds_src_val',
                         j.replace(cfg.data.extension, '_LABEL.nii.gz')))
            dice_arr = 100 * \
                np.reshape(dice_list, [len(img_lst), -1])

            dice_avg = np.mean(dice_arr, axis=1)
            dice_avg = np.expand_dims(dice_avg, axis=1)
            img_lst.append('Average')

            tmp_avg = np.mean(dice_arr, axis=0)
            dice_arr = np.concatenate(
                (dice_arr, np.expand_dims(tmp_avg, axis=0)), axis=0)

            tmp_avg = np.mean(dice_avg, axis=0)
            dice_avg = np.concatenate(
                (dice_avg, np.expand_dims(tmp_avg, axis=0)), axis=0)

            df = pandas.DataFrame(
                np.concatenate((np.expand_dims(img_lst, axis=1), dice_arr, dice_avg), axis=1))
            df.columns = ['img_name'] + ['dice_' + str(j)
                                         for j in range(1, cfg.train.cls_num)] + ['dice_avg']
            df.to_excel(os.path.join(
                exp_dir, 'test_result_'+i+'_srcval.xlsx'), index=False)
