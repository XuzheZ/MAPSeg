from .mae_cnn import MAE_CNN
from .mpl_segV2 import EMA_MPL
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
from .utils.visualizer import Visualizer
from .utils import util
import time
import os
import numpy as np
from collections import OrderedDict
import torchio as tio
import nibabel as nib
import random

model_zoo = {
    'mae': MAE_CNN,
    'mpl': EMA_MPL
}

optim_zoo = {
    # change the optimzier zoo if you use otehr optimizers
    'Adam': optim.Adam,
    'AdamW': optim.AdamW,
}


class mae_trainer(nn.Module):
    '''
    Solver for MAE pretraining
    There is no evaluation for MAE pretraining 
    There is also no inference for MAE solver
    '''

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = model_zoo[cfg.train.type](cfg)
        self.model.cuda()
        if self.cfg.model.pretrain_model is not None and self.cfg.model.load_pretrain:
            self.model.load_state_dict(torch.load(
                self.cfg.model.pretrain_model), strict=False)
            print('model initialized with pretrained weights: ' +
                  self.cfg.model.pretrain_model)
        self.optimizer = optim_zoo[cfg.train.optimizer](
            self.model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay, betas=cfg.train.betas)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=20, eta_min=1e-6)
        self.visualizer = Visualizer(cfg)
        self.vis_dir = os.path.join(
            cfg.system.ckpt_dir, cfg.system.project, cfg.system.exp_name, 'visulization')
        util.mkdirs(self.vis_dir)

        # we keep track of loss/val score here
        self.local_loss = []
        self.global_loss = []

    def _get_epoch(self):
        return len(self.local_loss)

    def _init_epoch(self):
        self.tmp_local_loss = 0
        self.tmp_global_loss = 0

    def _log_internal_epoch_res(self, steps):
        self.local_loss.append(self.tmp_local_loss/steps)
        self.global_loss.append(self.tmp_global_loss/steps)

    def _get_internal_loss(self):
        return {'avg_local_MSE': self.local_loss[-1], 'avg_global_MSE': self.global_loss[-1]}

    def train_step(self, data, epoch):
        self.model.train()
        local_loss, global_loss, local_pred, global_pred, local_mask, global_mask = \
            self.model.forward_train(
                data['local_patch'].float().cuda(), data['global_images'].float().cuda())
        self.loss_dict = dict(zip(['local_MSE', 'global_MSE', ],
                                  [local_loss.item(), global_loss.item()]))
        self.tmp_local_loss += local_loss.item()
        self.tmp_global_loss += global_loss.item()
        loss = local_loss + global_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.visual_dict = dict(zip(['loca_patch',  'local_mask', 'local_pred', 'global_scan', 'global_mask',  'global_pred'],
                                    [data['local_patch'],
                                     local_mask.detach(),
                                     local_pred.detach(),
                                     data['global_images'],
                                     global_mask.detach(),
                                     global_pred.detach()
                                     ]))

    def get_cur_loss(self):
        return self.loss_dict

    def print_cur_loss(self, epoch, epoch_iter, start_time):
        errors = {k: v if not isinstance(
            v, int) else v for k, v in self.loss_dict.items()}
        t = (time.time() - start_time) / self.cfg.train.print_freq
        self.visualizer.print_current_errors(
            epoch, epoch_iter, errors, t)

    def save_visualization(self, epoch):
        # save nifti first
        if self.cfg.system.save_nii:
            util.save_nii(self.visual_dict, self.vis_dir, epoch)
        # save images
        vis = OrderedDict()
        slc_num = self.cfg.data.patch_size[-1]//2
        for k, v in self.visual_dict.items():
            if 'mask' in k:
                vis[k] = util.tensor2label(v[0, :, :, :, slc_num], 2)
            else:
                vis[k] = util.tensor2im(v[0, :, :, :, slc_num])

        self.visualizer.display_current_results(vis, epoch)

    def scheduler_step(self):
        self.scheduler.step()
        print('Current Learning Rate changed to {}'.format(
            self.optimizer.param_groups[0]['lr']))


class mpl_trainer(nn.Module):
    '''
    Solver for Masked Pseudo Labeling

    '''

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # get the EMA model
        self.model = model_zoo[cfg.train.type](cfg)
        self.model.cuda()

        if self.cfg.model.pretrain_model is not None:
            if 'noMAE' in self.cfg.system.exp_name:
                self.model.load_state_dict(torch.load(
                    self.cfg.model.pretrain_model), strict=False)
                print('model initialized with pretrained segmentation weights (noMAE)')
            else:
                self.model.initialize_load()
        self.is_teacher_init = False

        self.optimizer = optim_zoo[cfg.train.optimizer](
            self.model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay, betas=cfg.train.betas)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-8)
        self.visualizer = Visualizer(cfg)
        self.vis_dir = os.path.join(
            cfg.system.ckpt_dir, cfg.system.project, cfg.system.exp_name, 'visulization')
        util.mkdirs(self.vis_dir)
        self.val_dir = os.path.join(
            cfg.system.ckpt_dir, cfg.system.project, cfg.system.exp_name, 'validation')
        util.mkdirs(self.val_dir)
        # we keep track of loss/val score here
        self.src_seg_loss = []
        self.src_seg_loss_masked = []
        self.src_seg_loss_aux = []
        self.src_seg_loss_aux_masked = []
        self.src_cos_reg = []
        self.src_cos_reg_masked = []
        self.tgt_pse_seg_loss = []
        self.tgt_pse_seg_loss_aux = []
        self.tgt_cos_reg = []
        self.val_score = []
        self.val_dice = []
        self.cumulative_no_improve = []
        self.total_step = 0

    def _get_epoch(self):
        return len(self.src_seg_loss)

    def _init_epoch(self):
        self.tmp_src_seg_loss = 0
        self.tmp_src_seg_loss_masked = 0
        self.tmp_src_seg_loss_aux = 0
        self.tmp_src_seg_loss_aux_masked = 0
        self.tmp_src_cos_reg = 0
        self.tmp_src_cos_reg_masked = 0
        self.tmp_tgt_pse_seg_loss = 0
        self.tmp_tgt_pse_seg_loss_aux = 0
        self.tmp_tgt_cos_reg = 0

    def _log_internal_epoch_res(self, steps):
        self.src_seg_loss.append(self.tmp_src_seg_loss/steps)
        self.src_seg_loss_masked.append(self.tmp_src_seg_loss_masked/steps)
        self.src_seg_loss_aux.append(self.tmp_src_seg_loss_aux/steps)
        self.src_seg_loss_aux_masked.append(
            self.tmp_src_seg_loss_aux_masked/steps)
        self.src_cos_reg.append(self.tmp_src_cos_reg/steps)
        self.src_cos_reg_masked.append(self.tmp_src_cos_reg_masked/steps)
        self.tgt_pse_seg_loss.append(self.tmp_tgt_pse_seg_loss/steps)
        self.tgt_pse_seg_loss_aux.append(self.tmp_tgt_pse_seg_loss_aux/steps)
        self.tgt_cos_reg.append(self.tmp_tgt_cos_reg/steps)

    def _get_internal_loss(self):
        return {
            'src_seg_loss': self.src_seg_loss[-1],
            'src_seg_loss_masked': self.src_seg_loss_masked[-1],
            'src_seg_loss_aux': self.src_seg_loss_aux[-1],
            'src_seg_loss_aux_masked': self.src_seg_loss_aux_masked[-1],
            'src_cos_reg': self.src_cos_reg[-1],
            'src_cos_reg_masked': self.src_cos_reg_masked[-1],
            'tgt_pse_seg_loss': self.tgt_pse_seg_loss[-1],
            'tgt_pse_seg_loss_aux': self.tgt_pse_seg_loss_aux[-1],
            'tgt_cos_reg': self.tgt_cos_reg[-1],
        }

    def train_step(self, data, epoch):
        torch.cuda.empty_cache()
        self.total_step += 1
        if epoch > self.cfg.train.warmup:
            self.model._update_ema(self.total_step)
        self.model.train()
        img_src, global_src, label_src, label_src_aux, cord_src, img_tgt, global_tgt, cord_tgt = \
            data['imgB'].float().cuda(), \
            data['downB'].float().cuda(), \
            data['labelB'].long().cuda(), \
            data['label_B_aux'].long().cuda(), \
            data['cord_B'], \
            data['imgA'].float().cuda(), \
            data['downA'].float().cuda(), \
            data['cord_A']
        if epoch <= self.cfg.train.warmup:
            # L_sup: seg_loss = L_Seg()
            # L_MPL: seg_loss_masked, pse_seg_loss
            # L_global: seg_loss_aux, seg_loss_aux_masked, cos_feat, cos_feat_masked, pse_seg_loss_aux, pse_cos_feat
            seg_loss, seg_loss_masked, seg_loss_aux, seg_loss_aux_masked, cos_feat, cos_feat_masked, pred_seg, pred_seg_masked, pred_aux, mask_seg = \
                self.model.train_source(cord_src, img_src, label_src,
                                        global_src, label_src_aux, self.cfg.train.mask_ratio)
            self.loss_dict = dict(zip(['src_seg_loss', 'src_seg_loss_masked', 'src_seg_loss_aux', 'src_seg_loss_aux_masked', 'src_cos_reg', 'src_cos_reg_masked'],
                                      [seg_loss.item(), seg_loss_masked.item(), seg_loss_aux.item(), seg_loss_aux_masked.item(), cos_feat.item(), cos_feat_masked.item()]))
            self.tmp_src_seg_loss += seg_loss.item()
            self.tmp_src_seg_loss_masked += seg_loss_masked.item()
            self.tmp_src_seg_loss_aux += seg_loss_aux.item()
            self.tmp_src_seg_loss_aux_masked += seg_loss_aux_masked.item()
            self.tmp_src_cos_reg += cos_feat.item()
            self.tmp_src_cos_reg_masked += cos_feat_masked.item()
            loss = (seg_loss + seg_loss_masked) * 0.5 + (seg_loss_aux + seg_loss_aux_masked) * 0.5 * 0.1 + (
                cos_feat + cos_feat_masked) * 0.5 * 0.05

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.visual_dict = dict(zip(['src_local', 'src_global', 'src_local_label', 'src_global_label',
                                         'src_local_pred', 'src_local_pred_masked', 'src_global_pred',
                                         'src_masked_map'],
                                        [img_src.detach(), global_src.detach(), label_src.detach(), label_src_aux.detach(),
                                         pred_seg.detach(), pred_seg_masked.detach(), pred_aux.detach(),
                                         mask_seg.detach()]))
        else:
            if not self.is_teacher_init:
                self.model._init_ema_weights()
                self.is_teacher_init = True

            if not self.cfg.train.test_time:

                seg_loss, seg_loss_aux, cos_feat, pred_seg, pred_aux = self.model.train_source_only1(cord_src, img_src, label_src,
                                                                                                     global_src, label_src_aux, self.cfg.train.mask_ratio)
                pseudo_label_loc_logit, pseudo_label_global_logit = self.model.get_pseudo_label(
                    img_tgt, global_tgt, cord_tgt)
                pseudo_label_loc = self.model.get_pseudo_label_and_weight(
                    pseudo_label_loc_logit)
                pseudo_label_global = self.model.get_pseudo_label_and_weight(
                    pseudo_label_global_logit)
                del pseudo_label_loc_logit, pseudo_label_global_logit

                # train on pseudo dataset
                pse_seg_loss, pse_seg_pred, pse_seg_loss_aux, pse_seg_pred_aux, pse_seg_mask, pse_cos_feat = \
                    self.model.train_pseudo(cord_tgt, img_tgt, pseudo_label_loc.long().cuda(), global_tgt, pseudo_label_global.long().cuda(),
                                            self.cfg.train.mask_ratio)

                loss = seg_loss + seg_loss_aux * 0.1 + \
                    cos_feat * 0.05 + \
                    (pse_seg_loss + 0.1 * pse_seg_loss_aux +
                        pse_cos_feat * 0.05)
                self.loss_dict = dict(zip(['src_seg_loss', 'src_seg_loss_aux', 'src_cos_reg',
                                           'tgt_pse_seg_loss', 'tgt_pse_seg_loss_aux', 'tgt_cos_reg'],
                                          [seg_loss.item(), seg_loss_aux.item(), cos_feat.item(),
                                           pse_seg_loss.item(), pse_seg_loss_aux.item(), pse_cos_feat.item()]))
                self.tmp_src_seg_loss += seg_loss.item()
                self.tmp_src_seg_loss_masked += 0
                self.tmp_src_seg_loss_aux += seg_loss_aux.item()
                self.tmp_src_seg_loss_aux_masked += 0
                self.tmp_src_cos_reg += cos_feat.item()
                self.tmp_src_cos_reg_masked += 0
                self.tmp_tgt_pse_seg_loss += pse_seg_loss.item()
                self.tmp_tgt_pse_seg_loss_aux += pse_seg_loss_aux.item()
                self.tmp_tgt_cos_reg += pse_cos_feat.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.visual_dict = dict(zip(['src_local', 'src_global', 'src_local_label', 'src_global_label',
                                            'src_local_pred',  'src_global_pred',
                                             'tgt_local', 'tgt_global', 'tgt_local_pse_label', 'tgt_global_pse_label',
                                             'tgt_local_pred', 'tgt_global_pred',
                                             'tgt_masked_map'],
                                            [img_src.detach(), global_src.detach(), label_src.detach(), label_src_aux.detach(),
                                            pred_seg.detach(), pred_aux.detach(),
                                            img_tgt.detach(), global_tgt.detach(
                                            ), pseudo_label_loc.detach(), pseudo_label_global.detach(),
                    pse_seg_pred.detach(), pse_seg_pred_aux.detach(), pse_seg_mask.detach()]))
            else:
                pseudo_label_loc_logit, pseudo_label_global_logit = self.model.get_pseudo_label(
                    img_tgt, global_tgt, cord_tgt)
                pseudo_label_loc = self.model.get_pseudo_label_and_weight(
                    pseudo_label_loc_logit)
                pseudo_label_global = self.model.get_pseudo_label_and_weight(
                    pseudo_label_global_logit)
                del pseudo_label_loc_logit, pseudo_label_global_logit

                # train on pseudo dataset
                pse_seg_loss1, pse_seg_pred, pse_seg_loss_aux1, pse_seg_pred_aux, pse_seg_mask, pse_cos_feat1 = \
                    self.model.train_pseudo(cord_tgt, img_tgt, pseudo_label_loc.long().cuda(), global_tgt, pseudo_label_global.long().cuda(),
                                            self.cfg.train.mask_ratio)
                pse_seg_loss2, _, pse_seg_loss_aux2, _, pse_cos_feat2 = \
                    self.model.train_pseudo(cord_tgt, img_tgt, pseudo_label_loc.long().cuda(), global_tgt, pseudo_label_global.long().cuda(),
                                            0)
                pse_seg_loss = (pse_seg_loss1 + pse_seg_loss2) * 0.5
                pse_seg_loss_aux = (pse_seg_loss_aux1 +
                                    pse_seg_loss_aux2) * 0.5
                pse_cos_feat = (pse_cos_feat1 + pse_cos_feat2) * 0.5

                loss = (pse_seg_loss + 0.1 * pse_seg_loss_aux +
                        pse_cos_feat * 0.05)
                self.loss_dict = dict(zip(['tgt_pse_seg_loss', 'tgt_pse_seg_loss_aux', 'tgt_cos_reg'],
                                          [pse_seg_loss.item(), pse_seg_loss_aux.item(), pse_cos_feat.item()]))
                self.tmp_src_seg_loss += 0
                self.tmp_src_seg_loss_masked += 0
                self.tmp_src_seg_loss_aux += 0
                self.tmp_src_seg_loss_aux_masked += 0
                self.tmp_src_cos_reg += 0
                self.tmp_src_cos_reg_masked += 0
                self.tmp_tgt_pse_seg_loss += pse_seg_loss.item()
                self.tmp_tgt_pse_seg_loss_aux += pse_seg_loss_aux.item()
                self.tmp_tgt_cos_reg += pse_cos_feat.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.visual_dict = dict(zip(['tgt_local', 'tgt_global', 'tgt_local_pse_label', 'tgt_global_pse_label',
                                             'tgt_local_pred', 'tgt_global_pred',
                                             'tgt_masked_map'],
                                            [img_tgt.detach(), global_tgt.detach(
                                            ), pseudo_label_loc.detach(), pseudo_label_global.detach(),
                    pse_seg_pred.detach(), pse_seg_pred_aux.detach(), pse_seg_mask.detach()]))

    def get_cur_loss(self):
        return self.loss_dict

    def print_cur_loss(self, epoch, epoch_iter, start_time):
        errors = {k: v if not isinstance(
            v, int) else v for k, v in self.loss_dict.items()}
        t = (time.time() - start_time) / self.cfg.train.print_freq
        self.visualizer.print_current_errors(
            epoch, epoch_iter, errors, t)

    def save_visualization(self, epoch):
        # save nifti first

        # save images
        vis = OrderedDict()
        slc_num = self.cfg.data.patch_size[-1]//2
        for k, v in self.visual_dict.items():
            if 'masked_map' in k:
                vis[k] = util.tensor2label(v[0, :, :, :, slc_num], 2)
                self.visual_dict[k] = v.squeeze().float()
            elif 'pred' in k:
                v = F.softmax(v, dim=1)
                v = torch.argmax(v, dim=1)
                self.visual_dict[k] = v.squeeze().float()
                vis[k] = util.tensor2label(
                    v.squeeze()[:, :, slc_num].unsqueeze(0), self.cfg.train.cls_num)
            elif 'label' in k:
                vis[k] = util.tensor2label(
                    v.squeeze()[:, :, slc_num].unsqueeze(0), self.cfg.train.cls_num)
                self.visual_dict[k] = v.squeeze().float()
            else:
                vis[k] = util.tensor2im(v[0, :, :, :, slc_num])
                self.visual_dict[k] = v.squeeze().float()

        if self.cfg.system.save_nii:
            util.save_nii(self.visual_dict, self.vis_dir, epoch)

        self.visualizer.display_current_results(vis, epoch)

    def scheduler_step(self):
        self.scheduler.step()
        print('Current Learning Rate changed to {}'.format(
            self.optimizer.param_groups[0]['lr']))

    @torch.no_grad()
    def infer_single_scan(self, tmp_scans):
        pad_flag = False
        self.model.eval()
        x, y, z = self.cfg.data.patch_size
        if self.cfg.data.normalize:
            tmp_scans = util.norm_img(tmp_scans, self.cfg.data.norm_perc)
        if min(tmp_scans.shape) < min(x, y, z):
            x_ori_size, y_ori_size, z_ori_size = tmp_scans.shape
            pad_flag = True
            x_diff = x-x_ori_size
            y_diff = y-y_ori_size
            z_diff = z-z_ori_size
            tmp_scans = np.pad(tmp_scans, ((max(0, int(x_diff/2)), max(0, x_diff-int(x_diff/2))), (max(0, int(
                y_diff/2)), max(0, y_diff-int(y_diff/2))), (max(0, int(z_diff/2)), max(0, z_diff-int(z_diff/2)))), constant_values=1e-4)  # cant pad with 0s, otherwise the local and global patches wont be the same location

        pred = np.zeros((self.cfg.train.cls_num,) + tmp_scans.shape)
        tmp_norm = np.zeros((self.cfg.train.cls_num,) + tmp_scans.shape)

        scan_patches, _, tmp_idx = util.patch_slicer(tmp_scans, tmp_scans, self.cfg.data.patch_size,
                                                     (x - 16, y -
                                                      16, z - 16),
                                                     remove_bg=self.cfg.data.remove_bg, test=True, ori_path=None)
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
            location[:, patch_idx[0]:patch_idx[1], patch_idx[2]:patch_idx[3], patch_idx[4]:patch_idx[5]] = 1

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
            tmp_pred, _ = self.model(ipt, down_scan.cuda().reshape([1, 1, x, y, z]),
                                     coordinates_A)

            patch_idx = (slice(0, self.cfg.train.cls_num),) + (
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

    def validation(self, epoch):
        if not self.cfg.train.test_time:
            val_lst = [tmp_file for tmp_file in os.listdir(
                self.cfg.data.val_img) if tmp_file.endswith(self.cfg.data.extension)]
            try:
                nib.load(os.path.join(self.cfg.data.val_img, val_lst[0]))
            except ValueError:
                nib.Nifti1Header.quaternion_threshold = -1e-06

            cur_dsc = 0
            for val_file in val_lst:
                val_scan = nib.load(os.path.join(
                    self.cfg.data.val_img, val_file))
                val_label = nib.load(os.path.join(
                    self.cfg.data.val_label, val_file))
                tar_orientation = ('R', 'A', 'S')
                if nib.aff2axcodes(val_scan.affine) == tar_orientation:
                    tmp_scans = val_scan.get_fdata()
                    tmp_label = val_label.get_fdata()
                else:
                    ori_scnas_new_ortn = nib.as_closest_canonical(val_scan)
                    ori_label_new_ortn = nib.as_closest_canonical(val_label)
                    tmp_label = ori_label_new_ortn.get_fdata()
                    tmp_scans = ori_scnas_new_ortn.get_fdata()

                tmp_scans = np.squeeze(tmp_scans)
                tmp_label = np.squeeze(tmp_label)
                tmp_scans[tmp_scans < 0] = 0

                tmp_pred = self.infer_single_scan(tmp_scans)
                ind_dsc = []
                for cls_idx in range(1, self.cfg.train.cls_num):
                    ind_dsc.append(util.cal_dice(tmp_pred, tmp_label, cls_idx))
                cur_dsc += np.mean(ind_dsc)

                # save the prediciton and GT
                if self.cfg.system.save_nii:
                    tmp_label = nib.Nifti1Image(tmp_label, np.eye(4))
                    tmp_pred = nib.Nifti1Image(
                        tmp_pred.astype(np.uint8), np.eye(4))
                    nib.save(tmp_label, os.path.join(
                        self.val_dir, str(epoch) + '_' + val_file.split('.')[0] + '_label.nii.gz'))
                    nib.save(tmp_pred, os.path.join(
                        self.val_dir, str(epoch) + '_' + val_file.split('.')[0] + '_pred.nii.gz'))

            cur_dsc /= len(val_lst)
            self.val_dice.append(cur_dsc)

            tmp_val_score = cur_dsc * 1 - self.tgt_pse_seg_loss[-1]*0.5
            torch.cuda.empty_cache()
        else:
            self.val_dice.append(0)
            tmp_val_score = - self.tgt_pse_seg_loss[-1]
        if len(self.val_score) == 0:
            self.val_score.append(tmp_val_score)
            self.cumulative_no_improve.append(0)
            save_best = True
        else:
            if tmp_val_score > max(self.val_score):
                self.cumulative_no_improve.append(0)
                save_best = True
            else:
                save_best = False
                self.cumulative_no_improve.append(
                    self.cumulative_no_improve[-1]+1)
            self.val_score.append(tmp_val_score)

        return save_best


solver_zoo = {
    'mae': mae_trainer,
    'mpl': mpl_trainer
}


def get_solver(cfg):
    assert cfg.train.type in solver_zoo.keys(), 'Solver type {} not supported'.format(
        cfg.train.type)
    return solver_zoo[cfg.train.type](cfg)
