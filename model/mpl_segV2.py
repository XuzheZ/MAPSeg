import torch
import torch.nn as nn
from .blocks import create_encoders, ExtResNetBlock, _ntuple, DeepLabHead
from .loss import DC_and_CE_loss


class Masked_seg(nn.Module):
    """ Masked Autoencoder with ResNet encoder + DeepLab segmentation header
    """

    def __init__(self, cfg):
        super().__init__()

        # --------------------------------------------------------------------------
        # ResNet encoder specifics
        self.cfg = cfg
        to_tuple = _ntuple(self.cfg.model.depth)
        embed_dim = self.cfg.model.embed_dim
        # encoder
        self.local_encoder = create_encoders(in_channels=1, f_maps=to_tuple(embed_dim), basic_module=ExtResNetBlock,
                                             conv_kernel_size=4, conv_stride_size=4, conv_padding=0, layer_order='gcr',
                                             num_groups=32)

        self.CE = nn.CrossEntropyLoss()
        self.seg_decoder = DeepLabHead(in_channels=embed_dim * 2, aspp_channel=embed_dim, num_classes=cfg.train.cls_num,
                                       ratio=4)

    def patchify(self, imgs, p):
        """

        imgs: (N, 1, H, W, D)
        x: (N, H*W*D/P***3, patch_size**3)
        """
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0 and imgs.shape[4] % p == 0
        h, w, d = [i // p for i in self.cfg.data.patch_size]

        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p, d, p))
        x = torch.einsum('nchpwqdr->nhwdpqrc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w * d, p ** 3))
        return x

    def unpatchify(self, x, p):
        """

        x: (N, H*W*D/P***3, patch_size**3)
        imgs: (N, 1, H, W, D)
        """
        h, w, d = [i // p for i in self.cfg.data.patch_size]

        assert h * w * d == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, d, p, p, p))
        x = torch.einsum('nhwdpqr->nhpwqdr', x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, w * p, d * p))
        return imgs

    def random_masking(self, x, mask_ratio, p):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        x = self.patchify(x, p)

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask_ = torch.zeros_like(x_masked)
        # generate the binary mask: 0 is keep, 1 is remove

        x_empty = torch.zeros((N, L - len_keep, D)).cuda()
        mask = torch.ones_like(x_empty)
        x_ = torch.cat([x_masked, x_empty], dim=1)
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        mask_ = torch.cat([mask_, mask], dim=1)
        mask_ = torch.gather(
            mask_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        x_masked = self.unpatchify(x_, p)

        mask = self.unpatchify(mask_, p)

        return x_masked, mask

    def forward_encoder(self, x, mask_ratio, p):

        # masking: length -> length * mask_ratio
        if mask_ratio > 0:
            x, mask = self.random_masking(x, mask_ratio, p)

        # apply Transformer blocks
        for blk in self.local_encoder:
            x = blk(x)
        if mask_ratio > 0:
            return x, mask
        else:
            return x

    def seg_loss(self, label, pred):
        # this version has confidence mask

        loss = DC_and_CE_loss(
            {'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {})
        loss_seg = loss(pred, label)
        return loss_seg

    def cos_regularization(self, pred, tar):
        loss = nn.CosineEmbeddingLoss()

        return loss(pred.flatten(start_dim=2).squeeze(), tar.flatten(start_dim=2).squeeze(),
                    target=torch.ones((pred.shape[1])).cuda())
    # THIS IS THE MAIN FORWARD FUNCTION to generate pseudo label

    def forward(self, coordinates, local_patch, local_label, global_img, global_label, mask_ratio=0,
                pseudo=False, real_label=True):

        # TODO: WARNING: The current implementation only supports batch size of 1
        # the way to extract feature via these coordinates can't be applied to multi batch

        if len(coordinates.shape) == 2 and coordinates.shape[0] == 1:
            coordinates = coordinates[0]
        if not pseudo:
            if real_label:
                # no masked out
                local_latent_1 = self.forward_encoder(
                    local_patch, mask_ratio=0, p=self.cfg.train.local_mae_patch)
                global_latent_1 = self.forward_encoder(global_img, mask_ratio=0,
                                                       p=self.cfg.train.global_mae_patch)
                global_latent_1_zoomed = global_latent_1[:, :, coordinates[0]:coordinates[1],
                                                         coordinates[2]:coordinates[3],
                                                         coordinates[4]:coordinates[5]].clone()
                upsample = nn.Upsample(
                    size=global_latent_1.shape[2:], mode='trilinear', align_corners=True)
                global_latent_1_zoomed = upsample(global_latent_1_zoomed)

                pred_1 = self.seg_decoder(torch.concat(
                    [local_latent_1, global_latent_1_zoomed], dim=1))
                loss_1 = self.seg_loss(local_label, pred_1)
                pred_aux_1 = self.seg_decoder(torch.concat(
                    [global_latent_1, global_latent_1], dim=1))
                loss_aux_1 = self.seg_loss(global_label, pred_aux_1)
                loss_feat_1 = self.cos_regularization(
                    local_latent_1, global_latent_1_zoomed)
                if mask_ratio > 0:
                    # with masked out:
                    local_latent_2, mask_local = self.forward_encoder(local_patch, mask_ratio=mask_ratio,
                                                                      p=self.cfg.train.local_mae_patch)
                    global_latent_2, _ = self.forward_encoder(global_img, mask_ratio=mask_ratio,
                                                              p=self.cfg.train.global_mae_patch)
                    global_latent_2_zoomed = global_latent_2[:, :, coordinates[0]:coordinates[1],
                                                             coordinates[2]:coordinates[3],
                                                             coordinates[4]:coordinates[5]].clone()
                    global_latent_2_zoomed = upsample(global_latent_2_zoomed)
                    pred_2 = self.seg_decoder(torch.concat(
                        [local_latent_2, global_latent_2_zoomed], dim=1))
                    loss_2 = self.seg_loss(local_label, pred_2)

                    pred_aux_2 = self.seg_decoder(torch.concat(
                        [global_latent_2, global_latent_2], dim=1))
                    loss_aux_2 = self.seg_loss(global_label, pred_aux_2)

                    loss_feat_2 = self.cos_regularization(
                        local_latent_2, global_latent_2_zoomed)
                    return loss_1, loss_2, loss_aux_1, loss_aux_2, loss_feat_1, loss_feat_2, pred_1, pred_2, pred_aux_1, mask_local  # , \
                else:
                    return loss_1, loss_aux_1, loss_feat_1, pred_1, pred_aux_1  # , \
            else:

                local_latent_1, mask_local = self.forward_encoder(local_patch, mask_ratio=mask_ratio,
                                                                  p=int(self.cfg.train.local_mae_patch))
                global_latent_1, _ = self.forward_encoder(global_img, mask_ratio=mask_ratio,
                                                          p=int(self.cfg.train.global_mae_patch))

                global_latent_1_zoomed = global_latent_1[:, :, coordinates[0]:coordinates[1],
                                                         coordinates[2]:coordinates[3],
                                                         coordinates[4]:coordinates[5]].clone()
                upsample = nn.Upsample(
                    size=global_latent_1.shape[2:], mode='trilinear', align_corners=True)
                global_latent_1_zoomed = upsample(global_latent_1_zoomed)
                pred_1 = self.seg_decoder(torch.concat(
                    [local_latent_1, global_latent_1_zoomed], dim=1))

                loss_1 = self.seg_loss(local_label, pred_1)
                loss_feat_1 = self.cos_regularization(
                    local_latent_1, global_latent_1_zoomed)
                pred_aux_1 = self.seg_decoder(torch.concat(
                    [global_latent_1, global_latent_1], dim=1))
                loss_aux1 = self.seg_loss(
                    global_label, pred_aux_1)

                local_latent_2 = self.forward_encoder(local_patch, mask_ratio=0,
                                                      p=int(self.cfg.train.local_mae_patch))
                global_latent_2 = self.forward_encoder(global_img, mask_ratio=0,
                                                       p=int(self.cfg.train.global_mae_patch))

                global_latent_2_zoomed = global_latent_2[:, :, coordinates[0]:coordinates[1],
                                                         coordinates[2]:coordinates[3],
                                                         coordinates[4]:coordinates[5]].clone()
                upsample = nn.Upsample(
                    size=global_latent_2.shape[2:], mode='trilinear', align_corners=True)
                global_latent_2_zoomed = upsample(global_latent_2_zoomed)
                pred_2 = self.seg_decoder(torch.concat(
                    [local_latent_2, global_latent_2_zoomed], dim=1))

                loss_2 = self.seg_loss(local_label, pred_2)
                loss_feat_2 = self.cos_regularization(
                    local_latent_2, global_latent_2_zoomed)
                pred_aux_2 = self.seg_decoder(torch.concat(
                    [global_latent_2, global_latent_2], dim=1))
                loss_aux2 = self.seg_loss(
                    global_label, pred_aux_2)

                loss = loss_1*0.5+loss_2*0.5
                loss_aux = loss_aux1*0.5+loss_aux2*0.5
                loss_feat = loss_feat_1*0.5+loss_feat_2*0.5
                return loss, pred_1, loss_aux, pred_aux_1, mask_local, loss_feat

        elif pseudo:
            local_latent_1 = self.forward_encoder(
                local_patch, mask_ratio=0, p=self.cfg.train.local_mae_patch)
            global_latent_1 = self.forward_encoder(
                global_img, mask_ratio=0, p=self.cfg.train.global_mae_patch)
            global_latent_1_zoomed = global_latent_1[:, :, coordinates[0]:coordinates[1],
                                                     coordinates[2]:coordinates[3],
                                                     coordinates[4]:coordinates[5]].clone()
            upsample = nn.Upsample(
                size=global_latent_1.shape[2:], mode='trilinear', align_corners=True)
            global_latent_1_zoomed = upsample(global_latent_1_zoomed)
            pred_1 = self.seg_decoder(torch.concat(
                [local_latent_1, global_latent_1_zoomed], dim=1))
            pred_aux = self.seg_decoder(torch.concat(
                [global_latent_1, global_latent_1], dim=1))
            return pred_1, pred_aux
    def forward_only1(self, coordinates, local_patch, local_label, global_img, global_label, mask_ratio=0,
                      pseudo=False, real_label=True):

        # TODO: WARNING: The current implementation only supports batch size of 1
        # the way to extract feature via these coordinates can't be applied to multi batch

        if len(coordinates.shape) == 2 and coordinates.shape[0] == 1:
            coordinates = coordinates[0]
        if not pseudo:
            if real_label:
                # no masked out
                local_latent_2, mask_local = self.forward_encoder(local_patch, mask_ratio=mask_ratio,
                                                                  p=self.cfg.train.local_mae_patch)
                global_latent_2, _ = self.forward_encoder(global_img, mask_ratio=mask_ratio,
                                                          p=self.cfg.train.global_mae_patch)
                upsample = nn.Upsample(
                    size=global_latent_2.shape[2:], mode='trilinear', align_corners=True)
                global_latent_2_zoomed = global_latent_2[:, :, coordinates[0]:coordinates[1],
                                                         coordinates[2]:coordinates[3],
                                                         coordinates[4]:coordinates[5]].clone()
                global_latent_2_zoomed = upsample(global_latent_2_zoomed)
                pred_2 = self.seg_decoder(torch.concat(
                    [local_latent_2, global_latent_2_zoomed], dim=1))
                loss_2 = self.seg_loss(local_label, pred_2)

                pred_aux_2 = self.seg_decoder(torch.concat(
                    [global_latent_2, global_latent_2], dim=1))
                loss_aux_2 = self.seg_loss(global_label, pred_aux_2)

                loss_feat_2 = self.cos_regularization(
                    local_latent_2, global_latent_2_zoomed)
                return loss_2, loss_aux_2, loss_feat_2, pred_2, pred_aux_2


class EMA_MPL(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.teacher = Masked_seg(cfg=cfg)
        self.student = Masked_seg(cfg=cfg)
        self.cfg = cfg

    def initialize_load(self):
        self.student.load_state_dict(
            torch.load(self.cfg.model.pretrain_model),
            strict=False)
        print('pretrained weights loaded, %s' % self.cfg.model.pretrain_model)

    def _init_ema_weights(self):

        for param in self.teacher.parameters():
            param.detach_()
        mp = list(self.student.parameters())
        mcp = list(self.teacher.parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()
        print('EMA weights initialized')

    @torch.no_grad()
    def _update_ema(self, iter):
        if self.cfg.model.large_scale:
            # for the model that was pretrained on large-scale data > 1000
            if iter < self.cfg.train.warmup*100 + 1000:
                # for the first 10 epochs after warmup
                # iteration per epoch is 100 and is never tuned
                # if that was altered, this part should be changed and the performance might be affected
                alpha_teacher = 0.999
            else:
                alpha_teacher = 0.9999
        else:
            # for the model that was pretrained on small-batch data: dozens to hundreds
            if iter < self.cfg.train.warmup*100 + 1000:  # for the first 10 epochs after warmup
                alpha_teacher = 0.99
            # for the 10-30th epochs after warmup
            elif iter >= self.cfg.train.warmup*100 + 1000 and iter < self.cfg.train.warmup*100 + 3000:
                alpha_teacher = 0.999
            else:
                alpha_teacher = 0.9999

        for ema_param, param in zip(self.teacher.parameters(),
                                    self.student.parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]
    # TO GET THE PSEUDO LABEL

    @torch.no_grad()
    def get_pseudo_label(self, local_patch, global_img, coordinates):
        pseudo, pseudo_aux = self.teacher(local_patch=local_patch, local_label=None, global_img=global_img,
                                          global_label=None, coordinates=coordinates, pseudo=True)
        return pseudo, pseudo_aux

    @torch.no_grad()
    def get_pseudo_label_and_weight(self, logits):
        ema_softmax = torch.softmax(logits.detach(), dim=1)
        _, pseudo_label = torch.max(ema_softmax, dim=1)

        # Below is a simple way to get the pseudo label with a certain threshold on confidence (prob.)
        # pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        # ps_large_p = pseudo_prob.ge(
        #     0.95).long() == 1
        # pseudo_label *= ps_large_p
        return pseudo_label

    # this is the training loop for source domain

    def train_source(self, cord_src, img_src, label_src, global_src, label_src_aux, src_mask_ratio):
        if src_mask_ratio > 0:
            seg_loss, seg_loss_masked, seg_loss_aux, seg_loss_aux_masked, cos_feat, cos_feat_masked, pred_seg, pred_seg_masked, pred_aux, mask_seg = \
                self.student(cord_src, img_src, label_src, global_src,
                             label_src_aux, src_mask_ratio)

            return seg_loss, seg_loss_masked, seg_loss_aux, seg_loss_aux_masked, cos_feat, cos_feat_masked, pred_seg, pred_seg_masked, pred_aux, mask_seg
        else:
            seg_loss, seg_loss_aux, cos_feat, pred_seg, pred_aux = \
                self.student(cord_src, img_src, label_src, global_src,
                             label_src_aux, src_mask_ratio)
            return seg_loss, seg_loss_aux, cos_feat, pred_seg, pred_aux

    def train_source_only1(self, cord_src, img_src, label_src, global_src, label_src_aux, src_mask_ratio):

        seg_loss, seg_loss_aux, cos_feat, pred_seg, pred_aux = \
            self.student.forward_only1(cord_src, img_src, label_src, global_src,
                                       label_src_aux, src_mask_ratio)
        return seg_loss, seg_loss_aux, cos_feat, pred_seg, pred_aux

    # THIS IS THE TRAINIGN LOOP FOR TARGET DOMAIN

    def train_pseudo(self, cord_tgt, img_tgt, pseudo_label_loc, global_tgt, pseudo_label_global, trg_mask_ratio):
        if trg_mask_ratio > 0:
            pse_seg_loss, pse_seg_pred, pse_seg_loss_aux, pse_seg_pred_aux, pse_seg_mask, pse_cos_feat = \
                self.student(cord_tgt, img_tgt, pseudo_label_loc, global_tgt, pseudo_label_global, trg_mask_ratio,
                             real_label=False)

            return pse_seg_loss, pse_seg_pred, pse_seg_loss_aux, pse_seg_pred_aux, pse_seg_mask, pse_cos_feat
        else:
            pse_seg_loss, pse_seg_pred, pse_seg_loss_aux, pse_seg_pred_aux, pse_cos_feat = \
                self.student(cord_tgt, img_tgt, pseudo_label_loc, global_tgt, pseudo_label_global, trg_mask_ratio,
                             real_label=False)

            return pse_seg_loss, pse_seg_pred, pse_seg_loss_aux, pse_seg_pred_aux, pse_cos_feat

    def forward(self, local_patch, global_img, coordinates):
        pseudo, pseudo_aux = self.student(local_patch=local_patch, local_label=None, global_img=global_img,
                                          global_label=None, coordinates=coordinates, pseudo=True)
        return pseudo, pseudo_aux
