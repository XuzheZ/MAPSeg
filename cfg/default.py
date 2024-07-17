'''
Default config file for MAPSeg
'''
from yacs.config import CfgNode as CN

# create the config according to ./options/base_options.py
_c = CN()

_c.system = CN()
_c.system.project = 'MAPSeg'
_c.system.exp_name = 'default'
# wheter to use wandb
_c.system.wandb = True
# Number of threads for data loading
_c.system.n_threads = 4
# random seed
_c.system.seed = 0
# whether to save nifti during training
_c.system.save_nii = True
# display image size
_c.system.display_winsize = 128
# save image output into HTML
_c.system.no_html = False
# checkpoint directory
_c.system.ckpt_dir = 'default'

_c.train = CN()
# training type
_c.train.type = 'mae'  # 'mae' or 'mpl'
# learning rate
_c.train.lr = 2e-4
# warmup epochs for mpl
_c.train.warmup = 10
# batch size
_c.train.batch_size = 4
# number of epochs with fixed initial learning rate
_c.train.niter = 100
# number of epochs with scheduler
_c.train.niter_decay = 0
# mask ratio
_c.train.mask_ratio = 0.7
# class number for segmentation (including background)
_c.train.cls_num = 8
# patch size for the local branch
_c.train.local_mae_patch = 8
# patch size for the global branch
_c.train.global_mae_patch = 4
# default optimizer
_c.train.optimizer = 'AdamW'
# weight decay
_c.train.weight_decay = 0.05
# betas
_c.train.betas = (0.9, 0.95)
# number of steps to print the loss
_c.train.print_freq = 100
# number of epochs to save the current checkpoints
_c.train.save_epoch_freq = 50
# whether to run test-time domain adaptation
_c.train.test_time = False
# how many of epochs without improvement will stop the trianing
_c.train.patience = 50

_c.data = CN()
# file extension fo the data
_c.data.extension = '.nii.gz'
# whether to normalize the data
_c.data.normalize = True
# normalization percentile for data
_c.data.norm_perc = 99.5
# whether to remove background
_c.data.remove_bg = True
# whether to do data augmentation
_c.data.aug = True
# probability of data augmentation
_c.data.aug_prob = 0.35
# size of 3D patch
_c.data.patch_size = (96, 96, 96)
# path of validation image (source domain) during DA
_c.data.val_img = 'default'
# path of validation label (source domain) during DA, it should have same file name with val_image
_c.data.val_label = 'default'
# For MAE TRAINING
# root for masked autoencoding training
'''
    data structure: 
    it support multiple source/target data domains, each domain should end with '_train':
    mae_root
        |--- src_1_train
        |--- src_2_train
        |--- tgt_1_train
        |--- tgt_2_train
'''
_c.data.mae_root = 'default'


# FOR MPL Training

# root for source data during pseudo label training
'''
    data structure:
    it support multiple source data domains, each domain should end with '_img' for image and '_label' for label:
    src_data 
        |--- src_1_img
        |--- src_1_label
        |--- src_2_img
        |--- src_2_label
'''
_c.data.src_data = 'default'
# root for target data
'''
    data structure:
    it support multiple target data domains, each target domain (only contains image) should end with '_train':
    test folder if for inference after training
    tgt_data
        |--- tgt_1_train
        |--- tgt_2_train
        |--- test
            |---- img
            |---- label
'''
_c.data.tgt_data = 'default'

_c.model = CN()
# depth of encoder
_c.model.depth = 8
# dimension of intermediate embedding
_c.model.embed_dim = 512
# whether to load pretrained model
_c.model.load_pretrain = True
# path to pretrained model
_c.model.pretrain_model = 'default'
# whetther the model was pretrained on large-scale dataset (say, 1000+ scans)
_c.model.large_scale = True


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _c.clone()
