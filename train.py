from cfg.default import get_cfg_defaults
import wandb
import torch
import numpy as np
from model.solver import get_solver
from data.get_dataloader import get_dataloader
import time
import os
import shutil
import yaml
import argparse
from datetime import datetime
from model.utils.util import gen_code_archive
import torch.backends.cudnn as cudnn
import random


def set_random_seed(seed):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


parser = argparse.ArgumentParser(description='Setting config file')
parser.add_argument('--config', type=str, required=True,
                    help='path to the config yaml file')
args = parser.parse_args()

if __name__ == '__main__':
    '''
    Initialize the config file
    '''

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)
    print('loaded configuration file {}'.format(args.config))

    cfg.freeze()
    if cfg.system.seed is not None:
        set_random_seed(cfg.system.seed)

    '''
    Initialize the checkpoint folder and save the config file
    '''

    ckpt_fld = os.path.join(cfg.system.ckpt_dir,
                            cfg.system.project, cfg.system.exp_name)

    if not os.path.exists(ckpt_fld):
        os.makedirs(ckpt_fld)

    if not os.path.exists(os.path.join(ckpt_fld, 'train_cfg.yaml')):
        with open(os.path.join(ckpt_fld, 'train_cfg.yaml'), 'w') as f:
            f.write(cfg.dump())
            f.close()
    else:
        time_now = datetime.now()
        cfg_fname = os.path.join(
            ckpt_fld, 'train_cfg_' + time_now.strftime('%Y%m%d_%H%M%S') + '.yaml')
        with open(cfg_fname, 'w') as f:
            f.write(cfg.dump())
            f.close()

    '''
    Get the solver and dataloader
    '''
    cudnn.benchmark = True
    train_solver = get_solver(cfg)

    # To detect if there is existing checkpoint
    if os.path.exists(os.path.join(ckpt_fld, 'solver_latest.pth')):
        train_solver = torch.load(os.path.join(ckpt_fld, 'solver_latest.pth'))
        print('Loaded the latest solver checkpoint')
        print('Previous epochs: ' + str(train_solver._get_epoch()))

    # initialize the wandb
    if cfg.system.wandb:
        wandb.init(project=cfg.system.project,
                   config=cfg, name=cfg.system.exp_name)

    # get data loader
    train_loader = get_dataloader(cfg, train=True)

    '''
    set up validation parameters
    '''
    run_val = False
    if cfg.train.type == 'mpl':
        run_val = True

    start_epoch = 1 + train_solver._get_epoch()

    # save the current code
    gen_code_archive(ckpt_fld)
    print('Start training with this config:')
    print(cfg)
    for epoch in range(start_epoch, cfg.train.niter + cfg.train.niter_decay+1):
        epoch_start_time = time.time()
        print_start_time = time.time()
        # training
        # first to initialize the internal log of loss
        train_solver._init_epoch()
        for i, data in enumerate(train_loader):
            train_solver.train_step(data, epoch)

            if i % cfg.train.print_freq == 0:
                # step-wise log into wandb was disabled because it is too messy
                # one can enable it if needed
                # if cfg.system.wandb:
                #     wandb.log(
                #         {k+'_steps': v for k, v in train_solver.get_cur_loss().items()})
                train_solver.print_cur_loss(epoch, i, print_start_time)
                print_start_time = time.time()
        # summarize this epoch's results
        train_solver._log_internal_epoch_res(len(train_loader))

        if cfg.system.wandb:
            wandb.log(
                {k+'_epoch': v for k, v in train_solver._get_internal_loss().items()})
        # validation
        if run_val:

            save_best = train_solver.validation(epoch)
            if cfg.system.wandb:
                wandb.log({'validation dice': train_solver.val_dice[-1],
                           'validation score': train_solver.val_score[-1]})
            print(
                f"Epoch: {epoch}, Validation Dice: {train_solver.val_dice[-1]}, Validation score: {train_solver.val_score[-1]}, target pseudo loss: {train_solver.tgt_pse_seg_loss[-1]}")

            if save_best:
                torch.save(train_solver.model.state_dict(),
                           os.path.join(ckpt_fld, 'best_model.pth'))

            print('Current cumulative epochs of no improvement: ' +
                  str(train_solver.cumulative_no_improve[-1]))
            if train_solver.cumulative_no_improve[-1] > cfg.train.patience:
                print('Early stopping')
                break

        # get the visualization
        train_solver.save_visualization(epoch)
        # save the model
        if epoch % cfg.train.save_epoch_freq == 0:
            torch.save(train_solver.model.state_dict(),
                       os.path.join(ckpt_fld, f'model_epoch_{epoch}.pth'))

        # save the latest solver status:
        torch.save(train_solver,
                   os.path.join(ckpt_fld, 'solver_latest.pth'))

        iter_end_time = time.time()
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, cfg.train.niter + cfg.train.niter_decay, time.time() - epoch_start_time))

        if epoch > cfg.train.niter:
            train_solver.scheduler_step()
        if cfg.system.wandb:
            wandb.log({'lr': train_solver.optimizer.param_groups[0]['lr']})
            wandb.log(
                {'epoch': epoch})

    torch.save(train_solver.model.state_dict(),
               os.path.join(ckpt_fld, 'model_final.pth'))
