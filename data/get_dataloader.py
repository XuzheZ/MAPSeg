from .datasets import mae_dataset, mpl_dataset
import torch


def get_dataloader(cfg, train=True):
    assert cfg.train.type in [
        'mae', 'mpl'], 'unknown training type, only support mae (masked autoencoding) and mpl (masked pseudo labeling)'
    if cfg.train.type == 'mae':
        dataset = mae_dataset(cfg)
    elif cfg.train.type == 'mpl':
        dataset = mpl_dataset(cfg)
    if train:
        train_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=cfg.train.batch_size,
                                                   shuffle=True,
                                                   num_workers=cfg.system.n_threads)
        return train_loader
    else:
        val_loader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 num_workers=0)
        return val_loader
