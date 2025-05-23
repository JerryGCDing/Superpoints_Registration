import os, argparse
from easydict import EasyDict
from cvhelpers.misc import prepare_logger
from cvhelpers.torch_helpers import setup_seed
from data_loaders import get_dataloader, get_multi_dataloader
from models import get_model
from trainer import Trainer
from utils.misc import load_config
import torch.multiprocessing as mp
from torch.distributed import init_process_group
import os
import torch
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from utils.comm import *


def main(opt, cfg):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    opt.world_size = torch.cuda.device_count()
    n_gpus_per_node = torch.cuda.device_count()

    mp.spawn(main_worker, nprocs=opt.world_size, args=(n_gpus_per_node, opt, cfg))


def main_worker(rank, n_gpus_per_node, opt, cfg):
    opt.local_rank = rank

    dist_url = 'env://12860'
    init_process_group(backend="nccl", init_method=dist_url, world_size=opt.world_size, rank=rank)

    Model = get_model(cfg.model)
    model = Model(cfg)

    if rank == 0:
        total_params = sum(param.numel() for param in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Total model params: {(total_params / 1000000):.6f}M")
        print(f"Total trainable model params: {(trainable_params / 1000000):.6f}M")

        # Save config to log
        config_out_fname = os.path.join(opt.log_path, 'config.yaml')
        with open(opt.config, 'r') as in_fid, open(config_out_fname, 'w') as out_fid:
            out_fid.write(f'# Original file name: {opt.config}\n')
            out_fid.write(f'# Total parameters: {(total_params / 1000000):.6f}M\n')
            out_fid.write(f'# Total trainable parameters: {(trainable_params / 1000000):.6f}M\n')
            out_fid.write(in_fid.read())

    opt.num_workers //= opt.world_size
    train_loader = get_multi_dataloader(cfg.dataloader, phase='train', num_workers=opt.num_workers,
                                        num_gpus=opt.num_gpus)
    val_loader = get_multi_dataloader(cfg.dataloader, phase='val', num_workers=opt.num_workers, num_gpus=opt.num_gpus)
    trainer = Trainer(opt, num_epochs=cfg.num_epochs, grad_clip=cfg.grad_clip)
    trainer.fit(model, train_loader, val_loader, opt.num_gpus, opt.local_rank)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the config file.')
    parser.add_argument('--logdir', type=str, default='../logs',
                        help='Directory to store logs, summaries, checkpoints.')
    parser.add_argument('--dev', action='store_true', help='If true, will ignore logdir and log to ../logdev instead')
    parser.add_argument('--testdev', action='store_true',
                        help='If true, will ignore logdir and log to ../logtestdev instead')
    parser.add_argument('--name', type=str, help='Experiment name (used to name output directory')
    parser.add_argument('--summary_every', type=int, default=500, help='Interval to save tensorboard summaries')
    parser.add_argument('--validate_every', type=int, default=-1, help='Validation interval. Default: every epoch')
    parser.add_argument('--debug', action='store_true', help='If set, will enable autograd anomaly detection')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPU for ddp')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads for dataloader')
    parser.add_argument('--resume', type=str, help='Checkpoint to resume from')
    parser.add_argument('--nb_sanity_val_steps', type=int, default=2,
                        help='Number of validation sanity steps to run before training.')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed data parallel.')

    opt = parser.parse_args()

    # Override config if --resume is passed
    if opt.config is None:
        if opt.resume is None or not os.path.exists(opt.resume):
            print('--config needs to be supplied unless resuming from checkpoint')
            exit(-1)
        else:
            resume_folder = opt.resume if os.path.isdir(opt.resume) else os.path.dirname(opt.resume)
            opt.config = os.path.normpath(os.path.join(resume_folder, '../config.yaml'))
            if os.path.exists(opt.config):
                print(f'Using config file from checkpoint directory: {opt.config}')
            else:
                print('Config not found in resume directory')
                exit(-2)

    cfg = EasyDict(load_config(opt.config))

    # Hack: Stores different datasets to its own subdirectory
    # opt.logdir = os.path.join(opt.logdir, cfg.dataset)

    if opt.name is None and len(cfg.get('expt_name', '')) > 0:
        opt.name = cfg.expt_name
    logger, opt.log_path = prepare_logger(opt)

    # # Save config to log
    # config_out_fname = os.path.join(opt.log_path, 'config.yaml')
    # with open(opt.config, 'r') as in_fid, open(config_out_fname, 'w') as out_fid:
    #     out_fid.write(f'# Original file name: {opt.config}\n')
    #     out_fid.write(in_fid.read())

    # Run the main function
    main(opt, cfg)
