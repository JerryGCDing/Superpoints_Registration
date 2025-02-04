import os, argparse

from easydict import EasyDict

from cvhelpers.misc import prepare_logger

from data_loaders import get_benchmark_dataset
from models import get_model
from trainer_vanilla import Trainer
from utils.misc import load_config
from cvhelpers.torch_helpers import setup_seed

setup_seed(0, cudnn_deterministic=False)

parser = argparse.ArgumentParser()
parser.add_argument('--benchmark', type=str, help='Benchmark dataset', default='3DMatch',
                    choices=['3DMatch', '3DLoMatch', 'ModelNet', 'ModelLoNet'])
# General
parser.add_argument('--config', type=str, help='Path to the config file.')
# Logging
parser.add_argument('--logdir', type=str, default='../logs',
                    help='Directory to store logs, summaries, checkpoints.')
parser.add_argument('--testdev', action='store_true',
                    help='If true, will ignore logdir and log to ../logtestdev instead')
parser.add_argument('--dev', action='store_true',
                    help='If true, will ignore logdir and log to ../logdev instead')
parser.add_argument('--name', type=str,
                    help='Prefix to add to logging directory')
# Misc
parser.add_argument('--num_workers', type=int, default=0,
                    help='Number of worker threads for dataloader')
# Training and model options
parser.add_argument('--resume', type=str, help='Checkpoint to resume from')

opt = parser.parse_args()
logger, opt.log_path = prepare_logger(opt)
# Override config if --resume is passed
if opt.resume is not None:
    resume_folder = opt.resume if os.path.isdir(opt.resume) else os.path.dirname(opt.resume)
    if os.path.exists(opt.config):
        print(f'Using config file from directory: {opt.config}')
    else:
        print('Config not found in resume directory')
        exit(-2)
cfg = EasyDict(load_config(opt.config))


def main():
    if opt.benchmark in ['3DMatch', '3DLoMatch']:
        cfg.dataloader.datasets['3dmatch'].benchmark = opt.benchmark
    elif opt.benchmark in ['ModelNet', 'ModelLoNet']:
        cfg.partial = [0.7, 0.7] if opt.benchmark == 'ModelNet' else [0.5, 0.5]
    else:
        raise NotImplementedError

    test_loader = get_benchmark_dataset(cfg.dataloader, benchmark=opt.benchmark, num_workers=opt.num_workers)
    Model = get_model(cfg.model)
    model = Model(cfg)
    trainer = Trainer(opt, num_epochs=cfg.num_epochs, grad_clip=cfg.grad_clip, benchmark=opt.benchmark)
    trainer.test(model, test_loader)


if __name__ == '__main__':
    main()
