import logging
import os
import sys
import time
import traceback
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from cvhelpers.misc import pretty_time_delta
from cvhelpers.torch_helpers import all_to_device, all_isfinite, CheckPointManager, TorchDebugger
from utils.misc import StatsMeter
from models.generic_model import GenericModel
from utils.misc import metrics_to_string
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.comm import *

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"


class Trainer:
    """
    Generic trainer class
    """

    def __init__(self, opt, num_epochs, grad_clip=0.0, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.opt = opt
        self.train_writer = SummaryWriter(os.path.join(self.opt.log_path, 'train'), flush_secs=10)
        self.val_writer = SummaryWriter(os.path.join(self.opt.log_path, 'val'), flush_secs=10)
        self.saver = CheckPointManager(os.path.join(self.opt.log_path, 'ckpt', 'model'),
                                       max_to_keep=6, keep_checkpoint_every_n_hours=3.0)
        self.num_epochs = num_epochs
        self.grad_clip = grad_clip
        self.log_path = self.opt.log_path

    def fit(self, model: GenericModel, train_loader, val_loader=None, num_gpus=1, local_rank=0):
        # Setup
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f'cuda:{local_rank}')
        else:
            device = torch.device('cpu')
            self.logger.warning('Using CPU for training. This can be slow...')

        model.to(device)
        model.configure_optimizers()
        model.set_trainer(self)
        if num_gpus > 1:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)

        # Initialize checkpoint manager and resume from checkpoint if necessary
        if self.opt.resume is not None:
            first_step = global_step = \
                self.saver.load(self.opt.resume, model,
                                optimizer=model.optimizer, scheduler=model.scheduler)
        else:
            first_step = global_step = 0

        # Configure anomaly detection
        torch.autograd.set_detect_anomaly(self.opt.debug)

        done = False
        loss_smooth = None
        stats_meter = StatsMeter()
        train_output, losses = {}, {}
        save_ckpt = True if local_rank == 0 else False

        if self.opt.validate_every < 0:
            # validation interval given in epochs, so convert to steps
            self.opt.validate_every = -self.opt.validate_every * len(train_loader)
            self.logger.info('Validation interval set to {} steps'.format(self.opt.validate_every))

        # Run validation and exit if validate_every = 0
        if self.opt.validate_every == 0 and local_rank == 0:
            self._run_validation(model, val_loader, step=global_step, save_ckpt=False, num_gpus=num_gpus,
                                 rank=local_rank)
            self.logger.info('Validation dry run passed')
            return

        # Validation dry run for sanity checks
        if self.opt.nb_sanity_val_steps > 0 and local_rank == 0:
            self._run_validation(model, val_loader, step=global_step,
                                 limit_steps=self.opt.nb_sanity_val_steps, save_ckpt=save_ckpt, num_gpus=num_gpus,
                                 rank=local_rank)
            self.logger.info('Validation dry run passed')

        # Main training loop
        for epoch in range(self.num_epochs):  # Loop over epochs
            if num_gpus > 1:
                train_loader.sampler.set_epoch(epoch)
            if local_rank == 0:
                self.logger.info('Starting epoch {} (steps {} - {})'.format(
                    epoch, global_step, global_step + len(train_loader)))

            # Train
            model.train()
            torch.set_grad_enabled(True)
            if num_gpus > 1:
                model.module.train_epoch_start()
            else:
                model.train_epoch_start()
            t_epoch_start = time.perf_counter()

            for batch_idx, batch in enumerate(train_loader):

                global_step += 1

                # train step
                # try:
                batch = all_to_device(batch, device)
                if num_gpus > 1:
                    train_output, losses = model.module.training_step(batch, global_step)
                    if model.module.optimizer_handled_by_trainer:
                        if model.module.optimizer is not None:
                            model.module.optimizer.zero_grad()

                        # Back propagate, take optimization step
                        if 'total' in losses and losses['total'].requires_grad:
                            if self.opt.debug:
                                with TorchDebugger():
                                    losses['total'].backward()
                            else:
                                losses['total'].backward()

                        if self.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.grad_clip)

                        if model.module.optimizer is not None:
                            model.module.optimizer.step()
                            model.module.scheduler.step()
                else:
                    train_output, losses = model.training_step(batch, global_step)

                    if model.optimizer_handled_by_trainer:
                        if model.optimizer is not None:
                            model.optimizer.zero_grad()

                        # Back propagate, take optimization step
                        if 'total' in losses and losses['total'].requires_grad:
                            if self.opt.debug:
                                with TorchDebugger():
                                    losses['total'].backward()
                            else:
                                losses['total'].backward()

                            if self.grad_clip > 0:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.grad_clip)

                            if model.optimizer is not None:
                                model.optimizer.step()
                                model.scheduler.step()

                # Increment counters
                for k in losses:
                    stats_meter[k].update(losses[k])

                if loss_smooth is None:
                    loss_smooth = losses['total'].item()
                elif not all_isfinite(losses['total']):
                    print(f"losses['feature']: {losses['feature']}")
                    print(f"losses['T']: {losses['T']}")
                    print(f"losses['overlap']: {losses['overlap']}")
                    self.logger.warning('Total loss is not finite, Ignoring...\n'
                                        'Instance {}, src_path: {}, tgt_path: {}'.format(
                        batch['item'], batch['src_path'], batch['tgt_path']))
                else:
                    loss_smooth = 0.99 * loss_smooth + 0.01 * losses['total'].item()

                if global_step == first_step + 1 or global_step % self.opt.summary_every == 0:
                    if num_gpus > 1:
                        if local_rank == 0:
                            self.logger.info(
                                f"Epoch {epoch}/Step {global_step}: losses['feature']: {losses['feature']}; losses['T']; "
                                f"{losses['T']}; losses['overlap']: {losses['overlap']}")
                            model.module.train_summary_fn(writer=self.train_writer,
                                                          step=global_step,
                                                          data_batch=batch,
                                                          train_output=train_output,
                                                          train_losses=losses)
                    else:
                        self.logger.info(
                            f"Epoch {epoch}/Step {global_step}: losses['feature']: {losses['feature']}; losses['T']; "
                            f"{losses['T']}; losses['overlap']: {losses['overlap']}")
                        model.train_summary_fn(writer=self.train_writer, step=global_step,
                                               data_batch=batch, train_output=train_output, train_losses=losses)

                    # Run validation, and save checkpoint.
            if local_rank == 0 and epoch % self.opt.validate_every == 0:
                self._run_validation(model,
                                     val_loader,
                                     step=global_step,
                                     save_ckpt=save_ckpt,
                                     num_gpus=num_gpus,
                                     rank=local_rank)

            if num_gpus > 1:
                model.module.train_epoch_end()
            else:
                model.train_epoch_end()

            losses_dict = {k: stats_meter[k].avg for k in stats_meter}
            if local_rank == 0:
                log_str = 'Epoch {} complete in {}. Average train losses: '.format(
                    epoch, pretty_time_delta(time.perf_counter() - t_epoch_start))
                log_str += metrics_to_string(losses_dict) + '\n'
                self.logger.info(log_str)
            stats_meter.clear()

        if local_rank == 0:
            self.logger.info('Ending training. Number of training steps = {}'.format(global_step))

    def test(self, model: GenericModel, test_loader):
        # Setup
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
            self.logger.warning('Using CPU for training. This can be slow...')
        model.to(device)
        model.set_trainer(self)

        # Initialize checkpoint manager and resume from checkpoint if necessary
        if self.opt.resume is not None and len(self.opt.resume) > 0:
            self.saver.load(self.opt.resume, model)
        else:
            self.logger.warning('No checkpoint given. Will perform inference '
                                'using random weights')

        # Run validation and exit if validate_every = 0
        model.eval()
        test_out_all = []
        with torch.no_grad():

            model.test_epoch_start()

            tbar_test = tqdm(total=len(test_loader), ncols=80, leave=False)
            for test_batch_idx, test_batch in enumerate(test_loader):
                test_batch = all_to_device(test_batch, model.device)
                test_out = model.test_step(test_batch, test_batch_idx)
                test_out_all.append(test_out)
                tbar_test.update(1)
            tbar_test.close()

            model.test_epoch_end(test_out_all)

        model.train()

    def _run_validation(self, model: GenericModel, val_loader, step, limit_steps=-1, save_ckpt=True, num_gpus=1,
                        rank=0):
        """Run validation on data from the validation data loader

        Args:
            model: Model
            val_loader: Validation data loader. If None, will skip validation
            limit_steps: If positive, will only run this number of validation
              steps. Useful as a sanity check on the validation code.
            save_ckpt: Whether to save checkpoint at the end

        Returns:
            val_score: Used for selecting the best checkpoint
        """
        if val_loader is None:
            return 0.0

        if limit_steps > 0:
            num_steps = limit_steps
            self.logger.info(f'Performing validation dry run with {num_steps} steps')
        else:
            num_steps = len(val_loader)
            self.logger.info(f'Running validation (step {step})...')

        model.eval()
        val_out_all = []
        with torch.no_grad():

            if num_gpus > 1:
                model.module.validation_epoch_start()
            else:
                model.validation_epoch_start()

            tbar_val = tqdm(total=num_steps, ncols=80, leave=False)
            for val_batch_idx, val_batch in enumerate(val_loader):
                if val_batch_idx >= num_steps:
                    break
                val_batch = all_to_device(val_batch, model.device)
                if num_gpus > 1:
                    val_out = model.module.validation_step(val_batch, val_batch_idx)
                else:
                    val_out = model.validation_step(val_batch, val_batch_idx)
                val_out_all.append(val_out)
                tbar_val.update(1)
            tbar_val.close()

            if num_gpus > 1:
                val_score, val_outputs = model.module.validation_epoch_end(val_out_all)
                model.module.validation_summary_fn(self.val_writer, step, val_outputs)
            else:
                val_score, val_outputs = model.validation_epoch_end(val_out_all)
                model.validation_summary_fn(self.val_writer, step, val_outputs)

            synchronize()
            log_str = ['Validation ended:']
            if 'losses' in val_outputs:
                log_str.append(metrics_to_string(val_outputs['losses'], '[Losses]'))
            if 'metrics' in val_outputs:
                log_str.append(metrics_to_string(val_outputs['metrics'], '[Metrics]'))
            log_str = '\n'.join(log_str)
            self.logger.info(log_str)

        if save_ckpt and rank == 0:
            if num_gpus > 1:
                self.saver.save(model.module, step, val_score,
                                optimizer=model.module.optimizer, scheduler=model.module.scheduler)
            else:
                self.saver.save(model, step, val_score,
                                optimizer=model.optimizer, scheduler=model.scheduler)

        model.train()
