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
INLIER_RATIO_THRESHOLD = 0.1
RMSE_THRESHOLD = 0.1


class Trainer:
    """
    Generic trainer class
    """

    def __init__(self, opt, num_epochs, grad_clip=0.0, benchmark=None, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.opt = opt
        self.train_writer = SummaryWriter(os.path.join(self.opt.log_path, 'train'), flush_secs=10)
        self.val_writer = SummaryWriter(os.path.join(self.opt.log_path, 'val'), flush_secs=10)
        self.saver = CheckPointManager(os.path.join(self.opt.log_path, 'ckpt', 'model'),
                                       max_to_keep=6, keep_checkpoint_every_n_hours=3.0)
        self.num_epochs = num_epochs
        self.grad_clip = grad_clip
        self.log_path = self.opt.log_path
        self.benchmark = benchmark

    def fit(self, model: GenericModel, train_loader, val_loader=None, num_gpus=1, local_rank=0):
        # Setup
        if torch.cuda.is_available():
            device = torch.device('cuda')
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
            epoch, global_step = self.saver.load(self.opt.resume, model, optimizer=model.optimizer,
                                                 scheduler=model.scheduler)
        else:
            epoch = 0
            global_step = 0

        # Configure anomaly detection
        torch.autograd.set_detect_anomaly(self.opt.debug)

        loss_smooth = None
        stats_meter = StatsMeter()
        train_output, losses = {}, {}
        save_ckpt = True if local_rank == 0 else False

        if self.opt.validate_every < 0:
            # validation interval given in epochs, so convert to steps
            self.opt.validate_every = -self.opt.validate_every * len(train_loader)
            self.logger.info('Validation interval set to {} steps'.format(self.opt.validate_every))

        # Run validation and exit if validate_every = 0
        if self.opt.validate_every == 0:
            self._run_validation(model, val_loader, epoch=epoch, global_step=global_step, save_ckpt=False,
                                 num_gpus=num_gpus, rank=local_rank)
            return

        # Validation dry run for sanity checks
        if self.opt.nb_sanity_val_steps > 0:
            self._run_validation(model, val_loader, epoch=epoch, global_step=global_step,
                                 limit_steps=self.opt.nb_sanity_val_steps, save_ckpt=save_ckpt, num_gpus=num_gpus,
                                 rank=local_rank)

        # Main training loop
        while epoch < self.num_epochs:  # Loop over epochs
            if num_gpus > 1:
                train_loader.sampler.set_epoch(epoch)
            self.logger.info('Starting epoch {}'.format(epoch))
            tbar = tqdm(total=len(train_loader), ncols=80, smoothing=0)

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
                    train_output, losses = model.module.training_step(batch, batch_idx)
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
                    train_output, losses = model.training_step(batch, batch_idx)

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
                tbar.set_description('Loss:{:.3g}'.format(loss_smooth))

                # except Exception as inst:
                #     # for i in range(len(batch['src_xyz'])):
                #     #     print(batch['src_xyz'][i].shape)
                #     exc_type, exc_obj, exc_tb = sys.exc_info()
                #     fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                #     self.logger.error(f'{exc_type} at {fname}:{exc_tb.tb_lineno} - {inst}')
                #     self.logger.debug(traceback.format_exc())

                tbar.update(1)
                # torch.cuda.empty_cache()

                if batch_idx % self.opt.summary_every == 0:
                    if num_gpus > 1:
                        model.module.train_summary_fn(writer=self.train_writer, step=global_step,
                                                      data_batch=batch, train_output=train_output, train_losses=losses)
                    else:
                        model.train_summary_fn(writer=self.train_writer, step=global_step,
                                               data_batch=batch, train_output=train_output, train_losses=losses)

                del batch, train_output, losses
                torch.cuda.empty_cache()

            tbar.close()  # we turn off the training progress bar since certain
            # environments (e.g. Pycharm) do not handle stacking well

            # Run validation, and save checkpoint.
            self._run_validation(model, val_loader, epoch=epoch, global_step=global_step, save_ckpt=save_ckpt,
                                 num_gpus=num_gpus, rank=local_rank)
            tbar = tqdm(total=len(train_loader), ncols=80, initial=batch_idx + 1, desc=tbar.desc[:-2])

            if num_gpus > 1:
                model.module.train_epoch_end()
            else:
                model.train_epoch_end()
            tbar.close()

            losses_dict = {k: stats_meter[k].avg for k in stats_meter}
            log_str = 'Epoch {} complete in {}. Average train losses: '.format(
                epoch, pretty_time_delta(time.perf_counter() - t_epoch_start))
            log_str += metrics_to_string(losses_dict) + '\n'
            self.logger.info(log_str)
            stats_meter.clear()
            torch.cuda.empty_cache()

            epoch += 1

        self.logger.info('Ending training. Number of training epochs = {}'.format(epoch + 1))

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
        inlier_ratio = []
        feature_matching_ratio = []
        registration_recall = []
        relative_rot_error = []
        relative_trans_error = []
        with torch.no_grad():

            model.test_epoch_start(self.benchmark)

            tbar_test = tqdm(total=len(test_loader), ncols=80, leave=False)
            for test_batch_idx, test_batch in enumerate(test_loader):
                test_batch = all_to_device(test_batch, model.device)
                metrics = model.test_step(test_batch, test_batch_idx, self.benchmark)
                inlier_ratio.append(metrics['inlier_ratio'])
                feature_matching_ratio.append(float(metrics['inlier_ratio'] >= INLIER_RATIO_THRESHOLD))
                registration_recall.append(float(metrics['rmse'] < RMSE_THRESHOLD))
                relative_rot_error.append(metrics['rot_err_deg'])
                relative_trans_error.append(metrics['trans_err'])
                tbar_test.update(1)
            tbar_test.close()

            # model.test_epoch_end(metrics_all, self.benchmark)
        inlier_ratio = torch.tensor(inlier_ratio).mean() * 100
        feature_matching_ratio = torch.tensor(feature_matching_ratio).mean() * 100
        registration_recall = torch.tensor(registration_recall).mean() * 100
        relative_rot_error = torch.tensor(relative_rot_error).mean()
        relative_trans_error = torch.tensor(relative_trans_error).mean()
        self.logger.info(
            f'Test results - IR: {inlier_ratio}; FMR: {feature_matching_ratio}; RR: {registration_recall}; RRE: {relative_rot_error}; RTE: {relative_trans_error}')

        model.train()

    def _run_validation(self, model: GenericModel, val_loader, epoch, global_step, limit_steps=-1, save_ckpt=True,
                        num_gpus=1, rank=0):
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
            self.logger.info(f'Running validation (epoch {epoch})...')

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
                model.module.validation_summary_fn(self.val_writer, global_step, val_outputs)
            else:
                val_score, val_outputs = model.validation_epoch_end(val_out_all)
                model.validation_summary_fn(self.val_writer, global_step, val_outputs)

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
                self.saver.save(model.module, epoch, global_step, val_score,
                                optimizer=model.module.optimizer, scheduler=model.module.scheduler)
            else:
                self.saver.save(model, epoch, global_step, val_score,
                                optimizer=model.optimizer, scheduler=model.scheduler)

        model.train()
