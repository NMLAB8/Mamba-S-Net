import torch
from pathlib import Path
from .utils import copy_state, EMA, new_sdr
from .apply import apply_model
from .ema import ModelEMA
from . import augment
from .loss import spec_rmse_loss
from tqdm import tqdm
from .log import logger
from accelerate import Accelerator
from torch.cuda.amp import autocast

def _summary(metrics):
    return " | ".join(f"{key.capitalize()}={val}" for key, val in metrics.items())

class Solver(object):
    def __init__(self, loaders, model, optimizer, config, args):
        self.config = config
        self.loaders = loaders

        self.model = model
        self.optimizer = optimizer
        self.device = next(iter(self.model.parameters())).device
        self.accelerator = Accelerator()

        self.stft_config = {
            'n_fft': config.model.nfft,
            'hop_length': config.model.hop_size,
            'win_length': config.model.win_size,
            'center': True,
            'normalized': config.model.normalized
        }
        self.emas = {'batch': [], 'epoch': []}
        for kind in self.emas.keys():
            decays = getattr(config.ema, kind)
            device = self.device if kind == 'batch' else 'cpu'
            if decays:
                for decay in decays:
                    self.emas[kind].append(ModelEMA(self.model, decay, device=device))
        augments = [augment.Shift(shift=int(config.data.samplerate * config.data.shift),
                                  same=config.augment.shift_same)]
        if config.augment.flip:
            augments += [augment.FlipChannels(), augment.FlipSign()]
        for aug in ['scale', 'remix']:
            kw = getattr(config.augment, aug)
            if kw.proba:
                augments.append(getattr(augment, aug.capitalize())(**kw))
        self.augment = torch.nn.Sequential(*augments)

        self.folder = args.save_path
        self.checkpoint_file = Path(args.save_path) / 'checkpoint.th'
        self.best_state = None
        self.best_nsdr = 0
        self.epoch = -1
        self._reset()

    def _serialize(self, epoch, steps=0):
        package = {}
        package['state'] = self.model.state_dict()
        package['best_nsdr'] = self.best_nsdr
        package['best_state'] = self.best_state
        package['optimizer'] = self.optimizer.state_dict()
        package['epoch'] = epoch
        for kind, emas in self.emas.items():
            for k, ema in enumerate(emas):
                package[f'ema_{kind}_{k}'] = ema.state_dict()

        checkpoint_with_epoch = Path(self.checkpoint_file).with_name(f'checkpoint_epoch{epoch+1}.th')
        self.accelerator.save(package, checkpoint_with_epoch)

        self.accelerator.save(package, self.checkpoint_file)

        if steps:
            checkpoint_with_steps = Path(self.checkpoint_file).with_name(f'checkpoint_{epoch+1}_{steps}.th')
            self.accelerator.save(package, checkpoint_with_steps)

    def _reset(self):
        if self.checkpoint_file.exists():
            logger.info(f'Loading checkpoint model: {self.checkpoint_file}')
            package = torch.load(self.checkpoint_file, map_location=self.accelerator.device)
            self.model.load_state_dict(package['state'])
            self.best_nsdr = package['best_nsdr']
            self.best_state = package['best_state']
            self.optimizer.load_state_dict(package['optimizer'])
            self.epoch = package['epoch']
            for kind, emas in self.emas.items():
                for k, ema in enumerate(emas):
                    ema.load_state_dict(package[f'ema_{kind}_{k}'])

    def _format_train(self, metrics: dict) -> dict:
        losses = {
            'Loss': format(metrics['loss'], ".4f"),
        }
        if 'nsdr' in metrics:
            losses['SDR'] = format(metrics['nsdr'], ".3f")
        if 'grad' in metrics:
            losses['Grad'] = format(metrics['grad'], ".4f")

        for source in self.config.model.sources:
            nsdr_key = f'nsdr_{source}'
            if nsdr_key in metrics:
                losses[f'SDR_{source}'] = format(metrics[nsdr_key], ".3f")

        return losses

    def _format_test(self, metrics: dict) -> dict:
        losses = {}
        if 'sdr' in metrics:
            losses['SDR'] = format(metrics['sdr'], '.3f')
        if 'nsdr' in metrics:
            losses['SDR'] = format(metrics['nsdr'], '.3f')
        for source in self.config.model.sources:
            key = f'sdr_{source}'
            if key in metrics:
                losses[f'SDR_{source}'] = format(metrics[key], '.3f')
            key = f'nsdr_{source}'
            if key in metrics:
                losses[f'SDR_{source}'] = format(metrics[key], '.3f')
        return losses

    def train(self):
        for epoch in range(self.epoch + 1, self.config.epochs):
            for param_group in self.optimizer.param_groups:
              param_group['lr'] = self.config.optim.lr * (self.config.optim.decay_rate**((epoch)//self.config.optim.decay_step))
              logger.info(f"Learning rate adjusted to {self.optimizer.param_groups[0]['lr']}")

            self.model.train()
            metrics = {}
            logger.info('-' * 70)
            logger.info(f'Training Epoch {epoch + 1} ...')

            metrics['train'] = self._run_one_epoch(epoch)
            formatted = self._format_train(metrics['train'])
            logger.info(
                f'Train Summary | Epoch {epoch + 1} | {_summary(formatted)}')

            logger.info('-' * 70)
            logger.info('Cross validation...')
            self.model.eval()
            with torch.no_grad():
                valid = self._run_one_epoch(epoch, train=False)
                bvalid = valid
                bname = 'main'
                state = copy_state(self.model.state_dict())
                metrics['valid'] = {}
                metrics['valid']['main'] = valid
                for kind, emas in self.emas.items():
                    for k, ema in enumerate(emas):
                        with ema.swap():
                            valid = self._run_one_epoch(epoch, train=False)
                        name = f'ema_{kind}_{k}'
                        metrics['valid'][name] = valid
                        a = valid['nsdr']
                        b = bvalid['nsdr']
                        if a > b:
                            bvalid = valid
                            state = ema.state
                            bname = name
                    metrics['valid'].update(bvalid)
                    metrics['valid']['bname'] = bname

            formatted = self._format_train(metrics['valid'])
            logger.info(
                f'Valid Summary | Epoch {epoch + 1} | {_summary(formatted)}')

            valid_nsdr = metrics['valid']['nsdr']
            save_best = False
            if valid_nsdr > self.best_nsdr:
              logger.info('New best valid nsdr %.4f', valid_nsdr)
              self.best_state = copy_state(state)
              self.best_nsdr = valid_nsdr
              save_best = True

            should_save = False
            save_reason = ""

            if save_best:
                should_save = True
                save_reason = f'best model (nsdr={valid_nsdr:.4f})'
            elif self.config.save_every and (epoch + 1) % self.config.save_every == 0:
                should_save = True
                save_reason = f'regular checkpoint (save_every={self.config.save_every})'
            elif epoch == self.config.epochs - 1:
                should_save = True
                save_reason = 'final checkpoint'
            elif not self.config.save_every:
                should_save = True
                save_reason = 'every epoch (save_every not set)'

            logger.info(f'Epoch {epoch + 1}: save_every={self.config.save_every}, (epoch+1)%save_every={(epoch + 1) % self.config.save_every if self.config.save_every else "N/A"}, should_save={should_save}, reason={save_reason}')

            if self.accelerator.is_main_process and should_save:
                checkpoint_name = f'checkpoint_epoch{epoch + 1}.th'
                logger.info(f'Saving checkpoint at epoch {epoch + 1}: {save_reason} -> {checkpoint_name}')
                self._serialize(epoch)
            if epoch == self.config.epochs - 1:
                break


    def _run_one_epoch(self, epoch, train=True):
        config = self.config
        data_loader = self.loaders['train'] if train else self.loaders['valid']
        data_loader.sampler.epoch = epoch

        label = ["Valid", "Train"][train]
        name = label + f" | Epoch {epoch + 1}"
        total = len(data_loader)

        averager = EMA()

        if self.accelerator.is_main_process:
            data_loader = tqdm(data_loader)

        for idx, sources in enumerate(data_loader):
            sources = sources.to(self.device)
            if train:
                sources = self.augment(sources)
                mix = sources.sum(dim=1)
            else:
                mix = sources[:, 0]
                sources = sources[:, 1:]

            if not train:
                estimate = apply_model(self.model, mix, split=True, overlap=0)
            else:
                with autocast():
                   estimate = self.model(mix)

            assert estimate.shape == sources.shape, (estimate.shape, sources.shape)

            loss = spec_rmse_loss(estimate, sources, self.stft_config)

            losses = {}

            losses['loss'] = loss
            if not train:
                nsdrs = new_sdr(sources, estimate.detach()).mean(0)
                nsdrs = self.accelerator.reduce(nsdrs, reduction="mean")
                total = 0
                for source, nsdr in zip(self.config.model.sources, nsdrs):
                    losses[f'nsdr_{source}'] = nsdr
                    total += nsdr
                losses['nsdr'] = total / len(self.config.model.sources)

            if train:
                self.accelerator.backward(loss)
                grad_norm = 0
                grads = []
                for p in self.model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm()**2
                        grads.append(p.grad.data)
                losses['grad'] = grad_norm ** 0.5

                self.optimizer.step()
                self.optimizer.zero_grad()
                for ema in self.emas['batch']:
                    ema.update()

            losses = averager(losses)

            del loss, estimate

        if train:
            for ema in self.emas['epoch']:
                ema.update()
        return losses
