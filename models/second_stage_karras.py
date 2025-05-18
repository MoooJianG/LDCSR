import copy
import functools

import numpy as np
import pytorch_lightning as pl
import torch as th
from easydict import EasyDict as edict
from pytorch_lightning.utilities.distributed import rank_zero_only
from torch.nn import functional as F
from torch.optim import RAdam

from data.transforms import denormalize, normalize, tensor2uint8
from metrics.psnr_ssim import calc_psnr_ssim
from modules.distributions.distributions import DiagonalGaussianDistribution
from utils import disabled_train, instantiate_from_config
from utils.karras.fp16_util import (
    get_param_groups_and_shapes,
    make_master_params,
    master_params_to_model_params,
)
from utils.karras.karras_diffusion import karras_sample
from utils.karras.nn import update_ema
from utils.karras.random_util import get_generator
from utils.karras.resample import LossAwareSampler, create_named_schedule_sampler
from utils.karras.script_util import (
    args_to_dict,
    create_ema_and_scales_fn,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)


def copy_params(source_module, target_module):
    for source_name, source_param in source_module.named_parameters():
        if source_name in target_module.state_dict():
            target_param = target_module.state_dict()[source_name]
            if source_param.size() == target_param.size():
                target_param.copy_(source_param)
            else:
                # pass
                print(
                    "Skipping parameter with different size:",
                    source_name,
                    source_param.size(),
                    target_param.size(),
                )
        else:
            print("Skipping parameter not found in target module:", source_name)
            # pass


class S2Model(pl.LightningModule):
    def __init__(
        self,
        training_mode="consistency_training",
        target_ema_mode="adaptive",
        start_ema=0.95,
        scale_mode="progressive",
        start_scales=2,
        end_scales=200,
        total_training_steps=400000,
        distill_steps_per_iter=50000,
        schedule_sampler="uniform",
        first_stage_config={},
        cond_stage_config={},
        ema_rate=(0.999, 0.9999, 0.9999432189950708),
        cond_stage_trainable=True,
        scale_factor=1.0,
        scale_by_std=False,
        l_kd_weight=0.0,
        l_consis_weight=0.0,
        **kwargs,
    ):
        super().__init__()

        # save hyperparameters
        self.save_hyperparameters()
        self.first_stage_config = first_stage_config
        self.cond_stage_config = cond_stage_config
        self.training_mode = training_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.scale_by_std = scale_by_std
        self.l_kd_weight = l_kd_weight
        self.l_consis_weight = l_consis_weight
        self.validation_results = []
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer("scale_factor", th.tensor(scale_factor))
        print("creating model and diffusion...")

        ema_scale_fn = create_ema_and_scales_fn(
            target_ema_mode=target_ema_mode,
            start_ema=start_ema,
            scale_mode=scale_mode,
            start_scales=start_scales,
            end_scales=end_scales,
            total_steps=total_training_steps,
            distill_steps_per_iter=distill_steps_per_iter,
        )
        if training_mode == "karras":
            distillation = False
        elif "consistency" in training_mode:
            distillation = True
        else:
            raise ValueError(f"unknown training mode {training_mode}")

        kwargs = edict(kwargs)
        model_and_diffusion_kwargs = args_to_dict(
            kwargs, model_and_diffusion_defaults()
        )

        model_and_diffusion_kwargs["distillation"] = distillation
        model, diffusion = create_model_and_diffusion(**model_and_diffusion_kwargs)
        schedule_sampler = create_named_schedule_sampler(schedule_sampler, diffusion)
        self.schedule_sampler = schedule_sampler

        if "consistency" in training_mode:
            target_model, _ = create_model_and_diffusion(
                **model_and_diffusion_kwargs,
            )
            target_model.train()

            for dst, src in zip(target_model.parameters(), model.parameters()):
                dst.data.copy_(src.data)
        else:
            target_model = None

        # instantiate models
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)

        self.model = model
        self.diffusion = diffusion
        self.target_model = target_model
        self.teacher_model = None
        self.teacher_diffusion = None
        self.total_training_steps = total_training_steps
        self.ema_scale_fn = ema_scale_fn
        self.ema_rate = ema_rate
        self.ema_params = [
            copy.deepcopy(list(self.model.parameters()))
            for _ in range(len(self.ema_rate))
        ]

        if self.target_model:
            self.target_model.requires_grad_(False)
            self.target_model.train()

            self.target_model_param_groups_and_shapes = get_param_groups_and_shapes(
                self.target_model.named_parameters()
            )
            self.target_model_master_params = make_master_params(
                self.target_model_param_groups_and_shapes
            )

        # data normalization
        self.mean, self.std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        self.normalize = lambda tensor: normalize(
            tensor, self.mean, self.std, inplace=False
        )
        self.denormalize = lambda tensor: denormalize(
            tensor, self.mean, self.std, inplace=False
        )

    @rank_zero_only
    @th.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        # only for very first batch
        if (
            self.scale_by_std
            and self.current_epoch == 0
            and self.global_step == 0
            and batch_idx == 0
        ):
            assert (
                self.scale_factor == 1.0
            ), "rather not use custom rescaling and std-rescaling simultaneously"
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")
            hr = batch["hr"].to(self.device)
            lr = batch["lr"].to(self.device)
            hr = self.normalize(hr)
            lr = self.normalize(lr)

            encoder_posterior = self.encode_first_stage(hr, lr)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer("scale_factor", 1.0 / z.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                print("Using first stage also as cond stage.")
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                print(f"Training {self.__class__.__name__} as an unconditional model.")
                self.cond_stage_model = None
                # self.be_unconditional = True
            else:
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()
                self.cond_stage_model.train = disabled_train
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False
        else:
            assert config != "__is_first_stage__"
            assert config != "__is_unconditional__"
            model = instantiate_from_config(config)
            self.cond_stage_model = model

    @th.no_grad()
    def encode_first_stage(self, hr, lr):
        return self.first_stage_model.encode(hr, lr)

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, th.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(
                f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented"
            )
        return self.scale_factor * z

    def get_cond_stage_encoding(self, lr, hr_size):
        lr_up = F.interpolate(lr, hr_size, mode="bicubic", align_corners=False)
        return self.cond_stage_model(None, lr_up)

    @th.no_grad()
    def decode_first_stage(self, z, lr, out_size):
        z = 1.0 / self.scale_factor * z
        return self.first_stage_model.decode(z, lr, out_size)

    @th.no_grad()
    def get_input(
        self,
        batch,
        return_first_stage_outputs=False,
    ):
        hr = batch["hr"].to(self.device)
        lr = batch["lr"].to(self.device)

        hr = self.normalize(hr)
        lr = self.normalize(lr)
        hr_size = hr.shape[2:]

        encoder_posterior = self.encode_first_stage(hr, lr)
        x0 = self.get_first_stage_encoding(encoder_posterior).detach()

        out = [x0, lr, hr]
        if return_first_stage_outputs:
            hr_rec = self.decode_first_stage(x0, lr, hr_size)
            out.extend([hr_rec])
        return out

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, list(self.model.parameters()), rate=rate)

    def _update_target_ema(self):
        target_ema, scales = self.ema_scale_fn(self.global_step)
        with th.no_grad():
            update_ema(
                self.target_model_master_params,
                make_master_params(
                    get_param_groups_and_shapes(self.model.named_parameters())
                ),
                rate=target_ema,
            )
            master_params_to_model_params(
                self.target_model_param_groups_and_shapes,
                self.target_model_master_params,
            )

    def shared_step(self, batch, t=None):
        log_prefix = "train" if self.training else "val"

        x0, lr, hr = self.get_input(batch)
        c_concat = self.get_cond_stage_encoding(lr, hr.shape[2:])

        t, weights = self.schedule_sampler.sample(x0.shape[0], self.device)
        cond = {"c_concat": c_concat}

        ema, num_scales = self.ema_scale_fn(self.global_step)
        if self.training_mode == "karras":
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.model,
                x0,
                t,
                model_kwargs=cond,
            )
        elif self.training_mode == "consistency_training":
            compute_losses = functools.partial(
                self.diffusion.consistency_losses,
                self.model,
                x0,
                num_scales,
                target_model=self.target_model,
                model_kwargs=cond,
            )
        else:
            raise ValueError(f"Unknown training mode {self.training_mode}")

        losses = compute_losses()

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(t, losses["loss"].detach())

        loss_karras = (losses["loss"] * weights).mean()
        self.log(f"{log_prefix}/loss_karras", loss_karras)

        loss = loss_karras

        if self.l_consis_weight > 0:
            # loss_consis
            sample = karras_sample(
                self.diffusion,
                self.model,
                x0.shape,
                steps=40,
                model_kwargs=cond,
                device=self.device,
                clip_denoised=True,
                sampler="heun",
                sigma_min=0.002,
                sigma_max=80,
                s_churn=0,
                s_tmin=0.0,
                s_tmax=float("inf"),
                s_noise=1.0,
                # generator=get_generator("determ", 10000),
                # ts=[0, 17, 39],
            )
            hr_hat = self.decode_first_stage(sample, lr, hr.shape[2:])
            loss_consis = th.abs(hr - hr_hat).mean()
            self.log(f"{log_prefix}/loss_consis", loss_consis)
            loss += loss_consis

        # kd loss
        if self.l_kd_weight > 0:
            loss_kd = th.abs(c_concat - x0).mean()
            self.log(f"{log_prefix}/loss_kd", loss_kd)
            loss += loss_kd

        self.log(f"{log_prefix}/loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        return loss

    def on_train_batch_end(self, *args, **kwargs):
        self._update_ema()
        if self.target_model:
            self._update_target_ema()

    def validation_step(self, batch, batch_idx):
        self.shared_step(batch)
        log = self.log_images(batch)

        for rslt, gt in zip(log["sr"].cpu(), log["hr"].cpu()):
            rslt_np, gt_np = tensor2uint8([rslt, gt], rgb_range=1)
            psnr, ssim = calc_psnr_ssim(rslt_np, gt_np, crop_border=4, test_Y=False)
            self.validation_results.append({"psnr": psnr, "ssim": ssim})
        return

    def on_validation_epoch_end(self, *args):
        psnr = np.array([x["psnr"] for x in self.validation_results]).mean()
        ssim = np.array([x["ssim"] for x in self.validation_results]).mean()

        self.log("val/psnr", psnr)
        self.log("val/ssim", ssim)
        self.validation_results = []

    def test_step(self, batch, *args):
        log = self.log_images(batch, steps=40, sampler="multistep", ts=[1, 17, 39])
        return log

    @th.no_grad()
    def log_images(
        self,
        batch,
        generator="determ",
        num_samples=10000,
        seed=42,
        clip_denoised=True,
        steps=40,
        sampler="heun",
        sigma_min=0.002,
        sigma_max=80,
        s_churn=0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        ts=None,
        **kwargs,
    ):
        log = dict()
        x0, lr, hr, hr_rec = self.get_input(
            batch,
            return_first_stage_outputs=True,
        )

        c_concat = self.get_cond_stage_encoding(lr, hr.shape[2:])
        cond = {"c_concat": c_concat}

        log["hr"] = hr
        log["hr_rec"] = hr_rec
        log["lr"] = lr

        generator = get_generator(generator, num_samples, seed)

        # sampling
        sample = karras_sample(
            self.diffusion,
            self.model,
            x0.shape,
            steps=steps,
            model_kwargs=cond,
            device=self.device,
            clip_denoised=clip_denoised,
            sampler=sampler,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            s_churn=s_churn,
            s_tmin=s_tmin,
            s_tmax=s_tmax,
            s_noise=s_noise,
            generator=generator,
            ts=ts,
        )

        sr = self.decode_first_stage(sample, lr, hr.shape[2:])
        sr_cond = self.decode_first_stage(cond["c_concat"], lr, hr.shape[2:])
        log["sr"] = sr
        log["sr_kd"] = sr_cond

        for key in list(log.keys()):
            log[key] = self.denormalize(log[key].clamp_(-1.0, 1.0))

        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters()) + list(
            self.cond_stage_model.parameters()
        )
        optimizer = RAdam(params, lr=lr, weight_decay=0.0)

        scheduler = th.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[200, 400, 600], gamma=0.5
        )

        return [optimizer], [scheduler]
