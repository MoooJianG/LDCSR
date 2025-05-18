# 使用三种损失
# 简化扩散模型的条件

from contextlib import contextmanager
import numpy as np
import pytorch_lightning as pl
import torch
from data.transforms import denormalize, normalize
from utils.diffusion import (
    Diffusion,
    samples_fn,
    progressive_samples_fn,
    fast_samples_fn,
)
from torch.nn import functional as F
from models.init_weights import init_weights
from modules.ema import LitEma
from utils import (
    default,
    disabled_train,
    instantiate_from_config,
)
from modules.distributions.distributions import DiagonalGaussianDistribution


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
        unet_config,
        diff_config,
        first_stage_config,
        cond_stage_config,
        sr_loss_weight=1.0,
        use_ema=False,
    ):
        super().__init__()

        # save hyperparameters
        self.save_hyperparameters()
        self.cond_stage_config = cond_stage_config
        self.sr_loss_weight = sr_loss_weight
        self.unet_config = unet_config

        # instantiate models
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.instantiate_unet_model(unet_config)

        # config ema
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.diffusion = Diffusion(**diff_config)
        assert diff_config.t_encode_mode == "continuous"
        self.num_steps = len(self.diffusion.betas)

        # data normalization
        self.mean, self.std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        self.normalize = lambda tensor: normalize(
            tensor, self.mean, self.std, inplace=False
        )
        self.denormalize = lambda tensor: denormalize(
            tensor, self.mean, self.std, inplace=False
        )

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        if config == "__identity__":
            self.cond_stage_model = torch.nn.Identity()
        else:
            self.cond_stage_model = instantiate_from_config(config)
            copy_params(self.first_stage_model.encoder, self.cond_stage_model)

    def instantiate_unet_model(self, config):
        model = instantiate_from_config(config)
        init_weights(model, init_type="orthogonal")
        self.unet = model

    def get_first_stage_encoding(self, hr, lr):
        h = self.first_stage_model.encode(hr, lr)
        if isinstance(h, DiagonalGaussianDistribution):
            h = h.mode()
        return h

    def get_cond_stage_encoding(self, lr, hr_size):
        return lr

    def decode_first_stage(self, z, lr, hr_size):
        return self.first_stage_model.decode(z, lr, hr_size)

    @torch.no_grad()
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

        x0 = self.get_first_stage_encoding(hr, lr).detach()

        out = [x0, lr, hr]
        if return_first_stage_outputs:
            hr_rec = self.decode_first_stage(x0, lr, hr_size)
            out.extend([hr_rec])
        return out

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def shared_step(self, batch, t=None, suffix=""):
        log_prefix = "train" if self.training else "val"

        x0, lr, hr = self.get_input(batch)

        c_concat = self.get_cond_stage_encoding(lr, hr.shape[2:])
        c = {"c_concat": c_concat}

        if t is None:
            t = torch.randint(
                0, self.num_steps, (x0.shape[0],), device=self.device
            ).long()

        loss_elbo, x0_hat = self.diffusion.get_loss(
            self.unet, x0, t, cond=c, return_pred_x0=True
        )
        loss_kd = F.l1_loss(x0, c["c_concat"])
        sr = self.decode_first_stage(x0_hat, lr, hr_size=hr.shape[2:])
        loss_consis = F.l1_loss(sr, hr)

        loss = loss_elbo + loss_kd + loss_consis

        self.log(f"{log_prefix}/loss_elbo{suffix}", loss_elbo)
        self.log(f"{log_prefix}/loss_consis{suffix}", loss_consis)
        self.log(f"{log_prefix}/loss_kd{suffix}", loss_kd)
        self.log(f"{log_prefix}/loss{suffix}", loss)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        return loss

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def validation_step(self, batch, batch_idx):
        batch_size = batch["lr"].shape[0]
        t = (
            torch.tensor(self.num_steps // 2, device=self.device)
            .repeat(batch_size)
            .long()
        )
        self.shared_step(batch, t)
        with self.ema_scope():
            self.shared_step(batch, t, "_ema")
        return

    def test_step(self, batch, *args):
        log = self.log_images(
            batch,
            N=2,
            sample=True,
            ddim_steps=200,
            ddim_eta=1.0,
        )
        return log

    @torch.no_grad()
    def log_images(
        self,
        batch,
        N=2,  # number of log images
        n_row=4,
        ddim_steps=200,
        ddim_eta=1.0,
        **kwargs,
    ):
        log = dict()
        x0, lr, hr, hr_rec = self.get_input(
            batch,
            return_first_stage_outputs=True,
        )

        c_concat = self.get_cond_stage_encoding(lr, hr.shape[2:])
        c = {"c_concat": c_concat}

        N = min(hr.shape[0], N)

        for key in list(c.keys()):
            c[key] = c[key][:N, ...]

        x0, c, lr, hr, hr_rec = (
            x0[:N, ...],
            c,
            lr[:N, ...],
            hr[:N, ...],
            hr_rec[:N, ...],
        )
        n_row = min(hr.shape[0], n_row)

        log["hr"] = hr
        log["hr_rec"] = hr_rec
        log["lr"] = lr

        # sampling
        skip = self.num_steps // ddim_steps
        with self.ema_scope("Plotting"):
            samples = fast_samples_fn(
                self.unet, self.diffusion, c, x0.shape, self.device, skip, ddim_eta
            )
            samples = samples["samples"]
        sr = self.decode_first_stage(samples, lr, hr.shape[2:])
        sr_cond = self.decode_first_stage(c["c_concat"], lr, hr.shape[2:])
        log["sr"] = sr
        log["sr_kd"] = sr_cond

        for key in list(log.keys()):
            log[key] = self.denormalize(log[key].clamp_(-1.0, 1.0))

        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.unet.parameters()) + list(self.cond_stage_model.parameters())
        optimizer = torch.optim.AdamW(params, lr=lr)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[200, 400, 600], gamma=0.5
        )

        return [optimizer], [scheduler]
