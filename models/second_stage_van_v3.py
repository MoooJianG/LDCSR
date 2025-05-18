import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from data.transforms import denormalize, normalize
from models.diffusion.ddim import DDIMSampler
from modules.diffusionmodules.util import extract_into_tensor, noise_like
from modules.distributions.distributions import DiagonalGaussianDistribution
from utils import default, instantiate_from_config
from pytorch_lightning.utilities.distributed import rank_zero_only

from .diffusion.ddpm import DDPM, disabled_train
from modules.e2sr.s1_v7c import Encoder as EncoderV7c


class S2Model(DDPM):
    def __init__(
        self,
        first_stage_config,
        cond_stage_config,
        cond_stage_key="lr",
        cond_stage_trainable=True,
        conditioning_key="concat",
        scale_factor=1.0,
        scale_by_std=False,
        l_consis_weight=1.0,
        l_kd_weight=1.0,
        *args,
        **kwargs,
    ):
        # for backwards compatibility after implementation of DiffusionWrapper

        if cond_stage_config == "__is_unconditional__":
            conditioning_key = None

        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        self.cond_stage_config = cond_stage_config
        self.scale_by_std = scale_by_std
        self.l_consis_weight = l_consis_weight
        self.l_kd_weight = l_kd_weight
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer("scale_factor", torch.tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.clip_denoised = False

        self.restarted_from_ckpt = False
        if ckpt_path:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

        self.mean, self.std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        self.normalize = lambda tensor: normalize(
            tensor, self.mean, self.std, inplace=False
        )
        self.denormalize = lambda tensor: denormalize(
            tensor, self.mean, self.std, inplace=False
        )

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        # only for very first batch
        if (
            self.scale_by_std
            and self.current_epoch == 0
            and self.global_step == 0
            and batch_idx == 0
            and not self.restarted_from_ckpt
        ):
            assert (
                self.scale_factor == 1.0
            ), "rather not use custom rescaling and std-rescaling simultaneously"
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")
            hr = super().get_input(batch, self.first_stage_key).to(self.device)
            lr = super().get_input(batch, self.first_stage_key).to(self.device)

            hr = self.normalize(hr)
            lr = self.normalize(lr)

            encoder_posterior = self.encode_first_stage(hr, lr)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer("scale_factor", 1.0 / z.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")

    def register_schedule(
        self,
        given_betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        super().register_schedule(
            given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s
        )

    # def on_train_epoch_end(self):
    #     if self.current_epoch < 100:
    #         max_scale = round(2.0 + (6.0 / 99) * self.current_epoch, 1)
    #     else:
    #         max_scale = 8
    #     self.trainer.datamodule.datasets["train"].random_sample_scale(1, max_scale)

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

    def get_cond_stage_encoding(self, lr, hr_size):
        xc_up = torch.nn.functional.interpolate(
            lr, hr_size, mode="bicubic", align_corners=False
        )
        if isinstance(self.cond_stage_model, EncoderV7c):
            return self.cond_stage_model(xc_up, lr, diff=False)
        else:
            return self.cond_stage_model(None, xc_up)

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(
                f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented"
            )
        return self.scale_factor * z

    @torch.no_grad()
    def get_input(
        self,
        batch,
        return_first_stage_outputs=False,
    ):
        hr = super().get_input(batch, self.first_stage_key).to(self.device)  # hr
        lr = super().get_input(batch, self.cond_stage_key).to(self.device)  # lr

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

    @torch.no_grad()
    def decode_first_stage(self, z, lr, out_size):
        z = 1.0 / self.scale_factor * z
        return self.first_stage_model.decode(z, lr, out_size)

    # same as above but without decorator
    def differentiable_decode_first_stage(self, z, lr, out_size):
        z = 1.0 / self.scale_factor * z
        return self.first_stage_model.decode(z, lr, out_size)

    @torch.no_grad()
    def encode_first_stage(self, hr, lr):
        z = self.first_stage_model.encode(hr, lr)
        return z
    
    def shared_step(self, batch, **kwargs):
        x0, lr, hr = self.get_input(batch)
        c_concat = self.get_cond_stage_encoding(lr, hr.shape[2:])
        scale = torch.tensor(hr.shape[2] / lr.shape[2]).repeat(x0.shape[0]).long().to(self.device)
        osize = torch.tensor(hr.shape[2]).repeat(x0.shape[0]).long().to(self.device)
        loss, log_dict = self.forward(x0, c_concat, scale, osize, lr=lr, hr=hr)
        return loss, log_dict

    def forward(self, x, c, scale, osize, *args, **kwargs):
        t = torch.randint(
            0, self.num_timesteps, (x.shape[0],), device=self.device
        ).long()

        if self.model.conditioning_key is not None:
            assert c is not None
        return self.p_losses(x, c, t, scale, osize, *args, **kwargs)

    def apply_model(self, x_noisy, t, cond_dict):
        if not isinstance(cond_dict['c_concat'], list):
            cond_dict['c_concat'] = [cond_dict['c_concat']]
        x_recon = self.model(x_noisy, t, **cond_dict)
        return x_recon

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def p_losses(self, x_start, cond, t, scale, osize, noise=None, *args, **kwargs):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        cond_dict = {"c_concat": cond, "c_scale": scale, "c_osize": osize}
        model_output = self.apply_model(x_noisy, t, cond_dict)

        loss_dict = {}
        prefix = "train" if self.training else "val"

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        # simple loss
        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f"{prefix}/loss_simple": loss_simple.mean()})

        self.logvar = self.logvar.to(self.device)

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        if self.learn_logvar:
            loss_dict.update({f"{prefix}/loss_gamma": loss.mean()})
            loss_dict.update({"logvar": self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        # loss_vlb
        loss_vlb = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f"{prefix}/loss_vlb": loss_vlb})
        loss += self.original_elbo_weight * loss_vlb

        # loss_consis
        if self.l_consis_weight > 0:
            if self.parameterization == "eps":
                x0_hat = self.predict_start_from_noise(x_noisy, t, model_output)
            else:
                x0_hat = model_output
            hr = kwargs["hr"]
            hr_hat = self.decode_first_stage(x0_hat, kwargs["lr"], hr.shape[2:])

            alpha_bar_t = extract_into_tensor(self.alphas_cumprod, t, x_start.shape)
            gamma = self.l_consis_weight * alpha_bar_t / (1 - alpha_bar_t)
            loss_consis = self.get_loss(hr_hat, hr, mean=False).mean([1, 2, 3]) * gamma
            loss_consis = self.l_consis_weight * loss_consis.mean()
            loss += self.l_consis_weight * loss_consis
            loss_dict.update({f"{prefix}/loss_consis": loss_consis})

        # loss_kd
        if self.l_kd_weight>0:
            loss_kd = self.get_loss(x_start, cond, mean=True)
            loss += self.l_kd_weight * loss_kd
            loss_dict.update({f"{prefix}/loss_kd": loss_kd})
        
        loss_dict.update({f"{prefix}/loss": loss})
        return loss, loss_dict

    def p_mean_variance(
        self,
        x,
        c,
        t,
        clip_denoised: bool,
        quantize_denoised=False,
        return_x0=False,
        score_corrector=None,
        corrector_kwargs=None,
    ):
        t_in = t
        model_out = self.apply_model(x, t_in, c)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(
                self, model_out, x, t, c, **corrector_kwargs
            )

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        if quantize_denoised:
            x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )

        if return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(
        self,
        x,
        c,
        t,
        clip_denoised=False,
        repeat_noise=False,
        quantize_denoised=False,
        return_x0=False,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
    ):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(
            x=x,
            c=c,
            t=t,
            clip_denoised=clip_denoised,
            quantize_denoised=quantize_denoised,
            return_x0=return_x0,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
        )

        if return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.0:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        if return_x0:
            return (
                model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise,
                x0,
            )
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def progressive_denoising(
        self,
        cond,
        shape,
        verbose=True,
        callback=None,
        quantize_denoised=False,
        img_callback=None,
        mask=None,
        x0=None,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        batch_size=None,
        x_T=None,
        start_T=None,
        log_every_t=None,
    ):
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=self.device)
        else:
            img = x_T
        intermediates = []
        if cond is not None:
            if isinstance(cond, dict):
                cond = {
                    key: (
                        cond[key][:batch_size]
                        if not isinstance(cond[key], list)
                        else list(map(lambda x: x[:batch_size], cond[key]))
                    )
                    for key in cond
                }
            else:
                cond = (
                    [c[:batch_size] for c in cond]
                    if isinstance(cond, list)
                    else cond[:batch_size]
                )

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = (
            tqdm(
                reversed(range(0, timesteps)),
                desc="Progressive Generation",
                total=timesteps,
            )
            if verbose
            else reversed(range(0, timesteps))
        )
        if type(temperature) == float:
            temperature = [temperature] * timesteps

        for i in iterator:
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)
            img, x0_partial = self.p_sample(
                img,
                cond,
                ts,
                clip_denoised=self.clip_denoised,
                quantize_denoised=quantize_denoised,
                return_x0=True,
                temperature=temperature[i],
                noise_dropout=noise_dropout,
                score_corrector=score_corrector,
                corrector_kwargs=corrector_kwargs,
            )
            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1.0 - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)
            if callback:
                callback(i)
            if img_callback:
                img_callback(img, i)
        return img, intermediates

    @torch.no_grad()
    def p_sample_loop(
        self,
        cond,
        shape,
        return_intermediates=False,
        x_T=None,
        verbose=True,
        callback=None,
        timesteps=None,
        quantize_denoised=False,
        mask=None,
        x0=None,
        img_callback=None,
        start_T=None,
        log_every_t=None,
    ):
        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = (
            tqdm(reversed(range(0, timesteps)), desc="Sampling t", total=timesteps)
            if verbose
            else reversed(range(0, timesteps))
        )

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)

            img = self.p_sample(
                img,
                cond,
                ts,
                clip_denoised=self.clip_denoised,
                quantize_denoised=quantize_denoised,
            )
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1.0 - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback:
                callback(i)
            if img_callback:
                img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(
        self,
        cond,
        batch_size=16,
        return_intermediates=False,
        x_T=None,
        verbose=True,
        timesteps=None,
        quantize_denoised=False,
        mask=None,
        x0=None,
        shape=None,
        **kwargs,
    ):
        if shape is None:
            raise ValueError
        if cond is not None:
            if isinstance(cond, dict):
                cond = {
                    key: (
                        cond[key][:batch_size]
                        if not isinstance(cond[key], list)
                        else list(map(lambda x: x[:batch_size], cond[key]))
                    )
                    for key in cond
                }
            else:
                cond = (
                    [c[:batch_size] for c in cond]
                    if isinstance(cond, list)
                    else cond[:batch_size]
                )
        return self.p_sample_loop(
            cond,
            shape,
            return_intermediates=return_intermediates,
            x_T=x_T,
            verbose=verbose,
            timesteps=timesteps,
            quantize_denoised=quantize_denoised,
            mask=mask,
            x0=x0,
        )

    @torch.no_grad()
    def sample_log(self, cond, batch_size, image_size, ddim, ddim_steps, **kwargs):
        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels, image_size[0], image_size[1])
            samples, intermediates = ddim_sampler.sample(
                ddim_steps, batch_size, shape, cond, verbose=False, **kwargs
            )
        else:
            samples, intermediates = self.sample(
                cond=cond, batch_size=batch_size, return_intermediates=True, **kwargs
            )

        return samples, intermediates

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        log = self.log_images(
            batch,
            N=1,
            sample=True,
            ddim_steps=200,
            ddim_eta=1.0,
            return_keys=None,
            quantize_denoised=True,
            plot_denoise_rows=False,
            plot_progressive_rows=False,
            plot_diffusion_rows=False,
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
        return_keys=None,
        **kwargs,
    ):
        use_ddim = ddim_steps is not None

        log = dict()
        x0, lr, hr, hr_rec = self.get_input(
            batch,
            return_first_stage_outputs=True,
        )

        c_concat = self.get_cond_stage_encoding(lr, hr.shape[2:])
        N = min(hr.shape[0], N)
        x0, c_concat, lr, hr, hr_rec = (
            x0[:N, ...],
            c_concat[:N, ...],
            lr[:N, ...],
            hr[:N, ...],
            hr_rec[:N, ...],
        )
        n_row = min(hr.shape[0], n_row)
        if self.current_epoch == 0 and not self.training:
            log["hr"] = hr
            log["hr_rec"] = hr_rec
            log["lr"] = lr

        # get denoise row
        scale = torch.tensor(hr.shape[2] / lr.shape[2]).repeat(x0.shape[0]).long().to(self.device)
        osize = torch.tensor(hr.shape[2]).repeat(x0.shape[0]).long().to(self.device)
        cond_dict = {"c_concat": c_concat, "c_scale": scale, "c_osize": osize}
        with self.ema_scope("Plotting"):
            samples, z_denoise_row = self.sample_log(
                cond=cond_dict,
                batch_size=N,
                image_size=x0.shape[2:],
                ddim=use_ddim,
                ddim_steps=ddim_steps,
                eta=ddim_eta,
            )
            # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
        sr = self.decode_first_stage(samples, lr, hr.shape[2:])
        sr_cond = self.decode_first_stage(c_concat, lr, hr.shape[2:])
        log["sr"] = sr
        log["sr_kd"] = sr_cond

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}

        for key in list(log.keys()):
            log[key] = self.denormalize(log[key].clamp_(-1.0, 1.0))
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())
        if self.learn_logvar:
            print("Diffusion model optimizing logvar")
            params.append(self.logvar)
        optimizer = torch.optim.AdamW(params, lr=lr)

        if self.use_scheduler:
            assert "target" in self.scheduler_config
            self.scheduler_config["params"]["optimizer"] = optimizer
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": LambdaLR(optimizer, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [optimizer], scheduler
        return optimizer
