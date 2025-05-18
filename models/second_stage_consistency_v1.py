import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
import time

from data.transforms import denormalize, normalize, tensor2uint8
from modules.diffusionmodules.util import extract_into_tensor, noise_like
from modules.distributions.distributions import DiagonalGaussianDistribution
from utils import default, instantiate_from_config
from pytorch_lightning.utilities.distributed import rank_zero_only
from metrics.psnr_ssim import calc_psnr_ssim

from .diffusion.ddpm import DDPM, disabled_train


def scalings_for_boundary_conditions(timestep, sigma_data=0.5, timestep_scaling=10.0):
    c_skip = sigma_data**2 / ((timestep / 0.1) ** 2 + sigma_data**2)
    c_out = (timestep / 0.1) / ((timestep / 0.1) ** 2 + sigma_data**2) ** 0.5
    return c_skip, c_out


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


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

        print(
            f"# params of the First Stage: {count_parameters(self.first_stage_model.decoder)}"
        )
        print(f"# params of the Cond Stage: {count_parameters(self.cond_stage_model)}")
        print(f"# params of the Diff Model: {count_parameters(self.model)}")

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

        self.validation_results = []

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
        linear_end=1e-2,
        cosine_s=8e-3,
    ):
        super().register_schedule(
            given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s
        )

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
        scale = (
            torch.tensor(hr.shape[2] / lr.shape[2])
            .repeat(x0.shape[0])
            .long()
            .to(self.device)
        )
        osize = torch.tensor(hr.shape[2]).repeat(x0.shape[0]).long().to(self.device)
        loss, log_dict = self.forward(x0, c_concat, scale, osize, lr=lr, hr=hr)
        return loss, log_dict

    def forward(self, x_start, cond, scale, osize, *args, **kwargs):
        """
        修改为使用 consistency loss 的版本
        """
        ## 是否会影响收敛速度？
        topk = 250
        ##

        t2 = torch.randint(
            0, self.num_timesteps, (x_start.shape[0],), device=self.device
        ).long()

        t1 = t2 - topk
        t1 = torch.where(t1 < 0, torch.zeros_like(t1), t1)  # 保证t1非负

        if self.model.conditioning_key is not None:
            assert cond is not None

        noise = torch.randn_like(x_start)

        x_noisy_t2 = self.q_sample(x_start=x_start, t=t2, noise=noise)
        x_noisy_t1 = self.q_sample(x_start=x_start, t=t1, noise=noise)

        cond_dict = {"c_concat": cond, "c_scale": scale, "c_osize": osize}

        pred_start_t2 = self.p_consistency_function(
            x=x_noisy_t2,
            c=cond_dict,
            t=t2,
            clip_denoised=False,
            quantize_denoised=False,
        )
        with torch.no_grad():
            with self.ema_scope():
                pred_start_t1 = self.p_consistency_function(
                    x=x_noisy_t1,
                    c=cond_dict,
                    t=t1,
                    clip_denoised=False,
                    quantize_denoised=False,
                )

        loss_dict = {}
        prefix = "train" if self.training else "val"

        # consistency loss
        loss_consistency = self.get_loss(pred_start_t1, pred_start_t2, mean=True)
        loss_dict.update({f"{prefix}/loss_consistency": loss_consistency})

        # simple loss
        loss_simple = self.get_loss(pred_start_t2, x_start, mean=True)
        loss_dict.update({f"{prefix}/loss_simple": loss_simple})

        loss = loss_consistency.mean()  # + loss_simple.mean()

        # loss_kd
        if self.l_kd_weight > 0:
            loss_kd = self.get_loss(x_start, cond, mean=True)
            loss += self.l_kd_weight * loss_kd
            loss_dict.update({f"{prefix}/loss_kd": loss_kd})

        loss_dict.update({f"{prefix}/loss": loss})
        return loss, loss_dict

    def apply_model(self, x_noisy, t, cond_dict):
        if not isinstance(cond_dict["c_concat"], list):
            cond_dict["c_concat"] = [cond_dict["c_concat"]]
        x_recon = self.model(x_noisy, t, **cond_dict)
        return x_recon

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def p_consistency_function(
        self,
        x,
        c,
        t,
        clip_denoised: bool,
        quantize_denoised=False,
    ):
        t_in = t
        model_out = self.apply_model(x, t_in, c)

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        # modify for consistency model
        # boundary conditions
        c_skip, c_out = scalings_for_boundary_conditions(t / self.num_timesteps)
        c_skip, c_out = [append_dims(x, x_recon.ndim) for x in [c_skip, c_out]]
        x_recon = c_skip * x + c_out * x_recon

        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        if quantize_denoised:
            x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)

        return x_recon

    def p_sample_loop(
        self,
        cond,
        shape,
        return_intermediates=False,
        x_T=None,
        verbose=True,
        timesteps=None,
        quantize_denoised=False,
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

        skip = self.num_timesteps // timesteps
        seq = list(range(0, self.num_timesteps, skip)) + [self.num_timesteps - 1]
        seq_prev = list(seq)[:-1]
        seq = list(seq)[1:]

        for t, t_prev in zip(reversed(seq), reversed(seq_prev)):
            ts = torch.full((b,), t, device=device, dtype=torch.long)
            ts_prev = torch.full((b,), t_prev, device=device, dtype=torch.long)

            x0 = self.p_consistency_function(
                img,
                cond,
                ts,
                clip_denoised=self.clip_denoised,
                quantize_denoised=quantize_denoised,
            )

            img = self.q_sample(x0, ts_prev)
            intermediates.append(img)

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
        verbose=False,
        timesteps=None,
        quantize_denoised=False,
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
        )

    @torch.no_grad()
    def sample_log(self, cond, batch_size, shape, timesteps, **kwargs):
        samples, intermediates = self.sample(
            cond=cond,
            batch_size=batch_size,
            return_intermediates=True,
            shape=shape,
            timesteps=timesteps,
            **kwargs,
        )

        return samples, intermediates

    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + "_ema": loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(
            loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True
        )
        self.log_dict(
            loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True
        )

        log = self.log_images(batch)
        for rslt, gt in zip(log["sr_s1"].cpu(), log["hr"].cpu()):
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

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        log = self.log_images(
            batch,
            N=128,
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
        return_keys=None,
        **kwargs,
    ):

        log = dict()
        x0, lr, hr, hr_rec = self.get_input(
            batch,
            return_first_stage_outputs=True,
        )

        torch.cuda.synchronize()
        tic = time.time()

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
        log["hr"] = hr
        log["hr_rec"] = hr_rec
        log["lr"] = lr

        # get denoise row
        scale = (
            torch.tensor(hr.shape[2] / lr.shape[2])
            .repeat(x0.shape[0])
            .long()
            .to(self.device)
        )
        osize = torch.tensor(hr.shape[2]).repeat(x0.shape[0]).long().to(self.device)
        cond_dict = {"c_concat": c_concat, "c_scale": scale, "c_osize": osize}
        with self.ema_scope("Plotting"):
            samples_s1, z_denoise_row = self.sample_log(
                cond=cond_dict,
                batch_size=N,
                shape=x0.shape,
                sample_log=1,
                timesteps=1,
            )
            # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
        sr_s1 = self.decode_first_stage(samples_s1, lr, hr.shape[2:])

        # with self.ema_scope("Plotting"):
        #     samples_s4, z_denoise_row = self.sample_log(
        #         cond=cond_dict, batch_size=N, shape=x0.shape, sample_log=1, timesteps=4
        #     )
        #     # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
        # sr_s4 = self.decode_first_stage(samples_s4, lr, hr.shape[2:])

        torch.cuda.synchronize()
        toc = time.time()
        # log["runtime"] = toc - tic

        sr_cond = self.decode_first_stage(c_concat, lr, hr.shape[2:])
        # log["sr_s1"] = sr_s1
        log["sr"] = sr_s1
        # log["sr_s4"] = sr_s4
        log["sr_kd"] = sr_cond

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}

        for key in list(log.keys()):
            if isinstance(log[key], torch.Tensor):
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
