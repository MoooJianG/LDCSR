import pytorch_lightning as pl
import torch
from modules.distributions.distributions import DiagonalGaussianDistribution
from utils import instantiate_from_config
from data.transforms import normalize, denormalize


class AutoencoderKL(pl.LightningModule):
    def __init__(
        self,
        encoder_config,
        decoder_config,
        lossconfig,
        embed_dim,
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
        monitor=None,
        **args,
    ):
        super().__init__()
        self.image_key = image_key
        self.encoder = instantiate_from_config(encoder_config)
        self.decoder = instantiate_from_config(decoder_config)
        self.loss = instantiate_from_config(lossconfig)
        assert encoder_config["params"]["double_z"]
        self.quant_conv = torch.nn.Conv2d(
            2 * encoder_config["params"]["z_channels"], 2 * embed_dim, 1
        )
        self.post_quant_conv = torch.nn.Conv2d(
            embed_dim, encoder_config["params"]["z_channels"], 1
        )
        self.embed_dim = embed_dim
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.mean, self.std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        self.normalize = lambda tensor: normalize(
            tensor, self.mean, self.std, inplace=False
        )
        self.denormalize = lambda tensor: denormalize(
            tensor, self.mean, self.std, inplace=False
        )

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(
            f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def encode(self, x, z_size):
        h = self.encoder(x, z_size)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z, out_size):
        z = self.post_quant_conv(z)
        dec = self.decoder(z, out_size)
        return dec

    def forward(self, input, z_size, sample_posterior=True):
        posterior = self.encode(input, z_size)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z, input.shape[2:])
        return dec, posterior

    def get_input(self, batch):
        hr, lr = batch["hr"], batch["lr"]
        hr = self.normalize(hr)
        return hr, lr

    def training_step(self, batch, batch_idx, optimizer_idx):
        hr, lr = self.get_input(batch)
        reconstructions, posterior = self(hr, lr.shape[2:])

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(
                hr,
                reconstructions,
                posterior,
                optimizer_idx,
                self.global_step,
                last_layer=self.get_last_layer(),
                split="train",
            )
            self.log(
                "aeloss",
                aeloss,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
            )
            self.log_dict(
                log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False
            )
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(
                hr,
                reconstructions,
                posterior,
                optimizer_idx,
                self.global_step,
                last_layer=self.get_last_layer(),
                split="train",
            )

            self.log(
                "discloss",
                discloss,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
            )
            self.log_dict(
                log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False
            )
            return discloss

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        return log_dict

    def _validation_step(self, batch, batch_idx):
        hr, lr = self.get_input(batch)
        reconstructions, posterior = self(hr, lr.shape[2:])
        aeloss, log_dict_ae = self.loss(
            hr,
            reconstructions,
            posterior,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val",
        )

        discloss, log_dict_disc = self.loss(
            hr,
            reconstructions,
            posterior,
            1,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val",
        )

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        log = self.log_images(batch)
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.quant_conv.parameters())
            + list(self.post_quant_conv.parameters()),
            lr=lr,
            betas=(0.5, 0.9),
        )
        opt_disc = torch.optim.Adam(
            self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9)
        )
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.get_last_layer()

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x, lr = self.get_input(batch)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x, lr.shape[2:])
            log["reconstructions"] = xrec
        log["inputs"] = x

        for key in list(log.keys()):
            log[key] = self.denormalize(log[key].clamp_(-1.0, 1.0))
        return log
