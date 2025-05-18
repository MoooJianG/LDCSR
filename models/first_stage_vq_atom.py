import pytorch_lightning as pl
import torch
from modules.distributions.distributions import DiagonalGaussianDistribution
from utils import instantiate_from_config
from data.transforms import normalize, denormalize
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

class AutoencoderVQ(pl.LightningModule):
    def __init__(
        self,
        encoder_config,
        decoder_config,
        lossconfig,
        n_embed,
        embed_dim,
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
        monitor=None,
        remap = None,
        sane_index_shape=False,
        **args,
    ):
        super().__init__()
        self.image_key = image_key
        self.encoder = instantiate_from_config(encoder_config)
        self.decoder = instantiate_from_config(decoder_config)
        if lossconfig:
            self.loss = instantiate_from_config(lossconfig)
            self.loss.n_classes = n_embed
        assert encoder_config["params"]["double_z"] == False
        self.quant_conv = torch.nn.Conv2d(
            encoder_config["params"]["z_channels"], embed_dim, 1
        )
        self.post_quant_conv = torch.nn.Conv2d(
            embed_dim, encoder_config["params"]["z_channels"], 1
        )
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.quantize = VectorQuantizer(
            n_embed,
            embed_dim,
            beta=0.25,
            remap=remap,
            sane_index_shape=sane_index_shape,
        )
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
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, hr, lr):
        h = self.encoder(hr, lr)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, z, lr, out_size):
        z = self.post_quant_conv(z)
        dec = self.decoder(z, lr, out_size)
        return dec
    
    def decode_code(self, code_b, lr, out_size):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b, lr, out_size)
        return dec

    def forward(self, hr, lr, return_pred_indices=False):
        quant, diff, (_, _, ind) = self.encode(hr, lr)
        dec = self.decode(quant, lr, hr.shape[2:])
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff

    def get_input(self, batch):
        hr, lr = batch["hr"].to(self.device), batch["lr"].to(self.device)
        hr = self.normalize(hr)
        lr = self.normalize(lr)
        return hr, lr

    def training_step(self, batch, batch_idx, optimizer_idx):
        hr, lr = self.get_input(batch)
        hr_rec, qloss, ind = self.forward(hr, lr, return_pred_indices=True)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(
                qloss,
                hr,
                hr_rec,
                optimizer_idx,
                self.global_step,
                last_layer=self.get_last_layer(),
                split="train",
                predicted_indices=ind,
            )

            self.log_dict(
                log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True
            )
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(
                qloss,
                hr,
                hr_rec,
                optimizer_idx,
                self.global_step,
                last_layer=self.get_last_layer(),
                split="train",
            )
            self.log_dict(
                log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True
            )
            return discloss

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        return log_dict

    def _validation_step(self, batch, batch_idx):
        hr, lr = self.get_input(batch)
        hr_rec, qloss, ind = self.forward(hr, lr, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(
            qloss,
            hr,
            hr_rec,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val",
            predicted_indices=ind,
        )

        discloss, log_dict_disc = self.loss(
            qloss,
            hr,
            hr_rec,
            1,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val",
            predicted_indices=ind,
        )
        self.log(
            f"val/aeloss",
            aeloss,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
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
            + list(self.quantize.parameters())
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
        if not only_inputs:
            xrec, _ = self.forward(x, lr)
            log["reconstructions"] = xrec
        log["inputs"] = x

        for key in list(log.keys()):
            log[key] = self.denormalize(log[key].clamp_(-1.0, 1.0))
        return log

class VQModelInterface(AutoencoderVQ):
    def __init__(self, embed_dim, *args, **kwargs):
        super().__init__(embed_dim=embed_dim, *args, **kwargs)
        self.embed_dim = embed_dim

    def encode(self, hr, lr):
        h = self.encoder(hr, lr)
        h = self.quant_conv(h)
        return h

    def decode(self, h, lr, out_size, force_not_quantize=False):
        # also go through quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant, lr, out_size)
        return dec