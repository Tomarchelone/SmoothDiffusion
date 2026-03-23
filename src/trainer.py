import random

import torch
import torch.nn as nn
import pytorch_lightning as pl

import time

from model import SNRLoss, SimpleModel, Sampler, LinearLogSNR

from adabelief_pytorch import AdaBelief

class SGMTrainer(pl.LightningModule):
    def __init__(self, T, B):
        super().__init__()

        self.B = B
        self.T = T

        self.snr_schedule = LinearLogSNR(5.0)

        self.automatic_optimization = False

        self.model = SimpleModel()

        self.sampler = Sampler(self.snr_schedule, T, B)

        self.loss = SNRLoss(self.snr_schedule)

        # [0.0, 0.0, 1/T, 2/T, ..., 1]. Total T + 2 points. First zero, last one. Second is zero because it will be shifted
        self.register_buffer("t_range", torch.tensor([0.0] + torch.linspace(0.0, 1.0, T + 1).tolist())) 

    def training_step(self, x0, batch_idx):
        # x0 (B, 2)

        self.t_range = self.sampler.sample_range(x0.device) # (T + 2). [0, shift/T, 2 * shift / T, ..., 1]

        idx, probs_of_sampled_times = self.sampler.sample_indices(x0.device) # (B), (B). idx != 0, idx != "-1"

        s = self.t_range[idx] # (B). For diffusion only

        snr = self.snr_schedule(s) # (B)

        x_s = self.sampler.sample(x0, snr) # B, 2

        x0_hat = self.model(x_s, snr.log()) # B, 2

        rec_loss, diff_loss, prior_loss = self.loss(x0, x0_hat, s) # B
        
        self.sampler.update(diff_loss.detach())

        #diff_loss = diff_loss.mean()
        diff_loss = (diff_loss / (probs_of_sampled_times * self.B * self.T)).sum()
        prior_loss = prior_loss.mean()

        loss = rec_loss + diff_loss + prior_loss

        opt = self.optimizers()
        
        opt.zero_grad()

        self.manual_backward(loss)

        grad_norm = torch.nn.utils.clip_grad_norm_((
            list(self.model.parameters())
        ), 1.0)
        

        self.log("rec_loss", rec_loss)
        self.log("diff_loss", diff_loss.item())
        self.log("prior_loss", prior_loss.item())
        self.log("loss", loss.item())

        self.log("grad_norm", grad_norm.item())

        opt.step()
    
    def configure_optimizers(self):
        return AdaBelief(list(self.model.parameters()), lr=3e-4, eps=1e-16)
    

class ScheduleTrainer(pl.LightningModule):
    def __init__(self, learnable_fn, snr_schedule, L):
        super().__init__()

        self.learnable_fn = learnable_fn
        self.snr_schedule = snr_schedule

        self.automatic_optimization = False

        self.register_buffer("t", torch.linspace(0.0, 1.0, 4097)) # T + 2, so T + 1 is a power of 2

    def training_step(self, _t):
        # t (L)
        
        t = self.t

        self.learnable_fn.compute_all(t)
        cmp = self.learnable_fn.computed

        log_snr_zero = cmp['log_SNR_zero']
        log_snr_one = cmp['log_SNR_one']

        target_log_snr_zero = 5.0
        target_log_snr_one = -5.0

        endpoints_loss = ((log_snr_zero - target_log_snr_zero) ** 2 + (log_snr_one - target_log_snr_one) ** 2).mean() #+ ((cmp['log_SNR'][len(cmp['log_SNR']) * 7 // 10] - 1) ** 2).mean()# ()

        beta_loss = (cmp['beta_one'] - 1) ** 2

        #regularizer_loss = (((cmp['beta'] + cmp['alpha'] ** 2 + cmp['nu'] + cmp['lambda'] ** 2) - (1 + cmp['beta_zero'] + cmp['nu_zero'])) ** 2).mean()

        #regularizer_loss = (((cmp['beta'] + cmp['alpha'] ** 2) - 1) ** 2).mean()

        # regularizer_loss = (
        #     (cmp['beta'] + cmp['alpha'] ** 2) * (cmp['nu'] + cmp['lambda'] ** 2) - (cmp['alpha'] * cmp['lambda'] + cmp['kappa']) ** 2
        # ).var()

        # regularizer_loss = (
        #     4 * cmp['kappa'][:-1]** 2
        #     + (cmp['nu'][:-1] - cmp['f'] * cmp['kappa'][:-1] - cmp['r'] * cmp['beta'][:-1]) ** 2
        #     + (cmp['g'] - 2 * cmp['f'] * cmp['nu'][:-1] - 2 * cmp['r'] * cmp['kappa'][:-1]) ** 2
        #     + (cmp['lambda'][:-1]) ** 2
        #     + (-cmp['f'] * cmp['lambda'][:-1] - cmp['r'] * cmp['alpha'][:-1]) ** 2
        # ).mean() * 0.01

        # regularizer_control_loss = (
        #     self.learnable_fn.computed['f'] ** 2 + self.learnable_fn.computed['r'] ** 2 + self.learnable_fn.computed['g']
        # ).mean() * 0.01
        
        #speed_loss = cmp['nu'].max() * 0.01
        #g_loss = ((1/cmp['g']).max() + cmp['g'].max()) * 0.01
        #nu_loss = ((10/cmp['nu']).max() + cmp['nu'].max()) * 0.01 #+ (cmp['nu_one'] - 1) ** 2

        rho = (cmp['kappa']) / (cmp['beta'] * cmp['nu']) ** 0.5
        rho_loss = (1/(1 - rho.abs())).max() * 0.01

        loss = endpoints_loss + rho_loss + beta_loss #+ speed_loss + g_loss + regularizer_control_loss

        self.log("endpoints_loss", endpoints_loss.item())
        #self.log("max_diff_loss", max_diff_loss.item())
        #self.log("regularizer_loss", regularizer_loss.item())
        #self.log("regularizer_control_loss", regularizer_control_loss.item())
        self.log("rho_loss", rho_loss.item())
        #self.log("g_loss", g_loss.item())
        #self.log("nu_loss", nu_loss.item())
        self.log("beta_loss", beta_loss.item())
        #self.log("speed_loss", speed_loss.item())
        #self.log("start_loss", start_loss.item())
        #self.log("beta_grow_loss", beta_grow_loss.item())
        #self.log("beta_one_loss", beta_one_loss.item())
        
        opt = self.optimizers()
        opt.zero_grad()

        self.manual_backward(loss)

        # grad_norm = torch.nn.utils.clip_grad_norm_((
        #     list(self.learnable_fn.parameters())
        # ), 1.0)

        opt.step()

        self.learnable_fn.flush()        

    def configure_optimizers(self):
        #return AdaBelief(list(self.learnable_fn.parameters()), lr=1e-3, eps=1e-16)
        return torch.optim.AdamW(list(self.learnable_fn.parameters()), lr=1e-3)