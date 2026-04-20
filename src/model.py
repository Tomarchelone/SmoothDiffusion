import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical

from tqdm import tqdm

from solver import Solver

import time

def cosine_log_snr(t):

    alpha2 = torch.cos(math.pi / 2 * t) ** 2
    beta = 1.0 - alpha2

    log_snr = torch.log(alpha2) - torch.log(beta)

    return log_snr

class CosineLogSNR(nn.Module):
    def __init__(self, start, end):
        super().__init__()

        self.start = start
        self.end = end

    def forward(self, outer_t):
        # t (B)

        t = self.start + outer_t * (self.end - self.start)

        alpha2 = torch.cos(math.pi / 2 * t) ** 2
        beta = 1.0 - alpha2

        log_snr = torch.log(alpha2) - torch.log(beta)

        return log_snr
    
class LinearLogSNR(nn.Module):
    def __init__(self, slope):
        super().__init__()

        self.slope = slope

        self.start_snr = math.exp(self.slope)
        self.end_snr = math.exp(-self.slope)

    def forward(self, t):
        # t (B)
        
        return (-(t - 0.5) * 2 * self.slope).exp()
    
    def log_snr(self, t):
        # t (B)
        
        return -(t - 0.5) * 2 * self.slope
    
    def derivative(self, t):

        return (-2 * self.slope) * (-(t - 0.5) * 2 * self.slope).exp()

class Function(nn.Module):
    def __init__(self, out_size=1):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(1, 256)
            , nn.GELU()
            , nn.Linear(256, 256)
            , nn.GELU()
            , nn.Linear(256, out_size)
        )

    def forward(self, t):
        # t (B)

        return self.layers(t.unsqueeze(-1)) # B, out_size

    
class PositiveFunction(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(1, 256)
            , nn.GELU()
            , nn.Linear(256, 256)
            , nn.GELU()
            , nn.Linear(256, 1)
        )

    def forward(self, t):
        # t (B)

        return F.softplus(self.layers(t.unsqueeze(-1)).squeeze(-1)) # B, out_size


class ExpandedSchedule(nn.Module):
    def __init__(self):
        super().__init__()

        self.fr = Function(out_size=2)
        self.g = PositiveFunction()

        self.solver = Solver()

        self.log_beta_nu_zero = nn.Parameter(
            torch.tensor([
                -5.0
                , 0.0
            ])
        ) # 2

        self.log_rho_zero = nn.Parameter(torch.tensor([0.0])) # 1     

        #self.log_alpha_zero = nn.Parameter(torch.tensor([0.0]))
        #self.lambda_zero = nn.Parameter(torch.tensor([0.0]))

        self.register_buffer("lambda_zero", torch.tensor([0.0]))
        self.register_buffer("alpha_zero", torch.tensor([1.0]))
        #self.register_buffer("rho_zero", torch.tensor([0.5]))

        self.computed = {}

    def compute_numerically(self, t_range):
        beta_zero, nu_zero = self.log_beta_nu_zero.exp().chunk(2) # (1), (1)
        rho_zero = torch.sigmoid(self.log_rho_zero)
        kappa_zero = rho_zero * (beta_zero ** 0.5 * nu_zero ** 0.5) # (1)
        
        print(f"beta {beta_zero.item():.6f}, rho {rho_zero.item():.4f}, nu {nu_zero.item():.4f}")

        # nu, kappa, beta, lambda, alpha, 1
        start_tensor = torch.cat((
            beta_zero
            , kappa_zero
            , nu_zero
            , self.alpha_zero
            , self.lambda_zero
            , torch.tensor([1.0]).to(beta_zero.device) # only for bias
        ), dim=0).unsqueeze(0) # 1, 6

        f = self.computed['f'].unsqueeze(-1) # T + 1, 1
        r = self.computed['r'].unsqueeze(-1) # T + 1, 1
        g = self.computed['g'].unsqueeze(-1) # T + 1, 1

        zeros = torch.zeros_like(f) # T + 1, 1
        ones = torch.ones_like(f) # T + 1, 1

        dts = t_range[1:] - t_range[:-1] # T + 1

        # T + 1 MUST be a degree of 2

        # beta' = 2 * kappa
        # kappa' = nu - f * kappa - r * beta
        # nu' = g - 2f * nu - 2r * kappa
        # alpha' = lambda
        # lambda' = -f * lambda -r * alpha
        # 1' = 0


        # DELTAS
        #                               OUT    
        #                beta    kappa      nu   alpha   lambda     1

        #      beta       0        -r       0       0       0       0
        #      kappa      2        -f       -2r     0       0       0
        # IN   nu         0        1        -2f     0       0       0 
        #      alpha      0        0        0       0       -r      0 
        #      lambda     0        0        0       1       -f      0 
        #      1          0        0        g       0       0       0   
        
        deltas = torch.cat((
            torch.cat((zeros, -r, zeros, zeros, zeros, zeros), dim=-1).unsqueeze(1) # T + 1, 1, 6
            , torch.cat((2 * ones, -f, -2 * r, zeros, zeros, zeros), dim=-1).unsqueeze(1) # T + 1, 1, 6
            , torch.cat((zeros, ones, -2 * f, zeros, zeros, zeros), dim=-1).unsqueeze(1) # T + 1, 1, 6
            , torch.cat((zeros, zeros, zeros, zeros, -r, zeros), dim=-1).unsqueeze(1) # T + 1, 1, 6
            , torch.cat((zeros, zeros, zeros, ones, -f, zeros), dim=-1).unsqueeze(1) # T + 1, 1, 6
            , torch.cat((zeros, zeros, g, zeros, zeros, zeros), dim=-1).unsqueeze(1) # T + 1, 1, 6
        ), dim=1) * dts.unsqueeze(-1).unsqueeze(-1) # T + 1, 6, 6

        transforms = deltas + torch.eye(6, device=deltas.device).unsqueeze(0) # (T + 1, 6, 6) + (1, 6, 6) = (T + 1, 6, 6)

        solved = self.solver(start_tensor,transforms) # T + 2, 6

        self.computed['beta'] = solved[:, 0] # T + 2
        self.computed['kappa'] = solved[:, 1] # T + 2
        self.computed['nu'] = solved[:, 2] # T + 2
        self.computed['alpha'] = solved[:, 3] # T + 2
        self.computed['lambda'] = solved[:, 4] # T + 2

    def compute_edges(self):
        for k in list(self.computed.keys()):
            self.computed[f'{k}_zero'] = self.computed[k][:1] # 1
            self.computed[f'{k}_one'] = self.computed[k][-1:] # 1

    def compute_all(self, t_range):
        # t_range (T + 2): first element is always zero, last - always one. Times are sorted

        assert not self.computed

        f, r = self.fr(t_range[:-1]).chunk(2, dim=-1) # (T + 1, 1), (T + 1, 1).

        g = self.g(t_range[:-1])

        self.computed['f'] = f.squeeze(-1) # T + 1
        self.computed['r'] = r.squeeze(-1) # T + 1
        self.computed['g'] = g.squeeze(-1) # T + 1

        self.compute_numerically(t_range)

        self.computed['mu'] = torch.cat(
            (
                self.computed['alpha'].unsqueeze(-1)
                , self.computed['lambda'].unsqueeze(-1)
            ), dim=-1
        ) # T, 2

        self.computed['cov_matrix'] = torch.cat(
            (
                torch.cat((self.computed['beta'].unsqueeze(-1), self.computed['kappa'].unsqueeze(-1)), dim=-1).unsqueeze(-2) # T, 1, 2
                , torch.cat((self.computed['kappa'].unsqueeze(-1), self.computed['nu'].unsqueeze(-1)), dim=-1).unsqueeze(-2) # T, 1, 2
            )
            , dim=-2
        ) # T, 2, 2

        # self.computed['loss_koeff'] = (
        #     0.5 * self.computed['g'] * (
        #         (self.computed['kappa'] * self.computed['alpha'] - self.computed['lambda'] * self.computed['beta'])
        #         /(self.computed['beta'] * self.computed['nu'] - self.computed['kappa']**2)
        #     ) ** 2
        # )

        self.computed['log_SNR'] = (
                (self.computed['beta'] * self.computed['lambda'] ** 2 + self.computed['nu'] * self.computed['alpha'] ** 2 - 2 * self.computed['kappa'] * self.computed['alpha'] * self.computed['lambda']).log()
                - (self.computed['beta'] * self.computed['nu'] - self.computed['kappa']**2).log()
        )

        self.compute_edges()

    def flush(self):
        self.computed = {}
    
    
class Sampler(nn.Module):
    def __init__(self, snr_schedule, T, B):
        super().__init__()

        self.T = T # Number of ODE steps
        self.B = B # Batch size

        self.register_buffer("loss_running_means", torch.ones(T))
        self.register_buffer("t_range", torch.tensor([0.0] + torch.linspace(0.0, 1.0, T + 1).tolist())) # [0.0, 0.0, 1/T, 2/T, ..., 1]. Total T + 2 points

        self.decay = 0.999

        self.last_indices = None

        self.snr_schedule = snr_schedule

    def sample_range(self, device):
        T = self.T

        shift = torch.rand(1).to(device) # (1)

        t_range = self.t_range.clone() # T + 2
        t_range[1:-1] += shift / T # [0, shift, 1/T + shift..., 1], total T + 2

        return t_range

    def sample_indices(self, device):
        B = self.B

        loss_running_means_sqrt = self.loss_running_means.to(device) ** 0.5 # T
        probs = loss_running_means_sqrt / loss_running_means_sqrt.sum() # T 

        indices = Categorical(probs=probs).sample((B,)).to(device) # B
        self.last_indices = indices.clone() # B

        probs_of_sampled_times = probs[indices] # B

        # Indices are now from 1 to T (inclusive). So all elements can be chosen except the first and the last
        indices = indices + 1

        return indices, probs_of_sampled_times

    def sample(self, x0, snr):
        # x_0 (B, 2)
        # snr (B)

        alpha = ((snr / (snr + 1)) ** 0.5).unsqueeze(-1) # (B, 1)
        beta = (1/(snr + 1)).unsqueeze(-1) # (B, 1)

        x_s = alpha * x0 + (beta ** 0.5) * torch.randn(x0.size(), device=x0.device)    
        
        return x_s # B, 2

    def update(self, loss):
        indices = self.last_indices

        losses_list = loss.cpu().tolist()
        idx_list = indices.cpu().tolist()

        loss_running_means_list = self.loss_running_means.cpu().tolist()

        for b in range(self.B):
            l = losses_list[b]
            idx = idx_list[b]

            loss_running_means_list[idx] = loss_running_means_list[idx] * self.decay + (1 - self.decay) * l ** 2

        self.loss_running_means = torch.tensor(loss_running_means_list).to("cuda")

        pass

class SNRLoss(nn.Module):
    def __init__(self, snr_schedule):
        super().__init__()

        self.snr_schedule = snr_schedule

    def forward(self, x0, x0_hat, s):
        # x0 (B, 2)
        # x0_hat (B, 2)
        # s (B)

        snr = self.snr_schedule(s) # B
        snr_d = self.snr_schedule.derivative(s) # B

        snr0 = self.snr_schedule.start_snr
        snr1 = self.snr_schedule.end_snr

        rec_loss = 0.5 * math.log(2 * math.pi) -0.5 * math.log(snr0 + 1) + 0.5 * (1 + 1/snr0)

        diff_loss = (-1/2 * snr_d.unsqueeze(-1) * (x0_hat - x0) ** 2).mean(dim=-1) # B
    
        prior_loss = 0.5 * snr1 * x0 ** 2 

        return rec_loss, diff_loss, prior_loss

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(3, 256)
            , nn.GELU()
            , nn.Linear(256, 256)
            , nn.GELU()
            , nn.Linear(256, 2)
        )
    
    def forward(self, x_estimate, log_snr):
        # x_estimate (B, 2)
        # snr (B)

        together = torch.cat((x_estimate, log_snr.unsqueeze(-1)), dim=1) # B, 3

        x0_hat = self.layers(together) # B, 2

        return x0_hat # B, 2
    
    
class Generator(nn.Module):
    def __init__(self, schedule, model, T, B):
        super().__init__()

        self.T = T # Number of ODE steps
        self.B = B # Batch size

        
        self.register_buffer("t_range", torch.linspace(0.0, 1.0, T)) # [0.0, 1/T, 2/T, ..., 1]. Total T points

        self.schedule = schedule

        self.model = model

    def generate(self):
         with torch.no_grad():
            self.schedule.compute_all(self.t_range)
        
            dt = 1 / (self.T-1)

            track = []

            z_t = self.sample_from_prior() # B, 2, 2

            track.append(z_t)

            for neg_idx in tqdm(range(1, self.T)):
                idx = -neg_idx

                noised_for_model = self.best_estimate(z_t, idx) # B, 2

                log_snr = self.schedule.computed['log_SNR'][idx]

                x0_hat = self.model(noised_for_model, torch.tensor([log_snr], device=z_t.device).expand((self.B,)))

                z_t = self.generative_step(z_t, idx, dt, x0_hat)

                track.append(z_t)

            x0_hat_final = self.best_estimate(z_t, 1)

            #x0_hat_final = z_t[:, 0]

            track = torch.stack(track, dim=0)

            return x0_hat_final, track

    def sample_from_prior(self):
        mu = torch.zeros((self.B, 2, 2)).to("cuda") # B, 2, 2

        cov_matrix = self.schedule.computed['cov_matrix_one'].unsqueeze(1) # B, 1, 2, 2

        z_1 = MultivariateNormal(mu, cov_matrix).rsample() # B, 2, 2

        return z_1 # B, 2, 2
        
    def best_estimate(self, z_t, idx):
        alpha = self.schedule.computed['alpha'][idx]
        beta = self.schedule.computed['beta'][idx]
        nu = self.schedule.computed['nu'][idx]
        kappa = self.schedule.computed['kappa'][idx]
        _lambda = self.schedule.computed['lambda'][idx]

        x_t, y_t = z_t[:, :, 0], z_t[:, :, 1] # B, 2

        normalizing_denom = (
            (nu * alpha ** 2 + beta * _lambda ** 2 - 2 * alpha * _lambda * kappa)
            * ( (alpha ** 2 + beta) * (_lambda ** 2 + nu) - (alpha * _lambda + kappa) ** 2)
        ) ** 0.5

        normalized_estimate =  ((alpha * nu - _lambda * kappa) * x_t + (beta* _lambda - alpha * kappa) * y_t) / normalizing_denom
    
        return normalized_estimate

    def generative_step(
                self
                , z_t
                , idx
                , dt
                , x0_hat
            ):
            # z_t (B, 2, 2)
            # idx (1)
            # dt (1)
            # x0_hat (B, 2)

            B = z_t.size(0)

            z_t = torch.flatten(z_t, 0, 1) # B * 2, 2
            x0_hat = torch.flatten(x0_hat, 0, 1) # B * 2

            g_s = self.schedule.computed['g'][idx] # ()
            beta_s = self.schedule.computed['beta'][idx] # ()
            kappa_s = self.schedule.computed['kappa'][idx] # ()
            nu_s = self.schedule.computed['nu'][idx] # ()
            #mu_s = self.schedule.computed['mu'][idx] # ()
            alpha_s = self.schedule.computed['alpha'][idx] # ()
            lmbda_s = self.schedule.computed['lambda'][idx] # ()

            f_s = self.schedule.computed['f'][idx] # ()
            r_s = self.schedule.computed['r'][idx] # ()
            
            #z_t = z_t.unsqueeze(-1) # B, 2, 1

            u_t = z_t[:, 0] # B * 2
            v_t = z_t[:, 1] # B * 2

            #mu_s = (x0_hat.unsqueeze(1) * mu_s.to(x0_hat.device)).unsqueeze(-1) # B, 2, 1

            Sigma_backward = g_s * torch.tensor([
                [1/3 * dt ** 3, -1/2 * dt**2]
                , [-1/2 * dt**2, dt]
            ]).unsqueeze(0).to(x0_hat.device) # 1, 2, 2

            delta_mu_st = dt * torch.stack(
                (
                    -v_t
                    , r_s * u_t + f_s * v_t + (g_s/ (beta_s * nu_s - kappa_s ** 2)) * (-kappa_s * (alpha_s * x0_hat - u_t) + beta_s * (lmbda_s * x0_hat - v_t))
                )
                , dim=-1
            )# B * 2, 2

            z_s = MultivariateNormal(z_t + delta_mu_st, Sigma_backward).rsample() # B, 2

            z_s = torch.unflatten(z_s, 0, (B, 2))

            return z_s
