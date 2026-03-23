from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from trainer import SGMTrainer, ScheduleTrainer
from data import TwoNormalsDataset, TimeDataset, collate_fn_two_dimensions, collate_fn_one_dimension

from model import ExpandedSchedule, CosineLogSNR, LinearLogSNR

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def train_schedule():
    dataset = TimeDataset(start=0.0, end=1.0)

    L = 4096

    train_dataloader = DataLoader(dataset, batch_size=L, collate_fn=collate_fn_one_dimension)

    model_fn = ExpandedSchedule()

    top_model = ScheduleTrainer(model_fn, LinearLogSNR(slope=5.0), L).to("cuda")

    checkpointing = ModelCheckpoint(
        every_n_train_steps=500
    )

    trainer = pl.Trainer(
        log_every_n_steps=10
        , max_epochs=1
        , accelerator='cuda'
        , callbacks=[checkpointing]
    )

    trainer.fit(top_model, train_dataloader)

def train():
    dataset = TwoNormalsDataset()

    T = 1000
    B = 4096

    train_dataloader = DataLoader(dataset, batch_size=B, collate_fn=collate_fn_two_dimensions)

    top_model = SGMTrainer(T, B)

    checkpointing = ModelCheckpoint(
        every_n_train_steps=500
    )

    trainer = pl.Trainer(
        log_every_n_steps=10
        , max_epochs=1
        , accelerator='cuda'
        , callbacks=[checkpointing]
    )

    trainer.fit(top_model, train_dataloader)

def test(model_path, save_name):
    T = 1000
    B = 10000

    top_model = SGMTrainer(T, B)

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    top_model.register_buffer("t_range", None)
    top_model.t_range = None
    top_model.load_state_dict(checkpoint["state_dict"], strict=False)

    model = top_model.model
    model = model.eval()

    model = model.to("cuda")

    with torch.no_grad():
        sampler = top_model.sampler.to("cuda")
        schedule = top_model.schedule.to("cuda")

        N = 1000
        dt = 1 / (N-1)

        t_range = torch.linspace(0.0, 1.0, N).to("cuda")
        schedule.compute_all(t_range)

        for name in ['beta', 'kappa', 'nu', 'g', 'alpha', 'f', 'r', 'loss_koeff']:
            plt.plot(t_range.cpu().numpy(), schedule.computed[name].cpu().numpy(), label=name)
        
            plt.legend()
            plt.savefig(f"images/ode_{name}.png")

            plt.clf()

        z_t = sampler.sample_from_prior(B)

        for neg_idx in tqdm(range(1, N)):
            idx = -neg_idx

            t = t_range[idx]

            x_t, y_t = z_t[:, 0], z_t[:, 1]

            x0_hat = model(x_t, y_t, t.expand((B,)))

            z_t = sampler.generative_step(z_t, idx, dt, x0_hat)

        x0_hat_final = z_t[:, 0]

    data = x0_hat_final.cpu().numpy()

    sns.histplot(data, kde=True, bins=100, color='purple')

    plt.savefig(f"images/{save_name}")


def visualize_reference_density():
    dataset = TwoNormalsDataset()

    points = []

    step = 0
    for point in dataset:
        points.append(point.item())

        step += 1
        if step == 4096:
            break

    sns.histplot(points, kde=True, bins=100, color='purple')

    plt.savefig('reference.png')

def main():
    train_schedule()
    
    #train()

    # test(
    #     "lightning_logs/uniform_no_h/checkpoints/epoch=0-step=5000.ckpt"
    #     , "ode_uniform_no_h.png"
    # )

    #reference()

if __name__ == "__main__":
    main()