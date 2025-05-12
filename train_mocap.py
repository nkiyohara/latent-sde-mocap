# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications Copyright 2025 Naoki Kiyohara
# Modified from the original latent SDE Lorenz attractor example (in torchsde) to:
# - Use Motion Capture dataset as training target
# - Implement architecture from "Scalable Gradients for Stochastic Differential
#   Equations" (Li et al., AISTATS 2020)
# - Add Weights & Biases logging and model checkpointing

import os
from typing import Literal, Union

import fire
import numpy as np
import time
import torch
import torchsde
import tqdm
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from loguru import logger
from scipy.io import loadmat
from torch import nn, optim
from torch.distributions import Normal
from torch.utils.flop_counter import FlopCounterMode  # Official FLOPs counter for PyTorch ≥1.13
import wandb
# torch.autograd.set_detect_anomaly(True)


def set_seed(seed: int):
    """Set random seed for reproducibility across all libraries."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # For reproducible operations on GPU
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # For reproducible operations on CPU
    # torch.use_deterministic_algorithms(True)


def load_mocap_data_many_walks(
    mocap_data_path: Union[str, os.PathLike],
    dataset_type: Literal["train", "val", "test"] = "train",
    dt: float = 0.1,
) -> tuple[Float[np.ndarray, "N L D"], Float[np.ndarray, "N L"]]:
    """
    Loads MoCap data (many separate walks) from a .mat file.

    Args:
        mocap_data_path: Path to the .mat file (e.g., 'mocap35.mat').
        dataset_type: One of 'train', 'val', or 'test'.
        dt: The time step used to construct the time axis.

    Returns:
        A tuple (X, t):
          - X: shape (N, T, D)
          - t: shape (N, T)
    """
    mocap_data = loadmat(mocap_data_path)

    # Select dataset
    if dataset_type == "train":
        X = mocap_data["Xtr"]
    elif dataset_type == "val":
        X = mocap_data["Xval"]
    elif dataset_type == "test":
        X = mocap_data["Xtest"]
    else:
        raise ValueError(f"Invalid dataset_type: {dataset_type}")

    # Create time arrays
    t = dt * np.arange(X.shape[1], dtype=np.float32)  # shape (T,)
    t = np.tile(t, [X.shape[0], 1])  # shape (N, T)

    # Convert to jax arrays
    X = np.array(X)
    t = np.array(t)

    return X, t


@jaxtyped(typechecker=beartype)
def create_sliding_windows_single_sequence(
    X_seq: Float[np.ndarray, "L D"], t_seq: Float[np.ndarray, "L"], window_size: int
) -> tuple[
    Float[np.ndarray, "T window_size D"], Float[np.ndarray, "T"], Int[np.ndarray, "T"]
]:
    """
    Creates sliding windows of size `window_size` for a single sequence (walk).
    """
    T, D = X_seq.shape

    X_windows_list = []
    t_windows_list = []
    idx_windows_list = []
    # Slide a window of length window_size along this single sequence
    for start_idx in range(T - window_size + 1):
        end_idx = start_idx + window_size
        X_win = X_seq[start_idx:end_idx, :]  # shape (window_size, D)
        t_win = t_seq[end_idx - 1]  # last time of the window

        X_windows_list.append(X_win)
        t_windows_list.append(t_win)
        idx_windows_list.append(start_idx)

    # Stack into single arrays using numpy instead of jax
    X_windows = np.stack(X_windows_list, axis=0)  # shape (W, window_size, D)
    t_windows = np.array(t_windows_list)  # shape (W,)
    idx_windows = np.array(idx_windows_list)  # shape (W,)

    return X_windows, t_windows, idx_windows


def make_dataset(
    mocap_data_path, dataset_type, device
) -> tuple[Float[torch.Tensor, "T B W D"], Float[torch.Tensor, "T"]]:
    window_size = 3
    X_raw, t_raw = load_mocap_data_many_walks(mocap_data_path, dataset_type)

    all_windows = []
    for seq_idx in range(X_raw.shape[0]):
        X_seq = X_raw[seq_idx]
        t_seq = t_raw[seq_idx]
        windows, t_windows, _ = create_sliding_windows_single_sequence(
            X_seq, t_seq, window_size
        )
        all_windows.append(windows)
    all_windows = np.stack(all_windows).transpose(1, 0, 2, 3)
    all_windows = torch.Tensor(all_windows).to(device)
    t_windows = torch.Tensor(t_windows).to(device)
    return all_windows, t_windows


class LinearScheduler(object):
    def __init__(self, iters, maxval=1.0):
        self._iters = max(1, iters)
        self._val = maxval / self._iters
        self._maxval = maxval

    def step(self):
        self._val = min(self._maxval, self._val + self._maxval / self._iters)

    @property
    def val(self):
        return self._val


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(start_dim=-2, end_dim=-1),
            nn.Linear(150, 30),
            nn.Softplus(),
            nn.Linear(30, 30),
            nn.Softplus(),
            nn.Linear(30, 15),
        )

    def forward(
        self, inp: Float[torch.Tensor, "T B 3 50"]
    ) -> tuple[
        Float[torch.Tensor, "T B 6"],
        Float[torch.Tensor, "T B 6"],
        Float[torch.Tensor, "T B 3"],
    ]:
        qz0_mean, qz0_logstd, context = torch.split(
            self.net(inp), split_size_or_sections=[6, 6, 3], dim=-1
        )
        return qz0_mean, qz0_logstd, context


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 30),
            nn.Softplus(),
            nn.Linear(30, 30),
            nn.Softplus(),
            nn.Linear(30, 100),
        )

    def forward(
        self, inp: Float[torch.Tensor, "T B 6"]
    ) -> tuple[Float[torch.Tensor, "T B 50"], Float[torch.Tensor, "T B 50"]]:
        px_mean, px_logstd = torch.split(
            self.net(inp), split_size_or_sections=[50, 50], dim=-1
        )
        return px_mean, px_logstd


class LatentSDE(nn.Module):
    noise_type = "diagonal"

    def __init__(self, sde_type="ito"):
        super(LatentSDE, self).__init__()
        self.sde_type = sde_type
        self.clamp_range = 6.0
        latent_size = 6
        context_size = 3
        self.encoder = Encoder()
        self.decoder = Decoder()

        self.f_net = nn.Sequential(
            nn.Linear(latent_size + 1 + context_size, 30),
            nn.Softplus(),
            nn.Linear(30, 6),
        )
        self.h_net = nn.Sequential(
            nn.Linear(latent_size + 1, 30), nn.Softplus(), nn.Linear(30, 6)
        )

        # This needs to be an element-wise function for the SDE to satisfy diagonal noise.
        self.g_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2, 30), nn.Softplus(), nn.Linear(30, 1), nn.Sigmoid()
                )
                for _ in range(latent_size)
            ]
        )

        self.pz0_mean = nn.Parameter(torch.zeros(1, latent_size))
        self.pz0_logstd = nn.Parameter(torch.zeros(1, latent_size))

        self._ctx = None

    def contextualize(
        self, ctx: tuple[Float[torch.Tensor, "T"], Float[torch.Tensor, "T B 3"]]
    ):
        self._ctx = ctx  # A tuple of tensors of sizes (T,), (T, batch_size, d).

    def f(self, t, y):
        ts, ctx = (
            self._ctx if self._ctx is not None else (None, None)
        )  # Handle None case
        if ts is None or ctx is None:
            raise ValueError("Context not set. Call contextualize() first.")

        idx = torch.searchsorted(ts, t, right=True)
        i = min(int(idx.item()), len(ts) - 1)  # Convert tensor to int
        t = t.expand(y.shape[0], 1)  # Expand t to match batch size of y
        return self.f_net(torch.cat((t, y, ctx[i]), dim=1))

    def h(self, t, y):
        t = t.expand(y.shape[0], 1)  # Expand t to match batch size of y
        return self.h_net(torch.cat((t, y), dim=1))

    def g(self, t, y):  # Diagonal diffusion.
        batch_size = y.shape[0]
        y = torch.split(y, split_size_or_sections=1, dim=1)
        t = t.expand(batch_size, 1)  # Expand t to match batch size of y
        out = [
            g_net_i(torch.cat((t, y_i), dim=1))
            for (g_net_i, y_i) in zip(self.g_nets, y)
        ]
        return torch.cat(out, dim=1)

    def forward(
        self,
        xs: Float[torch.Tensor, "T B 150"],
        ts: Float[torch.Tensor, "T"],
        adjoint=False,
        method="euler",
        dt=0.02,
    ):
        # Contextualization is only needed for posterior inference.
        qz0_mean, qz0_logstd, ctx = self.encoder(torch.flip(xs, dims=(0,)))
        qz0_mean, qz0_logstd = qz0_mean[-1], qz0_logstd[-1]  # first time step
        ctx = torch.flip(ctx, dims=(0,))
        self.contextualize((ts, ctx))

        # Use self.clamp_range instead of hardcoded 6.0
        z0 = qz0_mean + torch.exp(
            torch.clamp(qz0_logstd, -self.clamp_range, self.clamp_range)
        ) * torch.randn_like(qz0_mean)

        if adjoint:
            # Must use the argument `adjoint_params`, since `ctx` is not part of the input to `f`, `g`, and `h`.
            adjoint_params = (
                (ctx,)
                + tuple(self.f_net.parameters())
                + tuple(self.g_nets.parameters())
                + tuple(self.h_net.parameters())
            )
            zs, log_ratio = torchsde.sdeint_adjoint(
                self,
                z0,
                ts,
                adjoint_params=adjoint_params,
                dt=dt,
                logqp=True,
                method=method,
            )
        else:
            zs, log_ratio = torchsde.sdeint(
                self, z0, ts, dt=dt, logqp=True, method=method
            )

        _xs_mean, _xs_logstd = self.decoder(zs)
        # Use self.clamp_range instead of hardcoded 6.0
        xs_dist = Normal(
            loc=_xs_mean,
            scale=torch.exp(
                torch.clamp(_xs_logstd, -self.clamp_range, self.clamp_range)
            ),
        )
        log_pxs = xs_dist.log_prob(xs[:, :, -1, :]).sum(dim=(0, 2)).mean(dim=0)

        # Use self.clamp_range instead of hardcoded 6.0
        qz0 = torch.distributions.Normal(
            loc=qz0_mean,
            scale=torch.exp(
                torch.clamp(qz0_logstd, -self.clamp_range, self.clamp_range)
            ),
        )
        pz0 = torch.distributions.Normal(
            loc=self.pz0_mean,
            scale=torch.exp(
                torch.clamp(self.pz0_logstd, -self.clamp_range, self.clamp_range)
            ),
        )
        logqp0 = torch.distributions.kl_divergence(qz0, pz0).sum(dim=1).mean(dim=0)
        logqp_path = log_ratio.sum(dim=0).mean(dim=0)
        return log_pxs, logqp0 + logqp_path

    @torch.no_grad()
    def sample(self, batch_size, ts, bm=None, dt=0.02):
        eps = torch.randn(
            size=(batch_size, *self.pz0_mean.shape[1:]), device=self.pz0_mean.device
        )
        # Use self.clamp_range instead of hardcoded 6.0
        z0 = (
            self.pz0_mean
            + torch.exp(
                torch.clamp(self.pz0_logstd, -self.clamp_range, self.clamp_range)
            )
            * eps
        )
        zs = torchsde.sdeint(self, z0, ts, names={"drift": "h"}, dt=dt, bm=bm)
        # Most of the times in ML, we don't sample the observation noise for visualization purposes.
        _xs_mean, _ = self.decoder(zs)
        return _xs_mean

    @torch.no_grad()
    def sample_from_posterior(self, xs, ts, bm=None, dt=0.02):
        qz0_mean, qz0_logstd, ctx = self.encoder(torch.flip(xs, dims=(0,)))
        qz0_mean, qz0_logstd = qz0_mean[-1], qz0_logstd[-1]  # first time step
        ctx = torch.flip(ctx, dims=(0,))
        self.contextualize((ts, ctx))

        # Use self.clamp_range instead of hardcoded 6.0
        z0 = qz0_mean + torch.exp(
            torch.clamp(qz0_logstd, -self.clamp_range, self.clamp_range)
        ) * torch.randn_like(qz0_mean)
        zs = torchsde.sdeint(self, z0, ts, names={"drift": "h"}, dt=dt, bm=bm)
        _xs_mean, _ = self.decoder(zs)
        return _xs_mean

    @torch.no_grad()
    def mse(self, xs, ts, bm=None):
        xs_mean = self.sample_from_posterior(xs, ts, bm)
        return torch.mean((xs[:, :, -1] - xs_mean) ** 2)

    @torch.no_grad()
    def mean_mse(self, xs, ts, num_samples=100, bm=None):
        seq_len, batch_size, window_size, dim = xs.shape
        # Reshape inputs to (seq_len, num_samples * batch_size, dim)
        inputs = xs.repeat(1, num_samples, 1, 1).view(
            seq_len, num_samples * batch_size, window_size, dim
        )
        trajs = self.sample_from_posterior(inputs, ts, bm).view(
            seq_len, num_samples, batch_size, dim
        )
        traj_mean = torch.mean(trajs, dim=1)
        return torch.mean((traj_mean - xs[:, :, -1]) ** 2)


@torch.no_grad()
def benchmark_sampling(model: LatentSDE,
                       ts: torch.Tensor,
                       device: torch.device,
                       batch_size: int = 100,
                       repeats: int = 10):
    """Measurement: Sampling time (mean±std) and FLOPs"""
    # --------- Time measurement ---------
    times = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        _ = model.sample(batch_size=batch_size, ts=ts)   # Single sampling
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)
    times = np.asarray(times)
    mean_ms = times.mean() * 1e3
    std_ms  = times.std()  * 1e3

    # --------- FLOPs measurement ---------
    flop_counter = FlopCounterMode(display=False)
    with flop_counter:
        _ = model.sample(batch_size=batch_size, ts=ts)
    total_flops = flop_counter.get_total_flops()         # 64bit int
    gflops = total_flops / 1e9

    # --------- Log output ---------
    logger.info(f"[Benchmark] sample(bs={batch_size}) "
                f"{mean_ms:.2f} ± {std_ms:.2f} ms, {gflops:.2f} GFLOPs")

    wandb.log(
        {
            "bench/sample_ms_mean": mean_ms,
            "bench/sample_ms_std":  std_ms,
            "bench/sample_flops":   total_flops,
        },
        commit=False,  # When you want to record with training step
    )


def main(
    lr_init=1e-2,
    lr_gamma=0.999,
    num_iters=5000,
    kl_anneal_iters=400,
    kl_max_coeff=1.0,
    pause_every=50,
    save_every=50,
    checkpoint_dir="checkpoints",
    adjoint=False,
    mocap_data_path="./mocap35.mat",
    sde_type="ito",
    method="euler",
    grad_clip=0.5,
    seed=42,
    device=None,
    wandb_project="latent-sde-mocap",
    wandb_entity=None,
):
    # Set random seed for reproducibility
    set_seed(seed)

    # Initialize wandb
    _ = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        config={
            "lr_init": lr_init,
            "lr_gamma": lr_gamma,
            "num_iters": num_iters,
            "kl_anneal_iters": kl_anneal_iters,
            "kl_max_coeff": kl_max_coeff,
            "adjoint": adjoint,
            "method": method,
            "grad_clip": grad_clip,
            "seed": seed,
            "sde_type": sde_type,
        },
    )

    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger.info(f"Checkpoint directory: {checkpoint_dir}")

    # Device handling
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Log all settings
    logger.info("Training settings:")
    logger.info(f"Learning rate (initial): {lr_init}")
    logger.info(f"Learning rate decay (gamma): {lr_gamma}")
    logger.info(f"Number of iterations: {num_iters}")
    logger.info(f"KL annealing iterations: {kl_anneal_iters}")
    logger.info(f"KL max coefficient: {kl_max_coeff}")
    logger.info(f"Progress check interval: {pause_every}")
    logger.info(f"Using adjoint method: {adjoint}")
    logger.info(f"Data path: {mocap_data_path}")
    logger.info(f"SDE solver method: {method}")
    logger.info(f"Gradient clip value: {grad_clip}")
    logger.info(f"Random seed: {seed}")
    logger.info(f"Device: {device}")
    logger.info(f"SDE type: {sde_type}")

    xs, ts = make_dataset(mocap_data_path, "train", device)
    logger.info(f"xs: {xs.shape}, ts: {ts.shape}")
    logger.info(f"min(ts): {ts.min()}, max(ts): {ts.max()}")
    xs_val, ts_val = make_dataset(mocap_data_path, "val", device)
    logger.info(f"xs_val: {xs_val.shape}, ts_val: {ts_val.shape}")
    logger.info(f"min(ts_val): {ts_val.min()}, max(ts_val): {ts_val.max()}")
    xs_test, ts_test = make_dataset(mocap_data_path, "test", device)
    logger.info(f"xs_test: {xs_test.shape}, ts_test: {ts_test.shape}")
    logger.info(f"min(ts_test): {ts_test.min()}, max(ts_test): {ts_test.max()}")


    latent_sde = LatentSDE(sde_type=sde_type).to(device)
    optimizer = optim.Adam(params=latent_sde.parameters(), lr=lr_init)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer, gamma=lr_gamma
    )
    kl_scheduler = LinearScheduler(iters=kl_anneal_iters, maxval=kl_max_coeff)

    # Run benchmark before training
    # benchmark_sampling(latent_sde, ts, device)

    for global_step in tqdm.tqdm(range(1, num_iters + 1)):
        latent_sde.zero_grad()
        log_pxs, log_ratio = latent_sde(xs, ts, method=method, adjoint=adjoint)
        loss = -log_pxs + log_ratio * kl_scheduler.val
        loss.backward()
        torch.nn.utils.clip_grad_norm_(latent_sde.parameters(), max_norm=grad_clip)
        optimizer.step()
        scheduler.step()
        kl_scheduler.step()

        # Log metrics every step
        wandb.log(
            {
                "loss": loss.item(),
                "log_pxs": log_pxs.item(),
                "log_ratio": log_ratio.item(),
                "kl_coeff": kl_scheduler.val,
                "learning_rate": optimizer.param_groups[0]["lr"],
            },
            step=global_step,
        )

        if global_step % pause_every == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            logger.info(
                f"global_step: {global_step:06d}, lr: {lr_now:.5f}, log_pxs: {log_pxs:.4f}, log_ratio: {log_ratio:.4f}, loss: {loss:.4f}, kl_coeff: {kl_scheduler.val:.4f}"
            )

            # Calculate validation loss
            with torch.no_grad():
                val_log_pxs, val_log_ratio = latent_sde(
                    xs_val, ts_val, method=method, adjoint=adjoint
                )
                val_loss = -val_log_pxs + val_log_ratio * kl_scheduler.val
                logger.info(f"Validation loss: {val_loss:.4f}")

            # Calculate and log MSE metrics
            mse_train = latent_sde.mse(xs, ts)
            mse_val = latent_sde.mse(xs_val, ts_val)
            mse_test = latent_sde.mse(xs_test, ts_test)
            logger.info(
                f"MSE train: {mse_train:.4f}, val: {mse_val:.4f}, test: {mse_test:.4f}"
            )

            mean_mse_train = latent_sde.mean_mse(xs, ts)
            mean_mse_val = latent_sde.mean_mse(xs_val, ts_val)
            mean_mse_test = latent_sde.mean_mse(xs_test, ts_test)
            logger.info(
                f"Mean MSE train: {mean_mse_train:.4f}, val: {mean_mse_val:.4f}, test: {mean_mse_test:.4f}"
            )

            # Log evaluation metrics
            wandb.log(
                {
                    "loss/train": loss.item(),
                    "loss/val": val_loss.item(),
                    "log_pxs/train": log_pxs.item(),
                    "log_pxs/val": val_log_pxs.item(),
                    "log_ratio/train": log_ratio.item(),
                    "log_ratio/val": val_log_ratio.item(),
                    "mse/train": mse_train,
                    "mse/val": mse_val,
                    "mse/test": mse_test,
                    "mean_mse/train": mean_mse_train,
                    "mean_mse/val": mean_mse_val,
                    "mean_mse/test": mean_mse_test,
                },
                step=global_step,
            )

        # Save checkpoint
        if global_step % save_every == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir, f"model_step_{global_step:06d}.pt"
            )
            checkpoint = {
                "model_state_dict": latent_sde.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "global_step": global_step,
                "loss": loss.item(),
                "kl_coeff": kl_scheduler.val,
            }
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
            # Log checkpoint to wandb
            wandb.save(checkpoint_path)

    # Save final model
    final_checkpoint_path = os.path.join(checkpoint_dir, "model_final.pt")
    checkpoint = {
        "model_state_dict": latent_sde.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "global_step": num_iters,
        "loss": loss.item(),
        "kl_coeff": kl_scheduler.val,
    }
    torch.save(checkpoint, final_checkpoint_path)
    logger.info(f"Saved final checkpoint to {final_checkpoint_path}")
    wandb.save(final_checkpoint_path)

    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
