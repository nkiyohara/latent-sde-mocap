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
# Modifications Copyright 2023 Naoki Kiyohara
# Adapted from original latent SDE implementation for Motion Capture modeling

import os
from typing import Literal, Tuple, Union

import fire
import numpy as np
import torch
import torchsde
import tqdm
from torchdiffeq import odeint
from beartype import beartype
from jaxtyping import Float, jaxtyped
from loguru import logger
from scipy.io import loadmat
from torch import nn, optim
from torch.distributions import Normal
import wandb


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)


def load_mocap_data(
    mocap_data_path: Union[str, os.PathLike],
    dataset_type: Literal["train", "val", "test"] = "train",
    dt: float = 0.1,
) -> Tuple[
    Float[np.ndarray, "num_seqs seq_len obs_dim"], Float[np.ndarray, "num_seqs seq_len"]
]:
    """Load motion capture dataset from .mat file."""
    data = loadmat(mocap_data_path)
    key = {"train": "Xtr", "val": "Xval", "test": "Xtest"}[dataset_type]
    X = data[key].astype(np.float32)
    t = dt * np.arange(X.shape[1], dtype=np.float32)
    return X, np.tile(t, (X.shape[0], 1))


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


class ODEGRUDrift(nn.Module):
    """Neural ODE drift function for autonomous system."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.Tanh(), nn.Linear(64, hidden_size)
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self, t: Float[torch.Tensor, ""], h: Float[torch.Tensor, "batch hidden"]
    ) -> Float[torch.Tensor, "batch hidden"]:
        return self.net(h)


class ODEGRU(nn.Module):
    """Continuous-time GRU with autonomous ODE dynamics."""

    def __init__(self, input_dim: int, hidden_size: int):
        super().__init__()
        self.gru_cell = nn.GRUCell(input_dim, hidden_size)
        self.drift = ODEGRUDrift(hidden_size)
        self.hidden_size = hidden_size

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[torch.Tensor, "seq_len batch input_dim"],
        t: Float[torch.Tensor, "seq_len"],
    ) -> Float[torch.Tensor, "seq_len batch hidden"]:
        # Reverse sequence for encoder processing
        reversed_x = torch.flip(x, dims=[0])
        reversed_t = torch.flip(t, dims=[0])
        batch_size = reversed_x.size(1)
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        hidden_states = []

        for i in range(reversed_x.size(0)):
            h = self.gru_cell(reversed_x[i], h)
            if i < reversed_x.size(0) - 1:
                # Use actual time intervals from reversed sequence
                t_span = torch.tensor(
                    [reversed_t[i], reversed_t[i + 1]], device=x.device
                )
                h = odeint(self.drift, h, t_span, method="dopri5")[-1]
            hidden_states.append(h)

        return torch.flip(torch.stack(hidden_states), dims=[0])


class Encoder(nn.Module):
    """Encoder network for autonomous SDE system."""

    def __init__(self):
        super().__init__()
        self.odegru = ODEGRU(input_dim=50, hidden_size=30)
        self.context_net = nn.Linear(30, 3)
        self.qz0_net = nn.Linear(30, 20)  # 10 for mean, 10 for logstd

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[torch.Tensor, "seq_len batch input_dim"],
        t: Float[torch.Tensor, "seq_len"],
    ) -> Tuple[
        Float[torch.Tensor, "batch latent"],
        Float[torch.Tensor, "batch latent"],
        Float[torch.Tensor, "seq_len batch context"],
    ]:
        hidden = self.odegru(x, t)
        context = self.context_net(hidden)
        qz0_params = self.qz0_net(hidden[0])
        qz0_mean, qz0_logstd = torch.chunk(qz0_params, 2, dim=-1)
        return qz0_mean, qz0_logstd, context


class Decoder(nn.Module):
    """Decoder network for autonomous system."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 30),
            nn.Softplus(),
            nn.Linear(30, 30),
            nn.Softplus(),
            nn.Linear(30, 50 * 2),  # Mean and logstd
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self, z: Float[torch.Tensor, "seq_len batch latent"]
    ) -> Tuple[
        Float[torch.Tensor, "seq_len batch obs_dim"],
        Float[torch.Tensor, "seq_len batch obs_dim"],
    ]:
        out = self.net(z)
        px_mean, px_logstd = torch.split(out, 50, dim=-1)
        return px_mean, px_logstd


class LatentSDE(nn.Module):
    """Autonomous Latent SDE system using torchdiffeq."""

    noise_type = "diagonal"

    def __init__(self, sde_type: str = "ito"):
        super().__init__()
        self.sde_type = sde_type
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.clamp_range = 9.0

        # Autonomous drift networks (no explicit time dependence)
        self.f_net = nn.Sequential(
            nn.Linear(10 + 3, 30),  # latent + context
            nn.Softplus(),
            nn.Linear(30, 10),
        )

        self.h_net = nn.Sequential(nn.Linear(10, 30), nn.Softplus(), nn.Linear(30, 10))

        # Diagonal diffusion networks
        self.g_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, 30),
                    nn.Softplus(),
                    nn.Linear(30, 1),
                    nn.Sigmoid(),
                )
                for _ in range(10)
            ]
        )

        # Prior parameters
        self.pz0_mean = nn.Parameter(torch.zeros(1, 10))
        self.pz0_logstd = nn.Parameter(torch.zeros(1, 10))
        self._ctx = None

    def contextualize(self, ctx: Tuple[torch.Tensor, torch.Tensor]):
        """Set temporal context for autonomous SDE evaluation."""
        self._ctx = ctx

    @jaxtyped(typechecker=beartype)
    def f(
        self, t: Float[torch.Tensor, ""], y: Float[torch.Tensor, "batch latent"]
    ) -> Float[torch.Tensor, "batch latent"]:
        """Autonomous posterior drift: f(y) = f_net(y, context)"""
        ts, ctx = self._ctx
        idx = min(int(torch.searchsorted(ts, t)), len(ts) - 1)
        return self.f_net(torch.cat([y, ctx[idx]], dim=-1))

    @jaxtyped(typechecker=beartype)
    def h(
        self, t: Float[torch.Tensor, ""], y: Float[torch.Tensor, "batch latent"]
    ) -> Float[torch.Tensor, "batch latent"]:
        """Autonomous prior drift: h(y) = h_net(y)"""
        return self.h_net(y)

    @jaxtyped(typechecker=beartype)
    def g(
        self, t: Float[torch.Tensor, ""], y: Float[torch.Tensor, "batch latent"]
    ) -> Float[torch.Tensor, "batch latent"]:
        """Autonomous diagonal diffusion: g(y_i)"""
        return torch.cat(
            [net(y[:, i : i + 1]) for i, net in enumerate(self.g_nets)], dim=-1
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        xs: Float[torch.Tensor, "seq_len batch obs_dim"],
        ts: Float[torch.Tensor, "seq_len"],
        adjoint: bool = False,
        method: str = "euler",
        dt: float = 0.05,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for autonomous SDE."""
        qz0_mean, qz0_logstd, ctx = self.encoder(xs, ts)
        self.contextualize((ts, ctx))

        # Sample initial state
        z0 = qz0_mean + torch.exp(
            torch.clamp(qz0_logstd, -self.clamp_range, self.clamp_range)
        ) * torch.randn_like(qz0_mean)

        # Solve SDE
        if adjoint:
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

        # Decode
        px_mean, px_logstd = self.decoder(zs)
        xs_dist = Normal(
            loc=px_mean,
            scale=torch.exp(
                torch.clamp(px_logstd, -self.clamp_range, self.clamp_range)
            ),
        )
        log_pxs = xs_dist.log_prob(xs).sum(dim=(0, 2)).mean()

        # KL terms
        qz0 = Normal(
            qz0_mean,
            torch.exp(torch.clamp(qz0_logstd, -self.clamp_range, self.clamp_range)),
        )
        pz0 = Normal(
            self.pz0_mean,
            torch.exp(
                torch.clamp(self.pz0_logstd, -self.clamp_range, self.clamp_range)
            ),
        )
        logqp0 = torch.distributions.kl_divergence(qz0, pz0).sum(-1).mean()
        logqp_path = log_ratio.sum(0).mean()

        return log_pxs, logqp0 + logqp_path

    @jaxtyped(typechecker=beartype)
    @torch.no_grad()
    def predict(
        self,
        xs_input: Float[torch.Tensor, "input_seq batch obs_dim"],
        ts_input: Float[torch.Tensor, "input_seq"],
        ts_pred: Float[torch.Tensor, "pred_seq"],
        method: str = "euler",
        dt: float = 0.05,
    ) -> Float[torch.Tensor, "pred_seq batch obs_dim"]:
        """Autonomous prediction using prior dynamics."""
        qz0_mean, qz0_logstd, context = self.encoder(xs_input, ts_input)
        self.contextualize((ts_input, context))
        z0 = qz0_mean + torch.exp(qz0_logstd) * torch.randn_like(qz0_mean)
        zs_post = torchsde.sdeint(
            self, z0, ts_input, names={"drift": "f"}, method=method, dt=dt
        )
        zt_post = zs_post[-1]
        zs = torchsde.sdeint(
            self, zt_post, ts_pred, names={"drift": "h"}, method=method, dt=dt
        )
        px_mean, _ = self.decoder(zs)
        return px_mean


def main(
    seed: int = 42,
    num_iters: int = 5000,
    lr_init: float = 1e-2,
    lr_gamma: float = 0.999,
    kl_anneal_iters: int = 500,
    kl_max_coeff: float = 1.0,
    pause_every: int = 50,
    save_every: int = 50,
    checkpoint_dir: str = "checkpoints",
    adjoint: bool = False,
    mocap_data_path: str = "./mocap35.mat",
    sde_type: str = "ito",
    sde_method: str = "srk",
    sde_dt: float = 0.05,
    data_dt: float = 0.1,
    grad_clip: float = 0.5,
    device: str = None,
    wandb_project: str = "latent-sde-mocap2",
    wandb_entity: str = None,
):
    """Main training loop for autonomous SDE system."""
    set_seed(seed)
    wandb.init(project=wandb_project, entity=wandb_entity, config=locals())
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device(device or "cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    def load_dataset(data_type: str, truncate: int = None):
        X, t = load_mocap_data(mocap_data_path, data_type, dt=data_dt)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        t_tensor = torch.tensor(t[0], dtype=torch.float32).to(device)
        if truncate:
            return X_tensor[:, :truncate], t_tensor[:truncate]
        return X_tensor, t_tensor

    # Training data
    xs_train, ts_train = load_dataset("train", 200)
    xs_val, ts_val = load_dataset("val", 200)

    # Test data (first 100 steps input, next 200 target)
    xs_test, ts_test = load_dataset("test")
    xs_test_input = xs_test[:, :100]
    xs_test_target = xs_test[:, 100:300]  # 200 prediction steps
    ts_test_input = ts_test[:100]
    ts_test_target = ts_test[100:300]

    # Initialize model
    model = LatentSDE(sde_type=sde_type).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr_init)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)
    kl_scheduler = LinearScheduler(kl_anneal_iters, kl_max_coeff)

    # Training loop
    for step in tqdm.trange(1, num_iters + 1):
        model.zero_grad()

        log_pxs, log_ratio = model(
            xs_train.permute(1, 0, 2),
            ts_train,
            adjoint,
            method=sde_method,
            dt=sde_dt,
        )

        loss = -log_pxs + log_ratio * kl_scheduler.val

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        kl_scheduler.step()

        # Logging
        if step % pause_every == 0:
            with torch.no_grad():
                val_log_pxs, val_log_ratio = model(
                    xs_val.permute(1, 0, 2),
                    ts_val,
                    adjoint,
                    method=sde_method,
                    dt=sde_dt,
                )
                val_loss = -val_log_pxs + val_log_ratio * kl_scheduler.val

                # Testing (only evaluate prediction steps)
                pred = model.predict(
                    xs_test_input.permute(1, 0, 2),
                    ts_test_input,
                    ts_test_target,
                    method=sde_method,
                    dt=sde_dt,
                )
                # Ensure prediction and target dimensions match
                test_mse = torch.mean((pred - xs_test_target.permute(1, 0, 2)) ** 2)

            wandb.log(
                {
                    "train/loss": loss.item(),
                    "val/loss": val_loss.item(),
                    "test/mse": test_mse.item(),
                    "lr": scheduler.get_last_lr()[0],
                    "kl_coeff": kl_scheduler.val,
                    "step": step,
                }
            )

        # Save checkpoint
        if step % save_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"ckpt_{step}.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                },
                ckpt_path,
            )
            wandb.save(ckpt_path)

    # Final evaluation
    with torch.no_grad():
        pred = model.predict(
            xs_test_input.permute(1, 0, 2),
            ts_test_input,
            ts_test_target,
            method=sde_method,
            dt=sde_dt,
        )
        final_mse = torch.mean((pred - xs_test_target.permute(1, 0, 2)) ** 2)
        logger.info(f"Final Test MSE: {final_mse:.4f}")
        wandb.log({"final_test_mse": final_mse})

    wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
