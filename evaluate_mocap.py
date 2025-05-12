#!/usr/bin/env python3
"""Evaluate *train_mocap.py* checkpoints at multiple `dt` values.

For each specified wandb run (given **run name**), this script downloads the
`checkpoints/model_step_<STEP>.pt` checkpoint, restores the *LatentSDE* model
from **train_mocap.py**, and computes the *mean MSE* (average over multiple
posterior samples) on the MoCap **test** set for a list of Euler step-sizes
`dt`.  Results are saved to CSV and summarised in the console.

Key differences from the earlier script:
* Uses `make_dataset` from *train_mocap.py* to load sliding-window test data.
* Metric is `mean MSE` – reproduces the training-time evaluation but lets you
  vary `dt`.

Example
~~~~~~~
```bash
python evaluate_mocap_mean_mse.py \
  --run-names atomic-glade-71 youthful-water-67 glowing-cloud-63 \
                dainty-rain-63 snowy-glade-63 \
  --step 5000 \
  --dts 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0 \
  --device cuda
```
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Union

import pandas as pd
import torch
import wandb

# Import the original training utilities.
from train_mocap import LatentSDE, make_dataset

###############################################################################
# Helpers                                                                     #
###############################################################################


def load_model(checkpoint_path: Union[str, Path], device: torch.device) -> LatentSDE:
    """Instantiate *LatentSDE* and load weights from `checkpoint_path`."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    model = LatentSDE().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model


@torch.no_grad()
def evaluate_mean_mse(
    model: LatentSDE,
    xs: torch.Tensor,  # [T, B, W, D]
    ts: torch.Tensor,  # [T]
    dt: float,
    device: torch.device,
    num_samples: int = 100,
) -> float:
    """Compute mean-MSE across `num_samples` posterior trajectories for step-size *dt*."""
    # Move data to device lazily (keeps original tensors unchanged).
    xs = xs.to(device)
    ts = ts.to(device)

    T, B, W, D = xs.shape
    xs_rep = xs.repeat(1, num_samples, 1, 1).view(T, num_samples * B, W, D)

    trajs = model.sample_from_posterior(xs_rep, ts, dt=dt).view(T, num_samples, B, D)
    traj_mean = trajs.mean(dim=1)  # [T, B, D]

    mse = torch.mean((traj_mean - xs[:, :, -1]) ** 2).item()
    return mse


def download_checkpoint(run: wandb.apis.public.Run, step: int, dest: Path) -> Path:
    """Fetch `checkpoints/model_step_<step>.pt` from *run* and return its local path."""
    fname = f"checkpoints/model_step_{step:06d}.pt"
    try:
        wb_file = run.file(fname)
    except wandb.errors.CommError as err:
        raise FileNotFoundError(
            f"Checkpoint '{fname}' not found in run '{run.name}'."
        ) from err

    dest.mkdir(parents=True, exist_ok=True)
    local_obj = wb_file.download(root=str(dest), replace=True)

    # wandb may return str or a file object
    if hasattr(local_obj, "name"):
        path = Path(local_obj.name)
        local_obj.close()
    else:
        path = Path(local_obj)

    return path.resolve()


###############################################################################
# Main                                                                        #
###############################################################################


def main(argv: List[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Evaluate LatentSDE checkpoints (train_mocap.py) on the MoCap test set."
    )
    parser.add_argument(
        "--project", default="latent-sde-mocap", help="wandb project name"
    )
    parser.add_argument(
        "--entity",
        default=None,
        help="wandb entity (team/user); omit for default auth user",
    )
    parser.add_argument(
        "--run-names",
        nargs="+",
        required=True,
        help="List of wandb run names to evaluate",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=5000,
        help="Checkpoint step (global_step) to load for every run",
    )
    parser.add_argument(
        "--dts",
        nargs="+",
        type=float,
        required=True,
        help="Euler step sizes to test, e.g. 0.02 0.05",
    )
    parser.add_argument(
        "--mocap-path",
        default="./mocap35.mat",
        help="Path to MoCap .mat file used by training",
    )
    parser.add_argument(
        "--device", default="cpu", help="Device for inference, e.g. cuda or cuda:1"
    )
    parser.add_argument(
        "--output-dir", default="results", help="Directory to save the CSV output"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Posterior samples per sequence when computing mean MSE",
    )

    args = parser.parse_args(argv)

    dev = torch.device(args.device)
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Retrieve wandb runs ──────────────────────────────────────────────────
    api = wandb.Api()
    proj_path = args.project if args.entity is None else f"{args.entity}/{args.project}"
    runs = {r.name: r for r in api.runs(proj_path) if r.name in set(args.run_names)}
    missing = set(args.run_names) - runs.keys()
    if missing:
        raise ValueError(f"Run names not found in '{proj_path}': {sorted(missing)}")

    # ── Load dataset once (CPU) ─────────────────────────────────────────────
    print("Loading MoCap test set …")
    xs_test, ts_test = make_dataset(
        args.mocap_path, "test", device="cpu"
    )  # load on CPU first

    rows = []
    ckpt_cache = Path("checkpoints_dl")
    ckpt_cache.mkdir(exist_ok=True)

    for rname, run in runs.items():
        print(f"\n▶ Evaluating run '{rname}' (id: {run.id}) …")
        ckpt = download_checkpoint(run, args.step, ckpt_cache)
        model = load_model(ckpt, dev)

        for dt in args.dts:
            mse = evaluate_mean_mse(
                model,
                xs_test,
                ts_test,
                dt=dt,
                device=dev,
                num_samples=args.samples,
            )
            rows.append({"run": rname, "step": args.step, "dt": dt, "mean_mse": mse})
            print(f"    dt={dt:<5g}  meanMSE={mse:.6f}")

    # ── Save CSV and print summary ──────────────────────────────────────────
    df = pd.DataFrame(rows)
    csv_path = out_dir / "mocap_mean_mse_dt.csv"
    df.to_csv(csv_path, index=False)

    summary = (
        df.groupby("dt")
        .agg(mean=("mean_mse", "mean"), std=("mean_mse", "std"))
        .sort_index()
    )

    print("\n============= SUMMARY (mean ± std across runs) =============")
    for dt, row in summary.iterrows():
        print(f"dt={dt:>6g}   {row['mean']:.6f} ± {row['std']:.6f}")

    print(f"\nCSV saved to: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
