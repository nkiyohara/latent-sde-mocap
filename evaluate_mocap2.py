#!/usr/bin/env python3
"""Evaluate latent‑SDE MoCap checkpoints at multiple *dt* values.

Changes in this revision
-----------------------
* **Robust file download** – `wandb.File.download()` can return either a string
  path *or* an open file object, depending on `wandb` version.  The helper now
  normalises to `Path` in both cases, fixing the `TypeError` you observed.
* Minor tidy‑ups (better Path handling, early directory creation).

Usage example
~~~~~~~~~~~~~
```bash
python evaluate_mocap_checkpoints.py \
  --run-names dazzling-spaceship-26 grateful-sunset-25 whole-sky-23 \
  --run-names unique-eon-27 soft-surf-18 stellar-fire-17 toasty-sea-30 \
  --run-names stoic-terrain-16 dashing-plasma-28 usual-sun-32 \
  --dts 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0 \
  --step 5000 --device cuda
```

Make sure you are logged in to Weights & Biases (`wandb login`).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Union

import pandas as pd
import torch
import wandb

# Import the training components without redefining them.
from train_mocap2 import LatentSDE, load_mocap_data

################################################################################
# Utility helpers
################################################################################


def load_model(checkpoint_path: Union[str, Path], device: torch.device) -> LatentSDE:
    """Instantiate **LatentSDE** and load weights from *checkpoint_path*."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    model = LatentSDE().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model


@torch.no_grad()
def evaluate_mse(
    model: LatentSDE,
    xs_input: torch.Tensor,  # [batch, seq_in, dim]
    xs_target: torch.Tensor,  # [batch, seq_out, dim]
    ts_input: torch.Tensor,  # [seq_in]
    ts_target: torch.Tensor,  # [seq_out]
    dt: float,
    device: torch.device,
    repeats: int = 100,
    method: str = "euler",
) -> float:
    """Replicates the validation logic from *train_mocap2.py* and returns MSE."""

    xs_target = xs_target.to(device)
    ts_input = ts_input.to(device)
    ts_target = ts_target.to(device)

    batch_size = xs_input.size(0)
    xs_input_rep = xs_input.repeat_interleave(repeats, dim=0).to(device)

    pred_rep = model.predict(
        xs_input_rep.permute(1, 0, 2),  # [seq, batch*rep, dim]
        ts_input,
        ts_target,
        method=method,
        dt=dt,
    )

    pred_reshaped = pred_rep.reshape(
        pred_rep.shape[0], batch_size, repeats, pred_rep.shape[2]
    )
    pred_mean = pred_reshaped.mean(dim=2)  # [seq, batch, dim]

    mse = torch.mean((pred_mean - xs_target.permute(1, 0, 2)) ** 2).item()
    return mse


def download_checkpoint(
    run: wandb.apis.public.Run,
    step: int,
    dest_dir: Path,
) -> Path:
    """Download *mocap2_checkpoints/ckpt_{step}.pt* and return its local Path."""

    filename = f"mocap2_checkpoints/ckpt_{step}.pt"
    try:
        wb_file = run.file(filename)
    except wandb.errors.CommError as err:
        raise FileNotFoundError(
            f"Checkpoint '{filename}' not found in run '{run.name}'."
        ) from err

    # Ensure destination root exists before downloading.
    dest_dir.mkdir(parents=True, exist_ok=True)

    local_obj = wb_file.download(root=str(dest_dir), replace=True)

    # Depending on wandb version this returns a str or a *TextIOWrapper*
    if hasattr(local_obj, "name"):
        local_path = Path(local_obj.name)
        local_obj.close()
    else:
        local_path = Path(local_obj)

    return local_path.resolve()


################################################################################
# Main
################################################################################


def main(argv: List[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Evaluate Latent‑SDE checkpoints on the MoCap test set at various Euler step sizes (dt)."
    )

    parser.add_argument(
        "--project", default="latent-sde-mocap2", help="wandb project name"
    )
    parser.add_argument(
        "--entity",
        default=None,
        help="wandb entity (team or user). If omitted, default is used",
    )
    parser.add_argument(
        "--run-names", nargs="+", required=True, help="wandb *run.names* to evaluate"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=5000,
        help="Checkpoint step number (same for all runs)",
    )
    parser.add_argument(
        "--dts",
        nargs="+",
        type=float,
        required=True,
        help="Euler step sizes to test, e.g. 0.05 0.1",
    )
    parser.add_argument(
        "--mocap-path", default="./mocap35.mat", help="Path to mocap .mat file"
    )
    parser.add_argument(
        "--data-dt", type=float, default=0.1, help="Δt used when loading mocap data"
    )
    parser.add_argument(
        "--device", default="cpu", help="Device string for torch, e.g. cuda or cuda:1"
    )
    parser.add_argument(
        "--output-dir", default="results", help="Directory for CSV output"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=100,
        help="Stochastic forward passes per evaluation",
    )

    args = parser.parse_args(argv)

    device = torch.device(args.device)
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Fetch runs from wandb ────────────────────────────────────────────────
    api = wandb.Api()
    project_path = (
        args.project if args.entity is None else f"{args.entity}/{args.project}"
    )
    runs = {r.name: r for r in api.runs(project_path) if r.name in set(args.run_names)}

    missing = set(args.run_names) - runs.keys()
    if missing:
        raise ValueError(
            f"Could not find the following run names in '{project_path}': {sorted(missing)}"
        )

    # ── Load dataset once (CPU) ─────────────────────────────────────────────
    xs_test, ts_test = load_mocap_data(args.mocap_path, "test", dt=args.data_dt)
    xs_test = torch.tensor(xs_test, dtype=torch.float32)
    ts_test = torch.tensor(ts_test[0], dtype=torch.float32)

    xs_in = xs_test[:, :100]
    xs_tgt = xs_test[:, 100:300]
    ts_in = ts_test[:100]
    ts_tgt = ts_test[100:300]

    # ── Evaluation loop ─────────────────────────────────────────────────────
    rows = []
    ckpt_cache = Path("checkpoints_dl")

    for run_name, run in runs.items():
        print(f"\n▶ Evaluating run '{run_name}' (id: {run.id}) …")
        ckpt_path = download_checkpoint(run, args.step, ckpt_cache)
        model = load_model(ckpt_path, device)

        for dt in args.dts:
            mse = evaluate_mse(
                model,
                xs_in,
                xs_tgt,
                ts_in,
                ts_tgt,
                dt,
                device,
                repeats=args.repeats,
            )
            rows.append({"run": run_name, "step": args.step, "dt": dt, "mse": mse})
            print(f"    dt={dt:<5g}  MSE={mse:.6f}")

    # ── Save & summarise ────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    csv_path = out_dir / "mocap_dt_eval.csv"
    df.to_csv(csv_path, index=False)

    summary = (
        df.groupby("dt")
        .agg(mean_mse=("mse", "mean"), std_mse=("mse", "std"))
        .sort_index()
    )

    print("\n============= SUMMARY (mean ± std across runs) =============")
    for dt, row in summary.iterrows():
        print(f"dt={dt:>6g}   {row['mean_mse']:.6f} ± {row['std_mse']:.6f}")

    print(f"\nCSV saved to: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
