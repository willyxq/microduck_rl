#!/usr/bin/env python3
"""Train a minimal MicroDuck behavior-cloning policy from a motion file."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from mjlab_microduck.imitation import (
    BehaviorCloningPolicy,
    OBS_DIM,
    build_dataset,
    interpolate_actions,
    load_motion,
    phase_observation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )

    motion = load_motion(args.motion)
    observations, actions = build_dataset(motion, args.samples, args.seed)
    loader = DataLoader(
        TensorDataset(observations, actions),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    model = BehaviorCloningPolicy().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.MSELoss()

    started = time.perf_counter()
    first_loss = None
    final_loss = None
    for epoch in range(args.epochs):
        loss_sum = 0.0
        for batch_observations, batch_actions in loader:
            batch_observations = batch_observations.to(device)
            batch_actions = batch_actions.to(device)
            prediction = model(batch_observations)
            loss = loss_fn(prediction, batch_actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(batch_observations)
        final_loss = loss_sum / len(observations)
        if first_loss is None:
            first_loss = final_loss
        if epoch == 0 or (epoch + 1) % 50 == 0:
            print(f"epoch={epoch + 1:4d} mse={final_loss:.8f}")
    training_seconds = time.perf_counter() - started

    phases = np.linspace(0.0, 1.0, 401, endpoint=False, dtype=np.float32)
    validation_observations = torch.from_numpy(
        np.stack([phase_observation(float(phase)) for phase in phases])
    ).to(device)
    targets = interpolate_actions(motion, phases)
    model.eval()
    with torch.no_grad():
        predictions = model(validation_observations).cpu().numpy()
    validation_mse = float(np.mean((predictions - targets) ** 2))
    validation_max_abs = float(np.max(np.abs(predictions - targets)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "policy.pt"
    onnx_path = args.output_dir / "policy.onnx"
    metrics_path = args.output_dir / "metrics.json"
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "motion": motion.name,
            "observation_dim": OBS_DIM,
        },
        checkpoint_path,
    )
    dummy = torch.zeros(1, OBS_DIM)
    torch.onnx.export(
        model.cpu(),
        dummy,
        onnx_path,
        input_names=["observations"],
        output_names=["actions"],
        dynamic_axes={"observations": {0: "batch"}, "actions": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

    metrics = {
        "motion": motion.name,
        "device": device,
        "samples": args.samples,
        "epochs": args.epochs,
        "seed": args.seed,
        "training_seconds": training_seconds,
        "first_epoch_mse": first_loss,
        "final_training_mse": final_loss,
        "validation_mse": validation_mse,
        "validation_max_abs_error_rad": validation_max_abs,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"wrote {onnx_path}")


if __name__ == "__main__":
    main()
