"""Validate the AAA dysts subset against dysts itself (run in an environment with dysts 0.96).

    PYTHONPATH=. <python-with-dysts> tools/aaa_1k_v2_dysts_validation.py \
        --output docs/evidence/aaa_1k_v2/dysts_validation.json

For every selected system, both integrators start from dysts' own initial
condition with no transient and are sampled at 100 points per period:

* short-horizon agreement: normalized RMS difference of coordinate 0 over the
  first 25 samples (a quarter period) and the first 100 samples;
* attractor statistics: mean and standard deviation of coordinate 0 over
  samples 2000-5000, compared with each other and with dysts' published
  metadata.

This is a check of the reimplementation, not a benchmark result.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import dysts.flows as flows

    from research.aaa_1k_v2.external.dysts_subset import (
        METADATA_SHA256,
        integrate,
        load_metadata,
        select_systems,
    )

    metadata = load_metadata()
    records = []
    for entry in select_systems(metadata):
        name = entry["system"]
        model = getattr(flows, name)()
        ic = np.asarray(metadata[name]["initial_conditions"], dtype=float)
        reference = np.asarray(model.make_trajectory(5000, resample=True, pts_per_period=100, init_cond=ic))
        ours = integrate(name, 5000, seed=0, metadata=metadata, transient_periods=0, initial_condition=ic).raw
        std0 = float(metadata[name]["std"][0])
        short25 = float(np.sqrt(np.mean((reference[:25, 0] - ours[:25, 0]) ** 2)) / std0)
        short100 = float(np.sqrt(np.mean((reference[:100, 0] - ours[:100, 0]) ** 2)) / std0)
        ref_tail, our_tail = reference[2000:, 0], ours[2000:, 0]
        records.append(
            {
                "system": name,
                "normalized_rms_difference_first_25": short25,
                "normalized_rms_difference_first_100": short100,
                "dysts_mean": float(ref_tail.mean()),
                "aaa_mean": float(our_tail.mean()),
                "dysts_std": float(ref_tail.std()),
                "aaa_std": float(our_tail.std()),
                "metadata_mean": float(metadata[name]["mean"][0]),
                "metadata_std": std0,
                "mean_difference_in_std_units": float((our_tail.mean() - ref_tail.mean()) / std0),
                "std_ratio": float(our_tail.std() / ref_tail.std()),
            }
        )
        print(
            name,
            f"{short25:.2e} {short100:.2e}",
            f"{records[-1]['mean_difference_in_std_units']:+.3f}",
            f"{records[-1]['std_ratio']:.3f}",
            flush=True,
        )
    payload = {
        "schema": "aaa.1k.v2.dysts_validation.v1",
        "dysts_version": importlib.metadata.version("dysts"),
        "metadata_sha256": METADATA_SHA256,
        "python": sys.version.split()[0],
        "note": "reference = dysts make_trajectory(5000, resample=True, pts_per_period=100, init_cond=metadata IC); "
        "AAA = fixed-step RK4 from the same IC with no transient. Chaotic trajectories are expected to separate "
        "after a few Lyapunov times; the attractor statistics are the relevant comparison.",
        "systems": records,
    }
    from pathlib import Path

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
