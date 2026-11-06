"""Export a checkpoint's actor for the car (spec section 7.1).

Writes policy.npz: actor weights, log-std, observation normalizer mean/std
and a JSON config (beam indices, v_cmd_max, steering limit, policy rate,
frame-stack length, checkpoint hash). The car runs it with numpy_policy.py.

    python -m rl.export_policy runs/<run>/best.safetensors --out policy.npz
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from . import car_params as cp
from . import obs as ob
from .networks import NORM_CLIP, Normalizer, model_cfg_from


def export(checkpoint, out) -> dict:
    from .train_ppo import load_checkpoint

    ck = load_checkpoint(checkpoint)
    mc = model_cfg_from(ck["config"])
    if mc["arch"] != "mlp":
        raise NotImplementedError(f"export of arch {mc['arch']!r} (NumPy GRU forward pass) is not written yet")
    num_beams, frame_stack = mc["num_beams"], mc["frame_stack"]
    obs_dim = ob.obs_dim(frame_stack, num_beams)
    params = ck["actor_params"]["params"]
    mlp = params["mlp"]
    layers = [mlp[f"Dense_{i}"] for i in range(len(mlp))]  # hidden layers + output
    normalizer: Normalizer = ck["normalizer"]
    mean, std = normalizer.full_mean_std(obs_dim)
    config = {
        "format": "rl_phase1_policy_v1",
        "arch": "mlp",
        "obs_dim": obs_dim,
        "num_beams": num_beams,
        "car_scan_beams": ob.CAR_SCAN_BEAMS,
        "beam_indices": ob.beam_indices(num_beams).tolist(),
        "fov": ob.FOV,
        "max_range": ob.MAX_RANGE,
        "frame_stack": frame_stack,
        "v_cmd_max": ob.V_CMD_MAX,
        "steer_max": ob.S_MAX,
        "policy_rate_hz": cp.POLICY_RATE_HZ,
        "reset_action": list(ob.RESET_ACTION),
        "norm_clip": NORM_CLIP,
        "activation": "tanh",
        "action": "tanh(mean); a = (a_v, a_s); speed = (a_v + 1) / 2 * v_cmd_max; steer = a_s * steer_max",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": ck["sha256"],
        "checkpoint_update": ck["update"],
        "train_config_hash": ck["config_hash"],
    }
    arrays = {}
    for i, layer in enumerate(layers):
        arrays[f"W{i}"] = np.asarray(layer["kernel"], dtype=np.float32)
        arrays[f"b{i}"] = np.asarray(layer["bias"], dtype=np.float32)
    arrays["log_std"] = np.asarray(params["log_std"], dtype=np.float32)
    arrays["obs_mean"] = np.asarray(mean, dtype=np.float32)
    arrays["obs_std"] = np.asarray(std, dtype=np.float32)
    out = pathlib.Path(out)
    np.savez(out, config=np.array(json.dumps(config)), **arrays)
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--out", default="policy.npz")
    args = parser.parse_args(argv)
    config = export(args.checkpoint, args.out)
    print(f"wrote {args.out} (checkpoint update {config['checkpoint_update']}, sha256 {config['checkpoint_sha256'][:12]})")


if __name__ == "__main__":
    main()
