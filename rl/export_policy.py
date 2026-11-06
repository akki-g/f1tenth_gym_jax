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
from .networks import CONV_LAYERS, NORM_CLIP, Normalizer, model_cfg_from


def export(checkpoint, out) -> dict:
    from .train_ppo import load_checkpoint

    ck = load_checkpoint(checkpoint)
    mc = model_cfg_from(ck["config"])
    if mc["arch"] not in ("mlp", "cnn_gru"):
        raise ValueError(f"cannot export arch {mc['arch']!r}")
    num_beams, frame_stack = mc["num_beams"], mc["frame_stack"]
    obs_dim = ob.obs_dim(frame_stack, num_beams)  # the actor's inputs (a race policy's critic-only block is not exported)
    params = ck["actor_params"]["params"]
    normalizer: Normalizer = ck["normalizer"]
    mean, std = normalizer.full_mean_std(obs_dim)
    config = {
        "format": "rl_phase1_policy_v1" if mc["arch"] == "mlp" else "rl_phase1c_policy_v1",
        "arch": mc["arch"],
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
    f32 = lambda x: np.asarray(x, dtype=np.float32)  # noqa: E731
    arrays = {}
    if mc["arch"] == "mlp":
        mlp = params["mlp"]
        for i in range(len(mlp)):  # hidden layers + output
            arrays[f"W{i}"] = f32(mlp[f"Dense_{i}"]["kernel"])
            arrays[f"b{i}"] = f32(mlp[f"Dense_{i}"]["bias"])
    else:
        config.update(gru_dim=mc["gru_dim"], proj_dim=mc["proj_dim"], conv_layers=[list(c) for c in CONV_LAYERS],
                      scan_padding="edge", race_trained=bool(mc.get("priv_dim", 0)),
                      recurrent="reset() at start-up and every (re)activation; one step per policy tick")
        for i in range(len(CONV_LAYERS)):
            arrays[f"conv{i}_kernel"] = f32(params[f"convs_{i}"]["kernel"])
            arrays[f"conv{i}_bias"] = f32(params[f"convs_{i}"]["bias"])
        for name in ("proj", "actor_hidden", "actor_out"):
            arrays[f"{name}_kernel"] = f32(params[name]["kernel"])
            arrays[f"{name}_bias"] = f32(params[name]["bias"])
        g = params["gru"]
        for gate in ("ir", "iz", "in", "hr", "hz", "hn"):
            arrays[f"gru_{gate}_kernel"] = f32(g[gate]["kernel"])
            if "bias" in g[gate]:
                arrays[f"gru_{gate}_bias"] = f32(g[gate]["bias"])
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
