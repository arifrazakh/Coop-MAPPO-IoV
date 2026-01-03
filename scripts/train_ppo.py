"""Train a shared-policy PPO agent on the MultiAgentMobileNetwork environment.

This script is intentionally self-contained and tolerant to RLlib API differences
across Ray versions.

Examples
--------
python scripts/train_ppo.py --total-steps 1000000
python scripts/train_ppo.py --num-users 80 --num-channels 50 --deadline-ms 80
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Any, Dict, Tuple

import ray
import torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env

from mamn_marl.callbacks import EpisodeCSVLogger
from mamn_marl.env import MultiAgentMobileNetwork


ENV_ID = "multi_agent_mobile_network"


def rllib_env_creator(env_config: Dict[str, Any]):
    return MultiAgentMobileNetwork(
        num_base_stations=env_config.get("num_base_stations", 4),
        num_users=env_config.get("num_users", 100),
        num_channels_per_carrier=env_config.get("num_channels_per_carrier", 100),
        area_size=env_config.get("area_size", 5000.0),
        bs_loc=env_config.get(
            "bs_loc",
            [(1250, 1400), (3750, 1400), (1250, 3550), (3750, 3550)],
        ),
        max_steps=env_config.get("max_steps", 200),
        mobility_model=env_config.get("mobility_model", "manhattan"),
        seed=env_config.get("seed", 42),
        step_duration_s=env_config.get("step_duration_s", 1.0),
        deadline_ms=env_config.get("deadline_ms", 100.0),
        safety_traffic_ratio=env_config.get("safety_traffic_ratio", 0.3),
    )


def _build_algo(config: PPOConfig):
    """Build PPO algo across RLlib versions."""
    if hasattr(config, "build_algo"):
        return config.build_algo()
    return config.build()


def _get_total_env_steps(result_dict: Dict[str, Any]) -> int:
    for k in [
        "num_env_steps_sampled",
        "num_env_steps_sampled_lifetime",
        "timesteps_total",
    ]:
        v = result_dict.get(k, None)
        if v is not None:
            try:
                return int(v)
            except Exception:
                pass
    return 0


def _get_mean_return_and_len(result_dict: Dict[str, Any]) -> Tuple[float, float]:
    mean_ret = result_dict.get("episode_return_mean", result_dict.get("episode_reward_mean", float("nan")))
    mean_len = result_dict.get("episode_len_mean", float("nan"))

    env_runners = result_dict.get("env_runners", {})
    if isinstance(env_runners, dict):
        if mean_ret is None or (isinstance(mean_ret, float) and math.isnan(mean_ret)):
            mean_ret = env_runners.get("episode_return_mean", env_runners.get("episode_reward_mean", mean_ret))
        if mean_len is None or (isinstance(mean_len, float) and math.isnan(mean_len)):
            mean_len = env_runners.get("episode_len_mean", mean_len)

    return float(mean_ret), float(mean_len)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # stopping
    p.add_argument("--total-steps", type=int, default=1_000_000, help="Total env steps (RLlib-reported) to train for.")
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=10)

    # output
    p.add_argument("--out-dir", type=str, default="outputs", help="Where to store CSV + checkpoints.")

    # env
    p.add_argument("--num-base-stations", type=int, default=4)
    p.add_argument("--num-users", type=int, default=100)
    p.add_argument("--num-channels", type=int, default=100)
    p.add_argument("--area-size", type=float, default=5000.0)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--mobility-model", type=str, default="manhattan", choices=["manhattan", "random_waypoint"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--step-duration-s", type=float, default=1.0)
    p.add_argument("--deadline-ms", type=float, default=100.0)
    p.add_argument("--safety-traffic-ratio", type=float, default=0.3)

    # PPO
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lambda", dest="lambda_", type=float, default=0.95)
    p.add_argument("--train-batch-size", type=int, default=1024)
    p.add_argument("--sgd-minibatch-size", type=int, default=256)
    p.add_argument("--num-sgd-iter", type=int, default=10)

    # resources
    p.add_argument("--num-env-runners", type=int, default=1)
    p.add_argument("--num-gpus", type=int, default=None, help="Override detected GPUs. Default: auto-detect.")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.environ["PPO_CSV_BASE_DIR"] = os.path.abspath(args.out_dir)
    os.environ.setdefault("RAY_DEDUP_LOGS", "1")

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    register_env(ENV_ID, rllib_env_creator)

    bs_locs = [(1250, 1400), (3750, 1400), (1250, 3550), (3750, 3550)]
    env_config = dict(
        num_base_stations=args.num_base_stations,
        num_users=args.num_users,
        num_channels_per_carrier=args.num_channels,
        area_size=args.area_size,
        bs_loc=bs_locs,
        max_steps=args.max_steps,
        mobility_model=("manhattan" if args.mobility_model == "manhattan" else "random"),
        seed=args.seed,
        step_duration_s=args.step_duration_s,
        deadline_ms=args.deadline_ms,
        safety_traffic_ratio=args.safety_traffic_ratio,
    )

    # ----- PPO config -----
    config = PPOConfig()

    # Prefer old API stack for compatibility with a wide range of Ray versions.
    # Newer versions still accept these toggles.
    config.enable_rl_module_and_learner = False
    config.enable_env_runner_and_connector_v2 = False

    exploration_conf = {"type": "StochasticSampling"}
    config.exploration_config = exploration_conf

    config.environment(env=ENV_ID, env_config=env_config)
    config.framework("torch")

    # RLlib: env_runners replaces rollout workers in newer versions.
    config.env_runners(
        num_env_runners=args.num_env_runners,
        rollout_fragment_length="auto",
        batch_mode="complete_episodes",
    )

    config.training(
        gamma=args.gamma,
        lr=args.lr,
        lambda_=args.lambda_,
        clip_param=0.2,
        vf_clip_param=10.0,
        train_batch_size=args.train_batch_size,
        use_gae=True,
    )
    config.sgd_minibatch_size = args.sgd_minibatch_size
    config.num_sgd_iter = args.num_sgd_iter

    if args.num_gpus is None:
        num_gpus = int(torch.cuda.is_available())
    else:
        num_gpus = int(args.num_gpus)
    config.resources(num_gpus=num_gpus)

    config.normalize_actions = True

    config.multi_agent(
        policies={
            "shared_policy": PolicySpec(config={"exploration_config": exploration_conf})
        },
        policy_mapping_fn=lambda agent_id, *a, **k: "shared_policy",
        policies_to_train=["shared_policy"],
    )

    config.callbacks(EpisodeCSVLogger)

    algo = _build_algo(config)

    total_env_steps = 0
    it = 0

    while total_env_steps < args.total_steps:
        res = algo.train()
        it += 1

        total_env_steps = _get_total_env_steps(res)
        mean_reward, mean_len = _get_mean_return_and_len(res)

        if it % args.log_every == 0:
            print(
                f"[Iter {it:04d}] env_steps={total_env_steps} | "
                f"episode_return_mean={mean_reward:.3f} | episode_len_mean={mean_len:.1f}"
            )

        if it % args.save_every == 0:
            ckpt = algo.save(checkpoint_dir=os.path.join(args.out_dir, "checkpoints"))
            if hasattr(ckpt, "path"):
                ckpt_path = ckpt.path
            elif isinstance(ckpt, dict) and "checkpoint" in ckpt:
                ckpt_path = ckpt["checkpoint"]
            else:
                ckpt_path = str(ckpt)
            print(f"  ✓ checkpoint: {ckpt_path}")

    print("\n✅ Training finished.")
    ray.shutdown()


if __name__ == "__main__":
    main()
