"""Evaluate a saved RLlib checkpoint and optionally render episodes.

This script loads a checkpoint produced by `scripts/train_ppo.py` and runs a
fixed number of episodes in a fresh environment.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import ray
import torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env

from mamn_marl.env import MultiAgentMobileNetwork


def rllib_env_creator(env_config: dict):
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


def build_algo(env_id: str, env_config: dict):
    config = PPOConfig()

    # Old API stack for compatibility
    config.enable_rl_module_and_learner = False
    config.enable_env_runner_and_connector_v2 = False

    exploration_conf = {"type": "StochasticSampling"}
    config.exploration_config = exploration_conf

    config.environment(env=env_id, env_config=env_config)
    config.framework("torch")
    config.env_runners(
        num_env_runners=0,
        rollout_fragment_length="auto",
        batch_mode="complete_episodes",
    )

    config.resources(num_gpus=int(torch.cuda.is_available()))
    config.normalize_actions = True

    config.multi_agent(
        policies={
            "shared_policy": PolicySpec(config={"exploration_config": exploration_conf})
        },
        policy_mapping_fn=lambda *_args, **_kwargs: "shared_policy",
        policies_to_train=["shared_policy"],
    )

    if hasattr(config, "build_algo"):
        return config.build_algo()
    return config.build()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to RLlib checkpoint")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--render", action="store_true")
    args = p.parse_args()

    env_id = "mamn_marl_mobile_network"
    env_config = dict(
        num_base_stations=4,
        num_users=100,
        num_channels_per_carrier=100,
        area_size=5000.0,
        bs_loc=[(1250, 1400), (3750, 1400), (1250, 3550), (3750, 3550)],
        max_steps=200,
        mobility_model="manhattan",
        seed=42,
        step_duration_s=1.0,
        deadline_ms=100.0,
        safety_traffic_ratio=0.3,
    )

    register_env(env_id, rllib_env_creator)

    os.environ.setdefault("RAY_DEDUP_LOGS", "1")
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    algo = build_algo(env_id, env_config)
    algo.restore(args.checkpoint)

    env = rllib_env_creator(env_config)

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0

        while not done:
            action_dict = {}
            for agent_id, o in obs.items():
                a = algo.compute_single_action(o, policy_id="shared_policy")
                action_dict[agent_id] = a

            obs, rewards, _terminateds, truncateds, _infos = env.step(action_dict)
            ep_ret += float(np.mean(list(rewards.values())))
            done = bool(truncateds.get("__all__", False))
            if args.render:
                env.render()

        print(f"Episode {ep+1}/{args.episodes}: return~{ep_ret:.3f}")

    env.close()
    ray.shutdown()


if __name__ == "__main__":
    main()
