"""Quick smoke test for the environment (no RLlib needed).

This is useful to confirm that the simulator runs before training.
"""

from __future__ import annotations

import numpy as np

from mamn_marl.env import MultiAgentMobileNetwork


def main() -> None:
    env = MultiAgentMobileNetwork(
        num_base_stations=4,
        num_users=50,
        num_channels_per_carrier=50,
        area_size=2000.0,
        max_steps=50,
        mobility_model="manhattan",
        seed=42,
    )

    obs, info = env.reset(seed=42)
    print("Reset OK. obs keys:", list(obs.keys()))

    for t in range(5):
        action_dict = {
            a: np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
            for a in env.agents
        }
        obs, rew, terminateds, truncateds, infos = env.step(action_dict)
        print(
            f"t={t} reward={list(rew.values())[0]:+.3f} "
            f"throughput={infos[env.agents[0]]['total_throughput_Mbps']:.2f} Mbps "
            f"lat={infos[env.agents[0]]['avg_latency_ms']:.2f} ms"
        )

    env.close()


if __name__ == "__main__":
    main()
