"""RLlib callbacks used by training scripts.

Currently includes an episode-wise CSV logger that aggregates scalar `info` fields.
"""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, Optional

import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks


class EpisodeCSVLogger(DefaultCallbacks):
    """Episode-wise CSV logger for RLlib.

    - Accumulates scalar values from `info` into `episode.user_data`.
    - On episode end, writes a single row containing per-episode means.

    Output file:
        - `${PPO_CSV_BASE_DIR}/ppo_episode_metrics.csv` (defaults to cwd)

    Notes
    -----
    RLlib has multiple "Episode" implementations across versions (EpisodeV2, etc.).
    This callback is defensive and tries several access paths to retrieve info.
    """

    def __init__(self):
        super().__init__()

        base_dir = os.environ.get("PPO_CSV_BASE_DIR", os.getcwd())
        self.csv_path = os.path.join(base_dir, "ppo_episode_metrics.csv")

        self.header_written = False
        self.fieldnames: Optional[list[str]] = None

        # If CSV already exists, reuse its header
        if os.path.exists(self.csv_path):
            try:
                with open(self.csv_path, "r", newline="") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    if header:
                        self.fieldnames = list(header)
                        self.header_written = True
            except Exception:
                # We'll infer later from the first row
                self.fieldnames = None
                self.header_written = os.path.getsize(self.csv_path) > 0

    @staticmethod
    def _get_last_info(episode: Any) -> Optional[Dict[str, Any]]:
        """Best-effort extraction of a recent `info` dict from an episode object."""
        info = None

        # EpisodeV2 path
        if hasattr(episode, "get_agents") and hasattr(episode, "last_info_for"):
            agent_ids = list(episode.get_agents())
            if agent_ids:
                info = episode.last_info_for(agent_ids[0])

        # Older path
        elif hasattr(episode, "last_info_for") and hasattr(episode, "agent_ids"):
            agent_ids = list(getattr(episode, "agent_ids", []) or [])
            if agent_ids:
                info = episode.last_info_for(agent_ids[0])

        return info

    def on_episode_step(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        info = self._get_last_info(episode)
        if not info:
            return

        for k, v in info.items():
            if isinstance(v, (int, float)):
                episode.user_data.setdefault(k, []).append(float(v))

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        row: Dict[str, Any] = {
            "episode_id": getattr(episode, "episode_id", ""),
            "episode_len": getattr(episode, "length", ""),
        }

        # Episode-wide means for scalar metrics
        for k, values in getattr(episode, "user_data", {}).items():
            if not values:
                continue
            mean_val = float(np.mean(values))
            row[k] = mean_val
            # also expose as custom metric
            try:
                episode.custom_metrics[k] = mean_val
            except Exception:
                pass

        # Helpful: also include episode return (sum of global_reward if available)
        if "global_reward" in getattr(episode, "user_data", {}) and episode.user_data["global_reward"]:
            ep_return = float(np.sum(episode.user_data["global_reward"]))
            row["episode_return"] = ep_return
            try:
                episode.custom_metrics["episode_return"] = ep_return
            except Exception:
                pass

        # initialize header on first write
        if self.fieldnames is None:
            self.fieldnames = list(row.keys())

        for h in self.fieldnames:
            row.setdefault(h, "")

        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore")
            if not self.header_written or os.path.getsize(self.csv_path) == 0:
                writer.writeheader()
                self.header_written = True
            writer.writerow(row)
