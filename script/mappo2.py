"""
Complete experiment runner for the revised clustered IoV-VEC work.

This file is derived from the provided ``ppo_multi`` notebook and keeps the
radio/MEC environment, 20-D BS observation, 4-D BS action, shared QoE reward,
and PPO hyperparameters while adding reviewer-requested validation.  The default ``primary`` suite is
intended for the next experiment after Ours has already been trained: it runs
parameter-matched centralized-critic MAPPO and Cent-PPO for the requested
independent seeds, then merges their summaries with existing Ours results.
The original targeted/ablation/mobility modes remain available.

All raw training/evaluation outputs and paper-table summaries are written to CSV,
including streamed per-user/per-step evaluation samples.
The script intentionally does not invent or hard-code experimental results.
"""

CODE_VERSION = "2026-08-21-primary-v4-mappo-centppo"

import argparse
import csv
import json
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
from gymnasium.utils import seeding
import matplotlib.pyplot as plt
import numpy as np

# RLlib / Ray
import ray
from ray.rllib.algorithms.ppo import PPOConfig   # <<< PPO
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch
import torch.nn as nn


# ----------------------
# Basic PHY & helper functions
# ----------------------
def rx_sensitivity_dBm(bw_hz, nf_db=7.0, snr_req_db=-5.0):
    """Thermal-noise-based RX sensitivity in dBm."""
    return -174.0 + 10.0 * np.log10(max(bw_hz, 1.0)) + nf_db + snr_req_db


MOD_ORDER = 256
BITS_PER_CODEWORD = int(np.log2(MOD_ORDER))
NR_OVERHEAD = 0.85
SNR_GAP_DB = 1.5
MAX_SE = 7.5  # peak spectral efficiency (bit/s/Hz)

# Antenna / beamforming gains per BS type
BEAM_GAIN_DB = {
    "Ma": {
        "tx_main": 8.0,   # main-lobe TX gain (dB)
        "tx_side": -3.0,  # side-lobe TX gain (dB)
        "rx": 0.0,        # UE RX gain (dB)
    }
}

UE_INTERF_RX_DB = 0.0  # interference-side RX gain at UE
MACRO_REUSE_ONE = False
MIMO_MAX_RANK = {"Ma": 4}


def spectral_efficiency_from_sinr(SINR_dB, gap_db=SNR_GAP_DB, max_se=MAX_SE):
    """Shannon-like SE with SNR gap."""
    gamma_lin = 10.0 ** (SINR_dB / 10.0) / (10.0 ** (gap_db / 10.0))
    se = np.log2(1.0 + gamma_lin)
    return float(np.clip(se, 0.0, max_se))


def mimo_rank_and_total_se(
    SINR_dB: float,
    max_layers: int,
    gap_db: float = SNR_GAP_DB,
    max_se: float = MAX_SE,
):
    """Simple MIMO rank-selection and SE aggregation."""
    sinr_lin = 10.0 ** (SINR_dB / 10.0)
    best_L = 1
    best_sum_se = spectral_efficiency_from_sinr(SINR_dB, gap_db, max_se)

    for L in range(2, max_layers + 1):
        per_layer_snr_dB = 10.0 * np.log10(max(sinr_lin / max(L, 1), 1e-12))
        se_layer = spectral_efficiency_from_sinr(
            per_layer_snr_dB, gap_db, max_se
        )
        corr_eff = max(0.5, 1.0 - 0.07 * (L - 1))  # crude correlation penalty
        sum_se = L * se_layer * corr_eff
        if sum_se > best_sum_se:
            best_sum_se = sum_se
            best_L = L

    return best_L, float(best_sum_se)


# 20-D BS observation layout (0-1 normalized)
OBS_IDX = SimpleNamespace(
    BS_TYPE=0,
    TX_NORM=1,
    CH_UTIL=2,
    COV_UTIL=3,
    LOAD_RATIO_NORM=4,
    NEARBY_POT=5,
    AVG_SPEED=6,
    REQ_P_NORM=7,
    AVG_RADIAL_V=8,
    SP_VAR=9,
    NEIGHBOR_TX_NORM=10,
    AVG_DEMAND_NORM=11,
    INTER_NORM=12,
    MEC_QUEUE_NORM=13,
    CPU_UTIL_NORM=14,
    OFFLOAD_FRAC=15,
    SERVED_RATIO_NORM=16,  # local served throughput / max possible
    BLOCK_FRAC_NORM=17,    # fraction of in-coverage users not associated
    CH_MATCH_NORM=18,      # how well channels match covered users
    OFFLOAD_MAND_FRAC=19,  # fraction of covered users with mandatory-offload tasks
)

LOCAL_OBS_DIM = 20
# Neighbor-information ablation masks only the two explicitly cross-BS radio summaries.
NEIGHBOR_MASK_INDICES = (OBS_IDX.NEIGHBOR_TX_NORM,)


# ----------------
# Building blocks
# ----------------
class Channel:
    def __init__(self, id, frequency, bandwidth, noise_figure_db=7.0):
        self.id = int(id)
        self.frequency = float(frequency)
        self.bandwidth = float(bandwidth)
        self.noise_figure_db = float(noise_figure_db)
        self.users = []
        self.base_station = None

    def calculate_noise_power(self):
        """Noise power in Watts for this channel."""
        k = 1.380649e-23
        T = 293.15
        N = k * T * self.bandwidth
        NF = 10.0 ** (self.noise_figure_db / 10.0)
        return N * NF


class BaseStation:
    def __init__(
        self,
        id,
        transmit_power_mW,
        height_m,
        location_xy,
        type_bs="Ma",
        mec_cpu_capacity_cycles=5e9,
    ):
        self.id = int(id)
        self.transmit_power = float(transmit_power_mW)  # total TX power in mW
        self.height = float(height_m)
        self.location = np.array(location_xy, dtype=float)
        self.type_bs = type_bs
        self.assigned_channels = []
        self.per_channel_power = {}
        self.coverage_area = float(0.0)  # coverage radius (m)

        # MEC-related attributes (compute plane)
        self.mec_cpu_capacity = float(mec_cpu_capacity_cycles)  # F_i^{max}
        self.mec_queue_cycles = 0.0     # q_t^{bs,i} [cycles]
        self.offload_frac = 1.0         # x_t^i in [0,1]
        self.cpu_util_frac = 1.0        # aggregate CPU utilization fraction
        self.last_cpu_used = 0.0        # F_used this step [cycles/s]

        # simple type-dependent caps
        self.MAX_COVERAGE = {"Ma": 2000.0}
        self.COV_MARGIN_DB = {"Ma": 3.0}
        self.SNR_REQ_DB = {"Ma": -5.0}

    # ---- 3GPP-like UMa pathloss ----
    def _pl_uma_los(self, d3d, f_ghz, h_ut=1.5):
        c = 3e8
        f_hz = f_ghz * 1e9
        h_bs = float(self.height)

        d2d = max(np.sqrt(max(d3d**2 - (h_bs - h_ut) ** 2, 1e-9)), 1.0)
        h_bs_eff = h_bs - 1.0
        h_ut_eff = h_ut - 1.0
        d_bp = 4.0 * h_bs_eff * h_ut_eff * f_hz / c

        pl1 = 28.0 + 22.0 * np.log10(d3d) + 20.0 * np.log10(f_ghz)
        if d2d <= d_bp:
            return pl1
        pl2 = (
            28.0
            + 40.0 * np.log10(d3d)
            + 20.0 * np.log10(f_ghz)
            - 9.0 * np.log10(d_bp**2 + (h_bs - h_ut) ** 2)
        )
        return pl2

    def _pl_uma_nlos(self, d3d, f_ghz, h_ut=1.5):
        pl_los = self._pl_uma_los(d3d, f_ghz, h_ut)
        pl_nlos = (
            13.54
            + 39.08 * np.log10(d3d)
            + 20.0 * np.log10(f_ghz)
            - 0.6 * (h_ut - 1.5)
        )
        return max(pl_nlos, pl_los)

    def calculate_path_loss(self, distance_m, frequency_hz, user_height=1.5):
        """Average path loss (LOS-probability weighted)."""
        d2d = float(max(distance_m, 0.1))
        f_ghz = frequency_hz / 1e9
        d3d = np.sqrt(d2d**2 + (self.height - user_height) ** 2)

        p_los = min(18.0 / d2d, 1.0) * (1.0 - np.exp(-d2d / 63.0)) + np.exp(
            -d2d / 63.0
        )
        pl_los = self._pl_uma_los(d3d, f_ghz, h_ut=user_height)
        pl_nlos = self._pl_uma_nlos(d3d, f_ghz, h_ut=user_height)

        return float(p_los * pl_los + (1.0 - p_los) * pl_nlos)

    def _eirp_dbm(self, transmit_power_mW: float) -> float:
        g = BEAM_GAIN_DB[self.type_bs]
        tx_dbm = 10.0 * np.log10(max(transmit_power_mW, 1e-15))
        return tx_dbm + g["tx_main"] + g["rx"]

    def update_coverage_area_from_power(
        self, total_transmit_power_mW, frequency_hz
    ):
        """Update coverage radius from link budget."""
        if self.per_channel_power:
            values = list(self.per_channel_power.values())
            p_ch = float(np.percentile(values, 80))
        else:
            n = max(1, len(self.assigned_channels))
            p_ch = float(total_transmit_power_mW / n)

        if self.assigned_channels:
            bw_hz = self.assigned_channels[0].bandwidth
            nf_db = self.assigned_channels[0].noise_figure_db
        else:
            bw_hz = 20e6
            nf_db = 7.0

        sens_dBm = rx_sensitivity_dBm(
            bw_hz, nf_db=nf_db, snr_req_db=self.SNR_REQ_DB[self.type_bs]
        )
        eirp_plus_grx_dbm = self._eirp_dbm(p_ch)
        path_loss_budget_dB = (
            eirp_plus_grx_dbm - sens_dBm - self.COV_MARGIN_DB[self.type_bs]
        )

        radius_m = self.find_distance_for_path_loss(
            path_loss_budget_dB, frequency_hz
        )
        cap = self.MAX_COVERAGE.get(self.type_bs, None)
        if cap is not None:
            radius_m = min(radius_m, float(cap))

        self.coverage_area = float(max(radius_m, 1.0))
        return self.coverage_area

    def find_distance_for_path_loss(self, path_loss_dB, frequency_hz):
        """Invert pathloss model via binary search."""
        d_min, d_max = 1.0, 10000.0
        tol = 0.1
        d_mid = 0.0
        for _ in range(40):
            d_mid = 0.5 * (d_min + d_max)
            PL_mid = self.calculate_path_loss(d_mid, frequency_hz)
            if abs(PL_mid - path_loss_dB) < tol:
                return d_mid
            if PL_mid < path_loss_dB:
                d_min = d_mid
            else:
                d_max = d_mid
        return d_mid

    def assign_channels(self, channels):
        self.assigned_channels = list(channels)
        for ch in self.assigned_channels:
            ch.base_station = self

    def clear_assigned_channels(self):
        for ch in self.assigned_channels:
            ch.base_station = None
            ch.users = []
        self.assigned_channels = []
        self.per_channel_power = {}

    def find_available_channel(self):
        """Find a free (unoccupied) channel, if any."""
        for ch in self.assigned_channels:
            if len(ch.users) == 0:
                return ch
        return None


class User:
    def __init__(self, id, location_xy, velocity_xy, demand_bits=1e6):
        self.id = int(id)
        self.location = np.array(location_xy, dtype=float)
        self.velocity = np.array(velocity_xy, dtype=float)
        self.height = 1.5

        # radio link properties
        self.channel = []
        self.channel_SINR = []
        self.SINR = -100.0
        self.data_rate = 0.0  # instantaneous achievable rate (Mbps)
        self.demand = float(demand_bits)  # "required" rate (Mbps-like scale)
        self.mimo_layers = []

        # service class
        self.service_class = "infotainment"

        # task model (set properly by generate_task())
        self.task_type = "light"
        self.task_cycles = 0.0
        self.task_input_Mbit = 0.0
        self.task_offload_mandatory = False

        # mobility
        self.speed = float(np.linalg.norm(self.velocity))
        self.waypoint = None
        self.pause_time = 0
        self.dir_axis = None
        self.dir_sign = 1
        self.next_intersection = None
        self.mobility_regime = "urban"
        self.highway_lane_y = None

    def clear_channel(self):
        self.channel = []
        self.channel_SINR = []
        self.SINR = -100.0
        self.mimo_layers = []
        self.data_rate = 0.0

    def calculate_data_rate(self):
        """Aggregate rate across all assigned channels with MIMO."""
        self.data_rate = 0.0
        self.mimo_layers = []

        for i, ch in enumerate(self.channel):
            bw_eff = ch.bandwidth * NR_OVERHEAD
            SINR_dB = self.channel_SINR[i]
            max_rank = MIMO_MAX_RANK[ch.base_station.type_bs]

            L, total_se = mimo_rank_and_total_se(
                SINR_dB, max_rank, gap_db=SNR_GAP_DB, max_se=MAX_SE
            )
            dr_Mbps = (bw_eff * total_se) / 1e6
            self.mimo_layers.append(L)
            self.data_rate += dr_Mbps

    def calculate_demand_from_rng(self, rng):
        """Random traffic demand model."""
        if getattr(self, "service_class", "infotainment") == "safety":
            base_mbps = rng.uniform(0.5, 2.0)
        else:
            app_types = {"video": 5.0, "gaming": 10.0, "browsing": 2.0}
            app = rng.choice(list(app_types.keys()))
            base_mbps = app_types[app] * (1.0 + rng.uniform(0.0, 0.5))
        self.demand = float(base_mbps)

    def generate_task(self, rng, heavy_ratio, ue_cpu,
                      heavy_cyc_range, heavy_data_range,
                      light_cyc_range, light_data_range):
        """
        Assign a compute task to this user.

        heavy_ratio : fraction of users that get heavy (offload-mandatory) tasks
        ue_cpu      : UE local compute budget (cycles/step)
        *_range     : (lo, hi) for task cycles / input-data Mbit
        """
        if rng.random() < heavy_ratio:
            self.task_type = "heavy"    # MUST offload
            self.task_cycles = float(rng.uniform(*heavy_cyc_range))
            self.task_input_Mbit = float(rng.uniform(*heavy_data_range))
        else:
            self.task_type = "light"    # can run locally
            self.task_cycles = float(rng.uniform(*light_cyc_range))
            self.task_input_Mbit = float(rng.uniform(*light_data_range))
        self.task_offload_mandatory = (self.task_cycles > ue_cpu)

    def calculate_latency_ms(self, num_users_on_channel: float = None):
        """Very simplified base latency model (communication-centric)."""
        if not self.channel or self.data_rate <= 0:
            return 1000.0

        avg_d = np.mean(
            [
                np.linalg.norm(self.location - ch.base_station.location)
                for ch in self.channel
            ]
        )
        prop_delay_ms = (avg_d / 3e8) * 1e3
        proc_delay = 1.0
        sched_delay = 1.0

        if num_users_on_channel is None:
            num_users_on_channel = (
                np.mean([len(ch.users) for ch in self.channel])
                if self.channel
                else 1.0
            )

        queue_delay = 1.0 + num_users_on_channel / (self.data_rate + 1e-6)

        return float(
            np.clip(
                prop_delay_ms + proc_delay + sched_delay + queue_delay,
                0.0,
                1000.0,
            )
        )


# -----------------------
# Main environment class
# -----------------------
class MultiAgentMobileNetwork(MultiAgentEnv):
    """Multi-agent cellular + MEC environment (Gymnasium-style MA API)."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        num_base_stations=4,
        num_users=80,
        num_channels_per_carrier=50,
        area_size=2000.0,
        bs_loc=None,
        max_steps=200,
        mobility_model="urban",
        seed=42,
        step_duration_s=1.0,
        deadline_ms=100.0,
        safety_traffic_ratio=0.3,
        mask_neighbor_summaries=False,
        critic_observation_mode="local",
        reward_weights=None,
    ):
        super().__init__()

        # RNGs
        self.seed(seed)

        # topology / sim parameters
        self.num_base_stations = int(num_base_stations)
        self.num_users = int(num_users)
        self.num_channels_per_carrier = int(num_channels_per_carrier)
        self.area_size = float(area_size)
        self.max_steps = int(max_steps)
        scenario = str(mobility_model).lower().strip()
        aliases = {
            "manhattan": "urban",
            "urban_grid": "urban",
            "highway_like": "highway",
            "random_waypoint": "highway",
        }
        self.mobility_model = aliases.get(scenario, scenario)
        if self.mobility_model not in {"urban", "highway", "mixed"}:
            raise ValueError(
                f"mobility_model/scenario must be urban, highway, or mixed; got {mobility_model!r}"
            )
        self.mask_neighbor_summaries = bool(mask_neighbor_summaries)
        self.critic_observation_mode = str(critic_observation_mode).lower().strip()
        if self.critic_observation_mode not in {"local", "joint"}:
            raise ValueError("critic_observation_mode must be 'local' or 'joint'")

        # time / QoS
        self.step_duration_s = float(step_duration_s)
        self.deadline_ms = float(deadline_ms)
        self.safety_traffic_ratio = float(
            np.clip(safety_traffic_ratio, 0.0, 1.0)
        )

        # radio parameters
        self.ma_transmission_power = 40000.0  # mW (40 W)
        self.max_ma_channels = int(num_channels_per_carrier)
        self.macro_carrier_frequencies = (
            [3.5e9] if MACRO_REUSE_ONE else [3.4e9, 3.5e9, 3.6e9, 3.7e9]
        )
        self.macro_channel_bw = 180e3  # Hz

        # MEC / compute parameters
        self.mec_cpu_capacity_cycles = 50e9
        self.cycles_per_Mbit = 5e7
        self.cpu_kappa = 1e-27
        self.max_user_demand_Mbps = 10.0

        # ---- Task-model parameters (heterogeneous compute requirements) ----
        # Each user generates a task per step with compute_cycles & input_data.
        # "heavy" tasks EXCEED local UE capacity -> MUST offload to MEC.
        # "light" tasks fit on UE -> can run locally or be offloaded.
        self.heavy_task_ratio = 0.70          # 70% users get heavy tasks
        self.ue_local_cpu_cycles = 0.5e9      # UE compute budget per step (0.5 GHz)
        # Heavy tasks: 1-5 Gcycles  (well above UE capacity)
        self.heavy_task_cycles_range = (0.5e9, 2.5e9)
        self.heavy_task_data_range = (2.0, 8.0)     # input data in Mbit
        # Light tasks: 0.05-0.4 Gcycles (fit on UE)
        self.light_task_cycles_range = (0.05e9, 0.4e9)
        self.light_task_data_range = (0.2, 1.5)      # input data in Mbit
        # Penalty: attempting to run a heavy task locally -> huge delay
        self.local_heavy_penalty_ms = 4000.0

        # reward weights
        default_reward_weights = dict(
            w_lat=0.20,
            w_thr=0.30,
            w_dead=0.20,
            w_qoe=0.30,
            w_eng=0.05,
            w_block=0.20,
            w_fair=0.10,
        )
        if reward_weights is not None:
            default_reward_weights.update(dict(reward_weights))
        self.reward_weights = SimpleNamespace(**default_reward_weights)

        # objects
        self.users = []
        self.base_stations = []

        # BS placement
        if bs_loc is None:
            bs_loc = self._default_bs_locations(
                self.num_base_stations, self.area_size
            )
        assert len(bs_loc) >= self.num_base_stations

        for i in range(self.num_base_stations):
            bs = BaseStation(
                i,
                self.ma_transmission_power,
                30.0,
                bs_loc[i],
                type_bs="Ma",
                mec_cpu_capacity_cycles=self.mec_cpu_capacity_cycles,
            )
            self.base_stations.append(bs)

        # build macro channels
        self.macro_channels = []
        ch_id = 0
        for f in self.macro_carrier_frequencies:
            for _ in range(self.num_channels_per_carrier):
                self.macro_channels.append(
                    Channel(
                        ch_id,
                        f,
                        self.macro_channel_bw,
                        noise_figure_db=7.0,
                    )
                )
                ch_id += 1

        # spawn users according to the configured mobility regime
        rng = self.np_random
        for u_id in range(self.num_users):
            regime = self._user_regime_for_scenario(u_id)
            if regime == "urban":
                loc, vel = self._spawn_on_grid()
            else:
                loc, vel = self._spawn_highway()

            u = User(u_id, loc, vel)
            u.mobility_regime = regime
            svc = (
                "safety"
                if rng.random() < self.safety_traffic_ratio
                else "infotainment"
            )
            u.service_class = svc
            u.calculate_demand_from_rng(rng)
            u.generate_task(
                rng, self.heavy_task_ratio, self.ue_local_cpu_cycles,
                self.heavy_task_cycles_range, self.heavy_task_data_range,
                self.light_task_cycles_range, self.light_task_data_range,
            )
            self.users.append(u)

            if regime == "urban":
                self._init_user_manhattan(u)
            else:
                self._init_user_highway(u)

        # multi-agent API ids
        self.agents = [f"agent_{i}" for i in range(self.num_base_stations)]
        self.possible_agents = list(self.agents)
        self._agent_ids = set(self.agents)

        # action: (power_frac, channel_frac, offload_frac, cpu_frac)
        self.action_spaces = {
            a: gym.spaces.Box(
                low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
            for a in self.agents
        }

        # 20-D local observation. For MAPPO reference training, append the
        # normalized joint BS observation only for the centralized value function.
        self.local_obs_dim = LOCAL_OBS_DIM
        self.joint_obs_dim = self.local_obs_dim * self.num_base_stations
        obs_dim = self.local_obs_dim
        if self.critic_observation_mode == "joint":
            obs_dim += self.joint_obs_dim
        self.observation_spaces = {
            a: gym.spaces.Box(
                low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
            )
            for a in self.agents
        }

        # simulation state
        self.num_steps = 0
        self.current_episode_reward = 0.0
        self.assigned_channels = {}
        self.bs_carrier_frequency = {}

        self._assign_frequency_subsets()
        self.reset_user_info()

        # normalization constants
        # E_ref = Delta * sum_i(P_i^max + kappa_c * (F_i^max)^3).
        self.energy_normalization_J = self.step_duration_s * sum(
            (self.ma_transmission_power * 1e-3)
            + self.cpu_kappa * (bs.mec_cpu_capacity ** 3)
            for bs in self.base_stations
        )
        self.throughput_normalization_Mbps = (
            self.num_users * self.max_user_demand_Mbps
        )

        self.fig = None
        self._panel_axes = None

        # shared spaces for RLlib
        first_agent = self.agents[0]
        self.observation_space = self.observation_spaces[first_agent]
        self.action_space = self.action_spaces[first_agent]

    # -------------
    # RNG helpers
    # -------------
    def seed(self, seed=None):
        if seed is None:
            seed = random.randrange(2**31 - 1)
        seed_int = int(seed) & 0xFFFFFFFF
        self.np_random, _ = seeding.np_random(seed_int)
        random.seed(seed_int)
        np.random.seed(seed_int)
        return [seed_int]

    # -------------
    # Topology helpers
    # -------------
    def _default_bs_locations(self, n, L):
        """Simple grid-based macro deployment."""
        coords = []
        grid = int(math.ceil(math.sqrt(n)))
        xs = np.linspace(0.2 * L, 0.8 * L, grid)
        ys = np.linspace(0.2 * L, 0.8 * L, grid)
        for x in xs:
            for y in ys:
                coords.append((float(x), float(y)))
                if len(coords) >= n:
                    return coords[:n]
        return coords[:n]

    def _assign_frequency_subsets(self):
        """Assign each BS to a carrier (reuse pattern)."""
        idx = 0
        for bs in self.base_stations:
            if MACRO_REUSE_ONE:
                self.bs_carrier_frequency[bs.id] = (
                    self.macro_carrier_frequencies[0]
                )
            else:
                self.bs_carrier_frequency[bs.id] = (
                    self.macro_carrier_frequencies[
                        idx % len(self.macro_carrier_frequencies)
                    ]
                )
                idx += 1

    # -------------
    # Manhattan mobility
    # -------------
    def _build_road_grid(self):
        L = self.area_size
        spacing = max(50.0, float(250.0))
        self.grid_xs = np.arange(0.0, L + 1e-9, spacing)
        self.grid_ys = np.arange(0.0, L + 1e-9, spacing)

    def _spawn_on_grid(self):
        if not hasattr(self, "grid_xs"):
            self._build_road_grid()

        L = self.area_size
        if self.np_random.random() < 0.5:
            # horizontal road
            y = float(self.np_random.choice(self.grid_ys))
            x = float(self.np_random.uniform(0, L))
            dir_axis, dir_sign = "x", (1 if self.np_random.random() < 0.5 else -1)
        else:
            # vertical road
            x = float(self.np_random.choice(self.grid_xs))
            y = float(self.np_random.uniform(0, L))
            dir_axis, dir_sign = "y", (1 if self.np_random.random() < 0.5 else -1)

        speed = float(max(1.0, self.np_random.normal(12.0, 3.0)))
        if dir_axis == "x":
            vel = np.array([dir_sign * speed, 0.0], dtype=float)
        else:
            vel = np.array([0.0, dir_sign * speed], dtype=float)

        return np.array([x, y], dtype=float), vel

    def _init_user_manhattan(self, u: User):
        if not hasattr(self, "grid_xs"):
            self._build_road_grid()

        u.location = self._snap_to_grid(u.location)
        if abs(u.velocity[0]) >= abs(u.velocity[1]):
            u.dir_axis = "x"
            u.dir_sign = 1 if u.velocity[0] >= 0 else -1
        else:
            u.dir_axis = "y"
            u.dir_sign = 1 if u.velocity[1] >= 0 else -1

        u.speed = float(max(1.0, np.linalg.norm(u.velocity)))

        if u.dir_axis == "x":
            candidates = (
                self.grid_xs[self.grid_xs > u.location[0]]
                if u.dir_sign > 0
                else self.grid_xs[self.grid_xs < u.location[0]]
            )
            if u.dir_sign > 0:
                coord = candidates.min() if candidates.size else self.area_size
            else:
                coord = candidates.max() if candidates.size else 0.0
            u.next_intersection = np.array(
                [coord, self._snap_to_grid(u.location)[1]], dtype=float
            )
        else:
            candidates = (
                self.grid_ys[self.grid_ys > u.location[1]]
                if u.dir_sign > 0
                else self.grid_ys[self.grid_ys < u.location[1]]
            )
            if u.dir_sign > 0:
                coord = candidates.min() if candidates.size else self.area_size
            else:
                coord = candidates.max() if candidates.size else 0.0
            u.next_intersection = np.array(
                [self._snap_to_grid(u.location)[0], coord], dtype=float
            )
        u.pause_time = 0

    def _snap_to_grid(self, p):
        if not hasattr(self, "grid_xs"):
            self._build_road_grid()
        x = self.grid_xs[np.argmin(np.abs(self.grid_xs - p[0]))]
        y = self.grid_ys[np.argmin(np.abs(self.grid_ys - p[1]))]
        return np.array([x, y], dtype=float)

    def _advance_user_manhattan(self, u: User):
        if u.pause_time > 0:
            u.pause_time -= 1
            return

        delta = u.next_intersection - u.location
        dist = float(np.linalg.norm(delta))
        step = min(dist, u.speed)
        if dist > 1e-9:
            u.location += (delta / dist) * step

        # reached intersection or step exhausted
        if (
            np.linalg.norm(u.location - u.next_intersection) <= 1e-6
            or step == dist
        ):
            # occasional pause
            if self.np_random.random() < 0.15:
                u.pause_time = int(self.np_random.integers(0, 4))

            # turn decisions
            r = self.np_random.random()
            if r < 0.25:
                u.dir_axis = "y" if u.dir_axis == "x" else "x"
                u.dir_sign = -u.dir_sign
            elif r < 0.5:
                u.dir_axis = "y" if u.dir_axis == "x" else "x"

            u.speed = float(max(1.0, self.np_random.normal(12.0, 3.0)))

            if u.dir_axis == "x":
                if u.dir_sign > 0:
                    candidates = self.grid_xs[self.grid_xs > u.location[0]]
                    x_next = candidates.min() if candidates.size > 0 else self.area_size
                else:
                    candidates = self.grid_xs[self.grid_xs < u.location[0]]
                    x_next = candidates.max() if candidates.size > 0 else 0.0
                u.next_intersection = np.array(
                    [x_next, self._snap_to_grid(u.location)[1]], dtype=float
                )
                u.velocity = np.array([u.dir_sign * u.speed, 0.0], dtype=float)
            else:
                if u.dir_sign > 0:
                    candidates = self.grid_ys[self.grid_ys > u.location[1]]
                    y_next = candidates.min() if candidates.size > 0 else self.area_size
                else:
                    candidates = self.grid_ys[self.grid_ys < u.location[1]]
                    y_next = candidates.max() if candidates.size > 0 else 0.0
                u.next_intersection = np.array(
                    [self._snap_to_grid(u.location)[0], y_next], dtype=float
                )
                u.velocity = np.array([0.0, u.dir_sign * u.speed], dtype=float)

    def _user_regime_for_scenario(self, user_id: int) -> str:
        """Map the configured scenario to a per-user mobility regime."""
        if self.mobility_model == "mixed":
            return "urban" if (int(user_id) % 2 == 0) else "highway"
        return self.mobility_model

    def _spawn_highway(self):
        """High-speed multi-lane highway abstraction inside the common area."""
        L = self.area_size
        lanes = np.asarray([0.36, 0.44, 0.56, 0.64], dtype=float) * L
        lane_idx = int(self.np_random.integers(0, len(lanes)))
        y = float(lanes[lane_idx])
        x = float(self.np_random.uniform(0.0, L))
        direction = 1.0 if lane_idx < (len(lanes) // 2) else -1.0
        speed = float(np.clip(self.np_random.normal(28.0, 4.0), 18.0, 38.0))
        vel = np.array([direction * speed, 0.0], dtype=float)
        return np.array([x, y], dtype=float), vel

    def _init_user_highway(self, u: User):
        u.mobility_regime = "highway"
        u.highway_lane_y = float(u.location[1])
        u.speed = float(max(1.0, np.linalg.norm(u.velocity)))
        u.dir_axis = "x"
        u.dir_sign = 1 if u.velocity[0] >= 0 else -1
        u.pause_time = 0
        u.waypoint = None

    def _advance_user_highway(self, u: User):
        """Advance a highway vehicle with wrap-around longitudinal motion."""
        L = self.area_size
        if self.np_random.random() < 0.02:
            u.speed = float(np.clip(u.speed + self.np_random.normal(0.0, 1.0), 18.0, 38.0))
        u.velocity = np.array([u.dir_sign * u.speed, 0.0], dtype=float)
        u.location[0] = float((u.location[0] + u.velocity[0] * self.step_duration_s) % L)
        if u.highway_lane_y is not None:
            u.location[1] = float(u.highway_lane_y)

    # -------------
    # User / BS state helpers
    # -------------
    def reset_user_info(self):
        """Per-user association / channel bookkeeping."""
        self.user_info = {
            u.id: {
                "user": u.id,
                "ma_assigned_channel": [],
                "ma_base_station": [],
                "ma_base_station_location": [],
            }
            for u in self.users
        }

    def assign_channels_on_demand(self):
        """UE association and 1-channel assignment per UE to its best BS."""
        for u in self.users:
            u.clear_channel()

        for bs in self.base_stations:
            for ch in bs.assigned_channels:
                ch.users = []

        for u in self.users:
            best_bs, _ = self._best_bs_in_cov(u, type_filter="Ma")
            if best_bs:
                ch = best_bs.find_available_channel()
                if ch:
                    u.channel.append(ch)
                    ch.users.append(u)
                    info = self.user_info[u.id]
                    info["ma_assigned_channel"].append(ch.id)
                    info["ma_base_station"].append(best_bs.id)
                    info["ma_base_station_location"].append(
                        best_bs.location.copy()
                    )

    def _per_channel_tx_power_mW(self, bs: BaseStation):
        if bs.per_channel_power:
            vals = list(bs.per_channel_power.values())
            return float(np.mean(vals))
        n = max(1, len(bs.assigned_channels))
        return bs.transmit_power / n

    def _noise_mW_for(self, bs: BaseStation):
        """Noise (mW) for BS based on reference channel bandwidth."""
        if bs.assigned_channels:
            return bs.assigned_channels[0].calculate_noise_power() * 1e3
        # fallback
        bw_hz = self.macro_channel_bw
        k = 1.380649e-23
        T = 293.15
        NF = 10 ** (7 / 10)
        return k * T * bw_hz * NF * 1e3

    def _est_rate_one_channel_Mbps(self, u: User, bs: BaseStation):
        """Estimated per-channel rate for association decisions."""
        if not bs.assigned_channels:
            return 0.0

        f = self.bs_carrier_frequency[bs.id]
        d = float(np.linalg.norm(u.location - bs.location))
        PL_dB = bs.calculate_path_loss(d, f)

        n_ch = max(1, len(bs.assigned_channels))
        p_ch = bs.transmit_power / n_ch

        g = BEAM_GAIN_DB[bs.type_bs]
        g_tx = 10 ** (g["tx_main"] / 10.0)
        g_rx = 10 ** (g["rx"] / 10.0)

        sig_mW = p_ch * g_tx * g_rx / (10.0 ** (PL_dB / 10.0))
        noise_mW = self._noise_mW_for(bs)

        # ignore inter-cell interference in this quick estimate
        SINR = sig_mW / max(noise_mW, 1e-12)
        SINR_dB = 10.0 * np.log10(max(SINR, 1e-12))

        bw = bs.assigned_channels[0].bandwidth * NR_OVERHEAD
        _, total_se = mimo_rank_and_total_se(
            SINR_dB, MIMO_MAX_RANK[bs.type_bs], gap_db=SNR_GAP_DB, max_se=MAX_SE
        )
        return (bw * total_se) / 1e6

    def _best_bs_in_cov(self, u: User, type_filter=None):
        """Select BS with best rate*fairness score among those covering user."""
        best = None
        best_score = 0.0

        for bs in self.base_stations:
            if type_filter is not None and bs.type_bs != type_filter:
                continue
            if not bs.assigned_channels:
                continue

            d = np.linalg.norm(u.location - bs.location)
            if d > bs.coverage_area:
                continue

            r = self._est_rate_one_channel_Mbps(u, bs)
            load = (
                sum(1 for c in bs.assigned_channels if len(c.users) > 0)
                / max(1, len(bs.assigned_channels))
            )
            fairness = 1.0 - 0.5 * load
            score = r * fairness
            if score > best_score:
                best_score = score
                best = bs

        return best, best_score

    def _estimate_interference_mW(self, u: User, f, exclude_bs: BaseStation):
        """Downlink inter-cell interference (mW) from co-channel BSs."""
        co_bs = [
            b
            for b in self.base_stations
            if b is not exclude_bs and self.bs_carrier_frequency[b.id] == f
        ]
        I = 0.0
        g_rx_int = 10 ** (UE_INTERF_RX_DB / 10.0)

        for bi in co_bs:
            if not bi.assigned_channels:
                continue

            p_i = self._per_channel_tx_power_mW(bi)
            d_i = np.linalg.norm(u.location - bi.location)
            PL_i = bi.calculate_path_loss(d_i, f)

            gi = BEAM_GAIN_DB[bi.type_bs]
            g_tx_i = 10 ** (gi["tx_side"] / 10.0)

            users_in_cov_bi = [
                uu
                for uu in self.users
                if np.linalg.norm(uu.location - bi.location)
                <= bi.coverage_area
            ]
            util = min(
                1.0, len(users_in_cov_bi) / max(1, len(bi.assigned_channels))
            )

            I += (
                util
                * p_i
                * g_tx_i
                * g_rx_int
                / (10.0 ** (PL_i / 10.0))
            )

        return I

    # -------------
    # Water-filling power allocation
    # -------------
    @staticmethod
    def _waterfill(P_total, h_list, n_list, p_floor_list=None, tol=1e-6, max_it=200):
        """Classical water-filling with optional minimum powers (p_floor_list)."""
        K = len(h_list)
        if K == 0:
            return []

        h = np.asarray(h_list, dtype=float)
        n = np.asarray(n_list, dtype=float)
        p_floor = (
            np.zeros(K, dtype=float)
            if p_floor_list is None
            else np.asarray(p_floor_list, dtype=float)
        )

        mask = h > 0
        if not np.any(mask):
            return [0.0] * K

        p_floor_sum = float(np.sum(np.maximum(p_floor, 0.0)))
        if p_floor_sum > P_total and p_floor_sum > 0:
            scaled = P_total * (p_floor / p_floor_sum)
            return list(np.maximum(scaled, 0.0))

        def alloc(lmbd):
            base = 1.0 / max(lmbd, 1e-18)
            out = np.maximum(p_floor, base - n / np.maximum(h, 1e-18))
            return np.maximum(out, 0.0)

        lo, hi = 0.0, 1e12
        for _ in range(max_it):
            mid = 0.5 * (lo + hi)
            p = alloc(mid)
            s = float(np.sum(p))
            if abs(s - P_total) <= tol:
                return list(np.maximum(p, 0.0))
            if s > P_total:
                lo = mid
            else:
                hi = mid

        return list(np.maximum(alloc(hi), 0.0))

    # -------------
    # Core RL API (Gymnasium-style MA)
    # -------------
    def step(self, action_dict):
        """
        Gymnasium-style multi-agent step.

        Returns:
            obs (dict): agent_id -> observation
            rewards (dict): agent_id -> reward (float)
            terminateds (dict): agent_id -> bool, plus "__all__"
            truncateds (dict): agent_id -> bool, plus "__all__"
            infos (dict): agent_id -> info dict
        """
        start_wall = time.time()
        self.num_steps += 1

        # --------- parse / clip actions ----------
        for agent in self.agents:
            if agent not in action_dict:
                action_dict[agent] = np.array(
                    [0.5, 0.5, 0.5, 0.5], dtype=float
                )
            action_dict[agent] = np.clip(
                np.array(action_dict[agent], dtype=float),
                0.0,
                1.0,
            )

        # --------- traffic demand & task update ----------
        for u in self.users:
            if self.np_random.random() < 0.05:
                u.calculate_demand_from_rng(self.np_random)
            # Re-generate task every step (new task arrival each TTI)
            u.generate_task(
                self.np_random, self.heavy_task_ratio, self.ue_local_cpu_cycles,
                self.heavy_task_cycles_range, self.heavy_task_data_range,
                self.light_task_cycles_range, self.light_task_data_range,
            )

        # --------- clear previous associations & channel assignments ----------
        for u in self.users:
            u.clear_channel()

        for bs in self.base_stations:
            bs.clear_assigned_channels()
            # keep MEC queues across steps (do not reset)

        for ch in self.macro_channels:
            ch.users = []
            ch.base_station = None

        self.assigned_channels.clear()
        self.reset_user_info()

        # --------- resource allocation per BS: power, channels, offload, CPU ----------
        for i, bs in enumerate(self.base_stations):
            agent = f"agent_{i}"
            power_frac, channel_frac, offload_frac, cpu_frac = action_dict[
                agent
            ]

            # power allocation
            bs.transmit_power = float(
                np.clip(power_frac, 0.0, 1.0)
            ) * self.ma_transmission_power

            # channel allocation
            max_cap = int(self.max_ma_channels)
            req_ch = int(
                np.rint(
                    float(np.clip(channel_frac, 0.0, 1.0)) * max_cap
                )
            )

            if req_ch == 0 and bs.transmit_power > 1e-9:
                req_ch = 1

            freq = self.bs_carrier_frequency[bs.id]
            avail = [
                c
                for c in self.macro_channels
                if c.frequency == freq and c.base_station is None
            ]
            req_ch = min(len(avail), req_ch)

            if req_ch > 0:
                chosen = avail[:req_ch]
                bs.assign_channels(chosen)
                for ch in chosen:
                    self.assigned_channels[ch] = bs.id

            # MEC-related controls
            bs.offload_frac = float(np.clip(offload_frac, 0.0, 1.0))
            bs.cpu_util_frac = float(np.clip(cpu_frac, 0.0, 1.0))
            bs.last_cpu_used = bs.cpu_util_frac * bs.mec_cpu_capacity

            bs.update_coverage_area_from_power(bs.transmit_power, freq)

        # --------- user association and 1ch-per-user assignment ----------
        self.assign_channels_on_demand()

        # --------- per-BS water-filling ----------
        for bs in self.base_stations:
            bs.per_channel_power = {}
            ch_act = []
            H = []
            N = []
            floors = []

            for ch in bs.assigned_channels:
                if not ch.users:
                    continue

                # simple single-UE-per-channel view (first user)
                u = ch.users[0]
                f = ch.frequency
                d = float(np.linalg.norm(u.location - bs.location))
                PL_dB = bs.calculate_path_loss(d, f)

                g = BEAM_GAIN_DB[bs.type_bs]
                h_linear = (
                    10 ** (g["tx_main"] / 10.0)
                    * 10 ** (g["rx"] / 10.0)
                ) / (10.0 ** (PL_dB / 10.0))

                noise_mW = ch.calculate_noise_power() * 1e3
                I_mW = self._estimate_interference_mW(u, f, bs)

                prio = float(
                    np.clip(
                        0.5 + (u.demand / self.max_user_demand_Mbps),
                        0.5,
                        2.0,
                    )
                )
                N_eff = (noise_mW + I_mW) / prio

                ch_act.append(ch)
                H.append(h_linear)
                N.append(N_eff)
                floors.append(0.0)

            if ch_act:
                P_total = float(bs.transmit_power)
                p_list = self._waterfill(
                    P_total,
                    H,
                    N,
                    p_floor_list=floors,
                    tol=1e-6,
                    max_it=120,
                )
                for ch, p in zip(ch_act, p_list):
                    bs.per_channel_power[ch.id] = float(max(p, 0.0))
            else:
                bs.per_channel_power = {}

            f = self.bs_carrier_frequency[bs.id]
            bs.update_coverage_area_from_power(bs.transmit_power, f)

        # --------- SINR & rates ----------
        for u in self.users:
            self.calculate_SINR(u)
            u.calculate_data_rate()

        # --------- MEC queue dynamics & compute delays (task-aware) ----------
        bs_to_users = defaultdict(list)
        for u in self.users:
            if u.channel:
                bs = u.channel[0].base_station
                if bs is not None:
                    bs_to_users[bs.id].append(u)

        user_comp_delay_ms = {u.id: 0.0 for u in self.users}

        for bs in self.base_stations:
            assoc_users = bs_to_users.get(bs.id, [])

            off_frac = float(np.clip(bs.offload_frac, 0.0, 1.0))
            cpu_frac = float(np.clip(bs.cpu_util_frac, 0.0, 1.0))
            F_max = bs.mec_cpu_capacity
            F_used = cpu_frac * F_max
            bs.last_cpu_used = F_used

            # ---- Partition workload between MEC and local execution ----
            # Heavy tasks are mandatory-offload. For eligible light tasks, xi_i is
            # interpreted as a deterministic workload fraction, not a Bernoulli draw.
            offloaded = []       # (user, MEC cycles) pairs
            total_new_cycles = 0.0

            for u in assoc_users:
                C = float(u.task_cycles)

                if u.task_offload_mandatory:
                    mec_cycles = C
                    local_cycles = 0.0
                else:
                    mec_cycles = off_frac * C
                    local_cycles = (1.0 - off_frac) * C

                if mec_cycles > 1e-9:
                    offloaded.append((u, mec_cycles))
                    total_new_cycles += mec_cycles

                if local_cycles > 1e-9:
                    local_delay_s = local_cycles / max(self.ue_local_cpu_cycles, 1e-9)
                    user_comp_delay_ms[u.id] += local_delay_s * 1000.0

            # ---- MEC queue update ----
            q_prev = bs.mec_queue_cycles
            mu_cycles = F_used * self.step_duration_s
            q_after_serv = max(q_prev - mu_cycles, 0.0)
            q_next = q_after_serv + total_new_cycles

            if F_used > 0.0:
                queue_delay_s = q_after_serv / F_used
            else:
                queue_delay_s = (
                    self.deadline_ms / 1000.0 * 10.0
                )

            n_off = len(offloaded)
            for (u, C_cycles) in offloaded:
                if F_used > 0.0 and n_off > 0:
                    f_share = F_used / n_off
                    T_proc_s = C_cycles / max(f_share, 1e-9)
                    T_queue_s = queue_delay_s
                else:
                    T_proc_s = self.deadline_ms / 1000.0 * 10.0
                    T_queue_s = self.deadline_ms / 1000.0 * 10.0

                comp_ms = (T_proc_s + T_queue_s) * 1000.0
                comp_ms = float(np.clip(comp_ms, 0.0, 5000.0))
                user_comp_delay_ms[u.id] += comp_ms

            bs.mec_queue_cycles = float(q_next)

        # ---------------- BS-level throughput for fairness across BSs ----------------
        bs_served = []
        for bs in self.base_stations:
            assoc_users = bs_to_users.get(bs.id, [])
            local_served = sum(
                min(u.data_rate, u.demand) for u in assoc_users
            )
            bs_served.append(local_served)
        bs_served_arr = np.asarray(bs_served, dtype=float)
        if np.any(bs_served_arr > 0):
            jain_num_bs = float(np.sum(bs_served_arr) ** 2)
            jain_den_bs = float(
                len(bs_served_arr) * np.sum(bs_served_arr**2) + 1e-9
            )
            jain_fairness_bs = float(jain_num_bs / jain_den_bs)
        else:
            jain_fairness_bs = 0.0

        # --------- QoE metrics: throughput, latency, energy, fairness ----------
        served_rates = []
        latencies = []
        qoe_list = []
        deadline_hits = 0
        deadline_candidates = 0

        svc_stats = {
            "safety": {
                "served_rates": [],
                "latencies": [],
                "qoes": [],
                "deadline_hits": 0,
                "deadline_candidates": 0,
            },
            "infotainment": {
                "served_rates": [],
                "latencies": [],
                "qoes": [],
                "deadline_hits": 0,
                "deadline_candidates": 0,
            },
        }

        for u in self.users:
            svc = getattr(u, "service_class", "infotainment")

            served = min(u.data_rate, u.demand)
            served_rates.append(served)

            bucket = svc_stats.get(svc, None)
            if bucket is not None:
                bucket["served_rates"].append(served)

            base_lat = u.calculate_latency_ms()
            lat = base_lat + user_comp_delay_ms.get(u.id, 0.0)
            latencies.append(lat)
            if bucket is not None:
                bucket["latencies"].append(lat)

            if u.demand > 0:
                deadline_candidates += 1
                if lat <= self.deadline_ms:
                    deadline_hits += 1

                if bucket is not None:
                    bucket["deadline_candidates"] += 1
                    if lat <= self.deadline_ms:
                        bucket["deadline_hits"] += 1

            if u.demand > 0:
                thr_norm = np.clip(served / u.demand, 0.0, 1.0)
            else:
                thr_norm = 0.0
            if lat <= self.deadline_ms:
                lat_factor = 1.0
            else:
                lat_factor = max(
                    0.0,
                    1.0 - (lat - self.deadline_ms) / (5 * self.deadline_ms),
                )
            qoe = 0.5 * thr_norm + 0.5 * lat_factor
            qoe_clipped = float(np.clip(qoe, 0.0, 1.0))
            qoe_list.append(qoe_clipped)
            if bucket is not None:
                bucket["qoes"].append(qoe_clipped)

        # throughput metrics (Mbps)
        total_throughput_Mbps = float(np.sum(served_rates))
        avg_throughput_Mbps = float(
            total_throughput_Mbps / max(1, self.num_users)
        )

        # latency metrics
        if latencies:
            avg_latency_ms = float(np.mean(latencies))
            p95_latency_ms = float(np.percentile(latencies, 95))
        else:
            avg_latency_ms = 0.0
            p95_latency_ms = 0.0

        # deadline satisfaction
        if deadline_candidates > 0:
            deadline_satisfaction = float(
                deadline_hits / max(1, deadline_candidates)
            )
        else:
            deadline_satisfaction = 0.0

        # fairness (Jain's index) over users (for logging only)
        served_arr = np.asarray(served_rates, dtype=float)
        if np.any(served_arr > 0):
            jain_num_users = float(np.sum(served_arr) ** 2)
            jain_den_users = float(
                len(served_arr) * np.sum(served_arr**2) + 1e-9
            )
            jain_fairness_users = float(jain_num_users / jain_den_users)
        else:
            jain_fairness_users = 0.0

        # Use BS-level fairness in the reward
        jain_fairness = jain_fairness_bs

        # energy metrics
        total_radio_power_mW = float(
            np.sum([bs.transmit_power for bs in self.base_stations])
        )
        total_cpu_power_W = float(
            np.sum(
                [self.cpu_kappa * (bs.last_cpu_used**3) for bs in self.base_stations]
            )
        )
        step_energy_J = (
            total_radio_power_mW * 1e-3 * self.step_duration_s
            + total_cpu_power_W * self.step_duration_s
        )

        satisfied_tasks = deadline_hits
        if satisfied_tasks > 0:
            energy_per_task_J = float(step_energy_J / satisfied_tasks)
        else:
            energy_per_task_J = 0.0

        # blocking rate: users without any channel
        blocked_users = [u for u in self.users if not u.channel]
        blocking_rate = float(len(blocked_users) / max(1, self.num_users))

        # QoE statistics
        if qoe_list:
            avg_qoe = float(np.mean(qoe_list))
        else:
            avg_qoe = 0.0

        # ---------------- service-class metrics (for ITS-oriented analysis) ----------------
        def _svc_metrics(bucket):
            if not bucket["served_rates"]:
                return {
                    "avg_throughput_Mbps": 0.0,
                    "avg_latency_ms": 0.0,
                    "p95_latency_ms": 0.0,
                    "deadline_satisfaction": 0.0,
                    "avg_qoe": 0.0,
                }
            s_rates = np.asarray(bucket["served_rates"], dtype=float)
            lats = np.asarray(bucket["latencies"], dtype=float)
            qoes = np.asarray(bucket["qoes"], dtype=float)

            if lats.size > 0:
                avg_lat = float(np.mean(lats))
                p95_lat = float(np.percentile(lats, 95))
            else:
                avg_lat = 0.0
                p95_lat = 0.0

            if s_rates.size > 0:
                avg_tput = float(np.mean(s_rates))
            else:
                avg_tput = 0.0

            if bucket["deadline_candidates"] > 0:
                ds = float(
                    bucket["deadline_hits"]
                    / max(1, bucket["deadline_candidates"])
                )
            else:
                ds = 0.0

            if qoes.size > 0:
                avg_qoe_svc = float(np.mean(qoes))
            else:
                avg_qoe_svc = 0.0

            return {
                "avg_throughput_Mbps": avg_tput,
                "avg_latency_ms": avg_lat,
                "p95_latency_ms": p95_lat,
                "deadline_satisfaction": ds,
                "avg_qoe": avg_qoe_svc,
            }

        svc_safety = _svc_metrics(svc_stats["safety"])
        svc_infot = _svc_metrics(svc_stats["infotainment"])

        # ---------------- Normalized components (for logging) ----------------
        U_norm = float(
            np.clip(
                total_throughput_Mbps
                / max(1e-9, self.throughput_normalization_Mbps),
                0.0,
                1.0,
            )
        )

        if latencies:
            violations = [
                max(lat - self.deadline_ms, 0.0)
                / (5.0 * max(1.0, self.deadline_ms))
                for lat in latencies
            ]
            D_norm = float(np.clip(np.mean(violations), 0.0, 1.0))
        else:
            D_norm = 0.0

        E_norm = float(
            np.clip(
                step_energy_J / max(1e-9, self.energy_normalization_J),
                0.0,
                1.0,
            )
        )

        # ------------- Joint latency + throughput reward ----------------
        latency_ref_ms = self.deadline_ms if self.deadline_ms > 0 else 100.0

        if latency_ref_ms > 0:
            L_mean = avg_latency_ms / latency_ref_ms
            L_p95 = p95_latency_ms / (1.5 * latency_ref_ms)
        else:
            L_mean = 0.0
            L_p95 = 0.0

        L_norm = float(np.clip(0.7 * L_mean + 0.3 * L_p95, 0.0, 1.0))

        T_norm = U_norm
        V_fair = 1.0 - jain_fairness

        w = self.reward_weights
        positive = (
            w.w_thr * T_norm
            + w.w_lat * (1.0 - D_norm)
            + w.w_dead * deadline_satisfaction
            + w.w_qoe * avg_qoe
        )
        negative = (
            w.w_eng * E_norm
            + w.w_block * blocking_rate
            + w.w_fair * V_fair
        )
        global_reward = float(np.clip(positive - negative, -1.0, 1.0))

        rewards = {agent: global_reward for agent in self.agents}

        scenario_type = self.mobility_model

        # --------- per-BS infos ----------
        infos = {}
        association_fractions = []
        channel_rewards = []

        for i, bs in enumerate(self.base_stations):
            agent = f"agent_{i}"
            users_in_cov = [
                u
                for u in self.users
                if np.linalg.norm(u.location - bs.location)
                <= bs.coverage_area
            ]
            assoc_users = [
                u
                for u in users_in_cov
                if any(ch.base_station == bs for ch in u.channel)
            ]

            assoc_frac = (
                len(assoc_users) / max(1, len(users_in_cov))
                if users_in_cov
                else 0.0
            )
            association_fractions.append(assoc_frac)

            ch_num = max(1, len(bs.assigned_channels))
            usr_num = max(1, len(users_in_cov))
            channel_rewards.append(
                1.0 - abs(ch_num - usr_num) / (ch_num + usr_num)
            )

            local_served = sum(
                min(u.data_rate, u.demand) for u in assoc_users
            )
            local_demand = sum(u.demand for u in assoc_users) + 1e-9
            local_util = (
                local_served / local_demand if local_demand > 0 else 0.0
            )

            if assoc_users:
                loc_latencies = [
                    latencies[self.users.index(u)] for u in assoc_users
                ]
                loc_avg_lat_ms = float(np.mean(loc_latencies))
            else:
                loc_avg_lat_ms = 0.0

            infos[agent] = {
                "local_data_rate_Mbps": float(
                    sum(u.data_rate for u in assoc_users)
                ),
                "local_served_throughput_Mbps": float(local_served),
                "local_demand_Mbps": float(local_demand),
                "local_power_mW": float(bs.transmit_power),
                "local_channels": int(len(bs.assigned_channels)),
                "users_in_cov": int(len(users_in_cov)),
                "assoc_users": int(len(assoc_users)),
                "local_avg_latency_ms": loc_avg_lat_ms,
                "local_util": float(local_util),
                "local_mec_queue_cycles": float(bs.mec_queue_cycles),
                "local_cpu_used_cycles_per_s": float(bs.last_cpu_used),
                "local_offload_frac": float(bs.offload_frac),
                "local_cpu_util_frac": float(bs.cpu_util_frac),
            }

        if association_fractions:
            mean_assoc_frac = float(np.mean(association_fractions))
        else:
            mean_assoc_frac = 0.0

        if channel_rewards:
            mean_channel_reward = float(np.mean(channel_rewards))
        else:
            mean_channel_reward = 0.0

        global_info = {
            "total_throughput_Mbps": total_throughput_Mbps,
            "avg_throughput_Mbps": avg_throughput_Mbps,
            "offered_load_Mbps": float(np.sum([u.demand for u in self.users])),
            "avg_latency_ms": avg_latency_ms,
            "p95_latency_ms": p95_latency_ms,
            "latency_per_user": latencies,
            "deadline_satisfaction": deadline_satisfaction,
            "deadline_hits": int(deadline_hits),
            "deadline_candidates": int(deadline_candidates),
            "blocking_rate": blocking_rate,
            "blocked_users_count": int(len(blocked_users)),
            "avg_qoe": avg_qoe,
            "qoe_per_user": qoe_list,
            "total_radio_power_mW": total_radio_power_mW,
            "total_cpu_power_W": total_cpu_power_W,
            "step_energy_J": step_energy_J,
            "energy_per_task_J": energy_per_task_J,
            "jain_fairness": jain_fairness,
            "jain_fairness_bs": jain_fairness_bs,
            "jain_fairness_users": jain_fairness_users,
            "num_users": int(self.num_users),
            "num_base_stations": int(self.num_base_stations),
            "mean_assoc_frac": mean_assoc_frac,
            "mean_channel_reward": mean_channel_reward,
            "env_step_walltime_ms": float(
                (time.time() - start_wall) * 1e3
            ),
            "U_norm": U_norm,
            "D_norm": D_norm,
            "E_norm": E_norm,
            "L_norm": L_norm,
            "V_fair": V_fair,
            "global_reward": global_reward,
            "reward_w_lat": self.reward_weights.w_lat,
            "reward_w_thr": self.reward_weights.w_thr,
            "reward_w_dead": self.reward_weights.w_dead,
            "reward_w_qoe": self.reward_weights.w_qoe,
            "reward_w_eng": self.reward_weights.w_eng,
            "reward_w_block": self.reward_weights.w_block,
            "reward_w_fair": self.reward_weights.w_fair,
            "scenario_type": scenario_type,
            "svc_safety_avg_throughput_Mbps": svc_safety["avg_throughput_Mbps"],
            "svc_safety_avg_latency_ms": svc_safety["avg_latency_ms"],
            "svc_safety_p95_latency_ms": svc_safety["p95_latency_ms"],
            "svc_safety_deadline_satisfaction": svc_safety["deadline_satisfaction"],
            "svc_safety_avg_qoe": svc_safety["avg_qoe"],
            "svc_infotainment_avg_throughput_Mbps": svc_infot["avg_throughput_Mbps"],
            "svc_infotainment_avg_latency_ms": svc_infot["avg_latency_ms"],
            "svc_infotainment_p95_latency_ms": svc_infot["p95_latency_ms"],
            "svc_infotainment_deadline_satisfaction": svc_infot["deadline_satisfaction"],
            "svc_infotainment_avg_qoe": svc_infot["avg_qoe"],
            "qoe_safety": svc_stats["safety"]["qoes"],
            "qoe_infotainment": svc_stats["infotainment"]["qoes"],
        }

        for agent in self.agents:
            infos[agent].update(global_info)

        self.current_episode_reward += global_reward

        # Gymnasium-style: terminateds / truncateds
        done_all = self.num_steps >= self.max_steps

        terminateds = {agent: False for agent in self.agents}
        truncateds = {agent: done_all for agent in self.agents}
        terminateds["__all__"] = False
        truncateds["__all__"] = done_all

        # Mobility is part of the state transition. Move vehicles before exposing
        # the next observation so o(t+1) matches the state used by the next action.
        self.update_user_location()
        obs = self.get_observation()

        # step() -> obs, rewards, terminateds, truncateds, infos
        return obs, rewards, terminateds, truncateds, infos

    def reset(self, *, seed=None, options=None):
        """
        Gymnasium-style reset.

        Returns:
            obs (dict): agent_id -> observation
            infos (dict): agent_id -> info dict (empty at reset)
        """
        if seed is not None:
            self.seed(seed)

        self.num_steps = 0
        self.current_episode_reward = 0.0

        # reset users
        for u in self.users:
            regime = self._user_regime_for_scenario(u.id)
            u.mobility_regime = regime
            if regime == "urban":
                loc, vel = self._spawn_on_grid()
                u.location = loc
                u.velocity = vel
                self._init_user_manhattan(u)
            else:
                loc, vel = self._spawn_highway()
                u.location = loc
                u.velocity = vel
                self._init_user_highway(u)

            svc = (
                "safety"
                if self.np_random.random() < self.safety_traffic_ratio
                else "infotainment"
            )
            u.service_class = svc
            u.calculate_demand_from_rng(self.np_random)
            u.generate_task(
                self.np_random, self.heavy_task_ratio, self.ue_local_cpu_cycles,
                self.heavy_task_cycles_range, self.heavy_task_data_range,
                self.light_task_cycles_range, self.light_task_data_range,
            )

            u.data_rate = 0.0
            u.SINR = -100.0
            u.clear_channel()

        # reset channels & BSs
        for ch in self.macro_channels:
            ch.users = []
            ch.base_station = None

        self.assigned_channels.clear()

        for bs in self.base_stations:
            bs.clear_assigned_channels()
            bs.transmit_power = self.ma_transmission_power
            bs.mec_queue_cycles = 0.0
            bs.offload_frac = 1.0
            bs.cpu_util_frac = 1.0
            bs.last_cpu_used = bs.mec_cpu_capacity

        # simple initial allocation: 1 channel per BS if available
        for bs in self.base_stations:
            freq = self.bs_carrier_frequency[bs.id]
            avail = [
                c
                for c in self.macro_channels
                if c.frequency == freq and c.base_station is None
            ]
            if avail:
                chosen = [avail[0]]
                bs.assign_channels(chosen)
                self.assigned_channels[avail[0]] = bs.id
            bs.update_coverage_area_from_power(
                bs.transmit_power, self.bs_carrier_frequency[bs.id]
            )

        self.reset_user_info()
        self.prev_assoc_by_user = {u.id: {"Ma": None} for u in self.users}

        obs = self.get_observation()
        infos = {agent: {} for agent in self.agents}
        return obs, infos

    # -------------
    # Observation & global state
    # -------------
    def get_observation(self):
        obs = {}
        max_speed = 40.0

        for i, bs in enumerate(self.base_stations):
            agent = f"agent_{i}"
            users_in_cov = [
                u
                for u in self.users
                if np.linalg.norm(u.location - bs.location)
                <= bs.coverage_area
            ]

            max_power = self.ma_transmission_power
            max_ch_for_bs = float(max(1, self.max_ma_channels))

            bs_type = 0.5
            tx_norm = np.clip(bs.transmit_power / max_power, 0.0, 1.0)

            used = sum(1 for c in bs.assigned_channels if len(c.users) > 0)
            ch_util = np.clip(used / max_ch_for_bs, 0.0, 1.0)

            cov_util = len(users_in_cov) / max(1, self.num_users)

            load_ratio = (
                len(users_in_cov)
                / max(1, len(bs.assigned_channels))
                if bs.assigned_channels
                else 0.0
            )
            load_ratio_norm = float(np.clip(load_ratio / 2.0, 0.0, 1.0))

            nearby_pot = len(
                [
                    u
                    for u in self.users
                    if np.linalg.norm(u.location - bs.location)
                    <= bs.coverage_area * 1.5
                ]
            ) / max(1, self.num_users)

            avg_speed = np.clip(
                (
                    sum(u.speed for u in users_in_cov) / len(users_in_cov)
                    if users_in_cov
                    else 0.0
                )
                / max_speed,
                0.0,
                1.0,
            )

            # approximate required power so that edge user meets SNR target
            if users_in_cov:
                max_dist = max(
                    np.linalg.norm(u.location - bs.location)
                    for u in users_in_cov
                )
                req_p_per_ch = self.calculate_required_power_for_distance(
                    max_dist, bs
                )
                req_total = req_p_per_ch * max(
                    1, len(bs.assigned_channels)
                )
            else:
                req_total = 0.0

            req_p_norm = np.clip(
                req_total / max_power if max_power > 0 else 0.0, 0.0, 1.0
            )

            # radial velocity statistics
            avg_radial_v = 0.5
            if users_in_cov:
                comps = []
                for u in users_in_cov:
                    vec = bs.location - u.location
                    dist = np.linalg.norm(vec)
                    if dist > 1e-6:
                        comp = np.dot(u.velocity, vec / dist)
                        comps.append(comp)
                if comps:
                    avg_radial = float(np.mean(comps))
                    avg_radial_v = float(
                        np.clip((avg_radial / max_speed + 1) / 2.0, 0.0, 1.0)
                    )

            sp_var = np.clip(
                np.var([u.speed for u in users_in_cov])
                / ((max_speed**2) / 4.0)
                if len(users_in_cov) > 1
                else 0.0,
                0.0,
                1.0,
            )

            # coarse interference estimate
            f = self.bs_carrier_frequency[bs.id]
            co_bs = [
                o
                for o in self.base_stations
                if o is not bs and self.bs_carrier_frequency[o.id] == f
            ]

            sampled_users = users_in_cov[: min(len(users_in_cov), 8)]
            I_b = 0.0
            g_rx = 10.0 ** (UE_INTERF_RX_DB / 10.0)

            for b2 in co_bs:
                if not b2.assigned_channels:
                    continue
                p_i = self._per_channel_tx_power_mW(b2)
                gi = BEAM_GAIN_DB[b2.type_bs]
                g_tx_i = 10.0 ** (gi["tx_side"] / 10.0)

                users_in_cov_b2 = [
                    uu
                    for uu in self.users
                    if np.linalg.norm(uu.location - b2.location)
                    <= b2.coverage_area
                ]
                util = min(
                    1.0,
                    len(users_in_cov_b2) / max(1, len(b2.assigned_channels)),
                )

                if sampled_users:
                    acc = 0.0
                    for uu in sampled_users:
                        d_i = np.linalg.norm(uu.location - b2.location)
                        PL_i = b2.calculate_path_loss(d_i, f)
                        acc += (
                            p_i
                            * g_tx_i
                            * g_rx
                            / (10.0 ** (PL_i / 10.0))
                        )
                    I_b += util * (acc / len(sampled_users))

            I_dBm = 10.0 * np.log10(max(I_b, 1e-12))
            inter_norm = np.clip((I_dBm + 120.0) / 120.0, 0.0, 1.0)

            avg_demand_norm = (
                float(
                    np.clip(
                        np.mean(
                            [u.demand for u in users_in_cov]
                        )
                        / self.max_user_demand_Mbps,
                        0.0,
                        1.0,
                    )
                )
                if users_in_cov
                else 0.0
            )

            # Compact exchanged neighbor-activity summary.  Adjacent BSs in the
            # default 2-km square are ~1.2 km apart, so use a geometry-scaled
            # radius and aggregate both transmit load and RB activation.
            neighbor_radius_m = 0.75 * self.area_size
            neigh = []
            for o in self.base_stations:
                if o is bs:
                    continue
                if np.linalg.norm(bs.location - o.location) <= neighbor_radius_m:
                    tx_load = o.transmit_power / max(1.0, self.ma_transmission_power)
                    rb_load = len(o.assigned_channels) / max(1.0, self.max_ma_channels)
                    neigh.append(0.5 * (tx_load + rb_load))
            neighbor_tx_norm = float(np.clip(np.mean(neigh), 0.0, 1.0)) if neigh else 0.0

            mec_queue_norm = float(
                np.clip(
                    bs.mec_queue_cycles
                    / max(
                        1.0,
                        bs.mec_cpu_capacity
                        * self.step_duration_s
                        * 10.0,
                    ),
                    0.0,
                    1.0,
                )
            )
            cpu_util_norm = float(
                np.clip(
                    bs.last_cpu_used / max(1.0, bs.mec_cpu_capacity),
                    0.0,
                    1.0,
                )
            )
            offload_norm = float(np.clip(bs.offload_frac, 0.0, 1.0))

            assoc_users = [
                u
                for u in users_in_cov
                if any(ch.base_station == bs for ch in u.channel)
            ]

            if assoc_users:
                local_served = sum(
                    min(u.data_rate, u.demand) for u in assoc_users
                )
                max_served = len(assoc_users) * self.max_user_demand_Mbps
                served_ratio_norm = float(
                    np.clip(
                        local_served / max(1e-9, max_served),
                        0.0,
                        1.0,
                    )
                )
            else:
                served_ratio_norm = 0.0

            if users_in_cov:
                blocked_local = len(
                    [
                        u
                        for u in users_in_cov
                        if not any(ch.base_station == bs for ch in u.channel)
                    ]
                )
                block_frac_norm = float(
                    np.clip(
                        blocked_local / max(1, len(users_in_cov)), 0.0, 1.0
                    )
                )
            else:
                block_frac_norm = 0.0

            ch_num = len(bs.assigned_channels)
            usr_num = len(users_in_cov)
            if ch_num + usr_num > 0:
                ch_match_norm = float(
                    1.0
                    - min(1.0, abs(ch_num - usr_num) / (ch_num + usr_num))
                )
            else:
                ch_match_norm = 1.0

            # fraction of covered users whose tasks MUST be offloaded
            if users_in_cov:
                offload_mand_frac = float(
                    np.mean([1.0 if u.task_offload_mandatory else 0.0
                             for u in users_in_cov])
                )
            else:
                offload_mand_frac = 0.0

            vec_obs = np.array(
                [
                    bs_type,
                    tx_norm,
                    ch_util,
                    cov_util,
                    load_ratio_norm,
                    nearby_pot,
                    avg_speed,
                    req_p_norm,
                    avg_radial_v,
                    sp_var,
                    neighbor_tx_norm,
                    avg_demand_norm,
                    inter_norm,
                    mec_queue_norm,
                    cpu_util_norm,
                    offload_norm,
                    served_ratio_norm,
                    block_frac_norm,
                    ch_match_norm,
                    offload_mand_frac,
                ],
                dtype=np.float32,
            )
            if self.mask_neighbor_summaries:
                vec_obs = vec_obs.copy()
                for idx in NEIGHBOR_MASK_INDICES:
                    vec_obs[int(idx)] = 0.0
            obs[agent] = vec_obs

        if self.critic_observation_mode == "joint":
            joint_obs = np.concatenate([obs[a] for a in self.agents]).astype(np.float32)
            obs = {
                a: np.concatenate([obs[a], joint_obs]).astype(np.float32)
                for a in self.agents
            }

        return obs

    def get_global_state(self):
        """Optional centralized critic state."""
        per_bs = []
        for bs in self.base_stations:
            users_in_cov = [
                u
                for u in self.users
                if np.linalg.norm(u.location - bs.location)
                <= bs.coverage_area
            ]
            assoc_users = [
                u
                for u in users_in_cov
                if any(ch.base_station == bs for ch in u.channel)
            ]
            mean_rate = (
                float(np.mean([u.data_rate for u in assoc_users]))
                if assoc_users
                else 0.0
            )
            served_frac = (
                float(
                    sum(
                        min(u.data_rate, u.demand) for u in assoc_users
                    )
                    / max(
                        1e-9,
                        sum(u.demand for u in assoc_users),
                    )
                )
                if assoc_users
                else 0.0
            )
            per_bs.extend(
                [
                    bs.transmit_power,
                    len(bs.assigned_channels),
                    mean_rate,
                    served_frac,
                    bs.mec_queue_cycles,
                    bs.last_cpu_used,
                ]
            )
        return np.array(per_bs, dtype=np.float32)

    # -------------
    # PHY helpers
    # -------------
    def calculate_SINR(self, user: User):
        """Populate user.channel_SINR and aggregate SINR."""
        user.channel_SINR = []
        user.SINR = -100.0

        for ch in user.channel:
            bs = ch.base_station
            if bs is None:
                user.channel_SINR.append(-100.0)
                continue

            d = np.linalg.norm(user.location - bs.location)
            PL_dB = bs.calculate_path_loss(d, ch.frequency)

            p_tx_ch = bs.per_channel_power.get(ch.id, None)
            if p_tx_ch is None:
                n = max(1, len(bs.assigned_channels))
                p_tx_ch = bs.transmit_power / n

            g = BEAM_GAIN_DB[bs.type_bs]
            g_tx_main = 10.0 ** (g["tx_main"] / 10.0)
            g_rx_sig = 10.0 ** (g["rx"] / 10.0)

            signal_mW = (
                p_tx_ch * g_tx_main * g_rx_sig / (10.0 ** (PL_dB / 10.0))
            )

            interference_mW = self._estimate_interference_mW(
                user, ch.frequency, bs
            )
            noise_mW = ch.calculate_noise_power() * 1e3

            denom = max(interference_mW + noise_mW, 1e-15)
            SINR_lin = signal_mW / denom if signal_mW > 0 else 0.0
            if SINR_lin > 0:
                SINR_dB = 10.0 * np.log10(SINR_lin)
            else:
                SINR_dB = -100.0

            SINR_dB = float(np.clip(SINR_dB, -100.0, 60.0))
            user.channel_SINR.append(SINR_dB)

        if user.channel_SINR:
            s_ma = np.mean([10.0 ** (x / 10.0) for x in user.channel_SINR])
            user.SINR = 10.0 * np.log10(s_ma) if s_ma > 0 else -100.0
        else:
            user.SINR = -100.0

    def calculate_required_power_for_distance(
        self, distance_m, base_station: BaseStation
    ):
        """Required TX power (mW) so that UE at distance_m hits target SNR."""
        f = self.bs_carrier_frequency[base_station.id]
        pl_dB = base_station.calculate_path_loss(distance_m, f)

        if base_station.assigned_channels:
            bw_hz = base_station.assigned_channels[0].bandwidth
            nf_db = base_station.assigned_channels[0].noise_figure_db
        else:
            bw_hz = self.macro_channel_bw
            nf_db = 7.0

        sens_dBm = rx_sensitivity_dBm(
            bw_hz, nf_db=nf_db, snr_req_db=-5.0
        )
        g = BEAM_GAIN_DB[base_station.type_bs]
        req_tx_dBm = sens_dBm + pl_dB - (g["tx_main"] + g["rx"])

        return 10.0 ** (req_tx_dBm / 10.0)

    # -------------
    # Mobility update
    # -------------
    def update_user_location(self):
        if not hasattr(self, "grid_xs"):
            self._build_road_grid()
        for u in self.users:
            if getattr(u, "mobility_regime", "urban") == "urban":
                self._advance_user_manhattan(u)
                u.location = np.clip(u.location, 0.0, self.area_size)
            else:
                self._advance_user_highway(u)

    # -------------
    # Rendering
    # -------------
    def render(self, mode="human"):
        if mode != "human":
            return

        if self.fig is None:
            self.fig, self._panel_axes = plt.subplots(1, 1, figsize=(8, 8))

        ax = self._panel_axes
        ax.clear()

        L = self.area_size
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        ax.set_aspect("equal")

        # BSs and coverage
        for bs in self.base_stations:
            circle = plt.Circle(
                (bs.location[0], bs.location[1]),
                bs.coverage_area,
                color="C0",
                alpha=0.12,
            )
            ax.add_patch(circle)
            ax.scatter(
                bs.location[0], bs.location[1], marker="^", c="C0", s=80
            )
            ax.text(
                bs.location[0] + 5,
                bs.location[1] + 5,
                f"BS{bs.id}",
                fontsize=8,
            )

        # users and serving links
        for u in self.users:
            ax.plot(u.location[0], u.location[1], "ro", ms=4)
            for ch in u.channel:
                bs = ch.base_station
                if bs is None:
                    continue
                ax.plot(
                    [u.location[0], bs.location[0]],
                    [u.location[1], bs.location[1]],
                    "k-",
                    lw=0.6,
                    alpha=0.6,
                )

        plt.pause(0.001)

    def close(self):
        try:
            if self.fig is not None:
                plt.close(self.fig)
        except Exception:
            pass
        self.fig = None

# ============================================================================
# Targeted reviewer-requested experiment infrastructure
# ============================================================================

MULTI_ENV_ID = "clustered_iov_multi"
CENTRAL_ENV_ID = "clustered_iov_central"
MODEL_ID = "split_actor_central_critic"


class SplitActorCentralCriticModel(TorchModelV2, nn.Module):
    """Parameter-matched PPO actor with a centralized MAPPO-style value function.

    Observation layout for this reference is
        [local_obs_i (20), joint_bs_obs (20 * B)].
    The actor receives ONLY local_obs_i.  The value function receives ONLY the
    joint BS observation.  Therefore, centralized information changes critic
    training but not the decentralized actor input.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        custom = dict(model_config.get("custom_model_config", {}))
        self.local_obs_dim = int(custom["local_obs_dim"])
        self.joint_obs_dim = int(custom["joint_obs_dim"])

        actor_hiddens = list(custom.get("actor_hiddens", [256, 256]))
        actor_layers = []
        in_dim_actor = self.local_obs_dim
        for h in actor_hiddens:
            actor_layers.extend([nn.Linear(in_dim_actor, int(h)), nn.Tanh()])
            in_dim_actor = int(h)
        actor_layers.append(nn.Linear(in_dim_actor, int(num_outputs)))
        self.actor_net = nn.Sequential(*actor_layers)

        critic_hiddens = list(custom.get("critic_hiddens", [256, 256]))
        critic_layers = []
        in_dim = self.joint_obs_dim
        for h in critic_hiddens:
            critic_layers.extend([nn.Linear(in_dim, int(h)), nn.Tanh()])
            in_dim = int(h)
        critic_layers.append(nn.Linear(in_dim, 1))
        self.central_vf = nn.Sequential(*critic_layers)
        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float()
        local_obs = obs[..., : self.local_obs_dim]
        joint_obs = obs[
            ..., self.local_obs_dim : self.local_obs_dim + self.joint_obs_dim
        ]
        logits = self.actor_net(local_obs)
        self._value_out = self.central_vf(joint_obs).squeeze(-1)
        return logits, state

    def value_function(self):
        if self._value_out is None:
            raise RuntimeError("value_function() called before forward()")
        return self._value_out


class CentralizedPPOEnv(gym.Env):
    """Single-policy centralized PPO reference over the same physical simulator.

    The centralized observation is the concatenation of all 20-D BS observations
    and the action is the concatenation of all four BS controls.  It therefore
    changes only the information/control centralization, not the environment.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, env_config=None):
        super().__init__()
        cfg = dict(env_config or {})
        cfg["critic_observation_mode"] = "local"
        cfg["mask_neighbor_summaries"] = False
        self.base_env = rllib_env_creator(cfg)
        self.num_base_stations = self.base_env.num_base_stations
        self.local_obs_dim = self.base_env.local_obs_dim
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.local_obs_dim * self.num_base_stations,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4 * self.num_base_stations,),
            dtype=np.float32,
        )
        self.last_multi_infos = {}

    def _flatten_obs(self, obs_dict):
        return np.concatenate(
            [obs_dict[f"agent_{i}"] for i in range(self.num_base_stations)]
        ).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        obs, infos = self.base_env.reset(seed=seed, options=options)
        self.last_multi_infos = infos
        return self._flatten_obs(obs), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != 4 * self.num_base_stations:
            raise ValueError(
                f"central action has {action.size} elements; expected {4*self.num_base_stations}"
            )
        action_dict = {
            f"agent_{i}": np.clip(action[4 * i : 4 * (i + 1)], 0.0, 1.0)
            for i in range(self.num_base_stations)
        }
        obs, rewards, terminateds, truncateds, infos = self.base_env.step(action_dict)
        self.last_multi_infos = infos
        first = "agent_0"
        info = dict(infos.get(first, {}))
        reward = float(rewards.get(first, 0.0))
        terminated = bool(terminateds.get("__all__", False))
        truncated = bool(truncateds.get("__all__", False))
        return self._flatten_obs(obs), reward, terminated, truncated, info

    def render(self):
        return self.base_env.render()

    def close(self):
        return self.base_env.close()


@dataclass(frozen=True)
class ExperimentSpec:
    method: str
    scenario: str
    seed: int
    mask_neighbor_summaries: bool = False

    @property
    def variant(self) -> str:
        return "no_neighbor" if self.mask_neighbor_summaries else "full"

    @property
    def run_id(self) -> str:
        return f"{self.method}__{self.variant}__{self.scenario}__seed{self.seed}"


class EpisodeCSVLogger(DefaultCallbacks):
    """Write one row per TRAINING episode to a run-specific CSV.

    A unique file is used for every method/scenario/seed, avoiding the original
    header-collision problem when different experiments append to one CSV.
    """

    def __init__(self):
        super().__init__()
        path = os.environ.get("IOV_TRAIN_EPISODE_CSV")
        self.csv_path = Path(path) if path else None
        self.meta = {
            "run_id": os.environ.get("IOV_RUN_ID", "unknown"),
            "method": os.environ.get("IOV_METHOD", "unknown"),
            "variant": os.environ.get("IOV_VARIANT", "unknown"),
            "scenario": os.environ.get("IOV_SCENARIO", "unknown"),
            "seed": os.environ.get("IOV_SEED", ""),
        }
        self.header_written = False
        self.fieldnames = None
        if self.csv_path is not None:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            if self.csv_path.exists() and self.csv_path.stat().st_size > 0:
                try:
                    with self.csv_path.open("r", newline="") as f:
                        header = next(csv.reader(f), None)
                    if header:
                        self.fieldnames = header
                        self.header_written = True
                except Exception:
                    pass

    @staticmethod
    def _last_info(episode):
        if not hasattr(episode, "last_info_for"):
            return None
        try:
            if hasattr(episode, "get_agents"):
                agents = list(episode.get_agents())
                if agents:
                    return episode.last_info_for(agents[0])
            return episode.last_info_for()
        except Exception:
            try:
                return episode.last_info_for("agent_0")
            except Exception:
                return None

    def on_episode_step(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        info = self._last_info(episode)
        if not info:
            return
        for k, v in info.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                episode.user_data.setdefault(k, []).append(float(v))

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        if self.csv_path is None:
            return
        row = dict(self.meta)
        row.update(
            {
                "episode_id": getattr(episode, "episode_id", ""),
                "episode_len": getattr(episode, "length", ""),
            }
        )
        for k, values in episode.user_data.items():
            if not values:
                continue
            mean_val = float(np.mean(values))
            row[k] = mean_val
            try:
                episode.custom_metrics[k] = mean_val
            except Exception:
                pass

        if "global_reward" in episode.user_data and episode.user_data["global_reward"]:
            ep_return = float(np.sum(episode.user_data["global_reward"]))
            row["episode_return"] = ep_return
            try:
                episode.custom_metrics["episode_return"] = ep_return
            except Exception:
                pass

        if self.fieldnames is None:
            self.fieldnames = list(row.keys())
        for h in self.fieldnames:
            row.setdefault(h, "")
        with self.csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore")
            if not self.header_written or self.csv_path.stat().st_size == 0:
                writer.writeheader()
                self.header_written = True
            writer.writerow(row)


def _default_bs_locations_for_area(num_bs: int, area_size: float):
    grid = int(math.ceil(math.sqrt(num_bs)))
    xs = np.linspace(0.2 * area_size, 0.8 * area_size, grid)
    ys = np.linspace(0.2 * area_size, 0.8 * area_size, grid)
    out = []
    for x in xs:
        for y in ys:
            out.append((float(x), float(y)))
            if len(out) == num_bs:
                return out
    return out


def rllib_env_creator(env_config: dict):
    """Create the decentralized multi-BS environment using paper defaults."""
    cfg = dict(env_config or {})
    n_bs = int(cfg.get("num_base_stations", 4))
    area = float(cfg.get("area_size", 2000.0))
    return MultiAgentMobileNetwork(
        num_base_stations=n_bs,
        num_users=cfg.get("num_users", 80),
        num_channels_per_carrier=cfg.get("num_channels_per_carrier", 50),
        area_size=area,
        bs_loc=cfg.get("bs_loc", _default_bs_locations_for_area(n_bs, area)),
        max_steps=cfg.get("max_steps", 200),
        mobility_model=cfg.get("mobility_model", "urban"),
        seed=cfg.get("seed", 42),
        step_duration_s=cfg.get("step_duration_s", 1.0),
        deadline_ms=cfg.get("deadline_ms", 100.0),
        safety_traffic_ratio=cfg.get("safety_traffic_ratio", 0.3),
        mask_neighbor_summaries=cfg.get("mask_neighbor_summaries", False),
        critic_observation_mode=cfg.get("critic_observation_mode", "local"),
        reward_weights=cfg.get("reward_weights"),
    )


def central_env_creator(env_config: dict):
    return CentralizedPPOEnv(env_config)


def set_run_environment(spec: ExperimentSpec, run_dir: Path):
    os.environ["IOV_RUN_ID"] = spec.run_id
    os.environ["IOV_METHOD"] = spec.method
    os.environ["IOV_VARIANT"] = spec.variant
    os.environ["IOV_SCENARIO"] = spec.scenario
    os.environ["IOV_SEED"] = str(spec.seed)
    os.environ["IOV_TRAIN_EPISODE_CSV"] = str(run_dir / "train_episodes.csv")


def make_env_config(spec: ExperimentSpec, args):
    mode = "joint" if spec.method == "mappo" else "local"
    return {
        "num_base_stations": int(args.num_bs),
        "num_users": int(args.num_users),
        "num_channels_per_carrier": int(args.channels_per_carrier),
        "area_size": float(args.area_size),
        "bs_loc": _default_bs_locations_for_area(int(args.num_bs), float(args.area_size)),
        "max_steps": int(args.episode_len),
        "mobility_model": spec.scenario,
        "seed": int(spec.seed),
        "step_duration_s": 1.0,
        "deadline_ms": float(args.deadline_ms),
        "safety_traffic_ratio": 0.30,
        "mask_neighbor_summaries": bool(spec.mask_neighbor_summaries),
        "critic_observation_mode": mode,
    }


def _set_old_api_stack(config):
    """Stay compatible with the notebook's ModelV2/PolicySpec implementation."""
    if hasattr(config, "api_stack"):
        try:
            config.api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            return
        except Exception:
            pass
    if hasattr(config, "enable_rl_module_and_learner"):
        config.enable_rl_module_and_learner = False
    if hasattr(config, "enable_env_runner_and_connector_v2"):
        config.enable_env_runner_and_connector_v2 = False


def build_algorithm(spec: ExperimentSpec, args):
    env_cfg = make_env_config(spec, args)
    config = PPOConfig()
    _set_old_api_stack(config)

    env_id = CENTRAL_ENV_ID if spec.method == "cent_ppo" else MULTI_ENV_ID
    config.environment(env=env_id, env_config=env_cfg)
    config.framework("torch")

    # Keep complete episodes (as in the manuscript), but make the sampling
    # fragment match one environment episode by default. The previous "auto"
    # fragment combined with a heavy 200-step environment could exceed Ray's
    # synchronous sampling timeout and cause valid remote-worker samples to be
    # discarded.
    rollout_fragment_length = int(args.rollout_fragment_length)
    if rollout_fragment_length <= 0:
        rollout_fragment_length = int(args.episode_len)

    sample_timeout_s = float(args.sample_timeout_s)
    if sample_timeout_s <= 0:
        raise ValueError("--sample-timeout-s must be > 0")

    # Ray 2.x/old-stack compatibility: env_runners() accepts the old rollout
    # settings in current Ray releases; rollouts() remains the fallback for
    # older installations.
    configured_rollouts = False
    try:
        config.env_runners(
            num_env_runners=int(args.num_env_runners),
            rollout_fragment_length=rollout_fragment_length,
            batch_mode="complete_episodes",
            sample_timeout_s=sample_timeout_s,
        )
        configured_rollouts = True
    except (TypeError, AttributeError):
        pass

    if not configured_rollouts:
        try:
            config.rollouts(
                num_rollout_workers=int(args.num_env_runners),
                rollout_fragment_length=rollout_fragment_length,
                batch_mode="complete_episodes",
            )
            configured_rollouts = True
        except Exception as exc:
            raise RuntimeError(
                "Unable to configure RLlib rollout workers/EnvRunners for the "
                "installed Ray version."
            ) from exc

        # sample_timeout_s is a top-level AlgorithmConfig setting on the old
        # synchronous sampling path. Set it explicitly when rollouts() is used.
        if hasattr(config, "sample_timeout_s"):
            config.sample_timeout_s = sample_timeout_s

    config.training(
        gamma=0.99,
        lr=5e-5,
        lambda_=0.95,
        clip_param=0.2,
        vf_clip_param=10.0,
        train_batch_size=int(args.train_batch_size),
        use_gae=True,
    )
    # Old-stack PPO names retained for compatibility with the source notebook.
    if hasattr(config, "sgd_minibatch_size"):
        config.sgd_minibatch_size = int(args.minibatch_size)
    if hasattr(config, "num_sgd_iter"):
        config.num_sgd_iter = int(args.num_sgd_iter)

    try:
        config.resources(num_gpus=float(args.num_gpus))
    except Exception:
        pass
    config.normalize_actions = True
    try:
        config.debugging(seed=int(spec.seed), log_level="WARN")
    except Exception:
        config.seed = int(spec.seed)

    base_model = {
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "tanh",
        "vf_share_layers": False,
    }

    if spec.method == "mappo":
        model_cfg = dict(base_model)
        model_cfg.update(
            {
                "custom_model": MODEL_ID,
                "custom_model_config": {
                    "local_obs_dim": LOCAL_OBS_DIM,
                    "joint_obs_dim": LOCAL_OBS_DIM * int(args.num_bs),
                    "actor_hiddens": [256, 256],
                    "critic_hiddens": [256, 256],
                },
            }
        )
    else:
        model_cfg = base_model

    if spec.method != "cent_ppo":
        exploration_conf = {"type": "StochasticSampling"}
        config.multi_agent(
            policies={
                "shared_policy": PolicySpec(
                    config={
                        "exploration_config": exploration_conf,
                        "model": model_cfg,
                    }
                )
            },
            policy_mapping_fn=lambda agent_id, *a, **k: "shared_policy",
            policies_to_train=["shared_policy"],
        )
    else:
        # Centralized PPO receives joint observation/action through the wrapper.
        try:
            config.training(model=base_model)
        except Exception:
            config.model.update(base_model)

    config.callbacks(EpisodeCSVLogger)
    return config.build_algo() if hasattr(config, "build_algo") else config.build()


def get_total_env_steps(result_dict):
    candidates = [
        result_dict.get("num_env_steps_sampled_lifetime"),
        result_dict.get("num_env_steps_sampled"),
        result_dict.get("timesteps_total"),
    ]
    env_runners = result_dict.get("env_runners", {})
    if isinstance(env_runners, dict):
        candidates.extend(
            [
                env_runners.get("num_env_steps_sampled_lifetime"),
                env_runners.get("num_env_steps_sampled"),
            ]
        )
    for v in candidates:
        if isinstance(v, (int, float, np.integer, np.floating)) and v > 0:
            return int(v)
    return 0


def get_mean_return_and_len(result_dict):
    def first_num(*vals):
        for v in vals:
            if isinstance(v, (int, float, np.integer, np.floating)):
                return float(v)
        return float("nan")

    env_runners = result_dict.get("env_runners", {})
    if not isinstance(env_runners, dict):
        env_runners = {}
    mean_ret = first_num(
        result_dict.get("episode_return_mean"),
        result_dict.get("episode_reward_mean"),
        env_runners.get("episode_return_mean"),
        env_runners.get("episode_reward_mean"),
    )
    mean_len = first_num(
        result_dict.get("episode_len_mean"), env_runners.get("episode_len_mean")
    )
    return mean_ret, mean_len


def _clean_checkpoint_path(path_like) -> str:
    """Return a stable human-readable checkpoint path without object repr noise."""
    if path_like is None:
        return ""
    try:
        s = str(path_like).strip()
    except Exception:
        return ""
    if not s:
        return ""
    # Preserve cloud/object-store URIs. Resolve ordinary local paths.
    if "://" in s:
        return s
    try:
        return str(Path(s).expanduser().resolve())
    except Exception:
        return s


def _extract_checkpoint_path(save_result) -> str:
    """Extract a path from Ray's version-dependent checkpoint return objects."""
    if save_result is None:
        return ""

    if isinstance(save_result, (str, os.PathLike, Path)):
        return _clean_checkpoint_path(save_result)

    # New/old Ray result objects may expose .checkpoint.path through a
    # TrainingResult/Result wrapper. Prefer that over a top-level .path because
    # Result.path may denote the trial/result directory rather than the checkpoint.
    checkpoint_obj = getattr(save_result, "checkpoint", None)
    if checkpoint_obj is not None:
        cp_path = getattr(checkpoint_obj, "path", None)
        if cp_path:
            return _clean_checkpoint_path(cp_path)
        if isinstance(checkpoint_obj, (str, os.PathLike, Path)):
            return _clean_checkpoint_path(checkpoint_obj)
        if isinstance(checkpoint_obj, dict):
            for key in ("path", "checkpoint_path"):
                if checkpoint_obj.get(key):
                    return _clean_checkpoint_path(checkpoint_obj[key])

    direct_path = getattr(save_result, "path", None)
    if direct_path:
        return _clean_checkpoint_path(direct_path)

    if isinstance(save_result, dict):
        for key in ("checkpoint_path", "path"):
            if save_result.get(key):
                return _clean_checkpoint_path(save_result[key])
        if save_result.get("checkpoint") is not None:
            nested = _extract_checkpoint_path(save_result["checkpoint"])
            if nested:
                return nested

    return ""


def _find_checkpoint_dir(checkpoint_dir: Path) -> str:
    """Filesystem fallback: locate the newest RLlib checkpoint root below a directory."""
    checkpoint_dir = Path(checkpoint_dir)
    candidates = []
    if checkpoint_dir.exists():
        for marker in ("rllib_checkpoint.json", "algorithm_state.pkl", "metadata.json"):
            for marker_path in checkpoint_dir.rglob(marker):
                parent = marker_path.parent
                # A policy sub-checkpoint may also have rllib_checkpoint.json;
                # prefer roots that carry algorithm_state/class constructor state.
                score = 0
                if (parent / "algorithm_state.pkl").exists():
                    score += 10
                if (parent / "class_and_ctor_args.pkl").exists():
                    score += 5
                if (parent / "rllib_checkpoint.json").exists():
                    score += 2
                try:
                    mtime = parent.stat().st_mtime
                except Exception:
                    mtime = 0.0
                candidates.append((score, mtime, parent))
    if not candidates:
        return ""
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return _clean_checkpoint_path(candidates[0][2])


def save_checkpoint(algo, checkpoint_dir: Path):
    """Save an RLlib Algorithm and return only the actual checkpoint directory path."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Preferred modern API. Ray documents save_to_path() as returning a str path.
    try:
        save_result = algo.save_to_path(str(checkpoint_dir))
        clean = _extract_checkpoint_path(save_result)
        if clean:
            return clean
    except Exception:
        pass

    # Compatibility fallback for old-stack Algorithm.save(), which may return a
    # TrainingResult wrapper whose checkpoint lives at result.checkpoint.path.
    save_result = algo.save(str(checkpoint_dir))
    clean = _extract_checkpoint_path(save_result)
    if clean:
        return clean

    # Last-resort local filesystem discovery. Never store repr(TrainingResult)
    # in CSV because that is not a reusable checkpoint path.
    clean = _find_checkpoint_dir(checkpoint_dir)
    if clean:
        return clean

    raise RuntimeError(
        f"Checkpoint was saved but a clean checkpoint path could not be extracted under "
        f"{checkpoint_dir}"
    )


def write_dict_rows(path: Path, rows: Sequence[dict], fieldnames: Optional[Sequence[str]] = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    if fieldnames is None:
        seen = []
        for row in rows:
            for k in row.keys():
                if k not in seen:
                    seen.append(k)
        fieldnames = seen
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def append_dict_row(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open("r", newline="") as f:
            header = next(csv.reader(f), None)
        fieldnames = header or list(row.keys())
    else:
        fieldnames = list(row.keys())
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


class StreamingDictCsvWriter:
    """Write a large CSV incrementally without retaining all rows in RAM."""

    def __init__(self, path: Path, fieldnames: Sequence[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(fieldnames)
        self._fh = self.path.open("w", newline="")
        self._writer = csv.DictWriter(
            self._fh,
            fieldnames=self.fieldnames,
            extrasaction="ignore",
        )
        self._writer.writeheader()
        self._rows_since_flush = 0

    def write(self, row: dict):
        self._writer.writerow({k: row.get(k, "") for k in self.fieldnames})
        self._rows_since_flush += 1
        # Periodic flush gives useful partial output during long evaluations
        # without forcing a disk sync on every user/step row.
        if self._rows_since_flush >= 5000:
            self._fh.flush()
            self._rows_since_flush = 0

    def close(self):
        if getattr(self, "_fh", None) is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None


def _compute_action(algo, obs, policy_id=None):
    kwargs = {"explore": False}
    if policy_id is not None:
        kwargs["policy_id"] = policy_id
    out = algo.compute_single_action(obs, **kwargs)
    # Some RLlib versions return (action, state, info).
    if isinstance(out, tuple) and len(out) >= 1:
        out = out[0]
    return np.asarray(out, dtype=np.float32)


def _scalar_only(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, (int, float, np.integer, np.floating)):
            out[k] = float(v)
    return out


def evaluate_algorithm(algo, spec: ExperimentSpec, args, run_dir: Path):
    env_cfg = make_env_config(spec, args)
    env = CentralizedPPOEnv(env_cfg) if spec.method == "cent_ppo" else rllib_env_creator(env_cfg)

    step_rows = []
    bs_rows = []
    episode_rows = []
    user_qoe_acc = defaultdict(list)
    user_lat_acc = defaultdict(list)

    # Raw per-user/per-step evaluation log. This is streamed directly to disk so
    # a paper-scale run (e.g., 100 episodes x 200 steps x 80 users) does not keep
    # millions of Python dictionaries in memory.
    user_step_fields = [
        "run_id",
        "method",
        "variant",
        "scenario",
        "seed",
        "eval_seed",
        "eval_episode",
        "step",
        "user_id",
        "service_class",
        "mobility_regime",
        "task_type",
        "mandatory_offload",
        "demand_Mbps",
        "achievable_rate_Mbps",
        "served_rate_Mbps",
        "associated_bs",
        "channel_id",
        "latency_ms",
        "qoe",
        "deadline_candidate",
        "deadline_met",
    ]
    user_step_csv = StreamingDictCsvWriter(
        run_dir / "eval_user_steps.csv",
        user_step_fields,
    )

    all_latencies = []
    all_qoes = []
    total_deadline_hits = 0
    total_deadline_candidates = 0
    total_blocked = 0
    total_user_slots = 0
    total_rewards = []
    total_throughputs = []
    total_fairness = []
    total_energy = []
    total_walltime = []

    for ep in range(int(args.eval_episodes)):
        eval_seed = int(args.eval_seed_base) + ep
        obs, _ = env.reset(seed=eval_seed)
        ep_rewards = []
        ep_latencies = []
        ep_qoes = []
        ep_thr = []
        ep_hits = 0
        ep_candidates = 0
        ep_blocked = 0
        ep_user_slots = 0

        for step_idx in range(int(args.episode_len)):
            if spec.method == "cent_ppo":
                action = _compute_action(algo, obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                global_info = info
                multi_infos = env.last_multi_infos
                done = terminated or truncated
            else:
                actions = {
                    agent: _compute_action(algo, obs[agent], policy_id="shared_policy")
                    for agent in env.agents
                }
                next_obs, rewards, terminateds, truncateds, infos = env.step(actions)
                global_info = infos[env.agents[0]]
                multi_infos = infos
                reward = float(rewards[env.agents[0]])
                done = bool(terminateds.get("__all__", False) or truncateds.get("__all__", False))

            meta = {
                "run_id": spec.run_id,
                "method": spec.method,
                "variant": spec.variant,
                "scenario": spec.scenario,
                "seed": spec.seed,
                "eval_episode": ep,
                "step": step_idx,
            }
            step_row = dict(meta)
            step_row.update(_scalar_only(global_info))
            step_rows.append(step_row)

            for agent_id, agent_info in (multi_infos or {}).items():
                bs_row = dict(meta)
                try:
                    bs_row["bs_id"] = int(str(agent_id).split("_")[-1])
                except Exception:
                    bs_row["bs_id"] = str(agent_id)
                for k, v in agent_info.items():
                    if k.startswith("local_") or k in {"users_in_cov", "assoc_users"}:
                        if isinstance(v, (int, float, np.integer, np.floating)):
                            bs_row[k] = float(v)
                bs_rows.append(bs_row)

            lat_list = list(global_info.get("latency_per_user", []))
            qoe_list = list(global_info.get("qoe_per_user", []))
            for uid, val in enumerate(lat_list):
                user_lat_acc[uid].append(float(val))
            for uid, val in enumerate(qoe_list):
                user_qoe_acc[uid].append(float(val))

            ep_latencies.extend(float(x) for x in lat_list)
            ep_qoes.extend(float(x) for x in qoe_list)
            all_latencies.extend(float(x) for x in lat_list)
            all_qoes.extend(float(x) for x in qoe_list)

            # Save the exact user-level samples that feed the pooled latency/QoE
            # statistics. For Cent-PPO, metrics live in the wrapped base env.
            sim_env = env.base_env if isinstance(env, CentralizedPPOEnv) else env
            sim_users = list(getattr(sim_env, "users", []))
            for pos, u in enumerate(sim_users):
                latency_ms = (
                    float(lat_list[pos]) if pos < len(lat_list) else float("nan")
                )
                qoe = float(qoe_list[pos]) if pos < len(qoe_list) else float("nan")
                demand_mbps = float(getattr(u, "demand", 0.0))
                achievable_mbps = float(getattr(u, "data_rate", 0.0))
                served_mbps = float(min(achievable_mbps, demand_mbps))

                associated_bs = -1
                channel_id = -1
                channels = list(getattr(u, "channel", []) or [])
                if channels:
                    ch0 = channels[0]
                    channel_id = int(getattr(ch0, "id", -1))
                    bs0 = getattr(ch0, "base_station", None)
                    if bs0 is not None:
                        associated_bs = int(getattr(bs0, "id", -1))

                deadline_candidate = int(demand_mbps > 0.0)
                deadline_met = int(
                    deadline_candidate
                    and np.isfinite(latency_ms)
                    and latency_ms <= float(args.deadline_ms)
                )

                user_step_csv.write(
                    {
                        **meta,
                        "eval_seed": eval_seed,
                        "user_id": int(getattr(u, "id", pos)),
                        "service_class": str(
                            getattr(u, "service_class", "unknown")
                        ),
                        "mobility_regime": str(
                            getattr(u, "mobility_regime", spec.scenario)
                        ),
                        "task_type": str(getattr(u, "task_type", "unknown")),
                        "mandatory_offload": int(
                            bool(getattr(u, "task_offload_mandatory", False))
                        ),
                        "demand_Mbps": demand_mbps,
                        "achievable_rate_Mbps": achievable_mbps,
                        "served_rate_Mbps": served_mbps,
                        "associated_bs": associated_bs,
                        "channel_id": channel_id,
                        "latency_ms": latency_ms,
                        "qoe": qoe,
                        "deadline_candidate": deadline_candidate,
                        "deadline_met": deadline_met,
                    }
                )

            hits = int(round(float(global_info.get("deadline_hits", 0))))
            candidates = int(round(float(global_info.get("deadline_candidates", 0))))
            blocked = int(round(float(global_info.get("blocked_users_count", 0))))
            num_users = int(round(float(global_info.get("num_users", args.num_users))))
            ep_hits += hits
            ep_candidates += candidates
            ep_blocked += blocked
            ep_user_slots += num_users
            total_deadline_hits += hits
            total_deadline_candidates += candidates
            total_blocked += blocked
            total_user_slots += num_users

            thr = float(global_info.get("total_throughput_Mbps", 0.0))
            fair = float(global_info.get("jain_fairness", 0.0))
            energy = float(global_info.get("step_energy_J", 0.0))
            wall = float(global_info.get("env_step_walltime_ms", 0.0))
            ep_thr.append(thr)
            total_throughputs.append(thr)
            total_fairness.append(fair)
            total_energy.append(energy)
            total_walltime.append(wall)
            ep_rewards.append(float(reward))
            total_rewards.append(float(reward))

            obs = next_obs
            if done:
                break

        episode_rows.append(
            {
                "run_id": spec.run_id,
                "method": spec.method,
                "variant": spec.variant,
                "scenario": spec.scenario,
                "seed": spec.seed,
                "eval_episode": ep,
                "episode_return": float(np.sum(ep_rewards)) if ep_rewards else 0.0,
                "total_throughput_Mbps": float(np.mean(ep_thr)) if ep_thr else 0.0,
                "avg_latency_ms": float(np.mean(ep_latencies)) if ep_latencies else 0.0,
                "p95_latency_ms": float(np.percentile(ep_latencies, 95)) if ep_latencies else 0.0,
                "deadline_satisfaction": (ep_hits / ep_candidates) if ep_candidates else 0.0,
                "deadline_satisfaction_pct": 100.0 * ep_hits / ep_candidates if ep_candidates else 0.0,
                "blocking_rate": (ep_blocked / ep_user_slots) if ep_user_slots else 0.0,
                "blocking_pct": 100.0 * ep_blocked / ep_user_slots if ep_user_slots else 0.0,
                "avg_qoe": float(np.mean(ep_qoes)) if ep_qoes else 0.0,
            }
        )

    user_step_csv.close()
    env.close()

    user_rows = []
    all_uids = sorted(set(user_qoe_acc) | set(user_lat_acc))
    for uid in all_uids:
        user_rows.append(
            {
                "run_id": spec.run_id,
                "method": spec.method,
                "variant": spec.variant,
                "scenario": spec.scenario,
                "seed": spec.seed,
                "user_id": uid,
                "mean_qoe": float(np.mean(user_qoe_acc.get(uid, [0.0]))),
                "mean_latency_ms": float(np.mean(user_lat_acc.get(uid, [0.0]))),
            }
        )

    write_dict_rows(run_dir / "eval_steps.csv", step_rows)
    write_dict_rows(run_dir / "eval_bs_steps.csv", bs_rows)
    write_dict_rows(run_dir / "eval_episodes.csv", episode_rows)
    write_dict_rows(run_dir / "eval_user_metrics.csv", user_rows)

    summary = {
        "run_id": spec.run_id,
        "method": spec.method,
        "variant": spec.variant,
        "scenario": spec.scenario,
        "seed": spec.seed,
        "num_eval_episodes": int(args.eval_episodes),
        "throughput_Mbps": float(np.mean(total_throughputs)) if total_throughputs else 0.0,
        "mean_latency_ms": float(np.mean(all_latencies)) if all_latencies else 0.0,
        "p95_latency_ms": float(np.percentile(all_latencies, 95)) if all_latencies else 0.0,
        "deadline_satisfaction": (total_deadline_hits / total_deadline_candidates) if total_deadline_candidates else 0.0,
        "deadline_satisfaction_pct": 100.0 * total_deadline_hits / total_deadline_candidates if total_deadline_candidates else 0.0,
        "blocking_rate": (total_blocked / total_user_slots) if total_user_slots else 0.0,
        "blocking_pct": 100.0 * total_blocked / total_user_slots if total_user_slots else 0.0,
        "avg_qoe": float(np.mean(all_qoes)) if all_qoes else 0.0,
        "fairness": float(np.mean(total_fairness)) if total_fairness else 0.0,
        "energy_J_per_step": float(np.mean(total_energy)) if total_energy else 0.0,
        "walltime_ms_per_step": float(np.mean(total_walltime)) if total_walltime else 0.0,
        "mean_episode_return": float(np.mean([r["episode_return"] for r in episode_rows])) if episode_rows else 0.0,
        "eval_user_steps_csv": str((run_dir / "eval_user_steps.csv").resolve()),
    }
    write_dict_rows(run_dir / "run_summary.csv", [summary])
    return summary


def summarize_training_tail(csv_path: Path, tail_episodes: int = 500):
    """Summarize the final training episodes using the archived-paper convention."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return {}
    with csv_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    rows = rows[-int(tail_episodes):]
    metrics = [
        "episode_return",
        "global_reward",
        "total_throughput_Mbps",
        "avg_latency_ms",
        "p95_latency_ms",
        "deadline_satisfaction",
        "blocking_rate",
        "avg_qoe",
        "jain_fairness",
        "step_energy_J",
        "env_step_walltime_ms",
        "svc_safety_deadline_satisfaction",
        "svc_safety_avg_qoe",
        "svc_infotainment_deadline_satisfaction",
        "svc_infotainment_avg_qoe",
    ]
    out = {"tail_episodes": len(rows)}
    for metric in metrics:
        vals = []
        for row in rows:
            try:
                vals.append(float(row[metric]))
            except Exception:
                pass
        if vals:
            out[metric] = float(np.mean(vals))
            out[f"{metric}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    if "deadline_satisfaction" in out:
        out["deadline_satisfaction_pct"] = 100.0 * out["deadline_satisfaction"]
    if "blocking_rate" in out:
        out["blocking_pct"] = 100.0 * out["blocking_rate"]
    return out


def train_one(spec: ExperimentSpec, args, output_dir: Path):
    run_dir = output_dir / "runs" / spec.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "run_summary.csv"
    if args.skip_existing and summary_path.exists() and summary_path.stat().st_size > 0:
        print(f"[skip] {spec.run_id}")
        with summary_path.open("r", newline="") as f:
            return next(csv.DictReader(f))

    set_run_environment(spec, run_dir)
    env_cfg = make_env_config(spec, args)
    with (run_dir / "config.json").open("w") as f:
        json.dump({"spec": asdict(spec), "env_config": env_cfg, "args": vars(args)}, f, indent=2)

    algo = build_algorithm(spec, args)
    iter_rows = []
    total_steps = 0
    iteration = 0
    last_checkpoint = ""
    start_train = time.time()

    try:
        while total_steps < int(args.train_env_steps):
            iter_start = time.time()
            prev_total_steps = total_steps
            result = algo.train()
            iter_walltime_s = time.time() - iter_start

            iteration += 1
            total_steps_new = get_total_env_steps(result)
            if total_steps_new > 0:
                total_steps = total_steps_new
            else:
                # Conservative fallback: complete-episode runner with fixed episode length.
                total_steps += int(args.episode_len)

            sampled_this_iter = max(0, total_steps - prev_total_steps)
            mean_ret, mean_len = get_mean_return_and_len(result)
            iter_row = {
                "run_id": spec.run_id,
                "method": spec.method,
                "variant": spec.variant,
                "scenario": spec.scenario,
                "seed": spec.seed,
                "iteration": iteration,
                "env_steps": total_steps,
                "env_steps_this_iteration": sampled_this_iter,
                "episode_return_mean": mean_ret,
                "episode_len_mean": mean_len,
                "iteration_walltime_s": iter_walltime_s,
                "training_walltime_s": time.time() - start_train,
            }
            iter_rows.append(iter_row)
            append_dict_row(run_dir / "train_iterations.csv", iter_row)

            step_rate = (
                sampled_this_iter / iter_walltime_s
                if iter_walltime_s > 0 and sampled_this_iter > 0
                else float("nan")
            )
            print(
                f"[{spec.run_id}] iter={iteration:04d} env_steps={total_steps} "
                f"(+{sampled_this_iter}) return={mean_ret:.3f} len={mean_len:.1f} "
                f"iter_s={iter_walltime_s:.1f} env_steps/s={step_rate:.2f}"
            )
            if int(args.checkpoint_every) > 0 and iteration % int(args.checkpoint_every) == 0:
                last_checkpoint = save_checkpoint(algo, run_dir / "checkpoints")

        last_checkpoint = save_checkpoint(algo, run_dir / "checkpoints" / "final")
        summary = evaluate_algorithm(algo, spec, args, run_dir)
        train_tail = summarize_training_tail(run_dir / "train_episodes.csv", tail_episodes=500)
        if train_tail:
            write_dict_rows(run_dir / "train_tail_summary.csv", [train_tail])
            for k, v in train_tail.items():
                summary[f"train_tail_{k}"] = v
        summary["code_version"] = CODE_VERSION
        summary["ray_version"] = getattr(ray, "__version__", "")
        summary["torch_version"] = getattr(torch, "__version__", "")
        summary["final_checkpoint"] = last_checkpoint
        summary["training_iterations"] = iteration
        summary["training_env_steps"] = total_steps
        summary["training_walltime_s"] = time.time() - start_train
        write_dict_rows(run_dir / "run_summary.csv", [summary])
        return summary
    finally:
        try:
            algo.stop()
        except Exception:
            pass


def _to_float(row, key):
    try:
        return float(row[key])
    except Exception:
        return float("nan")


def t_critical_975(df: int) -> float:
    """Two-sided 95% Student-t critical value without requiring SciPy."""
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    }
    if df <= 0:
        return float("nan")
    return table.get(int(df), 1.96)


def aggregate_group(rows: Sequence[dict], group_name: str, group_value: str):
    metrics = [
        "throughput_Mbps",
        "mean_latency_ms",
        "p95_latency_ms",
        "deadline_satisfaction_pct",
        "blocking_pct",
        "avg_qoe",
        "fairness",
        "energy_J_per_step",
        "walltime_ms_per_step",
        "mean_episode_return",
    ]
    out = {group_name: group_value, "n_seeds": len(rows)}
    for m in metrics:
        vals = np.asarray([_to_float(r, m) for r in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            mean = sd = sem = ci = float("nan")
        else:
            mean = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
            sem = sd / math.sqrt(vals.size) if vals.size > 0 else float("nan")
            ci = t_critical_975(int(vals.size - 1)) * sem if vals.size > 1 else 0.0
        out[f"{m}_mean"] = mean
        out[f"{m}_sd"] = sd
        out[f"{m}_sem"] = sem
        out[f"{m}_ci95"] = ci
    return out


def _summary_identity(row: dict):
    """Stable identity for one trained/evaluated run summary."""
    return (
        str(row.get("method", "")),
        str(row.get("variant", "full")),
        str(row.get("scenario", "urban")),
        str(row.get("seed", "")),
    )


def merge_summaries(*groups: Sequence[dict]):
    """Merge run summaries and de-duplicate by method/variant/scenario/seed.

    Later groups override earlier groups.  This lets newly completed MAPPO or
    Cent-PPO runs replace an older copy while preserving previously completed
    Ours runs loaded from another result directory.
    """
    merged = {}
    order = []
    for group in groups:
        for raw in group or []:
            row = dict(raw)
            key = _summary_identity(row)
            if key not in merged:
                order.append(key)
            merged[key] = row
    return [merged[k] for k in order]


def load_existing_summaries(roots: Sequence[str]):
    """Recursively load run_summary.csv files from previous experiment roots.

    Each argument may be a result directory, a run directory, or a specific
    run_summary.csv file.  Aggregate copies under summaries/ are harmless
    because merge_summaries() de-duplicates them by run identity.
    """
    rows = []
    seen_files = set()
    for raw_root in roots or []:
        root = Path(raw_root).expanduser()
        if not root.exists():
            raise FileNotFoundError(f"Existing-results path does not exist: {root}")

        if root.is_file():
            candidates = [root]
        else:
            candidates = sorted(root.rglob("run_summary.csv"))

        if not candidates:
            raise FileNotFoundError(
                f"No run_summary.csv found under existing-results path: {root}"
            )

        for csv_path in candidates:
            resolved = str(csv_path.resolve())
            if resolved in seen_files:
                continue
            seen_files.add(resolved)
            try:
                with csv_path.open("r", newline="") as f:
                    for row in csv.DictReader(f):
                        if not row:
                            continue
                        if row.get("method") not in {"ours", "mappo", "cent_ppo"}:
                            continue
                        rows.append(dict(row))
            except Exception as exc:
                raise RuntimeError(f"Failed reading existing summary {csv_path}") from exc

    return merge_summaries(rows)


def validate_primary_existing_ours(rows: Sequence[dict], seeds: Sequence[int], args):
    """Reject accidental use of a short smoke-test Ours run in the final table."""
    wanted = {str(int(s)) for s in seeds}
    ours_rows = [
        r for r in rows
        if str(r.get("method")) == "ours"
        and str(r.get("variant", "full")) == "full"
        and str(r.get("scenario", "urban")) == "urban"
        and str(r.get("seed", "")) in wanted
    ]

    if not ours_rows:
        print(
            "WARNING: no existing Ours/full/urban summaries were loaded. "
            "The baselines will still run, but the final primary table will not "
            "contain Ours until --existing-results is supplied during aggregation."
        )
        return

    by_seed = {str(r.get("seed")): r for r in ours_rows}
    missing = sorted(wanted - set(by_seed))
    if missing:
        print(f"WARNING: existing Ours summaries are missing seeds: {', '.join(missing)}")

    if getattr(args, "allow_existing_step_mismatch", False):
        return

    requested = int(args.train_env_steps)
    too_short = []
    for seed, row in by_seed.items():
        try:
            trained = int(float(row.get("training_env_steps", "nan")))
        except Exception:
            continue
        # Complete-episode sampling can overshoot the requested budget; being
        # larger is fine.  Being smaller means this is probably a smoke test.
        if trained < requested:
            too_short.append((seed, trained))
    if too_short:
        details = ", ".join(f"seed {s}: {n} steps" for s, n in too_short)
        raise ValueError(
            "Existing Ours results have a smaller training budget than the "
            f"requested {requested} steps ({details}). Do not mix smoke-test and "
            "final baselines. Point --existing-results at the final Ours runs, "
            "or use --allow-existing-step-mismatch only if this is intentional."
        )


def write_primary_status(rows: Sequence[dict], output_dir: Path, seeds: Sequence[int]):
    """Write a compact completeness check for the reviewer-facing comparison."""
    summary_dir = output_dir / "summaries"
    wanted = [int(s) for s in seeds]
    status = []
    for method in ["ours", "mappo", "cent_ppo"]:
        selected = [
            r for r in rows
            if r.get("method") == method
            and r.get("scenario") == "urban"
            and r.get("variant") == "full"
        ]
        present = set()
        for r in selected:
            try:
                present.add(int(float(r.get("seed"))))
            except Exception:
                pass
        missing = [s for s in wanted if s not in present]
        status.append(
            {
                "method": method,
                "required_seeds": " ".join(map(str, wanted)),
                "present_seeds": " ".join(map(str, sorted(present & set(wanted)))),
                "missing_seeds": " ".join(map(str, missing)),
                "complete": int(len(missing) == 0),
            }
        )
    write_dict_rows(summary_dir / "primary_comparison_status.csv", status)


def build_paper_tables(all_summaries: Sequence[dict], output_dir: Path, seeds=None):
    summary_dir = output_dir / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    rows = merge_summaries(all_summaries)
    write_dict_rows(summary_dir / "run_summary.csv", rows)

    # Primary reviewer-facing table: Ours vs centralized-critic MAPPO vs Cent-PPO.
    # Only full/urban runs enter this table, and uncertainty is across independent
    # training seeds (not within-run episodes).
    table1 = []
    for method in ["ours", "mappo", "cent_ppo"]:
        selected = [
            r for r in rows
            if r.get("method") == method
            and r.get("scenario") == "urban"
            and r.get("variant") == "full"
        ]
        if seeds is not None:
            wanted = {str(int(s)) for s in seeds}
            selected = [r for r in selected if str(r.get("seed", "")) in wanted]
        if selected:
            table1.append(aggregate_group(selected, "method", method))
    write_dict_rows(summary_dir / "table_primary_comparison.csv", table1)
    # Keep the manuscript-compatible historical filename as an alias.
    write_dict_rows(summary_dir / "table_mappo_seed.csv", table1)

    # Table: controlled neighbor-information ablation (when those runs exist).
    table2 = []
    for variant in ["no_neighbor", "full"]:
        selected = [
            r for r in rows
            if r.get("method") == "ours"
            and r.get("scenario") == "urban"
            and r.get("variant") == variant
        ]
        if selected:
            table2.append(aggregate_group(selected, "variant", variant))
    write_dict_rows(summary_dir / "table_neighbor_ablation.csv", table2)

    # Table: proposed method by mobility regime (when those runs exist).
    table3 = []
    for scenario in ["urban", "highway", "mixed"]:
        selected = [
            r for r in rows
            if r.get("method") == "ours"
            and r.get("variant") == "full"
            and r.get("scenario") == scenario
        ]
        if selected:
            table3.append(aggregate_group(selected, "scenario", scenario))
    write_dict_rows(summary_dir / "table_mobility_breakdown.csv", table3)

    if seeds is not None:
        write_primary_status(rows, output_dir, seeds)


def primary_specs(seeds: Sequence[int], methods: Sequence[str]):
    """Runs needed after Ours is already complete for the primary comparison."""
    allowed = {"ours", "mappo", "cent_ppo"}
    methods = [str(m).lower() for m in methods]
    bad = [m for m in methods if m not in allowed]
    if bad:
        raise ValueError(f"Unsupported primary method(s): {bad}")

    specs = []
    for seed in seeds:
        for method in methods:
            specs.append(ExperimentSpec(method, "urban", int(seed), False))

    uniq = []
    seen = set()
    for s in specs:
        if s.run_id not in seen:
            uniq.append(s)
            seen.add(s.run_id)
    return uniq


def targeted_specs(seeds: Sequence[int]):
    """Original broader targeted suite retained for later experiments."""
    specs = []
    for seed in seeds:
        specs.extend(
            [
                ExperimentSpec("ours", "urban", seed, False),
                ExperimentSpec("mappo", "urban", seed, False),
                ExperimentSpec("cent_ppo", "urban", seed, False),
                ExperimentSpec("ours", "urban", seed, True),
                ExperimentSpec("ours", "highway", seed, False),
                ExperimentSpec("ours", "mixed", seed, False),
            ]
        )
    uniq = []
    seen = set()
    for s in specs:
        if s.run_id not in seen:
            uniq.append(s)
            seen.add(s.run_id)
    return uniq

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Reviewer-facing PPO/MAPPO experiments for clustered IoV-VEC. "
            "Default primary mode runs MAPPO and Cent-PPO only, assuming Ours "
            "has already been completed and is supplied with --existing-results."
        )
    )
    p.add_argument(
        "--suite",
        choices=["primary", "targeted", "single", "aggregate"],
        default="primary",
        help=(
            "primary: run the requested primary baselines; targeted: original "
            "broader suite; single: one run; aggregate: do not train, only merge "
            "existing summaries and rebuild comparison tables."
        ),
    )
    p.add_argument("--output-dir", default="primary_algorithm_comparison")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument(
        "--methods",
        nargs="+",
        choices=["ours", "mappo", "cent_ppo"],
        default=["mappo", "cent_ppo"],
        help=(
            "Methods trained by --suite primary. Defaults to MAPPO and Cent-PPO "
            "so an already-completed Ours experiment is not repeated."
        ),
    )
    p.add_argument(
        "--existing-results",
        nargs="*",
        default=[],
        help=(
            "Existing result root(s) or run_summary.csv file(s) to merge. The "
            "loader searches directories recursively. Point this at the final "
            "Ours seed runs when using the primary suite."
        ),
    )
    p.add_argument(
        "--allow-existing-step-mismatch",
        action="store_true",
        help="Allow an existing Ours result with fewer steps than --train-env-steps.",
    )
    p.add_argument("--train-env-steps", type=int, default=500_000)
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--eval-seed-base", type=int, default=900_000)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--skip-existing", action="store_true")

    # Paper-default system configuration.
    p.add_argument("--num-bs", type=int, default=4)
    p.add_argument("--num-users", type=int, default=80)
    p.add_argument("--channels-per-carrier", type=int, default=50)
    p.add_argument("--area-size", type=float, default=2000.0)
    p.add_argument("--episode-len", type=int, default=200)
    p.add_argument("--deadline-ms", type=float, default=100.0)

    # PPO configuration from the manuscript/notebook.
    p.add_argument("--train-batch-size", type=int, default=1024)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--num-sgd-iter", type=int, default=10)
    p.add_argument("--num-env-runners", type=int, default=1)
    p.add_argument(
        "--rollout-fragment-length",
        type=int,
        default=0,
        help=(
            "Sampling fragment length. 0 means use --episode-len. For the tested "
            "2-EnvRunner, batch-1024 setup, use 512 explicitly."
        ),
    )
    p.add_argument(
        "--sample-timeout-s",
        type=float,
        default=300.0,
        help=(
            "Maximum synchronous wait for a remote RLlib sampler. The heavier "
            "radio/MEC environment can exceed Ray's shorter default timeout."
        ),
    )
    p.add_argument("--num-gpus", type=float, default=0.0)

    # Single-run convenience mode.
    p.add_argument("--method", choices=["ours", "mappo", "cent_ppo"], default="ours")
    p.add_argument("--scenario", choices=["urban", "highway", "mixed"], default="urban")
    p.add_argument("--mask-neighbor-summaries", action="store_true")
    return p.parse_args()


def _print_output_paths(output_dir: Path):
    print("\nFinished. Key CSV outputs:")
    print(f"  {output_dir / 'summaries' / 'run_summary.csv'}")
    print(f"  {output_dir / 'summaries' / 'table_primary_comparison.csv'}")
    print(f"  {output_dir / 'summaries' / 'table_mappo_seed.csv'}")
    print(f"  {output_dir / 'summaries' / 'primary_comparison_status.csv'}")


def main():
    args = parse_args()

    if int(args.episode_len) <= 0:
        raise ValueError("--episode-len must be > 0")
    if int(args.train_batch_size) <= 0:
        raise ValueError("--train-batch-size must be > 0")
    if int(args.minibatch_size) <= 0:
        raise ValueError("--minibatch-size must be > 0")
    if int(args.minibatch_size) > int(args.train_batch_size):
        raise ValueError("--minibatch-size cannot exceed --train-batch-size")
    if int(args.num_env_runners) < 0:
        raise ValueError("--num-env-runners must be >= 0")

    effective_fragment = (
        int(args.rollout_fragment_length)
        if int(args.rollout_fragment_length) > 0
        else int(args.episode_len)
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = load_existing_summaries(args.existing_results)
    validate_primary_existing_ours(existing, args.seeds, args)

    # Aggregation-only mode is intentionally Ray-free.  Use it after running
    # MAPPO and Cent-PPO concurrently on separate GPUs/output directories.
    if args.suite == "aggregate":
        build_paper_tables(existing, output_dir, seeds=args.seeds)
        _print_output_paths(output_dir)
        return

    print(
        "RLlib sampling configuration: "
        f"env_runners={args.num_env_runners}, "
        f"episode_len={args.episode_len}, "
        f"fragment_len={effective_fragment}, "
        f"sample_timeout_s={args.sample_timeout_s}, "
        f"train_batch_size={args.train_batch_size}, "
        f"num_gpus={args.num_gpus}"
    )

    if args.suite == "primary":
        specs = primary_specs(args.seeds, args.methods)
        print(
            "Primary suite will train only: "
            + ", ".join(f"{s.method}/seed{s.seed}" for s in specs)
        )
        if any(s.method == "ours" for s in specs):
            print("NOTE: Ours was explicitly requested in --methods and will be retrained.")
    elif args.suite == "targeted":
        specs = targeted_specs(args.seeds)
    else:
        specs = [
            ExperimentSpec(
                args.method,
                args.scenario,
                int(args.seeds[0]),
                bool(args.mask_neighbor_summaries),
            )
        ]

    os.environ.setdefault("RAY_DEDUP_LOGS", "1")
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)

    register_env(MULTI_ENV_ID, rllib_env_creator)
    register_env(CENTRAL_ENV_ID, central_env_creator)
    try:
        ModelCatalog.register_custom_model(MODEL_ID, SplitActorCentralCriticModel)
    except Exception:
        pass

    new_summaries = []
    try:
        for spec in specs:
            summary = train_one(spec, args, output_dir)
            if isinstance(summary, dict):
                new_summaries.append(summary)
    finally:
        ray.shutdown()

    combined = merge_summaries(existing, new_summaries)
    build_paper_tables(combined, output_dir, seeds=args.seeds)
    _print_output_paths(output_dir)


if __name__ == "__main__":
    main()
