import math
import random
import time
from collections import defaultdict
from types import SimpleNamespace

import gymnasium as gym
from gymnasium.utils import seeding
import matplotlib.pyplot as plt
import numpy as np

# RLlib MultiAgentEnv (required for training)
try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
except Exception as e:  # pragma: no cover
    MultiAgentEnv = object  # allows import for docs/tools; training requires ray


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
        se_layer = spectral_efficiency_from_sinr(per_layer_snr_dB, gap_db, max_se)
        corr_eff = max(0.5, 1.0 - 0.07 * (L - 1))  # crude correlation penalty
        sum_se = L * se_layer * corr_eff
        if sum_se > best_sum_se:
            best_sum_se = sum_se
            best_L = L

    return best_L, float(best_sum_se)


# 19-D BS observation layout (0-1 normalized)
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
)


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

        p_los = min(18.0 / d2d, 1.0) * (1.0 - np.exp(-d2d / 63.0)) + np.exp(-d2d / 63.0)
        pl_los = self._pl_uma_los(d3d, f_ghz, h_ut=user_height)
        pl_nlos = self._pl_uma_nlos(d3d, f_ghz, h_ut=user_height)

        return float(p_los * pl_los + (1.0 - p_los) * pl_nlos)

    def _eirp_dbm(self, transmit_power_mW: float) -> float:
        g = BEAM_GAIN_DB[self.type_bs]
        tx_dbm = 10.0 * np.log10(max(transmit_power_mW, 1e-15))
        return tx_dbm + g["tx_main"] + g["rx"]

    def update_coverage_area_from_power(self, total_transmit_power_mW, frequency_hz):
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

        sens_dBm = rx_sensitivity_dBm(bw_hz, nf_db=nf_db, snr_req_db=self.SNR_REQ_DB[self.type_bs])
        eirp_plus_grx_dbm = self._eirp_dbm(p_ch)
        path_loss_budget_dB = eirp_plus_grx_dbm - sens_dBm - self.COV_MARGIN_DB[self.type_bs]

        radius_m = self.find_distance_for_path_loss(path_loss_budget_dB, frequency_hz)
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

        # mobility
        self.speed = float(np.linalg.norm(self.velocity))
        self.waypoint = None
        self.pause_time = 0
        self.dir_axis = None
        self.dir_sign = 1
        self.next_intersection = None

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

            L, total_se = mimo_rank_and_total_se(SINR_dB, max_rank, gap_db=SNR_GAP_DB, max_se=MAX_SE)
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

    def calculate_latency_ms(self, num_users_on_channel: float = None):
        """Very simplified base latency model (communication-centric)."""
        if not self.channel or self.data_rate <= 0:
            return 1000.0

        avg_d = np.mean([np.linalg.norm(self.location - ch.base_station.location) for ch in self.channel])
        prop_delay_ms = (avg_d / 3e8) * 1e3
        proc_delay = 1.0
        sched_delay = 1.0

        if num_users_on_channel is None:
            num_users_on_channel = np.mean([len(ch.users) for ch in self.channel]) if self.channel else 1.0

        queue_delay = 1.0 + num_users_on_channel / (self.data_rate + 1e-6)

        return float(np.clip(prop_delay_ms + proc_delay + sched_delay + queue_delay, 0.0, 1000.0))


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
        mobility_model="manhattan",
        seed=42,
        step_duration_s=1.0,
        deadline_ms=100.0,
        safety_traffic_ratio=0.3,
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
        self.mobility_model = mobility_model

        # time / QoS
        self.step_duration_s = float(step_duration_s)
        self.deadline_ms = float(deadline_ms)
        self.safety_traffic_ratio = float(np.clip(safety_traffic_ratio, 0.0, 1.0))

        # radio parameters
        self.ma_transmission_power = 40000.0  # mW (40 W)
        self.max_ma_channels = int(num_channels_per_carrier)
        self.macro_carrier_frequencies = [3.5e9] if MACRO_REUSE_ONE else [3.4e9, 3.5e9, 3.6e9, 3.7e9]
        self.macro_channel_bw = 180e3  # Hz

        # MEC / compute parameters
        self.mec_cpu_capacity_cycles = 5e9
        self.cycles_per_Mbit = 5e7
        self.cpu_kappa = 1e-27
        self.max_user_demand_Mbps = 10.0

        # reward weights
        self.reward_weights = SimpleNamespace(
            w_lat=0.2,
            w_thr=0.3,
            w_dead=0.2,
            w_qoe=0.3,
            w_eng=0.05,
            w_block=0.2,
            w_fair=0.1,
        )

        # objects
        self.users = []
        self.base_stations = []

        # BS placement
        if bs_loc is None:
            bs_loc = self._default_bs_locations(self.num_base_stations, self.area_size)
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
                self.macro_channels.append(Channel(ch_id, f, self.macro_channel_bw, noise_figure_db=7.0))
                ch_id += 1

        # spawn users
        rng = self.np_random
        for u_id in range(self.num_users):
            if self.mobility_model == "manhattan":
                loc, vel = self._spawn_on_grid()
            else:
                loc = rng.uniform(0, self.area_size, 2)
                vel = rng.uniform(-5.0, 5.0, 2)

            u = User(u_id, loc, vel)
            svc = "safety" if rng.random() < self.safety_traffic_ratio else "infotainment"
            u.service_class = svc
            u.calculate_demand_from_rng(rng)
            self.users.append(u)

            if self.mobility_model == "manhattan":
                self._init_user_manhattan(u)

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

        # 19-D observation
        obs_dim = len(vars(OBS_IDX))
        self.observation_spaces = {
            a: gym.spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
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
        self.energy_normalization_J = (
            self.num_base_stations * self.ma_transmission_power * 1e-3 * self.step_duration_s
        )
        self.throughput_normalization_Mbps = self.num_users * self.max_user_demand_Mbps

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
                self.bs_carrier_frequency[bs.id] = self.macro_carrier_frequencies[0]
            else:
                self.bs_carrier_frequency[bs.id] = self.macro_carrier_frequencies[idx % len(self.macro_carrier_frequencies)]
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
            candidates = self.grid_xs[self.grid_xs > u.location[0]] if u.dir_sign > 0 else self.grid_xs[self.grid_xs < u.location[0]]
            coord = candidates.min() if (candidates.size and u.dir_sign > 0) else (candidates.max() if candidates.size else (self.area_size if u.dir_sign > 0 else 0.0))
            u.next_intersection = np.array([coord, self._snap_to_grid(u.location)[1]], dtype=float)
        else:
            candidates = self.grid_ys[self.grid_ys > u.location[1]] if u.dir_sign > 0 else self.grid_ys[self.grid_ys < u.location[1]]
            coord = candidates.min() if (candidates.size and u.dir_sign > 0) else (candidates.max() if candidates.size else (self.area_size if u.dir_sign > 0 else 0.0))
            u.next_intersection = np.array([self._snap_to_grid(u.location)[0], coord], dtype=float)

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
        if np.linalg.norm(u.location - u.next_intersection) <= 1e-6 or step == dist:
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
                u.next_intersection = np.array([x_next, self._snap_to_grid(u.location)[1]], dtype=float)
                u.velocity = np.array([u.dir_sign * u.speed, 0.0], dtype=float)
            else:
                if u.dir_sign > 0:
                    candidates = self.grid_ys[self.grid_ys > u.location[1]]
                    y_next = candidates.min() if candidates.size > 0 else self.area_size
                else:
                    candidates = self.grid_ys[self.grid_ys < u.location[1]]
                    y_next = candidates.max() if candidates.size > 0 else 0.0
                u.next_intersection = np.array([self._snap_to_grid(u.location)[0], y_next], dtype=float)
                u.velocity = np.array([0.0, u.dir_sign * u.speed], dtype=float)

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
                    info["ma_base_station_location"].append(best_bs.location.copy())

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
        _, total_se = mimo_rank_and_total_se(SINR_dB, MIMO_MAX_RANK[bs.type_bs], gap_db=SNR_GAP_DB, max_se=MAX_SE)
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
        co_bs = [b for b in self.base_stations if b is not exclude_bs and self.bs_carrier_frequency[b.id] == f]
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
                uu for uu in self.users
                if np.linalg.norm(uu.location - bi.location) <= bi.coverage_area
            ]
            util = min(1.0, len(users_in_cov_bi) / max(1, len(bi.assigned_channels)))

            I += util * p_i * g_tx_i * g_rx_int / (10.0 ** (PL_i / 10.0))

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
        p_floor = np.zeros(K, dtype=float) if p_floor_list is None else np.asarray(p_floor_list, dtype=float)

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
                action_dict[agent] = np.array([0.5, 0.5, 0.5, 0.5], dtype=float)
            action_dict[agent] = np.clip(np.array(action_dict[agent], dtype=float), 0.0, 1.0)

        # --------- traffic demand update ----------
        for u in self.users:
            if self.np_random.random() < 0.05:
                u.calculate_demand_from_rng(self.np_random)

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
            power_frac, channel_frac, offload_frac, cpu_frac = action_dict[agent]

            # power allocation
            bs.transmit_power = float(np.clip(power_frac, 0.0, 1.0)) * self.ma_transmission_power

            # channel allocation
            max_cap = int(self.max_ma_channels)
            req_ch = int(np.rint(float(np.clip(channel_frac, 0.0, 1.0)) * max_cap))

            if req_ch == 0 and bs.transmit_power > 1e-9:
                req_ch = 1

            freq = self.bs_carrier_frequency[bs.id]
            avail = [c for c in self.macro_channels if c.frequency == freq and c.base_station is None]
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
                h_linear = (10 ** (g["tx_main"] / 10.0) * 10 ** (g["rx"] / 10.0)) / (10.0 ** (PL_dB / 10.0))

                noise_mW = ch.calculate_noise_power() * 1e3
                I_mW = self._estimate_interference_mW(u, f, bs)

                prio = float(np.clip(0.5 + (u.demand / self.max_user_demand_Mbps), 0.5, 2.0))
                N_eff = (noise_mW + I_mW) / prio

                ch_act.append(ch)
                H.append(h_linear)
                N.append(N_eff)
                floors.append(0.0)

            if ch_act:
                P_total = float(bs.transmit_power)
                p_list = self._waterfill(P_total, H, N, p_floor_list=floors, tol=1e-6, max_it=120)
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

        # --------- MEC queue dynamics & compute delays ----------
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

            # new workload
            offloaded = []
            total_new_cycles = 0.0

            for u in assoc_users:
                demand_Mbit = max(u.demand, 0.0) * self.step_duration_s
                off_Mbit = off_frac * demand_Mbit
                if off_Mbit <= 0.0:
                    continue
                C_cycles = off_Mbit * self.cycles_per_Mbit
                offloaded.append((u, C_cycles))
                total_new_cycles += C_cycles

            q_prev = bs.mec_queue_cycles
            mu_cycles = F_used * self.step_duration_s
            q_after_serv = max(q_prev - mu_cycles, 0.0)
            q_next = q_after_serv + total_new_cycles

            if F_used > 0.0:
                queue_delay_s = q_after_serv / F_used
            else:
                queue_delay_s = self.deadline_ms / 1000.0 * 10.0

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
            local_served = sum(min(u.data_rate, u.demand) for u in assoc_users)
            bs_served.append(local_served)
        bs_served_arr = np.asarray(bs_served, dtype=float)
        if np.any(bs_served_arr > 0):
            jain_num_bs = float(np.sum(bs_served_arr) ** 2)
            jain_den_bs = float(len(bs_served_arr) * np.sum(bs_served_arr**2) + 1e-9)
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
            "safety": {"served_rates": [], "latencies": [], "qoes": [], "deadline_hits": 0, "deadline_candidates": 0},
            "infotainment": {"served_rates": [], "latencies": [], "qoes": [], "deadline_hits": 0, "deadline_candidates": 0},
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
                lat_factor = max(0.0, 1.0 - (lat - self.deadline_ms) / (5 * self.deadline_ms))
            qoe = 0.5 * thr_norm + 0.5 * lat_factor
            qoe_clipped = float(np.clip(qoe, 0.0, 1.0))
            qoe_list.append(qoe_clipped)
            if bucket is not None:
                bucket["qoes"].append(qoe_clipped)

        # throughput metrics (Mbps)
        total_throughput_Mbps = float(np.sum(served_rates))
        avg_throughput_Mbps = float(total_throughput_Mbps / max(1, self.num_users))

        # latency metrics
        if latencies:
            avg_latency_ms = float(np.mean(latencies))
            p95_latency_ms = float(np.percentile(latencies, 95))
        else:
            avg_latency_ms = 0.0
            p95_latency_ms = 0.0

        # deadline satisfaction
        if deadline_candidates > 0:
            deadline_satisfaction = float(deadline_hits / max(1, deadline_candidates))
        else:
            deadline_satisfaction = 0.0

        # fairness (Jain's index) over users (for logging only)
        served_arr = np.asarray(served_rates, dtype=float)
        if np.any(served_arr > 0):
            jain_num_users = float(np.sum(served_arr) ** 2)
            jain_den_users = float(len(served_arr) * np.sum(served_arr**2) + 1e-9)
            jain_fairness_users = float(jain_num_users / jain_den_users)
        else:
            jain_fairness_users = 0.0

        # Use BS-level fairness in the reward
        jain_fairness = jain_fairness_bs

        # energy metrics
        total_radio_power_mW = float(np.sum([bs.transmit_power for bs in self.base_stations]))
        total_cpu_power_W = float(np.sum([self.cpu_kappa * (bs.last_cpu_used**3) for bs in self.base_stations]))
        step_energy_J = (total_radio_power_mW * 1e-3 * self.step_duration_s + total_cpu_power_W * self.step_duration_s)

        satisfied_tasks = deadline_hits
        if satisfied_tasks > 0:
            energy_per_task_J = float(step_energy_J / satisfied_tasks)
        else:
            energy_per_task_J = 0.0

        # blocking rate: users without any channel
        blocked_users = [u for u in self.users if not u.channel]
        blocking_rate = float(len(blocked_users) / max(1, self.num_users))

        # QoE statistics
        avg_qoe = float(np.mean(qoe_list)) if qoe_list else 0.0

        # ---------------- service-class metrics (for ITS-oriented analysis) ----------------
        def _svc_metrics(bucket):
            if not bucket["served_rates"]:
                return {"avg_throughput_Mbps": 0.0, "avg_latency_ms": 0.0, "p95_latency_ms": 0.0, "deadline_satisfaction": 0.0, "avg_qoe": 0.0}
            s_rates = np.asarray(bucket["served_rates"], dtype=float)
            lats = np.asarray(bucket["latencies"], dtype=float)
            qoes = np.asarray(bucket["qoes"], dtype=float)

            avg_lat = float(np.mean(lats)) if lats.size > 0 else 0.0
            p95_lat = float(np.percentile(lats, 95)) if lats.size > 0 else 0.0
            avg_tput = float(np.mean(s_rates)) if s_rates.size > 0 else 0.0

            if bucket["deadline_candidates"] > 0:
                ds = float(bucket["deadline_hits"] / max(1, bucket["deadline_candidates"]))
            else:
                ds = 0.0

            avg_qoe_svc = float(np.mean(qoes)) if qoes.size > 0 else 0.0

            return {"avg_throughput_Mbps": avg_tput, "avg_latency_ms": avg_lat, "p95_latency_ms": p95_lat, "deadline_satisfaction": ds, "avg_qoe": avg_qoe_svc}

        svc_safety = _svc_metrics(svc_stats["safety"])
        svc_infot = _svc_metrics(svc_stats["infotainment"])

        # ---------------- Normalized components (for logging) ----------------
        U_norm = float(np.clip(total_throughput_Mbps / max(1e-9, self.throughput_normalization_Mbps), 0.0, 1.0))

        if latencies:
            violations = [max(lat - self.deadline_ms, 0.0) / (5.0 * max(1.0, self.deadline_ms)) for lat in latencies]
            D_norm = float(np.clip(np.mean(violations), 0.0, 1.0))
        else:
            D_norm = 0.0

        E_norm = float(np.clip(step_energy_J / max(1e-9, 2.0 * self.energy_normalization_J), 0.0, 1.0))

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
        positive = (w.w_thr * T_norm + w.w_lat * (1.0 - D_norm) + w.w_dead * deadline_satisfaction + w.w_qoe * avg_qoe)
        negative = (w.w_eng * E_norm + w.w_block * blocking_rate + w.w_fair * V_fair)
        global_reward = float(np.clip(positive - negative, -1.0, 1.0))

        rewards = {agent: global_reward for agent in self.agents}

        scenario_type = "urban" if self.mobility_model == "manhattan" else "highway_like"

        # --------- per-BS infos ----------
        infos = {}
        association_fractions = []
        channel_rewards = []

        for i, bs in enumerate(self.base_stations):
            agent = f"agent_{i}"
            users_in_cov = [
                u for u in self.users
                if np.linalg.norm(u.location - bs.location) <= bs.coverage_area
            ]
            assoc_users = [
                u for u in users_in_cov
                if any(ch.base_station == bs for ch in u.channel)
            ]

            assoc_frac = (len(assoc_users) / max(1, len(users_in_cov)) if users_in_cov else 0.0)
            association_fractions.append(assoc_frac)

            ch_num = max(1, len(bs.assigned_channels))
            usr_num = max(1, len(users_in_cov))
            channel_rewards.append(1.0 - abs(ch_num - usr_num) / (ch_num + usr_num))

            local_served = sum(min(u.data_rate, u.demand) for u in assoc_users)
            local_demand = sum(u.demand for u in assoc_users) + 1e-9
            local_util = (local_served / local_demand if local_demand > 0 else 0.0)

            if assoc_users:
                # map users to their latency values (same order as self.users iteration)
                user_id_to_lat = {uu.id: lat for uu, lat in zip(self.users, latencies)}
                loc_latencies = [user_id_to_lat[u.id] for u in assoc_users]
                loc_avg_lat_ms = float(np.mean(loc_latencies))
            else:
                loc_avg_lat_ms = 0.0

            infos[agent] = {
                "local_data_rate_Mbps": float(sum(u.data_rate for u in assoc_users)),
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

        mean_assoc_frac = float(np.mean(association_fractions)) if association_fractions else 0.0
        mean_channel_reward = float(np.mean(channel_rewards)) if channel_rewards else 0.0

        global_info = {
            "total_throughput_Mbps": total_throughput_Mbps,
            "avg_throughput_Mbps": avg_throughput_Mbps,
            "avg_latency_ms": avg_latency_ms,
            "p95_latency_ms": p95_latency_ms,
            "deadline_satisfaction": deadline_satisfaction,
            "blocking_rate": blocking_rate,
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
            "env_step_walltime_ms": float((time.time() - start_wall) * 1e3),
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

        obs = self.get_observation()
        self.update_user_location()

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
            if self.mobility_model == "manhattan":
                loc, vel = self._spawn_on_grid()
                u.location = loc
                u.velocity = vel
                self._init_user_manhattan(u)
            else:
                u.location = self.np_random.uniform(0, self.area_size, 2)
                u.waypoint = self.np_random.uniform(0, self.area_size, 2)

            svc = "safety" if self.np_random.random() < self.safety_traffic_ratio else "infotainment"
            u.service_class = svc
            u.calculate_demand_from_rng(self.np_random)

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
            avail = [c for c in self.macro_channels if c.frequency == freq and c.base_station is None]
            if avail:
                chosen = [avail[0]]
                bs.assign_channels(chosen)
                self.assigned_channels[avail[0]] = bs.id
            bs.update_coverage_area_from_power(bs.transmit_power, self.bs_carrier_frequency[bs.id])

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
        max_speed = 25.0

        for i, bs in enumerate(self.base_stations):
            agent = f"agent_{i}"
            users_in_cov = [
                u for u in self.users
                if np.linalg.norm(u.location - bs.location) <= bs.coverage_area
            ]

            max_power = self.ma_transmission_power
            max_ch_for_bs = float(max(1, self.max_ma_channels))

            bs_type = 0.5
            tx_norm = np.clip(bs.transmit_power / max_power, 0.0, 1.0)

            used = sum(1 for c in bs.assigned_channels if len(c.users) > 0)
            ch_util = np.clip(used / max_ch_for_bs, 0.0, 1.0)

            cov_util = len(users_in_cov) / max(1, self.num_users)

            load_ratio = (
                len(users_in_cov) / max(1, len(bs.assigned_channels))
                if bs.assigned_channels else 0.0
            )
            load_ratio_norm = float(np.clip(load_ratio / 2.0, 0.0, 1.0))

            nearby_pot = len(
                [u for u in self.users if np.linalg.norm(u.location - bs.location) <= bs.coverage_area * 1.5]
            ) / max(1, self.num_users)

            avg_speed = np.clip(
                ((sum(u.speed for u in users_in_cov) / len(users_in_cov)) if users_in_cov else 0.0) / max_speed,
                0.0, 1.0
            )

            # approximate required power so that edge user meets SNR target
            if users_in_cov:
                max_dist = max(np.linalg.norm(u.location - bs.location) for u in users_in_cov)
                req_p_per_ch = self.calculate_required_power_for_distance(max_dist, bs)
                req_total = req_p_per_ch * max(1, len(bs.assigned_channels))
            else:
                req_total = 0.0

            req_p_norm = np.clip(req_total / max_power if max_power > 0 else 0.0, 0.0, 1.0)

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
                    avg_radial_v = float(np.clip((avg_radial / max_speed + 1) / 2.0, 0.0, 1.0))

            sp_var = np.clip(
                np.var([u.speed for u in users_in_cov]) / ((max_speed**2) / 4.0) if len(users_in_cov) > 1 else 0.0,
                0.0, 1.0
            )

            # coarse interference estimate
            f = self.bs_carrier_frequency[bs.id]
            co_bs = [o for o in self.base_stations if o is not bs and self.bs_carrier_frequency[o.id] == f]

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
                    uu for uu in self.users
                    if np.linalg.norm(uu.location - b2.location) <= b2.coverage_area
                ]
                util = min(1.0, len(users_in_cov_b2) / max(1, len(b2.assigned_channels)))

                if sampled_users:
                    acc = 0.0
                    for uu in sampled_users:
                        d_i = np.linalg.norm(uu.location - b2.location)
                        PL_i = b2.calculate_path_loss(d_i, f)
                        acc += p_i * g_tx_i * g_rx / (10.0 ** (PL_i / 10.0))
                    I_b += util * (acc / len(sampled_users))

            I_dBm = 10.0 * np.log10(max(I_b, 1e-12))
            inter_norm = np.clip((I_dBm + 120.0) / 120.0, 0.0, 1.0)

            avg_demand_norm = (
                float(np.clip(np.mean([u.demand for u in users_in_cov]) / self.max_user_demand_Mbps, 0.0, 1.0))
                if users_in_cov else 0.0
            )

            neigh = [
                o.transmit_power / max(1.0, self.ma_transmission_power)
                for o in self.base_stations
                if o is not bs and np.linalg.norm(bs.location - o.location) < 1000.0
            ]
            neighbor_tx_norm = float(np.mean(neigh)) if neigh else 0.0

            mec_queue_norm = float(
                np.clip(
                    bs.mec_queue_cycles / max(1.0, bs.mec_cpu_capacity * self.step_duration_s * 10.0),
                    0.0, 1.0
                )
            )
            cpu_util_norm = float(np.clip(bs.last_cpu_used / max(1.0, bs.mec_cpu_capacity), 0.0, 1.0))
            offload_norm = float(np.clip(bs.offload_frac, 0.0, 1.0))

            assoc_users = [u for u in users_in_cov if any(ch.base_station == bs for ch in u.channel)]

            if assoc_users:
                local_served = sum(min(u.data_rate, u.demand) for u in assoc_users)
                max_served = len(assoc_users) * self.max_user_demand_Mbps
                served_ratio_norm = float(np.clip(local_served / max(1e-9, max_served), 0.0, 1.0))
            else:
                served_ratio_norm = 0.0

            if users_in_cov:
                blocked_local = len([u for u in users_in_cov if not any(ch.base_station == bs for ch in u.channel)])
                block_frac_norm = float(np.clip(blocked_local / max(1, len(users_in_cov)), 0.0, 1.0))
            else:
                block_frac_norm = 0.0

            ch_num = len(bs.assigned_channels)
            usr_num = len(users_in_cov)
            if ch_num + usr_num > 0:
                ch_match_norm = float(1.0 - min(1.0, abs(ch_num - usr_num) / (ch_num + usr_num)))
            else:
                ch_match_norm = 1.0

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
                ],
                dtype=np.float32,
            )
            obs[agent] = vec_obs

        return obs

    def get_global_state(self):
        """Optional centralized critic state."""
        per_bs = []
        for bs in self.base_stations:
            users_in_cov = [
                u for u in self.users
                if np.linalg.norm(u.location - bs.location) <= bs.coverage_area
            ]
            assoc_users = [
                u for u in users_in_cov
                if any(ch.base_station == bs for ch in u.channel)
            ]
            mean_rate = float(np.mean([u.data_rate for u in assoc_users])) if assoc_users else 0.0
            served_frac = (
                float(sum(min(u.data_rate, u.demand) for u in assoc_users) / max(1e-9, sum(u.demand for u in assoc_users)))
                if assoc_users else 0.0
            )
            per_bs.extend([bs.transmit_power, len(bs.assigned_channels), mean_rate, served_frac, bs.mec_queue_cycles, bs.last_cpu_used])
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

            signal_mW = p_tx_ch * g_tx_main * g_rx_sig / (10.0 ** (PL_dB / 10.0))

            interference_mW = self._estimate_interference_mW(user, ch.frequency, bs)
            noise_mW = ch.calculate_noise_power() * 1e3

            denom = max(interference_mW + noise_mW, 1e-15)
            SINR_lin = signal_mW / denom if signal_mW > 0 else 0.0
            SINR_dB = 10.0 * np.log10(SINR_lin) if SINR_lin > 0 else -100.0

            SINR_dB = float(np.clip(SINR_dB, -100.0, 60.0))
            user.channel_SINR.append(SINR_dB)

        if user.channel_SINR:
            s_ma = np.mean([10.0 ** (x / 10.0) for x in user.channel_SINR])
            user.SINR = 10.0 * np.log10(s_ma) if s_ma > 0 else -100.0
        else:
            user.SINR = -100.0

    def calculate_required_power_for_distance(self, distance_m, base_station: BaseStation):
        """Required TX power (mW) so that UE at distance_m hits target SNR."""
        f = self.bs_carrier_frequency[base_station.id]
        pl_dB = base_station.calculate_path_loss(distance_m, f)

        if base_station.assigned_channels:
            bw_hz = base_station.assigned_channels[0].bandwidth
            nf_db = base_station.assigned_channels[0].noise_figure_db
        else:
            bw_hz = self.macro_channel_bw
            nf_db = 7.0

        sens_dBm = rx_sensitivity_dBm(bw_hz, nf_db=nf_db, snr_req_db=-5.0)
        g = BEAM_GAIN_DB[base_station.type_bs]
        req_tx_dBm = sens_dBm + pl_dB - (g["tx_main"] + g["rx"])

        return 10.0 ** (req_tx_dBm / 10.0)

    # -------------
    # Mobility update
    # -------------
    def update_user_location(self):
        if self.mobility_model == "manhattan":
            if not hasattr(self, "grid_xs"):
                self._build_road_grid()
            for u in self.users:
                self._advance_user_manhattan(u)
                u.location = np.clip(u.location, 0.0, self.area_size)
        else:
            L = self.area_size
            for u in self.users:
                if u.pause_time > 0:
                    u.pause_time -= 1
                    continue

                if u.waypoint is None:
                    u.waypoint = self.np_random.uniform(0, L, 2)

                delta = u.waypoint - u.location
                dist = np.linalg.norm(delta)

                if dist <= u.speed:
                    u.location = u.waypoint
                    u.pause_time = int(self.np_random.integers(0, 10))
                    u.waypoint = self.np_random.uniform(0, L, 2)
                else:
                    direction = delta / dist if dist > 0 else np.zeros(2)
                    u.velocity = direction * u.speed
                    u.location = np.clip(u.location + u.velocity, 0.0, L)

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
            circle = plt.Circle((bs.location[0], bs.location[1]), bs.coverage_area, color="C0", alpha=0.12)
            ax.add_patch(circle)
            ax.scatter(bs.location[0], bs.location[1], marker="^", c="C0", s=80)
            ax.text(bs.location[0] + 5, bs.location[1] + 5, f"BS{bs.id}", fontsize=8)

        # users and serving links
        for u in self.users:
            ax.plot(u.location[0], u.location[1], "ro", ms=4)
            for ch in u.channel:
                bs = ch.base_station
                if bs is None:
                    continue
                ax.plot([u.location[0], bs.location[0]], [u.location[1], bs.location[1]], "k-", lw=0.6, alpha=0.6)

        plt.pause(0.001)

    def close(self):
        try:
            if self.fig is not None:
                plt.close(self.fig)
        except Exception:
            pass
        self.fig = None
