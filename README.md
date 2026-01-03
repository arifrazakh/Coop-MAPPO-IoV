# MAMN-MARL: Multi-Agent Mobile Network + MEC (RLlib PPO)

This repository provides a **multi-agent cellular network + MEC (edge compute) simulator** and a **shared-policy PPO baseline** using **Ray RLlib**.

Each agent controls one macro base station (BS) and learns to jointly allocate:

1. **Downlink transmit power**
2. **Number of channels (PRB-like chunks)**
3. **MEC offloading ratio**
4. **MEC CPU utilization (service rate)**

The environment includes **Manhattan mobility**, **inter-cell interference**, **3GPP-like UMa pathloss**, **MIMO rank selection**, **water-filling power allocation**, and an **MEC queue**.

## What’s inside

- ✅ **Gymnasium-style** RL environment implemented as an RLlib `MultiAgentEnv`
- ✅ **Shared-policy PPO** training script (compatible across multiple RLlib versions)
- ✅ Episode-wise **CSV logger callback** (writes `ppo_episode_metrics.csv`)
- ✅ Optional plotting script for common metrics

## Repository layout

```
mamn-marl-mobile-network/
├─ src/
│  └─ mamn_marl/
│     ├─ __init__.py
│     ├─ env.py                  # the simulator (MultiAgentMobileNetwork)
│     └─ callbacks.py            # EpisodeCSVLogger callback
├─ scripts/
│  ├─ train_ppo.py               # RLlib PPO training entry-point
│  ├─ eval_policy.py             # evaluate a saved checkpoint
│  └─ plot_csv.py                # plot episode metrics from CSV
├─ configs/
│  ├─ env_default.yaml
│  └─ ppo_default.yaml
├─ requirements.txt
├─ pyproject.toml
├─ LICENSE
└─ .gitignore
```

## Quickstart

### 1) Create an environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt

# (optional) editable install so you can import `mamn_marl`
pip install -e .
```

### 2) Train PPO

```bash
python scripts/train_ppo.py \
  --total-env-steps 1000000 \
  --num-users 100 \
  --num-channels-per-carrier 100 \
  --area-size 5000 \
  --max-steps 200 \
  --mobility-model manhattan \
  --seed 42
```

Outputs:

- `ray_results/` (RLlib logs + checkpoints)
- `ppo_episode_metrics.csv` (episode-wise means of scalar `info` metrics)

### 3) Plot results from CSV

```bash
python scripts/plot_csv.py --csv ppo_episode_metrics.csv --out plots
```

### 4) Evaluate a checkpoint

```bash
python scripts/eval_policy.py --checkpoint /path/to/checkpoint_dir
```

## Environment: agents, actions, observations

### Agents

- `agent_0 ... agent_{N-1}` where **N = number of base stations**.
- RLlib is configured with a **single shared policy** (`shared_policy`).

### Action space (per agent)

Continuous 4-D Box in **[0, 1]**:

| Index | Name | Meaning |
|---:|---|---|
| 0 | `power_frac` | Total BS transmit power fraction (scaled by `ma_transmission_power`) |
| 1 | `channel_frac` | Fraction of max channels to request for the BS |
| 2 | `offload_frac` | Fraction of user workload offloaded to MEC |
| 3 | `cpu_frac` | Fraction of MEC CPU capacity to use this step |

### Observation space (per agent)

19-D normalized vector in **[0, 1]** describing BS load, radio conditions, interference, and MEC state.

See `OBS_IDX` in `src/mamn_marl/env.py` for the exact layout.

## Reward

The environment computes a **global reward** (same value for all agents each step), combining:

- throughput (served vs demanded)
- latency and deadline satisfaction
- QoE score per user
- energy (radio + MEC)
- blocking rate (users not associated)
- fairness (Jain’s index across BS-level served throughput)

The final reward is clipped to **[-1, 1]**.

## Notes on RLlib compatibility

RLlib’s configuration API has evolved over time.

- We use **`env_runners()`** (the successor of `rollouts()`), with `rollout_fragment_length="auto"` and `batch_mode="complete_episodes"` to avoid common batch-size vs fragment-length mismatches. (See RLlib docs.)
- The script checks for `config.build_algo()` vs `config.build()` so it runs on multiple RLlib versions.

## Reproducibility

- Deterministic seeding is applied to `random`, `numpy`, and Gymnasium RNG.
- For reproducible results, run on a fixed seed and keep Ray/RLlib version fixed.

## Citing / attribution

If you use this codebase in academic work, add a citation to your paper/project.

## License

MIT (see `LICENSE`).
