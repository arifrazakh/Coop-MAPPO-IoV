# Environment details

The main environment class is `MultiAgentMobileNetwork` (see `src/mamn_marl/env.py`).

## Agents

- **One agent per macro base station (BS)**.
- The training baseline uses a **shared policy** across all agents (parameter sharing).

## Action space

Each agent outputs a 4D continuous vector in **[0, 1]**:

| Index | Name | Meaning |
|---:|---|---|
| 0 | `power_frac` | Fraction of max BS transmit power |
| 1 | `channel_frac` | Fraction of the per-BS channel pool to request |
| 2 | `offload_frac` | Fraction of user demand to offload to MEC |
| 3 | `cpu_frac` | Fraction of MEC CPU cycles to use this step |

The environment clips the actions to [0, 1].

## Observation space (19D)

Observations are **per-BS**, normalized to **[0, 1]**. The indices match `OBS_IDX`.

| Index | Field | Description |
|---:|---|---|
| 0 | `BS_TYPE` | Encoded BS type (currently a constant for macro) |
| 1 | `TX_NORM` | BS transmit power / max power |
| 2 | `CH_UTIL` | Channel utilization (occupied channels / max) |
| 3 | `COV_UTIL` | Users-in-coverage / total users |
| 4 | `LOAD_RATIO_NORM` | Users-in-coverage / channels (normalized) |
| 5 | `NEARBY_POT` | Users within 1.5× coverage / total users |
| 6 | `AVG_SPEED` | Mean speed of users in coverage (normalized) |
| 7 | `REQ_P_NORM` | Approx required power to serve edge users (normalized) |
| 8 | `AVG_RADIAL_V` | Mean radial velocity component toward the BS (normalized) |
| 9 | `SP_VAR` | Speed variance in coverage (normalized) |
| 10 | `NEIGHBOR_TX_NORM` | Neighboring BS transmit power (normalized) |
| 11 | `AVG_DEMAND_NORM` | Mean demand of users in coverage (normalized) |
| 12 | `INTER_NORM` | Co-channel interference proxy (normalized) |
| 13 | `MEC_QUEUE_NORM` | MEC queue (cycles) normalized |
| 14 | `CPU_UTIL_NORM` | CPU used / CPU max |
| 15 | `OFFLOAD_FRAC` | Offload fraction applied at BS |
| 16 | `SERVED_RATIO_NORM` | Served throughput / max possible |
| 17 | `BLOCK_FRAC_NORM` | Fraction of in-coverage users not associated |
| 18 | `CH_MATCH_NORM` | Match between #channels and #users |

## Dynamics

- **Mobility**: Manhattan grid (default) or a simple random-waypoint style model.
- **Radio**: UMa-like pathloss, co-channel interference, simplified MIMO rank selection, and per-BS water-filling.
- **Compute**: MEC queue (cycles) with per-step service and queue delay.

## Rewards

The environment returns a single *global* reward replicated to all agents. It combines:

- throughput (served demand)
- latency and deadline satisfaction
- QoE proxy
- energy penalties (radio + compute)
- blocking rate penalty
- fairness penalty (Jain’s index over BS served throughput)

See `step()` in `src/mamn_marl/env.py` for the exact scalar construction.
