<h1 align="center">Joint Radio-Compute Resource Management for Clustered Vehicular Edge Networks</h1>

<p align="center"><b>Coop-MAPPO-IoV</b></p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/SUMO-Mobility%20Simulation-2E8B57" alt="SUMO">
  <img src="https://img.shields.io/badge/RLlib-Torch%20Backend-EE4C2C" alt="RLlib">
  <img src="https://img.shields.io/badge/MARL-CTDE-6A1B9A" alt="MARL">
  <img src="https://img.shields.io/badge/Domain-IoV%20%2F%20VEC-0A66C2" alt="Domain">
  <img src="https://img.shields.io/badge/License-See%20LICENSE-green" alt="License">
</p>

> **Arif Raza, Uddin Md. Borhan, Anam Nasir, Jie Chen, and Lu Wang**  
> College of Computer Science and Software Engineering, Shenzhen University, Shenzhen, China  
> School of Computer Science and Technology, Harbin Institute of Technology, Harbin, China  
> Corresponding authors: **Lu Wang** and **Jie Chen**

---

## Overview

<p align="justify">
This repository contains the implementation and evaluation assets for <b>joint radio-compute resource management in clustered vehicular edge networks</b>. The framework studies how clustered multi-cell base stations (BSs), each paired with a mobile edge computing (MEC) server, can cooperatively control <b>transmit power</b>, <b>resource-block activation</b>, <b>task offloading</b>, and <b>CPU allocation</b> under dynamic mobility, interference coupling, MEC queue evolution, and heterogeneous intelligent transportation system (ITS) traffic.
</p>

<p align="justify">
The core learning design is a <b>cooperative multi-agent Proximal Policy Optimization (MA-PPO)</b> framework under <b>centralized training and decentralized execution (CTDE)</b>. Each BS executes a local policy from compact neighbor-aware observations, while a shared learner performs centralized actor-critic updates using a QoE-oriented global reward. The repository also includes comparison assets for <b>Cent-PPO</b>, <b>MA-A2C</b>, <b>MA-SAC</b>, a <b>heuristic controller</b>, and a <b>radio-only</b> reference, together with ready-made figures and CSV logs for reproducible analysis.
</p>

The framework addresses four tightly coupled challenges in clustered IoV-VEC control:

1. <b>Unified radio-compute orchestration:</b> communication and computing are optimized jointly rather than as loosely coupled subproblems.
2. <b>Cooperative BS-level learning:</b> each BS acts from a local observation, but policies are coordinated through centralized training and a shared reward.
3. <b>Scalable state and reward design:</b> compact neighbor-aware summaries and a normalized multi-objective QoE reward keep learning stable in dense deployments.
4. <b>ITS-oriented evaluation:</b> the framework is validated under SUMO-driven urban, highway, and mixed mobility regimes with heterogeneous traffic classes, deadline sensitivity, and scalability studies.

---

## Clustered IoV-VEC System Model

<p align="center">
  <img src="./Figs/architecture.png" width="92%" alt="Clustered IoV-VEC architecture"/>
</p>

<p align="justify">
The system model couples clustered BSs, MEC queues, dynamic vehicle association, reuse-based interference, and delay-aware offloading into a single closed-loop control problem. Each BS-MEC pair acts as a cooperative agent serving vehicles in its coverage region while accounting for neighboring load, radio activity, and shared-spectrum interference.
</p>

---

## CTDE Learning Workflow

<p align="center">
  <img src="./Figs/clde.jpg" width="88%" alt="CTDE workflow for cooperative MA-PPO"/>
</p>

<p align="justify">
During rollout, each BS observes only its compact local state and samples a continuous action from the shared policy. The simulator then updates association, RB activation, interference, offloading workload, MEC service, and end-to-end delay, and returns a <b>global shared reward</b>. During training, trajectories from all BSs are aggregated and used by a shared PPO learner to update actor and critic parameters with policy, value, and entropy losses. Runtime execution remains fully decentralized at the BS side.
</p>

---

## Architecture

```text
SUMO mobility traces (urban / highway / mixed)
        |
        v
Clustered IoV-VEC environment
  Vehicles, BS coverage, MEC queues, task classes
  Reuse-based interference, RB activation, offloading, CPU service
        |
        v
Local BS observation o_i(t)   [20 normalized features]
  - local radio utilization
  - local MEC backlog / CPU status
  - served and blocked ratios
  - demand summaries
  - mobility cues
  - neighbor activity summaries
  - mandatory-offload summary
        |
        v
Shared cooperative actor  pi_theta(a_i | o_i)
  Continuous 4-D action per BS:
    [power fraction, RB activation fraction,
     offloading fraction, CPU utilization fraction]
        |
        v
Environment transition
  Association update
  Channel / RB assignment
  Interference and water-filling-style power shaping
  Rate, queue, delay, and QoE computation
        |
        v
Shared global reward r_t
  + throughput
  + delay satisfaction
  + deadline-related service quality
  + QoE
  - energy
  - blocking
  - unfairness
        |
        v
Shared PPO learner under CTDE
  Aggregated multi-BS trajectories
  PPO ratio clipping + value loss + entropy regularization
```

---

## Method Details

This section summarizes the main components of the framework and aligns them with the updated paper.

### 1. Cooperative MDP and BS Action Space

The clustered vehicular edge network is modeled as a cooperative Markov decision process in which each BS is an agent. At each control epoch, BS <i>i</i> selects a continuous action:

```text
a_i(t) = [alpha_i(t), kappa_i(t), xi_i(t), phi_i(t)] in [0,1]^4
```

with the following meanings:

```text
alpha_i(t)  -> transmit-power fraction
kappa_i(t)  -> RB activation fraction
xi_i(t)     -> task offloading fraction
phi_i(t)    -> MEC CPU utilization fraction
```

These normalized controls are mapped to physical resources as:

```text
P_i(t) = alpha_i(t) * P_i^max
K_i(t) = round(kappa_i(t) * K_i^max)
F_i(t) = phi_i(t) * F_i^max
```

This low-dimensional BS-level interface preserves tractability while still exposing the key control knobs needed for clustered radio-compute orchestration.

---

### 2. Joint Radio-Compute and QoE Modeling

The environment couples radio service, MEC queue evolution, and user QoE in one control loop.

**Radio layer.** Each BS activates a subset of RBs and distributes power over active RBs under a total power budget. Inter-cell interference is modeled through reuse-based coupling across neighboring BSs. Achievable rate is derived from an SINR-driven link model with protocol overhead, SNR-gap modeling, and capped spectral efficiency.

**Association.** Vehicles are associated to the BS that balances link quality and current load. Users that cannot be served after association contribute to the blocking term.

**Compute layer.** The BS action determines how much demand is offloaded to the MEC server. Offloaded tasks become workload arrivals that update the MEC backlog:

```text
q_i(t+1) = max{ q_i(t) - F_i(t) * Delta, 0 } + lambda_i(t)
```

**End-to-end delay.** Total latency is the sum of radio-side delay and MEC delay:

```text
T_u(t) = T_u^r(t) + T_u^m(t)
```

**User QoE.** QoE jointly reflects throughput fulfillment and delay satisfaction instead of optimizing one metric in isolation.

This joint formulation is central to the framework because it reduces queue buildup and tail-latency escalation while preserving a strong throughput operating point.

---

### 3. Cooperative MA-PPO Under CTDE

A single stochastic actor and a shared critic are used across all BSs. During execution, each BS performs only a local forward pass:

```text
a_i(t) ~ pi_theta(. | o_i(t))
```

During training, trajectories from all BSs are aggregated into a shared rollout buffer and optimized with PPO using:

```text
- clipped policy objective
- value regression loss
- entropy regularization
- rollout minibatch updates
```

The design preserves two useful properties at the same time:

1. <b>Cooperation</b>, because all BSs learn from a global reward and shared parameters.
2. <b>Scalability</b>, because runtime execution remains decentralized and requires only local observations.

This is the central learning mechanism of the proposed Coop-MAPPO-IoV framework.

---

### 4. Scalable Observation and Reward Design

Each BS receives a compact 20-dimensional observation that summarizes the most relevant local and neighborhood signals without growing with the number of vehicles. The observation includes normalized descriptors of:

```text
- power, RB, and CPU utilization
- MEC queue level
- demand and served-load summaries
- blocked-user and mobility cues
- neighboring RB use and transmit activity
- mandatory-offload summary
```

The shared reward is a normalized multi-objective QoE score of the form:

```text
reward = + throughput
         + delay satisfaction
         + deadline-related score
         + average QoE
         - energy cost
         - blocking
         - unfairness
```

The reward is clipped to a bounded interval so that on-policy PPO updates remain numerically stable under bursty traffic and congested network states.

---

### 5. ITS-Oriented Evaluation Protocol

The evaluation protocol is designed to reflect deployment-oriented ITS conditions rather than small synthetic toy settings.

**Mobility regimes**

```text
omega_u : urban grid with stop-and-go traffic and turning events
omega_h : highway traffic with higher-speed flow and transient coverage changes
omega_m : mixed regime combining both conditions
```

**Traffic classes**

```text
- safety-oriented traffic with stricter delay sensitivity
- infotainment traffic with more throughput-oriented demand
- heavy and light tasks under shared radio-compute resources
```

**Logged performance metrics**

```text
- throughput
- average latency
- P95 latency
- deadline-related score / satisfaction behavior
- QoE
- energy efficiency
- Jain's fairness index
- blocking probability
```

This evaluation logic enables direct comparison between the proposed method and the learning and non-learning baselines under common settings.

---

## Simulation Setup

The validation platform couples mobility generation, radio resource allocation, MEC queue evolution, heterogeneous task arrivals, and policy learning in a unified discrete-time simulator.

```text
Decision interval             : 1 s
Episode length                : 200 control epochs
Default deployment            : 4 macro BSs, 80 vehicles, 2 km x 2 km
BS height                     : 30 m
Channels per carrier          : 50
Channel bandwidth             : 180 kHz
Maximum BS transmit power     : 40 W
Carrier set                   : {3.4, 3.5, 3.6, 3.7} GHz
Receiver noise figure         : 7 dB
SNR-gap parameter             : 1.5 dB
Maximum spectral efficiency   : 7.5 bit/s/Hz
MIMO abstraction              : rank up to 4 for macro BSs
MEC capacity per BS           : 50 x 10^9 cycles/s
Workload intensity            : 5 x 10^7 cycles/Mbit
CPU energy coefficient        : 10^-27
Latency target                : 100 ms
Traffic mix                   : 30% safety, 70% infotainment
Demand refresh probability    : 0.05 per user per epoch
```

Heavy tasks are generated with probability 0.70 and are treated as mandatory-offloading tasks when they exceed the UE local compute budget. Light tasks may be processed locally or offloaded depending on the BS action.

---

## Repository Layout

```text
Coop-MAPPO-IoV-main/
├── CITATION.cff
├── Figs
│   ├── architecture.png            <- clustered IoV-VEC framework overview
│   └── clde.jpg                    <- CTDE learning workflow illustration
├── graphs
│   ├── fairness_vs_load.png        <- Jain's fairness across offered-load bins
│   ├── latency_mean_tail_single_scale.png
│   │                                <- mean latency and P95 latency comparison
│   ├── qoe_cdf.png                 <- per-user QoE cumulative distribution
│   ├── radar_summary.png           <- normalized multi-metric comparison
│   └── throughput_bar.png          <- steady-state throughput comparison
├── LICENSE
├── pyproject.toml
├── README.md
├── requirements.txt
├── results
│   ├── a2c_episode_metrics.csv     <- MA-A2C episode-level metrics
│   ├── heuristic_episode_metrics.csv
│   ├── ppo_central_episode_metrics.csv
│   ├── ppo_episode_metrics.csv     <- proposed cooperative MA-PPO metrics
│   ├── ppo_with_4_BS.csv           <- scalability study with 4 BSs
│   ├── ppo_with_6_BS.csv           <- scalability study with 6 BSs
│   ├── ppo_with_8_BS.csv           <- scalability study with 8 BSs
│   ├── radio_only.csv              <- radio-only reference variant
│   └── sac_episode_metrics.csv     <- MA-SAC episode-level metrics
└── scripts
    ├── a2c_multi.ipynb             <- MA-A2C baseline notebook
    ├── heuristic.ipynb             <- heuristic controller notebook
    ├── ppo_cent.ipynb              <- centralized PPO baseline notebook
    ├── ppo_multi.ipynb             <- proposed cooperative MA-PPO notebook
    ├── ppo_var_bs.ipynb            <- BS-density scalability study
    ├── radio_only.ipynb            <- radio-only diagnostic notebook
    └── sac_multi.ipynb             <- MA-SAC baseline notebook
```

---

## Code Tour

This repository is notebook-centered. The main logic is organized so that the proposed method, baselines, and scalability experiments can be reproduced from self-contained notebooks, while the `results/` and `graphs/` folders store the outputs used for analysis and paper-ready visualization.

### 1. `scripts/ppo_multi.ipynb` — main Coop-MAPPO-IoV pipeline

This is the main implementation notebook. It contains the full cooperative MA-PPO workflow for clustered IoV-VEC control, including environment design, PPO configuration, logging, training, and output generation.

The notebook is organized conceptually as follows:

```text
Imports and library setup
  -> Gymnasium, NumPy, Matplotlib
  -> Ray RLlib, Torch

Basic PHY and helper functions
  -> rx_sensitivity_dBm()
  -> spectral_efficiency_from_sinr()
  -> mimo_rank_and_total_se()

Environment building blocks
  -> Channel
  -> BaseStation
  -> User

Main multi-agent simulator
  -> MultiAgentMobileNetwork(MultiAgentEnv)
  -> mobility, association, RB assignment, interference
  -> MEC queues, reward, observation construction

Callbacks and logging
  -> EpisodeCSVLogger(DefaultCallbacks)
  -> episode-wise metric accumulation and CSV export

RLlib integration
  -> rllib_env_creator()
  -> shared-policy multi-agent setup
  -> PPOConfig and training loop
```

<p align="justify">
If you want to understand the full system end to end, start with `ppo_multi.ipynb`. It goes from radio and compute abstractions, to BS-user environment dynamics, to cooperative PPO training under CTDE in one place.
</p>

### 2. PHY helpers and link abstractions

At the beginning of the main notebook, helper functions define the lightweight radio model used by the environment. These include thermal-noise-based sensitivity, Shannon-like spectral efficiency with an SNR gap, capped modulation efficiency, and a simplified MIMO rank abstraction.

These helpers bridge continuous BS actions to measurable communication outcomes:

```text
power fraction
   -> per-channel power
   -> SINR
   -> spectral efficiency
   -> achievable throughput
```

This keeps the learning interface compact while preserving the radio-compute coupling required by the paper.

### 3. Environment entities: `Channel`, `BaseStation`, and `User`

The main notebook defines three building-block classes used throughout the simulator:

- `Channel` stores channel frequency, bandwidth, noise figure, and user occupancy.
- `BaseStation` stores BS-side radio and MEC state, coverage behavior, power mapping, and channel assignment.
- `User` stores vehicle location, velocity, service class, demand, task type, input-data size, and channel-specific link state.

These abstractions keep the simulator readable by separating physical objects from the higher-level control loop.

### 4. `MultiAgentMobileNetwork` — the main simulator

This is the main environment class and the core of the repository. It implements the clustered IoV-VEC world as an RLlib-compatible multi-agent environment.

Key responsibilities include:

```text
- BS placement and topology initialization
- channel subset assignment
- mobility initialization and updates
- best-BS association under coverage and load
- on-demand channel activation
- interference estimation
- rate calculation with lightweight MIMO abstraction
- task generation and mandatory-offload logic
- MEC queue updates and CPU service
- local 20-D observation construction
- global shared reward computation
- per-step KPI logging
```

In practical terms, this class is where the paper's MDP, queue model, radio-compute coupling, and reward design become executable.

### 5. `EpisodeCSVLogger` — experiment logging

`EpisodeCSVLogger` collects per-step scalar metrics over an episode and writes episode-wise summaries to CSV. In the PPO notebook, it appends rows to `ppo_episode_metrics.csv`, which later feed the result tables and figures.

This logger is important because it keeps the training loop clean while still exporting the metrics used for:

```text
- throughput comparison
- latency comparison
- QoE analysis
- fairness analysis
- reward and return tracking
```

### 6. RLlib integration and PPO configuration

The notebook uses RLlib with a shared-policy multi-agent setup. A single policy is mapped to all BS agents, which directly implements the paper's CTDE idea.

The PPO block configures the learner with parameters such as:

```text
- gamma = 0.99
- lr = 5e-5
- lambda = 0.95
- clip_param = 0.2
- train_batch_size = 1024
- sgd_minibatch_size = 256
- num_sgd_iter = 10
- batch_mode = complete_episodes
```

The shared-policy configuration is one of the most important parts of the repository because it operationalizes cooperative BS-level learning with centralized updates and decentralized runtime actions.

### 7. Baseline notebooks

The remaining notebooks follow the same overall environment logic but swap out the learning rule or control strategy:

```text
scripts/ppo_cent.ipynb   -> centralized PPO reference
scripts/a2c_multi.ipynb  -> MA-A2C baseline
scripts/sac_multi.ipynb  -> MA-SAC baseline
scripts/heuristic.ipynb  -> rule-based controller
scripts/radio_only.ipynb -> communication-only diagnostic baseline
scripts/ppo_var_bs.ipynb -> scalability study for 4 / 6 / 8 BS settings
```

This separation makes it easier to reproduce comparisons method by method while keeping the evaluation pipeline consistent.

---

## Requirements

The repository includes both `requirements.txt` and `pyproject.toml` for environment setup.

Typical dependencies include:

- Python 3.x
- Jupyter / IPython for notebook execution
- PyTorch
- Ray RLlib with Torch backend
- NumPy / Pandas
- Matplotlib
- Gymnasium
- SUMO-related preprocessing or trace-generation dependencies, depending on the workflow

A standard setup is:

```bash
git clone https://github.com/arifrazakh/coop-mappo-iov.git
cd coop-mappo-iov

python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

If you prefer editable installation through the project metadata:

```bash
pip install -e .
```

---

## Quick Start

### 1. Launch Jupyter

```bash
jupyter lab
```

or

```bash
jupyter notebook
```

### 2. Run the proposed cooperative PPO notebook

Open:

```text
scripts/ppo_multi.ipynb
```

This notebook contains the main cooperative MA-PPO training and evaluation pipeline for the clustered IoV-VEC setting.

### 3. Run comparison baselines

For baseline experiments, use the following notebooks:

```text
scripts/ppo_cent.ipynb   -> centralized PPO
scripts/a2c_multi.ipynb  -> MA-A2C
scripts/sac_multi.ipynb  -> MA-SAC
scripts/heuristic.ipynb  -> heuristic controller
scripts/radio_only.ipynb -> radio-only diagnostic baseline
```

### 4. Run the scalability study

Open:

```text
scripts/ppo_var_bs.ipynb
```

This notebook evaluates how the proposed method behaves as the number of BSs increases while vehicle density is held fixed.

### 5. Inspect generated outputs

Key experiment outputs are stored in:

```text
results/   -> episode-level CSV metrics
graphs/    -> ready-made publication figures
Figs/      -> architecture and CTDE illustrations
```

---

## Key Notebooks

| Notebook | Purpose |
|---|---|
| `scripts/ppo_multi.ipynb` | Proposed cooperative MA-PPO / Coop-MAPPO-IoV training and evaluation |
| `scripts/ppo_cent.ipynb` | Centralized PPO baseline |
| `scripts/a2c_multi.ipynb` | MA-A2C baseline |
| `scripts/sac_multi.ipynb` | MA-SAC baseline |
| `scripts/heuristic.ipynb` | Heuristic controller for reference comparison |
| `scripts/radio_only.ipynb` | Radio-only diagnostic baseline |
| `scripts/ppo_var_bs.ipynb` | Scalability analysis under different BS counts |

---

## Graphs and Visual Results

<p align="center">
  <img src="./graphs/throughput_bar.png" width="48%" alt="Throughput comparison"/>
  &nbsp;
  <img src="./graphs/latency_mean_tail_single_scale.png" width="48%" alt="Mean and tail latency comparison"/>
</p>

<p align="justify">
<b>Left: Steady-state throughput comparison.</b> The proposed cooperative MA-PPO method reaches the high-throughput operating region while preserving better end-to-end service quality than the weaker baselines.
</p>

<p align="justify">
<b>Right: Mean latency and P95 latency.</b> The proposed method achieves the lowest mean delay and the lowest tail delay among the learning baselines, showing the value of joint radio-compute control under queue dynamics and interference coupling.
</p>

<p align="center">
  <img src="./graphs/radar_summary.png" width="48%" alt="Radar summary"/>
  &nbsp;
  <img src="./graphs/qoe_cdf.png" width="48%" alt="QoE CDF"/>
</p>

<p align="justify">
<b>Left: Normalized radar summary.</b> The figure compares throughput, deadline-related service quality, QoE, fairness, low-energy behavior, and low-blocking performance in one view. The proposed method dominates most axes simultaneously, confirming that the reward does not collapse to a single objective.
</p>

<p align="justify">
<b>Right: QoE CDF.</b> The per-user QoE distribution of the proposed method is shifted furthest to the right, indicating that more users receive higher average service quality.
</p>

<p align="center">
  <img src="./graphs/fairness_vs_load.png" width="62%" alt="Fairness versus offered-load bins"/>
</p>

<p align="justify">
<b>Fairness across offered-load bins.</b> The proposed policy remains in the top fairness band over higher load regimes, indicating that the compact observation and multi-objective reward maintain balanced service allocation even when congestion increases.
</p>

---

## Key Results

Steady-state comparison of the learning baselines:

| Algorithm | Throughput (Mbps) | Mean Latency (ms) | P95 Latency (ms) | Ep. Return | Global Reward | Deadline Score | QoE | Block (%) | Fairness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Ours** | **115.9** | **562.5** | **1164.0** | **26.16** | **0.131** | **11.30** | **0.378** | **0.51** | 0.971 |
| Cent-PPO | 115.6 | 1733.0 | 4217.0 | 17.20 | 0.086 | 5.23 | 0.331 | 3.09 | **0.972** |
| MA-A2C | 58.6 | 1315.1 | 2575.0 | -31.34 | -0.157 | 1.05 | 0.120 | 56.76 | 0.515 |
| MA-SAC | 22.6 | 1104.8 | 1713.4 | -50.80 | -0.254 | 0.59 | 0.049 | 81.80 | 0.348 |

Class-aware comparison under the current safety / infotainment traffic mix:

| Algorithm | Safety Deadline Score | Safety QoE | Infotainment Deadline Score | Infotainment QoE |
|---|---:|---:|---:|---:|
| **Ours** | **11.24** | **0.529** | **11.33** | **0.313** |
| Cent-PPO | 5.22 | 0.482 | 5.25 | 0.267 |
| MA-A2C | 1.05 | 0.184 | 1.05 | 0.093 |
| MA-SAC | 0.60 | 0.074 | 0.58 | 0.039 |

Scalability of the proposed method with increasing BS density:

| BSs | Throughput (Mbps) | Avg. Latency (ms) | P95 Latency (ms) | QoE |
|---|---:|---:|---:|---:|
| 4 | 117.09 ± 6.86 | 567.44 ± 8.13 | 1193.92 ± 45.39 | 0.377 ± 0.014 |
| 6 | 140.88 ± 9.30 | 441.17 ± 15.42 | 1002.10 ± 12.64 | 0.457 ± 0.019 |
| 8 | **177.59 ± 8.20** | **318.12 ± 10.84** | **764.85 ± 69.54** | **0.576 ± 0.016** |

These results show that the cooperative BS-level PPO design improves throughput, latency, QoE, deadline-related behavior, and scalability jointly, while preserving strong fairness under dense traffic.

---

## Reference-Based Component View

The repository includes several meaningful reference variants that help interpret the role of each major framework component.

| Variant | Learning | Joint Radio-Compute | Role |
|---|---|---|---|
| Cent-PPO | Yes | Yes | Centralized policy reference |
| Heuristic | No | Yes | Rule-based reference |
| Radio-only | No / Partial | No | Communication-only diagnostic |
| Ours | Yes | Yes | Full proposed framework |

This view clarifies that the repository is not only a training implementation, but also a controlled comparison framework for analyzing decentralized cooperative learning and joint radio-compute control.

---

## Why the Proposed Method Works

The empirical profile of the repository can be understood through three design decisions:

1. <b>Joint radio-compute control</b> prevents the policy from optimizing bandwidth and offloading in isolation.
2. <b>Shared-policy CTDE learning</b> enables cooperation across BSs without requiring centralized runtime inference.
3. <b>Compact neighbor-aware state plus normalized reward</b> preserves scalability, fairness, and service quality under dense traffic and bursty demand.

Together, these choices reduce blocking and tail-latency escalation while preserving a strong throughput operating point.

---

## Reproducibility Notes

- All major comparisons are backed by CSV logs under `results/`.
- The repository already includes final publication-style figures under `graphs/`.
- `CITATION.cff` is included for repository citation support.
- The notebook structure separates the proposed method, baseline methods, and scalability experiments for easier reproduction.
- After training, each BS requires only a single forward pass of the shared policy per control epoch, while the remaining computation is local scheduling, queue updates, and power shaping.



---

## License

This project is released under the terms specified in `LICENSE`.

---

## Repository Link

Project page: `https://github.com/arifrazakh/coop-mappo-iov/`
