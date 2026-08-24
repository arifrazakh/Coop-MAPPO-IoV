# Joint Radio-Compute Resource Management for Clustered Vehicular Edge Networks

## Overview

This repository contains the implementation and evaluation assets for
joint radio-compute resource management in clustered vehicular edge
networks.

The framework studies BS-level cooperative control for Internet of
Vehicles (IoV) and vehicular edge computing (VEC). Each BS-MEC agent
controls four high-level resources:

-   transmit-power budget,
-   active resource-block budget,
-   task offloading ratio,
-   MEC CPU utilization.

Association, RB assignment, and per-RB power shaping remain
environment-side operations.

The proposed method is a parameter-shared cooperative PPO framework with
decentralized execution. Agents use compact local observations, while
training aggregates BS trajectories through a shared learner and a
common QoE-oriented reward.

## Key Features

-   Joint radio and MEC resource orchestration.
-   Cooperative BS-level MARL.
-   Mobility-aware IoV simulation.
-   MEC queue and deadline modeling.
-   Multi-objective reward using throughput, delay, QoE, energy,
    blocking, and fairness.
-   Comparison with MAPPO, Cent-PPO, MA-A2C, MA-SAC, heuristic, and
    radio-only references.

## Repository Structure

``` text
Coop-MAPPO-IoV/
├── Figs/
├── graphs/
├── results/
├── scripts/
├── requirements.txt
├── pyproject.toml
├── CITATION.cff
└── README.md
```

## Simulation Setup

  Parameter                              Value
  ---------------------- ---------------------
  BS deployment                    4 macro BSs
  Vehicles                                  80
  Area                             2 km x 2 km
  Episode length                    200 epochs
  Decision interval                        1 s
  Channels per carrier                      50
  Channel bandwidth                    180 kHz
  BS transmit power                       40 W
  MEC capacity             50 x 10\^9 cycles/s
  Deadline target                       100 ms

## Training Example

``` bash
CUDA_VISIBLE_DEVICES=0 python mappo.py   --suite single   --method ours   --scenario urban   --seeds 44   --train-env-steps 500000   --eval-episodes 20   --eval-seed-base 900000   --num-bs 2   --num-users 80   --channels-per-carrier 50   --area-size 2000   --episode-len 200   --deadline-ms 100   --train-batch-size 1024   --minibatch-size 256   --num-sgd-iter 10   --num-gpus 1   --output-dir ours_bs2_seed44
```

## Main Results

  -----------------------------------------------------------------------
  Method          Throughput   Mean Latency    P95 Latency            QoE
                      (Mbps)           (ms)           (ms) 
  ----------- -------------- -------------- -------------- --------------
  Cent-PPO            196.51         462.89        1017.28        0.55868

  MAPPO               194.97         463.75        1026.05        0.56058

  MA-A2C              194.82         483.56        1078.04        0.55384

  MA-SAC              195.54         476.92        1039.66        0.55526

  Ours            **197.10**         464.95        1034.31    **0.56264**
  -----------------------------------------------------------------------

The proposed framework achieves the highest throughput and QoE in the
matched evaluation while maintaining a decentralized execution design.

## Scalability Results

  -----------------------------------------------------------------------
  BSs             Throughput        Average    P95 Latency            QoE
                      (Mbps)   Latency (ms)           (ms) 
  ----------- -------------- -------------- -------------- --------------
  2                   120.69        3627.83        5009.24          0.285

  4                   197.31         465.73        1042.99          0.563

  8                   197.29         246.07         585.91          0.690
  -----------------------------------------------------------------------

## Reproducibility

The repository includes:

-   training notebooks,
-   evaluation CSV files,
-   publication figures,
-   baseline implementations,
-   scalability experiments.

## Limitations

The current evaluation uses a system-level simulator with large-scale
mobility-aware channel modeling. Explicit fast fading, HARQ, backhaul
congestion, UE energy, and heterogeneous BS capability studies are
outside the reported experiments.

