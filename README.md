# (NeurIPS 2025) COLA: Towards Efficient Multi-Objective Reinforcement Learning with Conflict Objective Regularization in Latent Space

This repository contains the official implementation of **COLA**, a *general-policy* Multi-Objective Reinforcement Learning (MORL) framework that learns in a shared latent space and mitigates optimization conflicts across preferences.

**Paper:** COLA: Towards Efficient Multi-Objective Reinforcement Learning with Conflict Objective Regularization in Latent Space (NeurIPS 2025). 

## ✨ Overview

COLA addresses two key challenges in general-policy MORL:
- **Objective-agnostic Latent Dynamics Model (OADM):** builds a shared latent space capturing environment dynamics via temporal consistency, enabling efficient knowledge sharing across diverse preferences.
- **Conflict Objective Regularization (COR):** regularizes value updates when optimization directions under different preferences conflict, stabilizing value approximation and improving policy learning.

We adopt **Envelope SAC** as the backbone (general-policy) algorithm and condition both value and policy on preferences to cover the entire preference space.

## 📚 Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Supported Environments](#supported-environments)
- [Training](#training)
- [Project Structure](#project-structure)
- [Baselines](#baselines)
- [Results at a Glance](#results-at-a-glance)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## ✅ Features

- **Objective-agnostic latent space (OADM):** compact state & state–action representations for efficient multi-objective optimization.
- **Conflict-aware value learning (COR):** reduces interference among preferences when their optimization directions conflict.
- **General policy conditioning:** learn a single policy \(\pi(a\mid s, \omega)\) that generalizes across preferences \(\omega\).
- **CPU-friendly:** the code supports CPU-only training for MuJoCo-based tasks.

## 🔧 Installation

We recommend using Conda.

```bash
git clone <your-repo-url>
cd COLA

# Option A: create from provided environment file
conda env create -f environment.yml
conda activate cola

# Option B: create a fresh env (example)
conda create -n cola python=3.10 -y
conda activate cola
pip install -r requirements.txt
```

## 🚀 Quick Start

Train COLA on a 2D multi-objective Ant task:

```bash
python main.py   --env_id "MO-Ant-2d"   --seed 1   --Use_Critic_Preference   --Use_Policy_Preference   --Policy_use_latent   --Policy_use_s   --Policy_use_w   --Critic_use_both   --Critic_use_s   --Critic_use_a   --latent_dim 50   --regular_alpha 0.001   --regular_bar 0.25
```

You can also use the pre-configured launcher in `run.sh` to reproduce experiments for all tasks.

## 🌍 Supported Environments

We follow the paper’s multi-objective MuJoCo tasks (2–5 objectives). Example task set:

- **2D:** `MO-HalfCheetah-2d`, `MO-Hopper-2d`, `MO-Walker-2d`, `MO-Ant-2d`  
- **3D:** `MO-Hopper-3d`, `MO-Ant-3d`  
- **4D:** `MO-Ant-4d`  
- **5D:** `MO-HalfCheetah-5d`, `MO-Hopper-5d`, `MO-Ant-5d`

Each task runs for 500 steps per episode, with objectives including forward/axis speed, jump height, and energy efficiency; some 4D/5D variants add per-limb energy costs.

## 📊 Training 

### Preference grids
During training/evaluation we use preference grids to cover the space and report:
- **HV (Hypervolume)** and **UT (Utility)** on the discovered policies.
- Typical preference step sizes (per paper):  
  - 2-objective: `0.005`  
  - 3-objective: `0.05`  
  - 4-objective: `0.2`  
  - 5-objective: `0.2`
  - 
More details can be found in the Appendix of paper.

### Hardware
Experiments can be run on CPU; GPU is optional.

## 📁 Project Structure

```
COLA/
├── agent.py              # SAC-based agent with COLA changes
├── main.py               # Training entry
├── model.py              # Networks (policy/critic/encoders)
├── base.py               # Replay & utils
├── utils.py              # Helpers (logging, eval, etc.)
├── hypervolume.py        # Hypervolume computation
├── compute_hv.py         # HV/UT evaluation utilities
├── multi_step.py         # Multi-step learning utils
├── environments/         # MuJoCo multi-objective tasks & assets
├── run.sh                # Repro scripts
├── requirements.txt
├── environment.yml
└── README.md
```

## 🧪 Baselines

We compare against representative MORL baselines used in the paper:
- **PGMORL**
- **Envelope SAC**
- **CAPQL**
- **Q-Pensive**

## 📈 Results at a Glance

Across a range of multi-objective continuous-control tasks (2–5 objectives), COLA exhibits **higher sample efficiency** and **better final HV/UT** than state-of-the-art general-policy methods, owing to OADM (efficient knowledge sharing) and COR (conflict-aware value learning). See the paper for full plots and numbers.

## 📝 Citation

If you use this repository, please cite the paper:

```bibtex
@inproceedings{Li2025COLA,
  title     = {COLA: Towards Efficient Multi-Objective Reinforcement Learning with Conflict Objective Regularization in Latent Space},
  author    = {Pengyi Li and Hongyao Tang and Yifu Yuan and Jianye Hao and Zibin Dong and Yan Zheng},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025},
  url       = {https://openreview.net/forum?id=Cldpn7H3NN}
}
```

## 📄 License

This project is released under the terms in `LICENSE`.

## 📫 Contact

For questions or issues, please open a GitHub issue or contact me.
