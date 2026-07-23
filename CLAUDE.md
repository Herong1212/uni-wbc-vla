# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`uni-wbc-vla` is a **Unified Whole-Body Control (WBC) + Vision-Language-Action (VLA)** framework for mobile manipulation robotics. It simulates and controls a mobile manipulator (omnidirectional base + UR5-style arm + gripper) using MuJoCo, with a VLA model (ViT vision encoder + Llama language model). The project integrates with the Hugging Face LeRobot ecosystem and supports Unitree hardware deployment.

**Action space dimension**: The VLA model config specifies 11-dim actions (6-DOF arm + 2-DOF base + 3-DOF gripper), but the current simulation environments use 7-dim actions (6-DOF arm/pose + 1 binary gripper). The full 11-dim action model is not yet wired to a simulation environment.

## Setup

```bash
pip install -r requirements.txt
cp .env .env.local  # edit HF_ENDPOINT, HF_HOME, HF_LEROBOT_HOME, etc.
```

`.env` is loaded at runtime via `dotenv.load_dotenv(".env")`.

## Running Tests

```bash
pytest tests/
pytest tests/test_load_xml.py
pytest tests/test_logging_utils.py
pytest tests/test_logging_utils.py::test_log
```

## Key Architecture

### Simulation Environments (`experiments/sim/mujoco_env/`)

Two environment variants, both driven by the `MuJoCoParserClass`:

- **`MobileSingleManipulatorEnv`** — 10 joints: 3 mobile base (`joint_x`, `joint_y`, `joint_th`) + 7 arm joints. Keyboard teleop (WASD/RF + QE/arrows + space gripper). 7-dim action: `[dx, dy, dz, droll, dpitch, dyaw, gripper]`. Uses IK for end-effector pose control.
- **`SimpleEnv` / `SimpleEnv2`** — 6-DOF arm only (`joint1`..`joint6`). 7-dim action same format. Used by the teleop data collection script. Simpler setup with fixed plate position.
- **`y_env2.py`** — Variant of SimpleEnv with fixed plate pose.

Action types: `"eef_pose"` (default, IK-based delta), `"delta_joint_angle"`, `"joint_angle"`. State types: `"joint_angle"`, `"ee_pose"`, `"delta_q"`.

Camera views: `agentview`, `egocentric`, `sideview`. Images captured via `grab_image()` returning `(rgb_agent, rgb_ego)`.

### MuJoCoParserClass (`experiments/sim/mujoco_env/mujoco_parser.py`)

A large (2500+ lines) custom MuJoCo XML parser/assembler. Composes simulations from XML components (arena + base + robot + gripper). Provides: FK/IK helpers, camera management, rendering with overlay, collision detection, physics stepping. Viewer uses GLFW.

### IK Solver (`experiments/sim/mujoco_env/ik.py`)

Custom damped-least-squares IK (`solve_ik`). Accepts: target body, target position/orientation, joint names, initial q. Configurable: max ticks, step size, convergence epsilon, angle threshold.

### Configuration (`config.py`)

`Config` class composes five dataclasses: `ModelConfig`, `SimulationConfig`, `TrainingConfig`, `RobotConfig`, `ExperimentConfig`. The training pipeline (`train_model.py`) uses LeRobot's `TrainPipelineConfig` instead.

### Data Collection & Pipeline

- **Keyboard teleop** → LeRobot dataset: `src/scripts/collect_dataset_teleoperation.py` — collects mug-on-plate demos via `SimpleEnv`, saves as LeRobot format (images at 256×256, EE pose, joint angles, gripper state).
- **LIBERO data processing**: `experiments/libero/re.py` replays LIBERO HDF5 demos to regenerate datasets (filtering no-ops, upscaling images). `utils/convert_libero_data_to_lerobot.py` converts LIBERO RLDS datasets to LeRobot format.
- **ALOHA scripts** (vendored at `third_party/aloha/`): one-side teleop, recording, replay, visualization.

### Training (`src/scripts/train_model.py`)

Uses LeRobot's training pipeline (`TrainPipelineConfig`, `make_policy`, `make_dataset`, `make_env`). Supports: offline training with `EpisodeAwareSampler`, mixed precision (AMP), gradient scaling, checkpointing, wandb logging, periodic eval in simulation.

Currently supported policy types (auto-resolved pretrained paths): `"pi0"` → `lerobot/pi0`, `"smolvla"` → `lerobot/smolvla_base`.

### Utilities (`utils/utils.py`)

- `init_logging()` — multi-GPU-aware logging (suppresses non-main-process console output)
- `TimerManager` — context-manager/start-stop timer with history, FPS, percentile stats
- `say()` / `log_say()` — cross-platform TTS (macOS `say`, Linux `spd-say`, Windows PowerShell)
- `SuppressProgressBars` — context manager to silence HuggingFace datasets progress bars

### Simulation Assets (`experiments/sim/assets/`)

Arenas: `pickandplace`, `playroom`, `scanned_objects` (includes objaverse models like mug/plate). Robots/objects/textures are in separate subdirectories.

### Vendored Third-Party

- `third_party/LIBERO/` — fork of LIBERO benchmark (task suites, data collection, evaluation)
- `third_party/aloha/` — ALOHA teleop scripts (one-side teleop, record/replay episodes)
- `mujoco_menagerie/`, `mujoco-py/`, `mujoco_learning/` — read-only references/tutorials

### Stub Files

`src/scripts/eval.py`, `finetune.py`, `load_model.py` are stubs (not yet implemented).
