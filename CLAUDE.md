# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`uni-wbc-vla` is a **Unified Whole-Body Control (WBC) + Vision-Language-Action (VLA)** framework for mobile manipulation robotics. It simulates and controls a mobile manipulator (omnidirectional base + arm + gripper) using MuJoCo, with a VLA model (ViT vision encoder + Llama language model). The project integrates with the Hugging Face LeRobot ecosystem and targets Unitree hardware deployment.

**Action space**: The root `config.py` defines 11-dim actions (6-DOF arm + 2-DOF base + 3-DOF gripper) for the VLA model. The simulation environments use either 7-dim actions (EEF delta pose + binary gripper) or 11-dim actions (full joint control). The 11-dim model config is not yet wired to a simulation environment.

## Setup

```bash
pip install -r requirements.txt
cp .env .env.local  # edit HF_ENDPOINT, HF_HOME, HF_LEROBOT_HOME, etc.
```

`.env` is loaded at runtime via `dotenv.load_dotenv(".env")`. It configures HuggingFace mirror endpoint, cache directories, and LeRobot paths.

## Running Tests

```bash
pytest tests/
pytest tests/test_load_xml.py
pytest tests/test_logging_utils.py
pytest tests/test_logging_utils.py::test_log
```

## Package Structure (`src/UniRobot/`)

All core code lives under the `src/UniRobot/` Python package. Imports should be rooted at `UniRobot` (e.g., `from UniRobot.sim.mujoco_env.tidybot_env import TidybotEnv`). Avoid hard-coded absolute machine paths.

### Simulation Environments (`src/UniRobot/sim/mujoco_env/`)

Three environment classes, all driven by the `MuJoCoParserClass` (in `mujoco_parser.py`, ~2500-line custom MuJoCo XML parser/assembler that composes scenes from XML components: arena + base + robot + gripper):

- **`TidybotEnv`** (`tidybot_env.py`) — Mobile manipulator: 3-DOF omnidirectional base (`joint_x`, `joint_y`, `joint_th`) + 7-DOF Kinova Gen3 arm (`joint_1`..`joint_7`) + Robotiq 2F-85 gripper (tendon-driven). Action space: 7-dim EEF delta pose `[dx, dy, dz, droll, dpitch, dyaw, gripper]` (default), or 11-dim joint control. IK target body: `bracelet_link`.
- **`SimpleEnv`** (`y_env.py`) — 6-DOF arm only (`joint1`..`joint6`). 7-dim action (same EEF pose format). Used for tabletop teleop data collection.
- **`SimpleEnv2`** (`y_env2.py`) — Variant of `SimpleEnv` with fixed plate pose.

All environments support three action types: `"eef_pose"` (default, IK-based delta), `"delta_joint_angle"`, `"joint_angle"`. State types: `"joint_angle"`, `"ee_pose"`, `"delta_q"`. Camera views: `agentview`, `egocentric`, `sideview` — captured via `grab_image()` returning `(rgb_agent, rgb_ego)`.

Supporting modules in the same package:
- **`ik.py`** — Custom damped-least-squares IK solver (`solve_ik`).
- **`transforms.py`** — Rotation matrix helpers (`rpy2r`, `r2rpy`, `t2p`, `t2r`, `t2pr`).
- **`utils.py`** — Rendering overlays, camera param computation, pyautogui helpers, XML prettifying.

### Robots (`src/UniRobot/robots/`)

Robot hardware abstraction (`robot.py`, `config.py`, `utils.py`). Currently mostly stub/placeholder — the primary robot logic is in the simulation environments.

### Scripts (`src/UniRobot/scripts/`)

- **`train_model.py`** — Main training entry point using LeRobot's `TrainPipelineConfig`, `make_policy`, `make_dataset`, `make_env`. Supports: offline training with `EpisodeAwareSampler`, mixed precision (AMP), gradient scaling, checkpointing, wandb logging, periodic eval in simulation. Supported policies: `"pi0"` → `lerobot/pi0`, `"smolvla"` → `lerobot/smolvla_base`.
- **`collect_dataset_teleoperation_mujoco.py`** — Keyboard teleop data collection: controls a MuJoCo `TidybotEnv` or `SimpleEnv`, records observations (images at 256×256, EE pose, joint angles, gripper state) into LeRobot dataset format.
- **`eval.py`**, **`finetune.py`**, **`load_model.py`** — Stubs, not yet implemented.

### Utils (`src/UniRobot/utils/`)

- **`import_utils.py`** — LeRobot package availability checks and third-party plugin discovery (scans installed packages for `lerobot_robot_*`, `lerobot_camera_*`, etc. prefixes).

## Top-Level Modules

### Configuration (`config.py`)

`Config` class composing five dataclasses: `ModelConfig` (vision/language model params, 11-dim action), `SimulationConfig`, `TrainingConfig`, `RobotConfig` (base/robot/gripper types), `ExperimentConfig`. This is separate from LeRobot's `TrainPipelineConfig` used in `train_model.py`.

### Utilities (`utils/utils.py`)

- `init_logging()` — Multi-GPU-aware logging (suppresses non-main-process console output).
- `TimerManager` — Context-manager/start-stop timer with history, FPS, percentile stats.
- `say()` / `log_say()` — Cross-platform TTS (macOS `say`, Linux `spd-say`, Windows PowerShell).
- `SuppressProgressBars` — Context manager to silence HuggingFace datasets progress bars.

### Data Conversion (`utils/convert_libero_data_to_lerobot.py`)

Converts LIBERO RLDS datasets to LeRobot format. Related LIBERO processing in `third_party/LIBERO/`.

## Simulation Assets (`src/UniRobot/sim/mujoco_env/assets/`)

Arenas (`pickandplace`, `scanned_objects` with objaverse mug/plate/can models), robots, objects, textures — each in separate subdirectories. Scene XML files compose these into complete simulation scenes.

## Vendored Third-Party

- `third_party/LIBERO/` — Fork of LIBERO benchmark (task suites, data collection, evaluation).
- `third_party/aloha/` — ALOHA teleop scripts (one-side teleop, record/replay episodes).

## Shell Scripts (`scripts/`)

Shell wrappers for common workflows — most are currently empty placeholders. The one with content (`train_pick_box.sh`) calls `python ./src/scripts/collect_dataset_teleoperation.py` (note: this path is stale; the actual script is at `src/UniRobot/scripts/collect_dataset_teleoperation_mujoco.py`).

## Stub / Placeholder Files

- `src/UniRobot/scripts/eval.py`, `finetune.py`, `load_model.py` — Not yet implemented.
- `src/UniRobot/robots/` — Mostly empty stubs for future hardware robot abstraction.
- `utils/datasets_utils.py` — Empty placeholder.
- Most shell scripts in `scripts/` — Empty placeholders.
