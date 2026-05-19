# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`uni-wbc-vla` is a **Unified Whole-Body Control (WBC) + Vision-Language-Action (VLA)** framework for mobile manipulation robotics. It simulates and controls a mobile manipulator (omnidirectional base + UR5-style arm + gripper) using MuJoCo, with a VLA model (ViT vision encoder + Llama language model) that outputs 11-dimensional actions: 6-DOF arm + 2-DOF base + 3-DOF gripper.

The project integrates with the Hugging Face LeRobot ecosystem and supports deployment to Unitree hardware.

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

Copy and configure environment variables:
```bash
cp .env .env.local  # edit HF_ENDPOINT, HF_HOME, HF_LEROBOT_HOME, etc.
```

The `.env` file is loaded at runtime via `dotenv.load_dotenv(".env")` (see `dom.py` and test files).

## Running Tests

```bash
pytest tests/
pytest tests/test_logging_utils.py          # run a single test file
pytest tests/test_logging_utils.py::test_log  # run a single test
```

## Architecture

### Configuration (`config.py`)
Single source of truth for all parameters. The top-level `Config` class composes five dataclasses:
- `ModelConfig` — VLA model (ViT + Llama, `action_dim=11`)
- `SimulationConfig` — MuJoCo env (`dual_arm_mobile_manipulation`, 0.02s timestep)
- `TrainingConfig` — optimizer, batch size, epochs
- `RobotConfig` — hardware types (`omni` base, `ur5` robot, `robotiq` gripper)
- `ExperimentConfig` — paths for `data/`, `logs/`, `results/`

Config can be serialized/loaded via `config.save(path)` / `Config.load(path)`.

### Assets (`assets/`)
MuJoCo XML (MJCF) definitions organized as composable components:
- `base.xml` — root physics config (global compiler/option/size settings)
- `arenas/` — environment scenes (table, bins, pegs, multi-table, empty)
- `bases/` — mobile base models (omni, floating-legged, null, turtlebot3)
- `robots/` — arm robot XMLs (UR5e, Panda, Baxter, G1, H1, Spot, etc.)
- `grippers/` — end-effector XMLs (Robotiq 85/140, Panda, Fourier hands, etc.)
- `objects/` — manipulable objects (can, bread, bottle, door, nuts, etc.)

Simulations are assembled by combining an arena + base + robot + gripper XML.

### Utilities (`utils/utils.py`)
- `init_logging()` — configures Python logging with multi-GPU/Accelerator awareness (suppresses non-main-process console output)
- `TimerManager` — context-manager/start-stop timer with history, FPS, and percentile stats
- `say()` / `log_say()` — cross-platform text-to-speech (macOS `say`, Linux `spd-say`, Windows PowerShell)
- `SuppressProgressBars` — context manager to silence HuggingFace datasets progress bars

### Models (`models/`)
Intended directories for model implementations: `base/`, `gripper/`, `robot/`. Currently being developed.

### MuJoCo Learning Resources (`mujoco_learning/`)
Chinese-language MuJoCo tutorials (MJCF modeling, Python API, C++ API, advanced topics). Separate from the main codebase — treat as reference material, not production code.

### External Libraries (vendored)
- `mujoco_menagerie/` — DeepMind's collection of high-quality MuJoCo robot models (read-only reference)
- `mujoco-py/` — Legacy MuJoCo Python bindings (read-only reference)
