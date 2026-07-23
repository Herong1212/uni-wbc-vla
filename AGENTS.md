# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python robotics project for MuJoCo simulation, data collection, and LeRobot/VLA training. Core package code lives in `src/UniRobot/`: robot interfaces under `robots/`, MuJoCo environments under `sim/mujoco_env/`, training/evaluation entry points under `scripts/`, and import helpers under `utils/`. Top-level `utils/` contains dataset conversion and logging/timing utilities. Shell workflows are in `scripts/`; tests are in `tests/`. Simulation XML, textures, and meshes are under `src/UniRobot/sim/mujoco_env/assets/`. Vendored code is kept in `third_party/` and should be changed only when intentionally updating that dependency.

## Build, Test, and Development Commands

- `pip install -r requirements.txt`: install MuJoCo, LeRobot, Transformers, and runtime dependencies.
- `pytest tests/`: run the full test suite.
- `pytest tests/test_load_xml.py`: run the MuJoCo XML loading test.
- `pytest tests/test_logging_utils.py::test_log`: run one targeted test.
- `bash scripts/train_pick_box.sh`: start the pick-box training workflow.
- `bash scripts/collect_data_pick_box_keyboard.sh`: collect pick-box demonstrations with keyboard teleoperation.
- `bash scripts/convert_h5_to_lerobot_pick_box.sh`: convert HDF5 data to LeRobot format.

Install `pytest` separately if it is missing from the active environment.

## Coding Style & Naming Conventions

Use Python 3 with 4-space indentation and clear module-level functions or dataclasses where practical. Keep source imports rooted at `UniRobot` and avoid hard-coded absolute machine paths. Use `snake_case` for files, functions, variables, and shell scripts; use `PascalCase` for classes and dataclasses. Prefer structured XML/MuJoCo APIs over string edits for model changes. Keep comments brief and focused on non-obvious robotics, simulation, or dataset assumptions.

## Testing Guidelines

Tests use `pytest` and follow the `tests/test_*.py` naming pattern. Add focused tests for utility functions, XML/model loading, and data conversion logic. For simulation code, prefer smoke tests that load assets or validate shapes/configuration without interactive viewer sessions. Run `pytest tests/` before submitting changes when dependencies are available.

## Commit & Pull Request Guidelines

The current Git history uses short imperative summaries, for example `Remove outputs/ from tracking and update .gitignor`. Keep commit subjects concise and action-oriented. For pull requests, include a short description, affected workflows, test commands run, and any dataset/model asset requirements. Link related issues when available and include screenshots or logs for viewer, rollout, or training changes.

## Security & Configuration Tips

Runtime configuration may load `.env`; do not commit local credentials, Hugging Face tokens, dataset cache paths, or machine-specific absolute paths. Keep large generated outputs and datasets out of Git unless they are intentional small fixtures.
