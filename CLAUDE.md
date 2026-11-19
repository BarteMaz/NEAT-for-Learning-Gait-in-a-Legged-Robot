# CLAUDE.md

Bachelor's thesis project: using the NEAT neuroevolution algorithm (`neat-python`) to teach a quadruped robot to walk in a PyBullet simulation. Everything is in Python.

Full context, plan and decisions live in the Claude project "NEAT usage for learning gait in legged robots" (doc `claude/project_context.md`). If this file and that doc disagree, ask the user.

## Working mode (most important)

The owner is writing this as an engineering thesis and wants to understand every part of it. He does NOT want the project built for him.

- Act as a mentor. Explain the theory and the reasoning first, check understanding with questions, then guide him to implement it himself.
- Do not write or rewrite project code (`src/`) unless he explicitly asks. Prefer explanations, small illustrative examples unrelated to the project code, exercises and hints.
- Review his code: point out bugs and risks, describe the fix, let him apply it.
- Raise criticism of the plan or design decisions right away.
- Talk to him in Polish.
- Never commit or push without explicit permission.

## Stack

- Windows 11, PyCharm, `uv` for Python and dependencies, Python 3.11.
- PyBullet via the `pybullet-arm64` package (3.2.8), not the official `pybullet`.
- `neat-python`, numpy, scipy, pandas, matplotlib. Dev: pytest, ruff, jupyter.

## Commands

```
uv sync
uv run pytest        
uv run ruff check .
```

## Layout

- `src/neat_walking_robot/environments`: module with classes and dataclasses for 3 different environments with factory
- `src/neat_walking_robot/simulation`: module that handles creating and preparing simulation
- `src/neat_walking_robot/robot`: module with class that handles reading robot's info, loading it into simulation, controls its actions and returns observations
- `src/neat_walking_robot/training`: module with functions that: run a training episode, calculate fitness, initializes training
- `configs`: folder with neat configs
- `scripts`: folder with scripts that will calculate a baseline, handles training loop and shows results
- `results`: folder with training results
- `guides/`: URDF guide and PyBullet quickstart.
- `scratch/`: folder for experimenting with PyBullet and learning new features 
- `tests/`: folder with tests for each module

## Plan (one learning session per stage)

1. DONE: PyBullet basics and three environments: flat fenced arena, fenced bumpy terrain, fenced arena with a ramp. 
2. DONE: URDF and manual robot control.
3. DONE: Quadruped model - Unitree A1, URDF provided by PyBullet
4. DONE: neat-python basics.
5. IN PROGRESS: Training process: fitness function, thresholds, checkpoints, progressive harder environments.
6. Analytics layer (data for the thesis).
7. IN PROGRESS: Tests.
