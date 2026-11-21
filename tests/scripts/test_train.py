"""Tests for the training script (scripts/train.py).

Contract (agreed 2026-11-19 / 2026-11-21):
- every run writes into its own directory results/<name>_<seed>_<date-time>/ and never
  reuses an existing one,
- the run directory holds everything needed to reproduce and replay the run: a copy of the
  NEAT .ini, TrainingConfig (pickle and JSON) pointing at that copy, run arguments,
  statistics CSVs, the best genome of every generation and the winner,
- --pop-size overrides pop_size from the .ini,
- checkpoints go to the run directory, not to the current working directory,
- the serial mode (--workers 0 or 1) and the parallel mode give identical results,
- the saved winner re-evaluates to exactly its saved fitness (determinism).

The script is not a package module, so it is loaded from its file path. RESULTS_PATH and
default_training_config are replaced in the loaded module: results go to a temporary
directory and episodes are short (1 s instead of 10 s), so the tests stay fast.
"""

import argparse
import csv
import importlib.util
import json
import multiprocessing as mp
import pickle
from dataclasses import replace
from pathlib import Path

import neat
import pybullet as p
import pytest

from neat_walking_robot.training import evaluation

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = REPO_ROOT / "scripts" / "train.py"
POP_SIZE = 10
SHORT_EPISODE = 1.0

EXPECTED_FILES = {
    "a1.ini",
    "training_config.pkl",
    "training_config.json",
    "run_data.json",
    "fitness_history.csv",
    "speciation.csv",
    "species_fitness.csv",
    "most_fit_genomes.pkl",
    "winner.pkl",
}


def load_train_module():
    spec = importlib.util.spec_from_file_location("train_script", TRAIN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def train(tmp_path, monkeypatch):
    """train.py loaded as a module, writing to tmp_path/results, with short episodes.

    The working directory is an empty folder, so files written relative to the
    current directory (e.g. checkpoints without a path) can be detected.
    """
    module = load_train_module()
    results = tmp_path / "results"
    monkeypatch.setattr(module, "RESULTS_PATH", results)

    original = module.default_training_config

    def short_config(neat_config_path, seed):
        config = original(neat_config_path, seed)
        episode = replace(config.episode_params, episode_time=SHORT_EPISODE)
        return replace(config, episode_params=episode)

    monkeypatch.setattr(module, "default_training_config", short_config)

    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    yield module
    # the serial mode runs init_worker in this process: close its server
    if "client" in evaluation._worker:
        p.disconnect(evaluation._worker["client"])
    evaluation._worker.clear()


def run(train, *, name="t", seed=3, generations=2, workers=0, pop_size=POP_SIZE) -> Path:
    """Runs main() and returns the new run directory."""
    results = train.RESULTS_PATH
    before = set(results.iterdir()) if results.exists() else set()
    args = argparse.Namespace(
        name=name, seed=seed, generations=generations, workers=workers, pop_size=pop_size
    )
    train.main(args)
    if "client" in evaluation._worker:
        p.disconnect(evaluation._worker["client"])
        evaluation._worker.clear()
    created = set(results.iterdir()) - before
    assert len(created) == 1, f"expected exactly one new run directory, got {created}"
    return created.pop()


def read_rows(path: Path) -> list[list[str]]:
    with open(path, newline="") as f:
        return [row for row in csv.reader(f) if row]


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------- run directory


def test_run_directory_name_and_contents(train):
    run_dir = run(train, name="abc", seed=7)
    assert run_dir.name.startswith("abc_7_")
    assert EXPECTED_FILES <= {f.name for f in run_dir.iterdir()}


def test_existing_run_directory_is_never_reused(train, monkeypatch):
    """Two runs in the same second with the same name and seed must not overwrite each other."""
    import datetime as dt

    class FixedDatetime(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 1, 1, 12, 0, 0)

    monkeypatch.setattr(train, "datetime", FixedDatetime)
    run(train, name="same", seed=1, generations=1)
    with pytest.raises(FileExistsError):
        run(train, name="same", seed=1, generations=1)


def test_training_config_points_to_the_copied_ini(train):
    """Replay must use the .ini the run was trained with, not the (changing) one in configs/."""
    run_dir = run(train, seed=5)
    config = load_pickle(run_dir / "training_config.pkl")
    assert Path(config.neat_config_path).resolve() == (run_dir / "a1.ini").resolve()
    assert (run_dir / "a1.ini").read_text() == (REPO_ROOT / "configs" / "a1.ini").read_text()
    assert config.seed == 5

    saved = json.loads((run_dir / "training_config.json").read_text(encoding="utf-8"))
    assert saved["seed"] == 5
    assert Path(saved["neat_config_path"]).resolve() == (run_dir / "a1.ini").resolve()


def test_run_data_records_the_arguments(train):
    run_dir = run(train, name="rd", generations=2, workers=0)
    data = json.loads((run_dir / "run_data.json").read_text(encoding="utf-8"))
    assert data["name"] == "rd"
    assert data["generations"] == 2
    assert data["num_workers"] == 0
    assert data["pop_size"] == POP_SIZE
    assert data["runtime"] > 0
    assert data["neat-python_version"]


# ---------------------------------------------------------------- statistics and genomes


def test_pop_size_overrides_the_ini(train):
    """speciation.csv: one row per generation, species sizes summing to the population size."""
    run_dir = run(train, generations=2)
    rows = read_rows(run_dir / "speciation.csv")
    assert len(rows) == 2
    for row in rows:
        assert sum(int(x) for x in row) == POP_SIZE


def test_one_statistics_row_and_one_best_genome_per_generation(train):
    run_dir = run(train, generations=3)
    assert len(read_rows(run_dir / "fitness_history.csv")) == 3
    assert len(load_pickle(run_dir / "most_fit_genomes.pkl")) == 3


def test_winner_is_the_best_genome_of_the_run(train):
    run_dir = run(train, generations=3)
    winner = load_pickle(run_dir / "winner.pkl")
    best = max(load_pickle(run_dir / "most_fit_genomes.pkl"), key=lambda g: g.fitness)
    assert isinstance(winner, neat.DefaultGenome)
    assert winner.fitness == best.fitness
    best_column = [float(row[0]) for row in read_rows(run_dir / "fitness_history.csv")]
    assert winner.fitness == pytest.approx(max(best_column), rel=0, abs=1e-12)


def test_saved_winner_reevaluates_to_its_saved_fitness(train):
    """Everything replay.py needs is in the run directory, and the result is bit-identical."""
    run_dir = run(train, generations=2)
    winner = load_pickle(run_dir / "winner.pkl")
    config = load_pickle(run_dir / "training_config.pkl")
    neat_config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
        neat.DefaultStagnation, str(config.neat_config_path),
    )
    evaluation._worker.clear()
    evaluation.init_worker(config)
    try:
        assert evaluation.eval_genome(winner, neat_config) == winner.fitness
    finally:
        p.disconnect(evaluation._worker["client"])
        evaluation._worker.clear()


# ---------------------------------------------------------------- checkpoints


def test_checkpoints_go_to_the_run_directory(train):
    """With interval 10, an 11-generation run writes checkpoint 10, and only in the run directory."""
    run_dir = run(train, generations=11, pop_size=POP_SIZE)
    assert any(f.name.endswith("checkpoint-10") for f in run_dir.iterdir())
    assert list(Path.cwd().iterdir()) == [], "files written to the current working directory"


# ---------------------------------------------------------------- serial vs parallel


@pytest.fixture
def spawn():
    """Forces the Windows start method on every platform for the duration of one test."""
    previous = mp.get_start_method(allow_none=True)
    mp.set_start_method("spawn", force=True)
    yield
    mp.set_start_method(previous, force=True)


@pytest.mark.parametrize("serial_workers", [0, 1])
def test_serial_and_parallel_runs_are_identical(train, spawn, serial_workers):
    serial = run(train, name="serial", seed=11, workers=serial_workers)
    parallel = run(train, name="parallel", seed=11, workers=2)
    for name in ("fitness_history.csv", "speciation.csv", "species_fitness.csv"):
        assert (serial / name).read_text() == (parallel / name).read_text(), name
    assert (
        load_pickle(serial / "winner.pkl").fitness
        == load_pickle(parallel / "winner.pkl").fitness
    )
