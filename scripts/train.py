import argparse
import importlib.metadata
import json
import pickle
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from time import perf_counter

import neat
from neat_walking_robot.training import (
    default_training_config,
    eval_genome,
    eval_genomes,
    init_worker,
    save_training_config_to_json,
    save_training_config_to_pickle,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "a1.ini"
RESULTS_PATH = PROJECT_ROOT / "results"


def git_info(repo_root: Path) -> dict:
    def run_git(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

    try:
        commit = run_git("rev-parse", "HEAD")
        dirty = run_git("status", "--porcelain") != ""
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def main(arguments):
    # last git commit
    last_commit_hash = git_info(PROJECT_ROOT)

    # training's folder
    now = datetime.now()
    date = now.strftime("%Y-%m-%d_%H-%M-%S")

    results_path = RESULTS_PATH / f"{arguments.name}_{arguments.seed}_{date}"
    results_path.mkdir(parents=True, exist_ok=False)

    # read and save config
    shutil.copy2(CONFIG_PATH, results_path)
    training_config = default_training_config(results_path / CONFIG_PATH.name, arguments.seed)
    save_training_config_to_pickle(training_config, results_path)
    save_training_config_to_json(training_config, results_path)


    # neat_config
    neat_config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        training_config.neat_config_path
    )

    if arguments.pop_size is not None:
        neat_config.pop_size = arguments.pop_size

    population = neat.Population(neat_config, seed=arguments.seed)

    stdout_reporter = neat.StdOutReporter(True)
    stats_reporter = neat.StatisticsReporter()
    checkpointer = neat.Checkpointer(generation_interval=10, time_interval_seconds=None, filename_prefix=f"{results_path!s}/neat-checkpoint-")

    population.add_reporter(stdout_reporter)
    population.add_reporter(stats_reporter)
    population.add_reporter(checkpointer)

    run_start_time = perf_counter()
    if arguments.workers > 1:
        with neat.ParallelEvaluator(
                num_workers=arguments.workers,
                eval_function=eval_genome,
                initializer=init_worker,
                initargs=(training_config,),
                seed=arguments.seed,
        ) as evaluator:
            winner = population.run(evaluator.evaluate, arguments.generations)
    else:
        init_worker(training_config)
        winner = population.run(eval_genomes, arguments.generations)
    run_end_time = perf_counter()

    run_data = {
        "name": arguments.name,
        "generations": arguments.generations,
        "num_workers": arguments.workers,
        "pop_size": neat_config.pop_size,
        "runtime": run_end_time - run_start_time,
        "neat-python_version" : importlib.metadata.version("neat-python"),
        "last_commit_hash" : last_commit_hash
    }

    # statistics
    stats_reporter.save_genome_fitness(filename=results_path / "fitness_history.csv", delimiter=",")
    stats_reporter.save_species_count(filename=results_path / "speciation.csv", delimiter=",")
    stats_reporter.save_species_fitness(filename=results_path / "species_fitness.csv", delimiter=",")

    # best genomes from all generations
    with open(results_path / "most_fit_genomes.pkl", 'wb') as f:
        pickle.dump(stats_reporter.most_fit_genomes, f)

    # best genome
    with open(results_path / "winner.pkl", 'wb') as f:
        pickle.dump(winner, f)

    # save run data
    with open(results_path / "run_data.json", 'w', encoding="utf-8") as f:
        json.dump(run_data, f, ensure_ascii=False, indent=2)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--pop-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--name", type=str, default="a1_train")
    args = parser.parse_args()

    main(args)

