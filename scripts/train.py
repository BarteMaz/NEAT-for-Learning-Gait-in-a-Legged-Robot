from neat_walking_robot.training import (
    default_training_config,
    save_training_config_to_json,
    save_training_config_to_pickle,
)

from pathlib import Path
from datetime import datetime

import argparse
import shutil
import neat


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "a1.ini"
RESULTS_PATH = PROJECT_ROOT / "results"

def main(arguments):

    # training's folder
    now = datetime.now()
    date = now.strftime("%Y-%m-%d_%H-%M-%S")

    results_path = RESULTS_PATH / f"{arguments.name}_{arguments.seed}_{date}"
    results_path.mkdir(parents=True, exist_ok=False)

    # read and save config
    training_config = default_training_config(CONFIG_PATH, arguments.seed)
    save_training_config_to_pickle(training_config, results_path)
    save_training_config_to_json(training_config, results_path)
    shutil.copy2(training_config.neat_config_path, results_path)

    # neat_config
    neat_config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        training_config.neat_config_path
    )
    population = neat.Population(neat_config, seed=arguments.seed)


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--generations", type=int, default=100)
parser.add_argument("--name", type=str, default="a1_train")
args = parser.parse_args()

main(args)

