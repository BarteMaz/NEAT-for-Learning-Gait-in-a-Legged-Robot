"""Mutation sensitivity test: how much do the children of a good genome resemble it?

Why
---
NEAT can only improve a good controller step by step if its mutated children stay close to it in fitness.
If almost every child falls over, evolution degenerates into random search around the elite and the best
fitness stalls (see guides/neat_config_analysis.md, section 4).

What it does
------------
1. Loads a finished run: its TrainingConfig (fitness function, action filter, ...), its winner and one
   checkpoint (the checkpoint is needed for the population and for the NEAT innovation tracker, see below).
2. For every mutation variant (dict VARIANTS below):
   - sets the variant's values on the genome config,
   - makes N children of the winner by mutation only (a copy of the winner + genome.mutate),
     or, with --crossover, by crossover with a random population member + mutation (what NEAT really does),
   - evaluates every child with exactly the same episode as in training (run_episode + the run's fitness),
   - restores the original values.
3. Prints and saves one row per variant: median / quartiles of child fitness, share of children within 90%
   of the parent, share at least as good as the parent, share that fell or hit a wall, median number of
   enabled connections.

How to read the result
----------------------
- "within 90%" high and "fell" low = gentle mutation: evolution can climb step by step.
- "within 90%" near 0 = destructive mutation: each child is effectively a new random controller.
- "at least parent" is the chance that one mutation is an improvement (or a neutral copy).
- Compare variants against "current" (the run's own .ini values).

Notes
-----
- Episodes are deterministic, so differences between variants come only from the mutations.
- The same random seed is set before every variant (common random numbers), so variants are compared on
  "the same luck" as far as possible.
- neat-python 2.0 mutations require config.genome_config.innovation_tracker. It is created by the
  reproduction module, so a plain neat.Config built from the .ini does not have it; a restored checkpoint does.
  That is why the script uses the checkpoint's config.
- Mutation in neat-python mutates every connection gene, also disabled ones (they only matter after being
  re-enabled).
- Runs in one process: about 0.3-0.5 s per child on one core, so 6 variants x 30 children take 1-2 minutes.

Usage
-----
    uv run scratch/etap5/mutation_sensitivity.py --run_name v2_filter_1_2026-11-25_12-25-13
    uv run scratch/etap5/mutation_sensitivity.py --run_name <run> --children 50 --crossover
    uv run scratch/etap5/mutation_sensitivity.py --run_name <run> --variants current gentle
"""

import argparse
import copy
import csv
import random
from pathlib import Path

import neat
import numpy as np

from neat_walking_robot.training import evaluation, load_run_config, make_policy, run_episode
from neat_walking_robot.training.episode import EndReason
from neat_walking_robot.training.fitness import FITNESS

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Each variant overrides some [DefaultGenome] values; everything not listed keeps the run's .ini value.
# Names are the attribute names of neat's genome config (the same as the keys in the .ini file).
VARIANTS: dict[str, dict[str, float]] = {
    "current": {},
    "power_0.1": {
        "weight_mutate_power": 0.1, "bias_mutate_power": 0.1,
    },
    "power_0.1_no_replace": {
        "weight_mutate_power": 0.1, "bias_mutate_power": 0.1,
        "weight_replace_rate": 0.0, "bias_replace_rate": 0.0,
    },
    "rate_0.1_power_0.5_no_replace": {
        "weight_mutate_rate": 0.1, "bias_mutate_rate": 0.1,
        "weight_replace_rate": 0.0, "bias_replace_rate": 0.0,
    },
    "gentle": {  # rate 0.1, power 0.1, no replace
        "weight_mutate_rate": 0.1, "bias_mutate_rate": 0.1,
        "weight_mutate_power": 0.1, "bias_mutate_power": 0.1,
        "weight_replace_rate": 0.0, "bias_replace_rate": 0.0,
    },
    "gentle_no_structure": {  # as gentle, and no structural mutations at all
        "weight_mutate_rate": 0.1, "bias_mutate_rate": 0.1,
        "weight_mutate_power": 0.1, "bias_mutate_power": 0.1,
        "weight_replace_rate": 0.0, "bias_replace_rate": 0.0,
        "conn_add_prob": 0.0, "conn_delete_prob": 0.0,
        "node_add_prob": 0.0, "node_delete_prob": 0.0,
        "enabled_mutate_rate": 0.0,
    },
}


def latest_checkpoint(run_dir: Path) -> Path:
    """The checkpoint with the highest generation number (the number is the next generation to evaluate)."""
    checkpoints = sorted(run_dir.glob("neat-checkpoint-*"), key=lambda f: int(f.name.rsplit("-", 1)[1]))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoint in {run_dir}")
    return checkpoints[-1]


def evaluate(genome, neat_config, training_config):
    """Same episode and fitness as eval_genome in training, but also returns the metrics (end reason)."""
    net = neat.nn.FeedForwardNetwork.create(genome, neat_config)
    policy = make_policy(net, training_config.episode_params)
    metrics = run_episode(
        evaluation._worker["robot"], evaluation._worker["env"], policy,
        training_config.simulation_data.dt, training_config.episode_params,
    )
    fitness = FITNESS[training_config.fitness_function](metrics, training_config.fitness_params)
    return fitness, metrics


def enabled_connections(genome) -> int:
    return sum(1 for c in genome.connections.values() if c.enabled)


def make_child(key, parent, population, genome_config, crossover: bool):
    """Mutation only: a copy of the parent, mutated once.
    With crossover: child of the parent (the fitter one) and a random population member, then mutated,
    which is what DefaultReproduction does (including the 75% disable rule for disabled genes)."""
    if crossover:
        other = random.choice(population)
        if other.fitness is None:          # population members restored from a checkpoint are not evaluated
            other.fitness = float("-inf")  # keeps the winner as the fitter parent
        child = neat.DefaultGenome(key)
        child.configure_crossover(parent, other, genome_config)
    else:
        child = copy.deepcopy(parent)
        child.key = key
        child.fitness = None
    child.mutate(genome_config)
    return child


def run_variant(name, overrides, winner, population, config, training_config, args):
    genome_config = config.genome_config
    original = {k: getattr(genome_config, k) for k in overrides}
    for k, v in overrides.items():
        setattr(genome_config, k, v)
    random.seed(args.seed)  # the same random stream for every variant

    try:
        fitness, failed, conns = [], 0, []
        next_key = max(g.key for g in population) + 1
        for i in range(args.children):
            child = make_child(next_key + i, winner, population, genome_config, args.crossover)
            f, metrics = evaluate(child, config, training_config)
            fitness.append(f)
            failed += metrics.end_reason in (EndReason.FALL, EndReason.WALL)
            conns.append(enabled_connections(child))
    finally:
        for k, v in original.items():  # the genome config is shared: always restore it
            setattr(genome_config, k, v)

    fitness = np.array(fitness)
    return {
        "variant": name,
        "children": args.children,
        "median": float(np.median(fitness)),
        "q25": float(np.percentile(fitness, 25)),
        "q75": float(np.percentile(fitness, 75)),
        "within_90pct": float(np.mean(fitness >= 0.9 * winner.fitness)),
        "at_least_parent": float(np.mean(fitness >= winner.fitness - 1e-12)),
        "fell_or_wall": failed / args.children,
        "enabled_conns_median": float(np.median(conns)),
    }


def main(args):
    run_dir = PROJECT_ROOT / "results" / args.run_name
    training_config, winner, _ = load_run_config(run_dir)
    checkpoint = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(run_dir)
    restored = neat.Checkpointer.restore_checkpoint(str(checkpoint))
    config = restored.config  # has the innovation tracker needed by mutate()
    population = list(restored.population.values())

    evaluation.init_worker(training_config)  # one PyBullet server, robot and env in this process
    try:
        parent_fitness, _ = evaluate(winner, config, training_config)
        print(f"run {args.run_name}, checkpoint {checkpoint.name}, fitness function "
              f"{training_config.fitness_function.name}, filter alpha {training_config.robot_spec.action_filter_alpha}")
        print(f"winner: saved fitness {winner.fitness:.4f}, re-evaluated {parent_fitness:.4f}, "
              f"enabled connections {enabled_connections(winner)}")
        mode = "crossover with a random member + mutation" if args.crossover else "mutation only"
        print(f"{args.children} children per variant, {mode}, seed {args.seed}\n")

        rows = []
        for name in args.variants:
            row = run_variant(name, VARIANTS[name], winner, population, config, training_config, args)
            rows.append(row)
            print(f"{name:30s} median {row['median']:6.2f} [{row['q25']:5.2f}, {row['q75']:5.2f}]  "
                  f">=90% {row['within_90pct']:.2f}  >=parent {row['at_least_parent']:.2f}  "
                  f"fell/wall {row['fell_or_wall']:.2f}  conns {row['enabled_conns_median']:.0f}")
    finally:
        import pybullet as p
        p.disconnect(physicsClientId=evaluation._worker["client"])
        evaluation._worker.clear()

    out_dir = run_dir / "diagnose"
    out_dir.mkdir(exist_ok=True)
    suffix = "crossover" if args.crossover else "mutation"
    out_file = out_dir / f"mutation_sensitivity_{suffix}_seed{args.seed}.csv"
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nsaved {out_file.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run_name", type=str, required=True, help="directory in results/")
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint file (default: the latest)")
    parser.add_argument("--children", type=int, default=30, help="children per variant")
    parser.add_argument("--seed", type=int, default=0, help="random seed, the same for every variant")
    parser.add_argument("--crossover", action="store_true", help="crossover with a random member before mutation")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    main(parser.parse_args())
