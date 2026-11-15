from pathlib import Path
import neat

XOR = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0)]

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = 4.0 - sum((net.activate(x)[0] - y) ** 2 for x, y in XOR)

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation, str(Path(__file__).parent / "xor.ini"))

pop = neat.Population(config)
stats = neat.StatisticsReporter(); pop.add_reporter(stats)
winner = pop.run(eval_genomes, 300)
net = neat.nn.FeedForwardNetwork.create(winner, config)
print("gens", len(stats.most_fit_genomes), "fitness", round(winner.fitness,3), "size", winner.size())
for x, y in XOR: print(x, y, round(net.activate(x)[0], 3))
