"""
ai.py
-----------------------------------------
A very simple Machine Learning approach: a Genetic Algorithm (GA)
training a tiny "perceptron" (a single-layer, weighted-sum brain).

Why this instead of deep learning?
  - Easy to understand for a college mini-project.
  - No heavy libraries (no TensorFlow / PyTorch) needed.
  - Still demonstrates real ML concepts: population, fitness,
    selection, crossover, mutation, generations.

How the "brain" works (Genome):
  4 inputs (bird_y, velocity, distance_to_pipe, gap_y)
      -> multiplied by 4 weights, summed, plus 1 bias
      -> if the result > 0: FLAP, otherwise: do nothing

How learning happens (GeneticAlgorithm):
  1. Start with a population of random genomes (random weights).
  2. Let each genome play a full game (using game_logic.Game).
  3. Fitness = the score the genome achieved.
  4. Keep the best genomes, breed new ones from them (crossover),
     and randomly mutate a few weights (mutation).
  5. Repeat for many generations -> average performance improves.
"""

import random
from game_logic import Game

INPUT_SIZE = 4          # bird_y, velocity, dist_to_pipe, gap_y
GENOME_SIZE = INPUT_SIZE + 1   # + 1 bias term

MAX_FRAMES = 4000        # safety cap so a "perfect" bird doesn't run forever


class Genome:
    """A single AI 'brain': a list of weights + a bias."""

    def __init__(self, weights=None):
        if weights is None:
            # Random weights between -1 and 1
            weights = [random.uniform(-1, 1) for _ in range(GENOME_SIZE)]
        self.weights = weights
        self.fitness = 0

    def decide(self, inputs):
        """
        inputs: list of 4 floats -> returns 1 (flap) or 0 (do nothing)
        This is just a weighted sum (a single artificial neuron).
        """
        total = self.weights[-1]  # bias
        for w, x in zip(self.weights, inputs):
            total += w * x
        return 1 if total > 0 else 0

    def clone(self):
        return Genome(list(self.weights))


def crossover(parent_a, parent_b):
    """Mix two parents' weights to create a child (simple average + random pick)."""
    child_weights = []
    for wa, wb in zip(parent_a.weights, parent_b.weights):
        child_weights.append(wa if random.random() < 0.5 else wb)
    return Genome(child_weights)


def mutate(genome, mutation_rate=0.15, mutation_strength=0.5):
    """Randomly tweak a few weights to introduce new behaviour."""
    new_weights = []
    for w in genome.weights:
        if random.random() < mutation_rate:
            w += random.uniform(-mutation_strength, mutation_strength)
        new_weights.append(w)
    return Genome(new_weights)


def evaluate_genome(genome):
    """Run one full headless game with this genome controlling the bird."""
    game = Game()
    frames = 0
    while game.alive and frames < MAX_FRAMES:
        state = game.get_state()
        action = genome.decide(state)
        game.step(action)
        frames += 1
    genome.fitness = game.score
    return game.score


class GeneticAlgorithm:
    """Manages the population of genomes across generations."""

    def __init__(self, population_size=40):
        self.population_size = population_size
        self.population = [Genome() for _ in range(population_size)]
        self.generation = 0
        self.best_genome = self.population[0]
        self.best_score_ever = 0

    def run_generation(self):
        """
        Evaluate every genome in the current population, then breed
        the next generation from the best performers.
        Returns a dict summarizing the results (used by the Flask API).
        """
        for genome in self.population:
            evaluate_genome(genome)

        # Sort best-first
        self.population.sort(key=lambda g: g.fitness, reverse=True)

        best_this_gen = self.population[0]
        avg_score = sum(g.fitness for g in self.population) / len(self.population)

        if best_this_gen.fitness > self.best_score_ever:
            self.best_score_ever = best_this_gen.fitness
            self.best_genome = best_this_gen.clone()

        self.generation += 1

        result = {
            "generation": self.generation,
            "best_score_this_gen": best_this_gen.fitness,
            "avg_score_this_gen": round(avg_score, 2),
            "best_score_ever": self.best_score_ever,
            "weights": self.best_genome.weights,
        }

        # ---- Build the next generation ----
        next_population = []

        # Elitism: keep the top 10% unchanged so we never lose progress
        elite_count = max(1, self.population_size // 10)
        for i in range(elite_count):
            next_population.append(self.population[i].clone())

        # Breed the rest from the top half of the population
        top_half = self.population[: max(2, self.population_size // 2)]
        while len(next_population) < self.population_size:
            parent_a = random.choice(top_half)
            parent_b = random.choice(top_half)
            child = crossover(parent_a, parent_b)
            child = mutate(child)
            next_population.append(child)

        self.population = next_population
        return result

    def reset(self):
        self.__init__(self.population_size)
