import random
import matplotlib.pyplot as plt

# Define the target function
def target_function(x):
    return x**2 + x + 1

# Define primitive functions
def add(x, y):
    return x + y

def sub(x, y):
    return x - y

def mul(x, y):
    return x * y

# Create a list of primitive functions
primitives = [add, sub, mul]

# Generate a random expression tree
def gen_random_expr(max_depth, depth=0):
    if depth == max_depth or random.random() < 0.1:
        return "x"
    primitive = random.choice(primitives)
    return [primitive.__name__, gen_random_expr(max_depth, depth+1), gen_random_expr(max_depth, depth+1)]

# Evaluate the expression tree for a given input x
def eval_expr(expr, x):
    if expr == "x":
        return x
    func_name, a, b = expr
    if func_name == "add":
        return add(eval_expr(a, x), eval_expr(b, x))
    elif func_name == "sub":
        return sub(eval_expr(a, x), eval_expr(b, x))
    elif func_name == "mul":
        return mul(eval_expr(a, x), eval_expr(b, x))
    else:
        return 0

# Calculate the fitness of an individual based on the error between its output and the target function's output
def fitness(individual, points):
    error = 0
    for x in points:
        error += abs(eval_expr(individual, x) - target_function(x))
    return error

# Calculate the fitness of an individual using predefined points
def fitness_with_points(ind):
    points = [x / 10. for x in range(-10, 10)]
    return fitness(ind, points)

# Perform crossover between two parent trees
def crossover(parent1, parent2):
    if isinstance(parent1, list) and isinstance(parent2, list) and random.random() < 0.7:
        index = random.randint(1, len(parent1) - 1)
        child1 = parent1[:index] + parent2[index:]
        child2 = parent2[:index] + parent1[index:]
        
        # Simple depth control
        if depth(child1) > 5:
            child1 = parent1
        if depth(child2) > 5:
            child2 = parent2

        return child1, child2
    return parent1, parent2

# Calculate the depth of an expression tree
def depth(expr):
    if isinstance(expr, list):
        return 1 + max(depth(expr[1]), depth(expr[2]))
    return 0

# Mutate an individual tree
def mutate(individual, mutation_rate):
    if random.random() < mutation_rate:
        return gen_random_expr(4)
    if isinstance(individual, list):
        return [mutate(elem, mutation_rate) for elem in individual]
    return individual

# Run the genetic programming algorithm
def run_genetic_programming(population_size, crossover_probability, mutation_probability, number_of_generations, max_depth, elitism_size):
    best_fitness_values = []

    # Initialize the population
    population = [gen_random_expr(max_depth) for _ in range(population_size)]
    points = [x / 10. for x in range(-10, 10)]

    for generation in range(number_of_generations):
        print(f"Generation {generation+1}/{number_of_generations}")

        # Evaluate the population
        fitness_values = [fitness(ind, points) for ind in population]
        # Select the best individuals for elitism
        elite_individuals = sorted(population, key=fitness_with_points)[:elitism_size]

        # Select parents
        parents = []
        for _ in range((population_size - elitism_size) // 2):
            p1 = min(random.sample(population, 10), key=fitness_with_points)  # Increase selection pressure
            p2 = min(random.sample(population, 10), key=fitness_with_points)  # Increase selection pressure
            parents.append((p1, p2))

        # Create offspring
        offspring = []
        for parent1, parent2 in parents:
            if random.random() < crossover_probability:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2
            offspring.append(mutate(child1, mutation_probability))
            offspring.append(mutate(child2, mutation_probability))

        population = elite_individuals + offspring

        best_fitness = min(fitness_values)
        best_fitness_values.append(best_fitness)

    plt.plot(best_fitness_values)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("Fitness Curve")
    plt.show()

    best_individual = min(population, key=lambda ind: fitness(ind, points))
    print("Best individual:", best_individual)
    print("Fitness:", fitness(best_individual, points))

if __name__ == '__main__':
    population_size = 100
    crossover_probability = 0.8
    mutation_probability = 0.3
    number_of_generations = 50
    max_depth = 6
    elitism_size = 5

    # Run the genetic programming algorithm with the given hyperparameters
    run_genetic_programming(population_size, crossover_probability, mutation_probability, number_of_generations, max_depth, elitism_size)
