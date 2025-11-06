import random
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score


def initialize_population(population_size, params_dt):
    """
    Generate the initial population for the genetic algorithm.

    Parameters
    ----------
    population_size : int
        Number of individuals in the population.
    params_dt : dict
        Dictionary of parameter names and possible values for decision tree hyperparameters.

    Returns
    -------
    list of dict
        Population represented as a list of individuals,
        where each individual is a dictionary of hyperparameters.
    """
    return [
        {key: random.choice(values) for key, values in params_dt.items()}
        for _ in range(population_size)
    ]


def crossover(parent1, parent2):
    """
    Perform single-point crossover between two parents.

    Parameters
    ----------
    parent1 : dict
        Hyperparameters of the first parent.
    parent2 : dict
        Hyperparameters of the second parent.

    Returns
    -------
    (dict, dict)
        Two offspring individuals generated from the parents.
    """
    crossover_point = np.random.randint(1, len(parent1))
    items1 = list(parent1.items())
    items2 = list(parent2.items())

    child1 = dict(items1[:crossover_point] + items2[crossover_point:])
    child2 = dict(items2[:crossover_point] + items1[crossover_point:])

    return child1, child2


def mutate(individual, mutation_rate, params_dt):
    """
    Apply mutation to an individual with a given mutation rate.

    Parameters
    ----------
    individual : dict
        A dictionary representing hyperparameters.
    mutation_rate : float
        Probability of mutating each hyperparameter.
    params_dt : dict
        Mapping of parameter names to allowed values.

    Returns
    -------
    dict
        Mutated individual.
    """
    mask = np.random.rand(len(individual)) < mutation_rate
    genes = [item for i, item in enumerate(individual.items()) if mask[i]]

    new_genes = [(param, random.choice(params_dt[param])) for param, value in genes]

    for param, value in new_genes:
        individual[param] = value

    return individual


def calculate_fitness(X_train, X_val, train_targets, val_targets, parameters):
    """
    Compute the fitness of an individual using ROC AUC on validation data.

    Parameters
    ----------
    X_train : array-like
        Training features.
    X_val : array-like
        Validation features.
    train_targets : array-like
        Training labels.
    val_targets : array-like
        Validation labels.
    parameters : dict
        Hyperparameters for the DecisionTreeClassifier.

    Returns
    -------
    float
        ROC AUC score on the validation set.
    """
    model = DecisionTreeClassifier(random_state=42, **parameters)
    model.fit(X_train, train_targets)
    y_proba = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(val_targets, y_proba)


def genetic_algorithm_improved(
    X_train,
    X_val,
    train_targets,
    val_targets,
    params_dt,
    population_size=10,
    generations=10,
    mutation_rate=0.1
):
    """
    Perform hyperparameter optimization using a genetic algorithm.

    Parameters
    ----------
    X_train : array-like
        Training dataset.
    X_val : array-like
        Validation dataset.
    train_targets : array-like
        Labels for training data.
    val_targets : array-like
        Labels for validation data.
    params_dt : dict
        Hyperparameter search space.
    population_size : int, optional
        Number of individuals in the population.
    generations : int, optional
        Number of evolutionary generations.
    mutation_rate : float, optional
        Probability of mutation.

    Returns
    -------
    dict
        Best hyperparameters found during evolution.
    float
        Corresponding ROC AUC score.
    """
    population = initialize_population(population_size, params_dt)
    best_overall_parameters = None
    best_overall_score = -np.inf

    for generation in range(generations):

        fitness_scores = np.array([
            calculate_fitness(X_train, X_val, train_targets, val_targets, params)
            for params in population
        ])

        max_score_gen = np.max(fitness_scores)
        if max_score_gen > best_overall_score:
            best_overall_score = max_score_gen
            best_overall_parameters = population[np.argmax(fitness_scores)]

        new_population = [best_overall_parameters]  # Elite preservation

        total_fitness = np.sum(fitness_scores)
        probabilities = (
            fitness_scores / total_fitness if total_fitness > 0
            else np.ones(population_size) / population_size
        )

        num_couples = (population_size - 1) // 2

        for _ in range(num_couples):
            parent_indices = np.random.choice(
                range(population_size), size=2, p=probabilities, replace=True
            )

            parent1 = population[parent_indices[0]]
            parent2 = population[parent_indices[1]]

            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate, params_dt)
            child2 = mutate(child2, mutation_rate, params_dt)

            new_population.extend([child1, child2])

        if len(new_population) < population_size:
            clone_idx = np.random.choice(range(population_size), p=probabilities)
            clone = population[clone_idx].copy()
            new_population.append(mutate(clone, mutation_rate, params_dt))

        population = np.array(new_population[:population_size])

    return best_overall_parameters, best_overall_score
