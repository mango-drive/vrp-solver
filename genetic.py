
import random
from dataclasses import dataclass
from math import inf
from random import randint
from typing import List

import pyqtgraph as pg

from debug_utils import log, timeit, profile
from fast_utils import (crossover_ero3, evaluate_cGA3, get_neighbors2, decode,
                        sample)
from table import tabulate
from utils import euclidean, load_csv, load_solomun_problem, plot_customers, plot_problem_solution
from vrp import Customer

import time

SPEED = 1
WIDTH = 15
TIME_LIMIT = 1000
ITERATIONS = 300
N_EQUAL_BEST = 50

DEPOT = 0
LATE_PENALTY = 10
DEP_UNREACH_PENALTY = 10
CAP_EXCEEDED_PENALTY = 10


class Chromosome:
    X: List[str]       # string representation
    int_X: List[int]

    @timeit(True)
    def __init__(self, X: List):
        # assert uniqueness
        seen = set()
        for x in X:
            if x in seen:
                assert False
            else:
                seen.add(x)

        self.X = [str(x) for x in X]

    @timeit(True)
    def as_list(self) -> List:
        self.int_X = list([int(x) for x in self.X])
        return self.int_X

    def __repr__(self) -> str:
        return 'X(' + ','.join(self.X) + ')'

    def __str__(self) -> str:
        return self.__repr__()


@dataclass
class Conditions:
    start_time: float = 0
    time_elapsed = 0
    iterations = 0
    n_equal_exp = 0
    best = None

    def set_best(self, best):
        if self.best is not None:
            if self.best == best:
                self.n_equal_exp += 1
        self.best = best


@dataclass
class Algo:
    C: List[Customer]
    D: List[list]
    T: int
    Q: int
    conditions = Conditions()
    should_print = False
    parameters = {}
    performance = {}

    def print(self, *args, **kwargs):
        if self.should_print:
            print(*args, **kwargs)

    def __repr__(self):
        return "Algo(params = {self.parameters})"

    def __str__(self):
        return self.__repr__()


@timeit(True)
def init_population_1d(pop_size, cust_locs, start_pop=None, stop_perc=0.25):
    if start_pop is None:
        start_pop = []

    assert len(start_pop) < pop_size

    nodes = list(range(len(cust_locs)))
    # stop signals
    N = randint(0, int(stop_perc * len(nodes)))
    stps = [-1 * s for s in list(range(1, N))]
    nodes.extend(stps)

    Xs = [random.sample(nodes, len(nodes))
          for _ in range(pop_size - len(start_pop))]

    for x in Xs:
        start_pop.append(Chromosome(x))

    assert len(start_pop) <= pop_size
    return start_pop


@timeit(True)
@log(False)
def init_population_2d(N, C, *args):
    pop_2d = []
    for _ in range(N):
        pop_2d.append(init_population_1d(N, C, *args))
    return pop_2d


@timeit(True)
@log(False)
def constraints_satisfied(node, V, tp, tpp, T, q, Q):
    if node in V:
        return (False, 'visited')
    if tp > node.li:
        return (False, 'late arr')
    if tpp > T:
        return (False, 'dep unreach')
    if q + node.q > Q:
        return (False, 'cap ex')
    return (True, 'del poss')


@timeit(True)
@log(False)
def mutate(C):
    def swap(C, m, n):
        t = C[m]
        C[m] = C[n]
        C[n] = t
        return C

    def inversion(C, m, n):
        a = min(m, n)
        b = max(m, n)
        while a < b:
            t = C[a]
            C[a] = C[b]
            C[b] = t
            a += 1
            b -= 1
        return C

    def insertion(C, m, n):
        dir = -1 if (n - m) < 0 else 1
        while m != n:
            t = C[m + dir]
            C[m+dir] = C[m]
            C[m] = t
            m += dir
        return C

    fns = [swap, inversion, insertion]
    fn = sample(fns)
    m = randint(0, len(C)-1)
    n = randint(0, len(C)-1)
    return fn(C, m, n)


@timeit(True)
@log(False)
def evolve_cGA(population, algo=None, decoded_memo=None):
    table = []
    aux_pop = [[0 for _ in range(WIDTH)] for _ in range(WIDTH)]

    Es = [[None for _ in range(WIDTH)] for _ in range(WIDTH)]

    for x in range(WIDTH):
        for y in range(WIDTH):
            row = []
            n_list = get_neighbors2(population, (x, y))
            row.append(n_list)

            n_Es = [evaluate_cGA3(decode(X, algo=algo, decoded_memo=decoded_memo), memo=Es, xy=(x, y), algo=algo)
                    for X in n_list]
            row.append([n_Es])

            # binary tournament
            min_i = 0
            min = max(n_Es)
            for i, E in enumerate(n_Es):
                if E < min:
                    min = E
                    min_i = i
            p1 = n_list[min_i]

            # local select
            p2 = population[x][y]
            row.append([p1, p2])
            # recombination
            aux = crossover_ero3(p1.as_list(), p2.as_list(), algo=algo)
            row.append(aux)
            # mutation
            aux = Chromosome(mutate(aux))

            aux_E = evaluate_cGA3(decode(aux, algo=algo, decoded_memo=decoded_memo),
                                  memo=Es, xy=(x, y), algo=algo)
            local_E = evaluate_cGA3(decode(p2, algo=algo, decoded_memo=decoded_memo),
                                    memo=Es, xy=(x, x), algo=algo)
            if aux_E < local_E:
                aux_pop[x][y] = aux
            else:
                aux_pop[x][y] = p2

            table.append(row)

    population = aux_pop

    algo.print(tabulate(table, headers=["n_list", "Es", "parents", "aux"]))

    return population, Es


def terminating(conditions: Conditions):
    if conditions.time_elapsed > TIME_LIMIT:
        print("\n\nTIME LIMIT EXCEEDED")
        return True
    if conditions.iterations > ITERATIONS:
        print("\n\nITERATION LIMIT EXCEEDED")
        return True
    if conditions.n_equal_exp > N_EQUAL_BEST:
        print("\n\nN_EQUAL_BEST LIMIT EXCEEDED")
        return True


@profile
def evolution(population, algo: Algo = None):

    algo.iterations = 0
    while not terminating(algo.conditions):
        if algo.conditions.iterations % 100 == 0:
            print(f"Generation: ", algo.conditions.iterations)

        decoded_memo = {"first": None}
        population, Es = evolve_cGA(
            population, algo=algo, decoded_memo=decoded_memo)
        algo.conditions.time_elapsed = time.time() - algo.conditions.start_time
        algo.conditions.iterations += 1

        # find the best solution
        min = inf
        min_xy = None
        for x in range(WIDTH):
            for y in range(WIDTH):
                if Es[x][y] < min:
                    min = Es[x][y]
                    min_xy = (x, y)
        x, y = min_xy
        best_C = population[x][y]
        best_E = Es[x][y]

        algo.conditions.set_best(best_E)

    return best_C


def init(file):
    problem = load_solomun_problem(file)

    C = problem['customers']
    P = init_population_2d(WIDTH, C)

    lats = [c.lat for c in C]
    lngs = [c.lng for c in C]
    xy = list(zip(lats, lngs))
    D = euclidean(xy)

    algo = Algo(C, D, 24, 40)
    algo.parameters['l'] = 100
    algo.parameters['m'] = 100

    algo.conditions.start_time = time.time()

    return P, algo


def solve(file):
    P, algo = init(file)
    best = evolution(P, algo=algo)
    print(f"\n\n\nBEST SOLUTION: \n {best}")
    print("\n PERFORMANCE")
    print(algo.performance)
    return decode(best, algo=algo), algo


def plot_route(path, algo):
    plt = pg.plot()
    plt.setBackground('w')
    plot_problem_solution(plt, path, algo)
    pg.exec()


if __name__ == "__main__":
    best, algo = solve("data/text/C101_simple.txt")
    plot_route(best, algo)
