import csv
import json
from dataclasses import dataclass
from datetime import timedelta
from math import sqrt
from random import randint
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg

from math import inf

from vrp import Customer

def find_min_index(mat, WIDTH):
    min = inf
    min_xy = None
    for x in range(WIDTH):
        for y in range(WIDTH):
            if mat[x][y] < min:
                min = mat[x][y]
                min_xy = (x, y)
    return min_xy


def union(A, B):
    U = {}
    for c in A.keys():
        row = []
        U[c] = A[c].union(B[c])
    return U


def adj(x):
    L = len(x)
    A = {c: set((x[(n-1) % L], x[(n+1) % L])) for n, c in enumerate(x)}
    return A


def sample(l):
    return l[randint(0, len(l) - 1)]


def get_neighbors(M, xy):
    # assuming Matrix is square
    N = len(M)
    x, y = xy
    nidxs = []
    ndirs = ((0, 1), (1, 0), (0, -1), (-1, 0))
    for ndir in ndirs:
        nidxs.append(((x + ndir[0]) % N, (y + ndir[1]) % N))

    neighs = [M[x][y] for (x, y) in nidxs]
    return neighs


def manhattan(l):
    # l: [l1, ... ln]
    N = len(l)
    D = [[[0]*len(l) for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            D[i][j] = abs(l[j][0] - l[i][0])
    return D


def euclidean(l):
    # l: [l1, ... ln]
    N = len(l)
    D = [[[0]*len(l) for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            D[i][j] = sqrt((l[j][0] - l[i][0]) ** 2 + (l[j][1] - l[i][1]) ** 2)

    return D


def load_csv(file, delimiter=','):
    # read from csv line by line, rstrip helps to remove '\n' at the end of line
    f = open(file, 'r')
    lines = [line.rstrip() for line in f]

    results = []
    for line in lines[1:]:
        words = line.split(delimiter)  # get each item in one line
        words = [w.strip() for w in words]
        results.append([float(w) for w in words if w != ''])

    return results


def load_solomun_problem(file):
    problem = {}

    f = open(file, 'r')
    lines = [line.rstrip() for line in f]
    lines = [line.split() for line in lines]

    TITLE_LINE = 0
    VEHICLE_LINE = 4
    FIRST_CUSTOMER_LINE = 9

    problem['title'] = lines[TITLE_LINE]
    problem['num_veh'] = int(lines[VEHICLE_LINE][0])
    problem['capacity'] = int(lines[VEHICLE_LINE][1])

    customers = []
    for line in lines[FIRST_CUSTOMER_LINE:]:
        line = [int(n) for n in line]
        customers.append(Customer(*line))

    problem['customers'] = customers

    return problem


def load_problem_definition(file):

    return json.load(open(file))


def load_csv_as_dict(file):
    # read from csv line by line, rstrip helps to remove '\n' at the end of line
    f = open(file, 'r')
    lines = [line.rstrip() for line in f]
    lines = [r.split(',') for r in lines]

    d = {}
    headers = lines[0]
    headers = [h.lstrip() for h in headers]
    for n, h in enumerate(headers):
        d[h] = []
        for l in lines[1:]:
            d[h].append(float(l[n]))
    return d


def write_ldict_to_csv(d, file):
    with open(file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=d[0].keys())
        writer.writeheader()
        for data in d:
            writer.writerow(data)


def export_vehicle_routes(vehicle_routes, file):
    tls = []
    for k, vr in vehicle_routes.items():
        tl = []
        for s in vr:  # step in vehicle route
            tl.append([s.lat, s.lng])
        tls.append({'vehicle': k, 'steps': tl})
    write_ldict_to_csv(tls, file)


def plot_customers(plot, cs):
    x = [c.lat for c in cs]
    y = [c.lng for c in cs]
    ns = [str(c.number) for c in cs]
    scatter = pg.ScatterPlotItem(x, y)
    plot.addItem(scatter)
    for i, n in enumerate(ns):
        label = pg.TextItem(text=n, color=('k'))
        label.setPos(x[i], y[i])
        plot.addItem(label)


def plot_line_between(plot, c1, c2, color):
    x = [c1.lat, c2.lat]
    y = [c1.lng, c2.lng]
    line = pg.PlotCurveItem(x, y, pen=color)
    plot.addItem(line)


def plot_path(plot, xys, path, color='b'):
    for sp in path:
        for idx in range(len(sp))[:-1]:
            c1, c2 = xys[sp[idx]], xys[sp[idx+1]]
            plot_line_between(plot, c1, c2, color)
        label = pg.TextItem(f"path={str(path)}")
        plot.addItem(label)


def plot_problem_solution(plot, path, algo):
    plot_path(plot, algo.C, path)
    plot_customers(plot, algo.C)

def plot_fitness_over_generations(evolution):
    x = []
    y = []
    for generation in evolution:
        x.append(generation['generation'])
        y.append(generation['best_fitness'])

    plt.plot(x, y)
    plt.show()

