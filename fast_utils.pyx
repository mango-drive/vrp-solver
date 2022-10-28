from random import randint
from debug_utils import log, timeit

SPEED = 10
DEPOT = 0

def union(A, B):
    U = {}
    for c in A.keys():
        row = []
        U[c] = A[c].union(B[c])
    return U

def adj(x):
    L = len(x)
    A = {c: set((x[(n-1)%L], x[(n+1)%L])) for n, c in enumerate(x)}
    return A

def sample(l):
    return l[randint(0, len(l) - 1)]



@timeit(True)
def get_neighbors2(M, xy):
    # assuming Matrix is square
    N = len(M)
    x, y = xy
    nidxs = []
    ndirs = ((0, 1), (1, 0), (0, -1), (-1, 0))
    for ndir in ndirs:
        nidxs.append(((x + ndir[0]) % N, (y + ndir[1]) % N))

    neighs = [M[x][y] for (x, y) in nidxs]
    return neighs

@timeit(True)
@log(False)
def crossover_ero3(p1, p2, algo=None):

    def _in(C, v):
        for c in C: 
            if v == c: return True
        return False

    A = adj(p1); B = adj(p2); U = union(A, B)

    C = []; cp = (p1, p2)[randint(0, 1)]; c = cp[0]

    table = []
    while len(C) < len(cp):
        row = []
        C.append(c) 
        
        U = {n: U[n] - {c} for n in U.keys()};                  

        LUN = len(U[c]);                                           
                     
        if len(U[c]) > 0:                                   
            nbs = {n: U[n] for n in U[c] if n in U};              
            
            if len(nbs) == 1: 
                c = list(nbs.keys())[0]
            else:
                # each node has a maximum of 4 neighbors after Union
                MAX = 4; min = MAX; min_is = []
                for k, v in nbs.items():
                    L = len(v)
                    if L < min: min=L; min_is=[k]
                    if L == min: min_is.append(k)
                       
                c = min_is[0] if len(min_is) == 1 else sample(min_is);  
        else:
            not_visited = [k for k in U.keys() if not _in(C, k)]    
            if len(not_visited): c = sample(not_visited)

    return C
                                       

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
def evaluate_cGA3(S, memo=[], xy=None, algo=None):
    if len(memo) and xy:
        x, y = xy
        if memo[x][y] != None:
            return memo[x][y]

    t = E = q = m = 0           # fitness, last visited idx, vehicle capacity, time


    params = algo.parameters

    for r in S:
        for n in r:
            row = []
            node = algo.C[n]

            dmn = algo.D[m][n]                           # cost
            tmn = dmn / SPEED                            # time to customer
            tp = t+tmn if t+tmn > node.ei else node.ei   # arrival time at customer        
            if n == 0: t = q = 0                         # return to depot
            else: t, q = tp, q + node.q                  # time and capacity update

            # Costs
            ot = params['m']*(tp - node.li) if tp > node.li else 0
            oc = params['l']*(q - algo.Q) if q > algo.Q else 0
            E += dmn + oc + ot

            m = n

    if len(memo) and xy:
        x, y = xy
        memo[x][y] = E
    return E

@timeit(True)
def decode(c, algo, decoded_memo=None):
    key = ",".join(c.X)
    if decoded_memo:  
        if key in decoded_memo:
            return decoded_memo[key]
    R = []
    r = []
    # break path into subpaths, delimited by stop signals
    N = len(c.as_list())
    for i, x in enumerate(c.as_list()):
        if len(r) == 0:
            r.append(DEPOT)
            if x == DEPOT: continue
        
        if x != 0: r.append(x)

        if x == 0 or i == N - 1:
            r.append(0); R.append(r.copy()) 
            r = []

    if decoded_memo:
        decoded_memo[key] = R
        
    return R

