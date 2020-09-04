import os
import numpy as np
import re
from warnings import warn
import subprocess
import uuid
from .solver import Solver

# ATSP: Find a HAMILTONIAN CIRCUIT (Tour)  whose global cost is minimum (Asymmetric Travelling Salesman Problem: ATSP)
# https://github.com/coin-or/metslib-examples/tree/master/atsp
# http://www.localsolver.com/documentation/exampletour/tsp.html
# http://or.dei.unibo.it/research_pages/tspsoft.html

class SolverConcorde(Solver):

    def __init__(self, maximize=False, max_precision=3, seed=None):

        self.solution = None
        self.maximize = maximize
        self.max_precision = max_precision
        self.seed = seed


    def solve(self, instance, fname=None):

        instance = np.array(instance)
        if self.maximize:
            np.fill_diagonal(instance, 0)
            instance = instance.max() - instance # transformation function (similarity -> distance)
        instance = np.pad(instance, ((0, 1), (0, 1)), mode='constant', constant_values=0) # dummy node
        num_cities = instance.shape[0]

        solverATSP = ConcordeATSPSolver(self.max_precision, seed=self.seed)
        solution = solverATSP.solve(instance, fname).solution
        if solution is None:
            self.solution = None
            return self

        # remove repeated element
        solution = solution[: -1]
        # removing dummy node from solution
        dummy_idx = solution.index(num_cities - 1)

        self.solution = solution[dummy_idx + 1 :] + solution[: dummy_idx]
        return self


    def id(self):

        return 'Concorde'


class ConcordeATSPSolver:
    ''' Solver for ATSP using Concorde.'''

    @staticmethod
    def load_tsplib(filename):
        ''' Load a tsplib instance for testing. '''

        lines = open(filename).readlines()
        regex_non_numeric = re.compile(r'[^\d]+')
        n = int(next(regex_non_numeric.sub('', line)
                     for line in lines if line.startswith('DIMENSION')))
        start = next(i for i, line in enumerate(lines) if line.startswith('EDGE_WEIGHT_SECTION'))
        end = next(i for i, line in enumerate(lines) if line.startswith('EOF'))
        matrix = np.array([float(v) for v in ' '.join(lines[start + 1:end]).split()], dtype=np.int32).reshape((n, -1))
        return matrix

    @staticmethod
    def dump_tsplib(matrix, filename):
        ''' Dump a tsplib instance.

        For detais on tsplib format, check: http://ftp.uni-bayreuth.de/math/statlib/R/CRAN/doc/packages/TSP.pdf
        '''

        assert matrix.ndim == 2
        assert matrix.shape[0] == matrix.shape[1]

        template = '''NAME: {name}
TYPE: TSP
COMMENT: {name}
DIMENSION: {n}
EDGE_WEIGHT_TYPE: EXPLICIT
EDGE_WEIGHT_FORMAT: FULL_MATRIX
EDGE_WEIGHT_SECTION
{matrix_str}EOF'''

        name = os.path.splitext(os.path.basename(filename))[0]
        n = matrix.shape[0]

        # space delimited string
        matrix_str = ' '
        for row in matrix:
            matrix_str += ' '.join([str(val) for val in row])
            matrix_str += '\n'
        open(filename, 'w').write(template.format(
            **{'name': name, 'n': n, 'matrix_str': matrix_str}))


    def __init__(self, max_precision=3, seed=None):
        ''' Class constructor. '''

        self.max_precision = max_precision
        self.seed = seed
        self.solution = None
        self.cost = 0


    def _run_concorde(self, matrix, fname):
        ''' Run Concorde solver for instance named with filename.

        Check https://github.com/mhahsler/TSP/blob/master/R/tsp_concorde.R for some tricks.
        '''

        # fix negative values
        if matrix.min() < 0: matrix -= matrix.min()

        # fix small values
        if matrix.max() < 1: matrix += 1 - matrix.max()

        for i in range(self.max_precision):
            if np.all(np.modf(matrix)[0] == 0) or (matrix * 10).max() >= (2 ** 31 - 1):
                print('solverconcorde.py: precision reduced to {}'.format(i))
                break
            matrix = matrix * 10

        n = matrix.shape[0]

        assert matrix.max() < 2 ** 31 - 1

        # dump matrix in int32 format
        #tsp_filename = '/tmp/{}.tsp'.format(str(uuid.uuid4()))
        tsp_filename = fname if fname is not None else '/tmp/{}.tsp'.format(str(uuid.uuid4()))
        ConcordeATSPSolver.dump_tsplib(matrix.astype(np.int32), tsp_filename)

        # call Concorde solver
        curr_dir = os.path.abspath('.')
        dir_ = os.path.dirname(tsp_filename)
        os.chdir(dir_)
        #print(os.path.abspath('.'))

        sol_filename = '{}/{}.sol'.format(dir_, os.path.splitext(os.path.basename(tsp_filename))[0])
        seed = self.seed
        if seed is not None:
            cmd = ['concorde', '-s', str(seed), '-o', sol_filename, tsp_filename]
        else:
            cmd = ['concorde', '-o', sol_filename, tsp_filename]
        try:
            # print('1')
            with open(os.devnull, 'w') as devnull:
                # print('2')
                try:
                    # print('3')
                    output = subprocess.check_output(cmd, stderr=devnull)
                    # print('-----------------------------------------------')
                    #output = subprocess.check_output(cmd) #, stderr=devnull)
                    # print(output, '++++++++++++++++++++++++++++++++++++++++++')
                except subprocess.CalledProcessError:
                    os.chdir(curr_dir)
                    return None
        except OSError as exc:
            # print('4', exc)
            if 'No such file or directory' in str(exc):
                raise Exception('ERROR: Concorde solver not found.')
            raise exc
        os.chdir(curr_dir)
        tour = [int(v) for v in open(sol_filename).read().split()[1:]]
        return tour


    def _atsp_to_tsp(self, C):
        '''
        Reformulate an asymmetric TSP as a symmetric TSP:
        "Jonker and Volgenant 1983"
        This is possible by doubling the number of nodes. For each city a dummy
        node is added: (a, b, c) => (a, a', b, b', c, c')

        distance = "value"
        distance (for each pair of dummy nodes and pair of nodes is INF)
        distance (for each pair node and its dummy node is -INF)
        ------------------------------------------------------------------------
          |a    |b    |c    |a'   |b'   |c'   |
        a |0    |INF  |INF  |-INF |dBA  |dCA  |
        b |INF  |0    |INF  |dAB  |-INF |dCB  |
        c |INF  |INF  |0    |dAC  |dBC  |-INF |
        a'|-INF |dAB  |dAC  |0    |INF  |INF  |
        b'|dBA  |-INF |dBC  |INF  |0    |INF  |
        c'|dCA  |dCB  |-INF |INF  |INF  |0    |

        @return: new symmetric matrix

        [INF][C.T]
        [C  ][INF]
        '''

        n = C.shape[0]
        n_tilde = 2 * n
        C_tilde = np.empty((n_tilde, n_tilde), dtype=np.float64)
        C_tilde[:, :] = np.inf
        np.fill_diagonal(C_tilde, 0.0)
        C_tilde[n:, :n] = C
        C_tilde[:n, n:] = C.T
        np.fill_diagonal(C_tilde[n:, :n], -np.inf)
        np.fill_diagonal(C_tilde[:n, n:], -np.inf)
        return C_tilde


    def solve(self, matrix, fname):
        ''' Solve ATSP instance. '''

        matrix_ = self._atsp_to_tsp(matrix)
        masked_matrix = np.ma.masked_array(
            matrix_, mask=np.logical_or(matrix_ == np.inf, matrix_ == -np.inf)
        )
        min_val, max_val = masked_matrix.min(), masked_matrix.max()

        # https://rdrr.io/cran/TSP/man/TSPLIB.html
        # Infinity = val +/- 2*range
        pinf = max_val + 2 * (max_val - min_val)
        ninf = min_val - 2 * (max_val - min_val)

        matrix_[matrix_ == np.inf] = pinf
        matrix_[matrix_ == -np.inf] = ninf

        # TSP solution
        solution_tsp = self._run_concorde(matrix_, fname=fname)
        if solution_tsp is None:
            self.solution = None
            self.cost = -1
        else:
            # convert to ATSP solution
            solution = solution_tsp[:: 2] + [solution_tsp[0]]

            # TSP - Infrastructure for the Traveling Salesperson Problem (Hahsler and Hornik)
            # "Note that the tour needs to be reversed if the dummy cities appear before and
            # not after the original cities in the solution of the TSP."
            N = matrix.shape[0]
            if solution_tsp[1] != N:
                solution = solution[:: -1]
            self.cost = matrix[solution[: -1], solution[1 :]].sum()
            self.solution = solution
        return self


# Testing
if __name__ == '__main__':
    '''
    Best known solutions for asymmetric TSPs
    br17: 39
    ft53: 6905
    ft70: 38673
    ftv33: 1286
    ftv35: 1473
    ftv38: 1530
    ftv44: 1613
    ftv47: 1776
    ftv55: 1608
    ftv64: 1839
    ftv70: 1950
    ftv90: 1579
    ftv100: 1788
    ftv110: 1958
    ftv120: 2166
    ftv130: 2307
    ftv140: 2420
    ftv150: 2611
    ftv160: 2683
    ftv170: 2755
    kro124: 36230
    p43: 5620
    rbg323: 1326
    rbg358: 1163
    rbg403: 2465
    rbg443: 2720
    ry48p: 14422
    '''

    import sys
    import time
    import matplotlib.pyplot as plt
    path = '/home/thiagopx/software/tsplib'

    print('TSPLIB instances')

    optimal_costs = {
        'br17': 39, 'ft53': 6905, 'ft70': 38673, 'ftv33': 1286, 'ftv35': 1473, 'ftv38': 1530, 'ftv44': 1613,
        'ftv47': 1776, 'ftv55': 1608, 'ftv64': 1839, 'ftv70': 1950, 'ftv90': 1579, 'ftv100': 1788,
        'ftv110': 1958, 'ftv120': 2166, 'ftv130': 2307, 'ftv140': 2420, 'ftv150': 2611, 'ftv160': 2683,
        'ftv170': 2755, 'kro124p': 36230, 'p43': 5620, 'rbg323': 1326, 'rbg358': 1163, 'rbg403': 2465,
        'rbg443': 2720, 'ry48p': 14422
    }
    filenames = [filename for filename in os.listdir(path) if filename.endswith('.atsp')]
    T = []
    N = []
    solver = ConcordeATSPSolver(max_precision=0)
    for filename in filenames[:5]:
        t0 = time.time()
        matrix = ConcordeATSPSolver.load_tsplib('{}/{}'.format(path, filename))
        cost = solver.solve(matrix).cost
        t = time.time() - t0
        N.append(matrix.shape[0])
        T.append(t)
        print('{} - {}/{} [{:.10f}s]'.format(filename, cost, optimal_costs[filename[: -5]], t))
    idx = np.argsort(N)
    N = [N[i] for i in idx]
    T = [T[i] for i in idx]
    plt.plot(N, T)
    plt.savefig('preliminar/tsplib-time.pdf', bbox_inches='tight')