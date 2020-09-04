import numpy as np
from time import time

from .strips import Strips


class Pipeline:

    def __init__(self, algorithm, solver):

        self.algorithm = algorithm
        self.solver = solver
        self.comp_time = 0
        self.opt_time = 0


    def run(self, strips, d=0, verbose=False):

        t0 = time()
        self.algorithm.run(strips, d, verbose=verbose)
        self.comp_time = time() - t0
        self.solver.solve(self.algorithm.compatibilities)
        self.opt_time = time() - self.comp_time - t0
        if self.algorithm.name() in ['proposed', 'sib18']:
            return self.solver.solution, self.algorithm.compatibilities, self.algorithm.displacements
        return self.solver.solution, self.algorithm.compatibilities
