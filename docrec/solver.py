import abc


class Solver(object):
    ''' Base class for solvers. '''

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def id(self):
        ''' Return the id (short name) of the solver. '''


    def set_params(self, **kwargs):
        ''' Set the algorithm parameters described in kwargs.'''

        for key, value in kwargs.items():
            _key = key
            if _key not in self.__dict__:
                raise Exception('%s: invalid parameter %s.' % (__name__, key))
            self.__dict__[_key] = value


    def __call__(self, **kwargs):
        ''' Short call for solve. '''

        return self.solve(**kwargs)