import abc


class Algorithm(object):
    ''' Base class for cost functions algorithms. '''

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def name(self):
        ''' Return the name of algorithm. '''


    def set_params(self, **kwargs):
        ''' Set the algorithm parameters described in kwargs.'''

        for key, value in kwargs.items():
            _key = key
            if _key not in self.__dict__:
                raise Exception('%s: invalid parameter %s.' % (__name__, key))
            self.__dict__[_key] = value


    def __call__(self, **kwargs):
        ''' Short call for run. '''

        return self.run(**kwargs)