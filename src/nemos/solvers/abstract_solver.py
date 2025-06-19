import abc


class AbstractSolver(abc.ABC):
    @abc.abstractmethod
    def init_state(self, init_params, *args):
        pass

    @abc.abstractmethod
    def update(self, params, state, *args):
        pass

    @abc.abstractmethod
    def run(self, init_params, *args):
        pass
