from game_objectives.abstract_objective import AbstractObjective


class AbstractLearning(object):
    def __init__(self, game_objective, g, f):
        assert isinstance(game_objective, AbstractObjective)
        self.game_objective = game_objective
        self.g = g
        self.f = f

    def fit_from_tensors(self, x_train, y_train, z_train, x_dev, z_dev, y_dev,
                         g_dev=None, w_train=None):
        raise NotImplementedError()

    def fit(self, scenario):
        """
        fits model using PyTorch tensors x, z, and y, and possibly some
        extra arguments in args
        """
        raise NotImplementedError()
