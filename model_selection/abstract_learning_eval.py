

class AbstractLearningEval(object):
    def __init__(self):
        pass

    def eval(self, x_train, z_train, y_train, x_dev, z_dev, y_dev,
             g, f, g_optimizer, f_optimizer, game_objective):
        raise NotImplementedError()

