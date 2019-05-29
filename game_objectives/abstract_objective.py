

class AbstractObjective(object):
    def __init__(self):
        pass

    def calc_objective(self, g, f, x, z, y):
        """
        returns tuple (g_objective, f_objective), which are to be minimized
        by f and g respectively
        assumes that all data passed as PyTorch tensors
        """
        raise NotImplementedError()
