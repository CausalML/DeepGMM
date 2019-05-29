import numpy as np


class AbstractBaseline(object):
    def __init__(self):
        self._model = None
        self._fitted_with_context = None
        self._x_dim = None

    def fit(self, x, y, z, context=None):
        self._check_arguments(x, y, z, context)
        self._fitted_with_context = (context is not None)
        self._x_dim = x.shape[1]
        self._fit(x, y, z, context)
        return self

    def _fit(self, x, y, z, context):
        raise NotImplementedError()

    def display(self):
        pass

    def predict(self, x, context=None):
        # returns numpy arrays
        if self._model is None:
            raise AttributeError("Model has not been fit!")
        elif (context is not None) != self._fitted_with_context:
            map = {True: "without", False: "with"}
            raise AttributeError("Model was fitted " +
                                 map[self._fitted_with_context] +
                                 " context features, but now got called "
                                 + map[not self._fitted_with_context] + ".")
        elif self._x_dim != x.shape[1]:
            raise ValueError(
                "Was fitted with %d-dimensional x, but now got called with %-d-dimensional x" % (
                self._x_dim, x.shape[1]))
        else:
            result = self._predict(x, context)
            if result.ndim != 2:
                raise ValueError("Class returned incorrect shape in predict()!")
            return result

    def _predict(self, x, context):
        raise NotImplementedError()

    @staticmethod
    def augment(var, context):
        if context is not None:
            return np.concatenate([context, var], axis=1)
        else:
            return var

    @staticmethod
    def add_constant(var):
        return np.append(var, np.ones_like(var[:, 0:1]), axis=1)

    @staticmethod
    def arr2str(array):
        return np.array2string(array,
                               formatter={'float_kind': '{0:.3f}'.format})

    @staticmethod
    def _check_arguments(x, y, z, context):
        all_var = [x, y, z] + ([context] if context is not None else [])

        if not all([isinstance(var, np.ndarray) for var in all_var]):
            raise ValueError("All variables need to be numpy arrays")
        if not all([var.ndim == 2 for var in all_var]):
            raise ValueError(
                "All variables need to be 2-dimensional, but dimensions given are " + str(
                    [var.ndim for var in all_var]))
        if not all([var.shape[0] == x.shape[0] for var in all_var]):
            raise ValueError(
                "All variables need to have same number of samples, but sizes given are " + str(
                    [var.shape for var in all_var]))
        if not y.shape[1] == 1:
            raise ValueError(
                "Outcome variable needs to be n x 1, but given " + str(y.shape))
