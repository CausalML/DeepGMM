import numpy as np

from scenarios.abstract_scenario import AbstractScenario

class Standardizer(AbstractScenario):
    def __init__(self, scenario):
        AbstractScenario.__init__(self)
        self._scenario = scenario

        self._mean = None
        self._std = None

    def generate_data(self, num_data, **kwargs):
        x, z, y, g, w = self._scenario.generate_data(num_data, **kwargs)
        if self._mean is None:
            self._mean = y.mean()
            self._std = y.std()

        y = self.normalize(y)
        g = self.normalize(g)

        return x, z, y, g, w

    def normalize(self, y):
        return (y - self._mean) / self._std

    def denormalize(self, y):
        return y*self._std + self._mean

class HingeLinearScenario(AbstractScenario):
    def __init__(self, slope_1=3.0, slope_2=0.15,
                 intercept_1=0.0, intercept_2=0.0):
        AbstractScenario.__init__(self)
        self.slope_1 = slope_1
        self.slope_2 = slope_2
        self.intercept_1 = intercept_1
        self.intercept_2 = intercept_2

    def generate_data(self, num_data, **kwargs):

        epsilon = np.random.normal(size=(num_data,1))
        eta = np.random.normal(size=(num_data,1))
        z = np.random.normal(size=(num_data, 2))

        x = z[:, 0:1] + z[:, 1:] + epsilon * 2.0
        g = self._true_g_function_np(x)
        y = g + epsilon * 5.0 + eta * (2.0 ** -0.5)
        
        return x, z, y, g, x

    def _true_g_function_np(self, x):
        return np.maximum(self.slope_1 * x + self.intercept_1,
                          self.slope_2 * x + self.intercept_2)


class Zoo(HingeLinearScenario):
    def __init__(self, name='linear'):
        HingeLinearScenario.__init__(self)
        self._function_name = name

    def _generate_random_pw_linear(self, lb=-2, ub=2, n_pieces=5):
        splits = np.random.choice(np.arange(lb, ub, 0.1), n_pieces-1, replace=False)
        splits.sort()
        slopes = np.random.uniform(-4, 4, size=n_pieces)
        start = []
        start.append(np.random.uniform(-1, 1))
        for t in range(n_pieces-1):
            start.append(start[t] + slopes[t] * (splits[t] - (lb if t==0 else splits[t-1])))
        return lambda x: [start[ind] + slopes[ind] * (x - (lb if ind==0 else splits[ind-1])) for ind in [np.searchsorted(splits, x)]][0]
    
    def _true_g_function_np(self, x):
        func = self._function_name
        if func=='abs':
            return np.abs(x)
        elif func=='2dpoly':
            return -1.5 * x + .9 * (x**2)
        elif func=='sigmoid':
            return 2/(1+np.exp(-2*x))
        elif func=='sin':
            return np.sin(x)
        elif func=='step':
            return 1. * (x<0) + 2.5 * (x>=0)
        elif func=='3dpoly':
            return -1.5 * x + .9 * (x**2) + x**3
        elif func=='linear':
            return x
        elif func=='rand_pw':
            pw_linear = self._generate_random_pw_linear()
            return np.reshape(np.array([pw_linear(x_i) for x_i in x.flatten()]), x.shape)
        else:
            raise NotImplementedError()

class AGMMZoo(Zoo):
    def __init__(self, g_function='linear', two_gps=True, n_instruments=2, iv_strength=0.5):
        HingeLinearScenario.__init__(self)
        self._function_name = g_function
        self._two_gps = two_gps
        self._n_instruments = n_instruments
        self._iv_strength = iv_strength
        
    def generate_data(self, num_data, **kwargs):
        confounder = np.random.normal(0, 1, size=(num_data, 1))
        z = np.random.uniform(-3, 3, size=(num_data, self._n_instruments))
        iv_strength = self._iv_strength
        if self._two_gps:
            x = 2 * z[:, 0].reshape(-1, 1) * (z[:, 0] > 0).reshape(-1, 1) * iv_strength \
                + 2 * z[:, 1].reshape(-1, 1) * (z[:, 1] < 0).reshape(-1, 1) * iv_strength \
                + 2 * confounder * (1 - iv_strength) + \
                np.random.normal(0, .1, size=(num_data, 1))
        else:
            x = 2 * z[:, 0].reshape(-1, 1) * iv_strength \
                + 2 * confounder * (1 - iv_strength) + \
                np.random.normal(0, .1, size=(num_data, 1))
        g = self._true_g_function_np(x) 
        y = g + 2 * confounder + \
            np.random.normal(0, .1, size=(num_data, 1))
        
        return x, z, y, g, x

class HeaviSideScenario(HingeLinearScenario):
    def __init__(self, step_height=10.0):
        HingeLinearScenario.__init__(self)
        self._step_height = step_height

    def _true_g_function(self, x):
        return np.heaviside(x, 1) * self._step_height
