from models.cnn_models import DefaultCNN

# if __name__ == "__main__":
#     from abstract_baseline import AbstractBaseline
#     from agmm.deep_gmm import DeepGMM
# else:
#     from .abstract_baseline import AbstractBaseline
#     from .agmm.deep_gmm import DeepGMM
from baselines.abstract_baseline import AbstractBaseline
from baselines.agmm.deep_gmm import DeepGMM
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, \
    StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import neural_network
from sklearn.mixture import GaussianMixture
import sklearn.metrics.pairwise

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data_utils

import keras
from econml.deepiv import DeepIVEstimator

import statsmodels.sandbox.regression.gmm
import statsmodels.tools.tools

import os


class SklearnBaseline(AbstractBaseline):
    def _predict(self, x, context):
        return self._model.predict(self.augment(x, context))


# class MNISTNet(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = torch.nn.Conv2d(1, 20, 5, 1)
#         self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
#         self.fc1 = torch.nn.Linear(4 * 4 * 50, 500)
#         self.fc2 = torch.nn.Linear(500, 10)
#         self.fc3 = torch.nn.Linear(10, 1)
#
#     def forward(self, x):
#         x = x.view(x.shape[0], 1, 28, 28)
#         x = F.leaky_relu(self.conv1(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.leaky_relu(self.conv2(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = x.view(-1, 4 * 4 * 50)
#         x = F.leaky_relu(self.fc1(x))
#         x = self.fc2(x)
#         x = F.leaky_relu(x)
#         return self.fc3(x)


class DirectLinearRegression(SklearnBaseline):
    def _fit(self, x, y, z, context=None):
        x = self.augment(x, context)
        direct_regression = sklearn.linear_model.LinearRegression()
        direct_regression.fit(x, y)
        self._model = direct_regression
        return self


class DirectRidge(SklearnBaseline):
    def _fit(self, x, y, z, context=None):
        x = self.augment(x, context)

        params = dict(ridge__alpha=np.logspace(-5, 5, 11))
        pipe = Pipeline([('ridge', Ridge())])
        direct_regression = GridSearchCV(pipe, param_grid=params, cv=5)

        direct_regression.fit(x, y)
        self._model = direct_regression
        return self


class DirectPoly(SklearnBaseline):
    def _fit(self, x, y, z, context=None):
        x = self.augment(x, context)

        params = dict(poly__degree=range(1, 4),
                      ridge__alpha=np.logspace(-5, 5, 11))
        pipe = Pipeline([('poly', PolynomialFeatures()),
                         ('ridge', Ridge())])
        direct_regression = GridSearchCV(pipe, param_grid=params, cv=5)

        direct_regression.fit(x, y)
        self._model = direct_regression
        return self


class DirectMNIST(AbstractBaseline):
    def __init__(self, n_epochs=6, batch_size=128, lr=0.005):
        super().__init__()
        self._n_epochs = n_epochs
        self._n_batch_size = batch_size
        self._lr = lr

    def _fit(self, x, y, z, context=None):
        model = DefaultCNN(cuda=torch.cuda.is_available())
        model.float()
        optimizer = torch.optim.Adam(model.parameters(), lr=self._lr)
        model.train()

        x = self.augment(x, context)
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        train = data_utils.DataLoader(data_utils.TensorDataset(x, y),
                                      batch_size=self._n_batch_size,
                                      shuffle=True)
        for epoch in range(self._n_epochs):
            losses = list()
            print("Epoch: ", epoch + 1, "/", self._n_epochs, " batch size: ",
                  self._n_batch_size)
            for i, (x, y) in enumerate(train):
                optimizer.zero_grad()
                y_pred = model(x)
                loss = F.mse_loss(y_pred, y)
                losses += [loss.data.cpu().numpy()]
                loss.backward()
                optimizer.step()
            print("   train loss", np.mean(losses))
        self._model = model
        return self

    def _predict(self, x, context):
        self._model.eval()
        x = self.augment(x, context)
        x = torch.tensor(x, dtype=torch.float)
        if torch.cuda.is_available():
            x = x.cuda()
        return self._model(x).data.cpu().numpy()


class DirectNN(SklearnBaseline):
    def _fit(self, x, y, z, context=None):
        x = self.augment(x, context)

        params = dict(nn__alpha=np.logspace(-5, 5, 5),
                      nn__hidden_layer_sizes=[(10,), (20,), (10, 10), (20, 10),
                                              (10, 10, 10), (20, 10, 5)])
        pipe = Pipeline([('standard', MinMaxScaler()),
                         ('nn',
                          sklearn.neural_network.MLPRegressor(solver="lbfgs"))])
        direct_regression = GridSearchCV(pipe, param_grid=params, cv=5)

        direct_regression.fit(x, y.flatten())
        self._model = direct_regression
        return self

    def _predict(self, x, context):
        return self._model.predict(self.augment(x, context)).reshape((-1, 1))


class DeepIV(AbstractBaseline):
    def __init__(self, treatment_model=None):
        if treatment_model is None:
            print("Using standard treatment model...")
            self._treatment_model = lambda input_shape: keras.Sequential(
                [keras.layers.Dense(128,
                                    activation='relu',
                                    input_shape=input_shape),
                 keras.layers.Dropout(0.17),
                 keras.layers.Dense(64,
                                    activation='relu'),
                 keras.layers.Dropout(0.17),
                 keras.layers.Dense(32,
                                    activation='relu'),
                 keras.layers.Dropout(0.17)])

        else:
            if keras.backend.image_data_format() == "channels_first":
                image_shape = (1, 28, 28)
            else:
                image_shape = (28, 28, 1)

            self._treatment_model = lambda input_shape: keras.Sequential([
                keras.layers.Reshape(image_shape, input_shape=input_shape),
                keras.layers.Conv2D(16, kernel_size=(3, 3),
                                    activation='relu'),
                keras.layers.Conv2D(32, (3, 3), activation='relu'),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),
                keras.layers.Dropout(0.1),
                keras.layers.Flatten(),
                keras.layers.Dense(128, activation='relu'),
                keras.layers.Dropout(0.1)])

    def _fit(self, x, y, z, context=None):
        if context is None:
            context = np.empty((x.shape[0], 0))

        x_dim = x.shape[1]
        z_dim = z.shape[1]
        context_dim = context.shape[1]

        treatment_model = self._treatment_model((context_dim + z_dim,))

        response_model = keras.Sequential([keras.layers.Dense(128,
                                                              activation='relu',
                                                              input_shape=(
                                                                  context_dim + x_dim,)),
                                           keras.layers.Dropout(0.17),
                                           keras.layers.Dense(64,
                                                              activation='relu'),
                                           keras.layers.Dropout(0.17),
                                           keras.layers.Dense(32,
                                                              activation='relu'),
                                           keras.layers.Dropout(0.17),
                                           keras.layers.Dense(1)])

        self._model = DeepIVEstimator(n_components=10,
                                      # Number of gaussians in the mixture density networks)
                                      m=lambda _z, _context: treatment_model(
                                          keras.layers.concatenate(
                                              [_z, _context])),
                                      # Treatment model
                                      h=lambda _t, _context: response_model(
                                          keras.layers.concatenate(
                                              [_t, _context])),
                                      # Response model
                                      n_samples=1
                                      )
        self._model.fit(y, x, context, z)

    def _predict(self, x, context):
        if context is None:
            context = np.empty((x.shape[0], 0))

        return self._model.predict(x, context)


class AGMM(AbstractBaseline):
    def _fit(self, x, y, z, context=None):
        _z = self.augment(z, context)
        _x = self.augment(x, context)

        self._model = DeepGMM(n_critics=50, num_steps=100,
                              learning_rate_modeler=0.01,
                              learning_rate_critics=0.1, critics_jitter=True,
                              eta_hedge=0.16, bootstrap_hedge=False,
                              l1_reg_weight_modeler=0.0,
                              l2_reg_weight_modeler=0.0,
                              dnn_layers=[1000, 1000, 1000], dnn_poly_degree=1,
                              log_summary=False, summary_dir='', random_seed=30)
        self._model.fit(_z, _x, y)

    def _predict(self, x, context):
        _x = self.augment(x, context)

        return self._model.predict(_x).reshape(-1, 1)


class Poly2SLS(SklearnBaseline):
    def __init__(self, poly_degree=range(1, 4),
                 ridge_alpha=np.logspace(-5, 5, 11)):
        super().__init__()
        self._poly_degree = poly_degree
        self._ridge_alpha = ridge_alpha

    def _fit(self, x, y, z, context=None):
        '''
        Two stage least squares with polynomial basis function.
        - x: treatment
        - y: outcome
        - z: instrument
        - context: additional information
        '''
        params = dict(poly__degree=self._poly_degree,
                      ridge__alpha=self._ridge_alpha)
        pipe = Pipeline([('poly', PolynomialFeatures()),
                         ('ridge', Ridge())])
        stage_1 = GridSearchCV(pipe, param_grid=params, cv=5)
        _z = self.augment(z, context)
        x_hat = stage_1.fit(_z, x)

        x_hat = stage_1.predict(_z)

        pipe2 = Pipeline([('poly', PolynomialFeatures()),
                          ('ridge', Ridge())])
        stage_2 = GridSearchCV(pipe2, param_grid=params, cv=5)
        stage_2.fit(self.augment(x_hat, context), y)

        self._model = stage_2


class Vanilla2SLS(Poly2SLS):
    def display(self):
        weights = self.arr2str(self._model.coef_)
        bias = self.arr2str(self._model.intercept_)
        print("%s*x + %s" % (weights, bias))

    def _fit(self, x, y, z, context=None):
        '''
        Vanilla two stage least squares.
        - x: treatment
        - y: outcome
        - z: instrument
        - context: additional information
        '''
        stage_1 = LinearRegression()
        _z = self.augment(z, context)
        x_hat = stage_1.fit(_z, x)
        x_hat = stage_1.predict(_z)

        stage_2 = LinearRegression()
        stage_2.fit(self.augment(x_hat, context), y)

        self._model = stage_2


class GMMfromStatsmodels(AbstractBaseline):
    def _fit(self, x, y, z, context=None):
        z = self.augment(z, context)
        z = statsmodels.tools.tools.add_constant(z, prepend=False)

        x = self.augment(x, context)
        x = statsmodels.tools.tools.add_constant(x, prepend=False)

        resultIV = statsmodels.sandbox.regression.gmm.IVGMM(y, x, z).fit(
            optim_args={"disp": True, "gtol": 1e-08, "epsilon": 1e-10,
                        "maxiter": 250}, maxiter=1,
            inv_weights=np.eye(z.shape[1]))
        print(resultIV.model.gmmobjective(resultIV.params, np.eye(z.shape[1])))
        # print(resultIV.model.momcond_mean(resultIV.params))

        self._model = resultIV

    def display(self):
        weights = self.arr2str(self._model.params[:-1])
        bias = self.arr2str(self._model.params[-1])
        print("%s*x + %s" % (weights, bias))

    def _predict(self, x, context):
        x = self.augment(x, context)
        x = statsmodels.tools.tools.add_constant(x, prepend=False)

        return self._model.predict(x)


class Featurizer(object):
    def transform(self, X):
        if isinstance(X, torch.Tensor):
            return torch.from_numpy(self._transform(X.data.cpu().numpy()))
        else:
            return self._transform(X)

    def is_initialized(self):
        return self._n_features is not None

    def n_features(self):
        if self.is_initialized():
            return self._n_features
        else:
            raise ValueError("Need to call transform first")


class VanillaFeatures(Featurizer):
    def __init__(self, add_constant=True):
        self._add_constant = add_constant

    def _transform(self, X):
        self._n_features = X.shape[1] + int(self._add_constant)
        if self._add_constant:
            return np.append(X, np.ones_like(X[:, 0:1]), axis=1)
        else:
            return X


class PolyFeatures(Featurizer):
    def __init__(self, degree=2):
        self._scaler = self._scaler = Pipeline([('pre_scale', MinMaxScaler()),
                                                ('poly',
                                                 PolynomialFeatures(degree)),
                                                (
                                                    'after_scale',
                                                    MinMaxScaler())])
        self._n_features = None

    def _transform(self, X):
        if self.is_initialized():
            return self._scaler.transform(X)
        else:
            r = self._scaler.fit_transform(X)
            self._n_features = self._scaler.named_steps[
                'poly'].n_output_features_
            return r


class GaussianKernelFeatures(Featurizer):
    def __init__(self, n_kernel_fcts=10):
        self._n_kernel_fcts = n_kernel_fcts
        self._n_features = None

    def _transform(self, X):
        if not self.is_initialized():
            # fit a (spherical) Gaussian mixture model to estimate kernel params
            gmix = GaussianMixture(n_components=self._n_kernel_fcts,
                                   covariance_type="spherical", max_iter=100,
                                   random_state=0)

            gmix.fit(X)

            kernels = []
            for k in range(self._n_kernel_fcts):
                kernels.append(
                    (np.atleast_2d(gmix.means_[k]), gmix.precisions_[k]))

            self._n_features = self._n_kernel_fcts

        transformed = list()
        for kernel in kernels:
            gamma = None if X.shape[1] > 10 else kernel[
                1]  # only use precision for low-dim X
            shift = sklearn.metrics.pairwise.rbf_kernel(X, kernel[0], gamma)
            transformed += [shift]

        transformed = np.hstack(transformed)
        return transformed


class GMM(AbstractBaseline):
    models = {
        "linear": lambda input_dim: torch.nn.Linear(input_dim, 1),
        "2-layer": lambda input_dim: torch.nn.Sequential(
            torch.nn.Linear(input_dim, 20),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(20, 1)
        ),
        "mnist": lambda input_dim: DefaultCNN(cuda=torch.cuda.is_available())
    }

    def __init__(self, g_model="linear", f_feature_mapping=None,
                 g_feature_mapping=None, n_steps=1, g_epochs=200):
        '''
        Generalized methods of moments.
        - g_model: Model to estimate for g
        - f_feature_mapping: mapping of raw instruments z
        - g_feature_mapping: mapping of raw features x
        - norm: additional information
        '''
        super().__init__()

        if f_feature_mapping is None:
            self.f_mapping = VanillaFeatures()
        else:
            self.f_mapping = f_feature_mapping

        if g_feature_mapping is None:
            self.g_mapping = VanillaFeatures(add_constant=False)
        else:
            self.g_mapping = g_feature_mapping

        if g_model in self.models:
            self._g = self.models[g_model]
        else:
            raise ValueError("g_model has invalid value " + str(g_model))
        self._optimizer = None
        self._n_steps = n_steps
        self._g_epochs = g_epochs

    def display(self):
        for name, param in self._model.named_parameters():
            print(name, self.arr2str(param.data.cpu().numpy()))

    def fit_g_minibatch(self, train, loss):
        losses = list()
        for i, (x_b, y_b, z_b) in enumerate(train):
            if torch.cuda.is_available():
                x_b = x_b.cuda()
                y_b = y_b.cuda()
                z_b = z_b.cuda()
            loss_val = self._optimizer.step(lambda: loss(x_b, y_b, z_b))
            losses += [loss_val.data.cpu().numpy()]
        print("  train loss ", np.mean(losses))

    def fit_g_batch(self, x, y, z, loss):
        _ = self._optimizer.step(lambda: loss(x, y, z))

    def _fit(self, x, y, z, context=None):
        z = self.augment(z, context)
        z = self.f_mapping.transform(z)
        x = self.augment(x, context)
        x = self.g_mapping.transform(x)
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        z = torch.tensor(z, dtype=torch.float)
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
            z = z.cuda()

        n_samples = x.size(0)
        x_dim, z_dim = x.size(1), z.size(1)

        g_model = self._g(x_dim)
        if torch.cuda.is_available():
            g_model = g_model.cuda()
        g_model.float()
        self._optimizer = torch.optim.Adam(g_model.parameters(), lr=0.01)
        weights = torch.eye(z_dim)
        if torch.cuda.is_available():
            weights = weights.cuda()
        self._model = g_model

        def loss(x_b, y_b, z_b):
            moment_conditions = z_b.mul(y_b - g_model(x_b))
            moms = moment_conditions.mean(dim=0, keepdim=True)
            loss = torch.mm(torch.mm(moms, weights), moms.t())
            self._optimizer.zero_grad()
            loss.backward()
            return loss

        batch_mode = "mini" if n_samples > 5000 else "full"
        train = data_utils.DataLoader(data_utils.TensorDataset(x, y, z),
                                      batch_size=128, shuffle=True)

        for step in range(self._n_steps):
            print("GMM step %d/%d" % (step + 1, self._n_steps))
            if step > 0:
                # optimize weights
                with torch.no_grad():
                    moment_conditions = z.mul(y - g_model(x))
                    covariance_matrix = torch.mm(moment_conditions.t(),
                                                 moment_conditions) / n_samples
                    weights = torch.as_tensor(
                        np.linalg.pinv(covariance_matrix.cpu().numpy(),
                                       rcond=1e-9))
                    if torch.cuda.is_available():
                        weights = weights.cuda()

            for epoch in range(self._g_epochs):
                if batch_mode == "full":
                    self.fit_g_batch(x, y, z, loss)
                else:
                    print("g epoch %d / %d" % (epoch + 1, self._g_epochs))
                    self.fit_g_minibatch(train, loss)
            self._model = g_model
        return self

    def _predict(self, x, context):
        x = self.augment(x, context)
        x = self.g_mapping.transform(x)
        x = torch.tensor(x, dtype=torch.float)
        if torch.cuda.is_available():
            x = x.cuda()
        return self._model(x).data.cpu().numpy()


def main():
    def quick_scenario(n=1000, train=True):
        z = np.random.normal(size=(n, 2))
        context = np.zeros((n, 1))
        intercept = 0.0
        slope = 0.2
        g_true = lambda x: np.maximum(slope * x + intercept,
                                      slope * x / 0.2 + intercept)
        epsilon, eta = np.random.normal(size=(n, 1)), np.random.normal(
            size=(n, 1))
        x = z[:, 0:1] + z[:, 1:] + epsilon * 2.0
        y = g_true(x) + epsilon * 7.0 + eta / np.sqrt(2)
        y_true = g_true(x)
        return x, y, y_true, z, context

    np.random.seed(1)
    torch.manual_seed(1)

    x, y, _, z, context = quick_scenario()
    x_t, y_t_observed, y_t, _, context_t = quick_scenario(train=False)

    def eval(model):
        y_pred = model.predict(x_t, context_t)
        return ((y_pred - y_t) ** 2).mean()

    def save(model):
        os.makedirs("quick_scenario", exist_ok=True)
        y_pred = model.predict(x_t, context_t)
        np.savez("quick_scenario/" + type(model).__name__, x=x_t,
                 y=y_t_observed, g_true=y_t, g_hat=y_pred)

    for method in [DirectPoly(), DirectLinearRegression(),
                   GMM(f_feature_mapping=PolyFeatures(),
                       g_feature_mapping=PolyFeatures()), Vanilla2SLS(),
                   Poly2SLS()]:
        model = method.fit(x, y, z, context)

        print("Test MSE of %s: %f" % (type(model).__name__, eval(model)))


if __name__ == "__main__":
    main()
