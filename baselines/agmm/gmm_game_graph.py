import tensorflow as tf
import math
import numpy as np

def gaussian_kernel(x, precision=None, normalizer=None):
    '''
    A multi-dimensional symmetric gaussian kernel
    '''
    dimension = x.get_shape().as_list()[1]
    kernel = tf.scalar_mul(tf.pow(tf.abs(precision), dimension), tf.exp(- tf.scalar_mul(tf.pow(precision, 2) / 2., tf.reduce_sum(tf.pow(x, 2), axis=1, keepdims=True)))) / (math.pow(2. * math.pi, dimension / 2.))
    return tf.scalar_mul(1./normalizer, kernel)


def uniform_kernel(leaf, leaf_id=None, normalizer=None):
    tree_id = int(leaf_id[0])
    tree_leaf_id = int(leaf_id[1])
    return tf.scalar_mul(1./normalizer, tf.cast(tf.equal(tf.cast(leaf[:, tree_id], dtype=tf.int32), tree_leaf_id), dtype=tf.float32))

def dnn_regressor(x, n_outputs, degree, layers, drop_prob):
    '''
    A multi-layer neural network. All dense layers
    '''
    n_inputs = x.get_shape().as_list()[1]
    poly_x = x
    for d in range(2, degree + 1):
        poly_x = tf.concat([poly_x, tf.pow(x, d)], axis=1)
    cur_out = poly_x
    layers = [n_inputs * degree] + layers
    weights = []
    biases = []
    for l_id in range(1, len(layers)):
        with tf.name_scope("Layer{}".format(l_id)):
            with tf.name_scope("Weights"):
                weight_l = tf.Variable(tf.random_normal(
                    [layers[l_id - 1], layers[l_id]], 0, 0.1), name="weights")
                tf.summary.histogram('Weight Histogram', weight_l)
            with tf.name_scope("Bias"):
                bias_l = tf.Variable(tf.random_normal(
                    [layers[l_id]], 0, 0.1), name="biases")
                tf.summary.histogram('Bias Histogram', bias_l)
            cur_out = tf.add(tf.matmul(cur_out, weight_l), bias_l)
            cur_out = tf.nn.relu(tf.nn.dropout(cur_out, keep_prob=drop_prob))
            weights.append(weight_l)
            biases.append(bias_l)

    # Final layer
    with tf.name_scope("OutLayer"):
        with tf.name_scope("Weights"):
            weight_l = tf.Variable(tf.random_normal(
                [layers[-1], n_outputs], 0, 0.1))
            tf.summary.histogram('Weight Histogram', weight_l)
        with tf.name_scope("bias"):
            bias_l = tf.Variable(tf.random_normal([n_outputs], 0, 0.1))
            tf.summary.histogram('Bias Histogram', bias_l)
        cur_out = tf.add(tf.matmul(cur_out, weight_l), bias_l)
        weights.append(weight_l)
        biases.append(bias_l)

    return cur_out, weights, biases


class Modeler:
    ''' Stores the tf expressions related to the modeler '''

    def __init__(self, P, num_outcomes, dnn_layers, dnn_poly_degree, drop_prob, optimizer):
        with tf.name_scope("Modeler"):
            self._output, self._weights, self._biases = dnn_regressor(
                P, num_outcomes, dnn_poly_degree, dnn_layers, drop_prob)
            self._optimizer = optimizer
            for w in self._weights:
                tf.add_to_collection("ModelerModelVariables", w)
            for b in self._biases:
                tf.add_to_collection("ModelerModelVariables", b)

    @property
    def output(self):
        return self._output
    @property
    def weights(self):
        return self._weights
    @property
    def biases(self):
        return self._biases
    @property
    def optimizer(self):
        return self._optimizer
    @property
    def trainable_vars(self):
        return self._weights + self._biases


class GaussianCritic:
    ''' Stores the tf expressions related to a critic '''

    def __init__(self, Z, optimizer, id=0, num_reduced_dims=2, center=None, precision=None, jitter=False, normalizer=None):
        with tf.name_scope("Critic_{}".format(id)):
            n_instruments = Z.get_shape().as_list()[1]
            self._center = tf.Variable(tf.random_normal(
                [n_instruments])) if center is None else tf.Variable(center, dtype=tf.float32, trainable=jitter)
            self._precision = tf.Variable(tf.random_normal(
                [])) if precision is None else tf.Variable(precision, dtype=tf.float32, trainable=jitter)
            tf.summary.scalar("Precision", self._precision)
            [tf.summary.scalar("Center_{}".format(d), self._center[d]) for d in range(n_instruments)]
            self._translation = tf.Variable(tf.random_normal([n_instruments, min(n_instruments, num_reduced_dims)], 0, 1))
            self._normalized_translation = tf.nn.l2_normalize(self._translation, 1, epsilon=1e-12)
            self._output = gaussian_kernel(
                tf.matmul(Z - self._center, self._normalized_translation), precision=self._precision, normalizer=normalizer)
            self._optimizer = optimizer
            self._center_trainable = (center is None) or jitter
            self._precision_trainable = (precision is None) or jitter
            self._translation_trainable = True
            self._precision_l = 0.2 * precision if precision is not None else 0
            self._precision_u = 1.2 * precision if precision is not None else math.inf
            self._center_l = center - 0.4 * np.abs(center) if center is not None else - math.inf
            self._center_u = center + 0.4 * np.abs(center) if center is not None else math.inf

    @property
    def precision_l(self):
        return self._precision_l
    @property
    def precision_u(self):
        return self._precision_u
    @property
    def center_l(self):
        return self._center_l
    @property
    def center_u(self):
        return self._center_u
    @property
    def output(self):
        return self._output
    @property
    def center(self):
        return self._center
    @property
    def precision(self):
        return self._precision
    @property
    def weights(self):
        return self._translation
    @property
    def optimizer(self):
        return self._optimizer
    @property
    def trainable_vars(self):
        vars = []
        if self._center_trainable:
            vars.append(self._center)
        if self._precision_trainable:
            vars.append(self._precision)
        if self._translation_trainable:
            vars.append(self._translation)
        return vars

class BinCritic:
    ''' Stores the tf expressions related to a critic '''

    def __init__(self, Z, Leaf, leaf_id, optimizer, id=0, normalizer=None):
        with tf.name_scope("Critic_{}".format(id)):
            self._output = uniform_kernel(
                Leaf, leaf_id=leaf_id, normalizer=normalizer)
            self._optimizer = optimizer
    @property
    def output(self):
        return self._output
    @property
    def optimizer(self):
        return self._optimizer
    @property
    def trainable_vars(self):
        return []

class GMMGameGraph:

    def __init__(self, Z, P, Y, Leaf, drop_prob,
                 eta_hedge=0.16, loss_clip_hedge=2,
                 learning_rate_critics=0.01, critics_jitter=False, critic_type='Gaussian',
                 l1_reg_weight_modeler=0., l2_reg_weight_modeler=0.,
                 learning_rate_modeler=0.01, dnn_layers=[1000, 1000, 1000], dnn_poly_degree=1,
                 dissimilarity_eta=0.0):
        self._Z = Z
        self._P = P
        self._Y = Y
        self._Leaf = Leaf
        self._drop_prob = drop_prob
        self._num_instruments = Z.get_shape().as_list()[1]
        self._num_treatments = P.get_shape().as_list()[1]
        self._num_outcomes = Y.get_shape().as_list()[1]
        self._eta_hedge = eta_hedge
        self._loss_clip_hedge = loss_clip_hedge
        self._learning_rate_modeler = learning_rate_modeler
        self._learning_rate_critics = learning_rate_critics
        self._critics_jitter = critics_jitter
        self._l1_reg_weight_modeler = l1_reg_weight_modeler
        self._l2_reg_weight_modeler = l2_reg_weight_modeler
        self._dnn_layers = dnn_layers
        self._dnn_poly_degree = dnn_poly_degree
        self._dissimilarity_eta = dissimilarity_eta
        self._critic_type = critic_type

    def _create_modeler(self):
        ''' Creates the modeler tf expression '''
        self._modeler = Modeler(self._P, self._num_outcomes, self._dnn_layers, self._dnn_poly_degree,
                                self._drop_prob, tf.train.AdamOptimizer(learning_rate=self._learning_rate_modeler))

    def _create_critics(self, normalizers=None, leaf_list=None, center_grid=None, precision_grid=None):
        ''' Creates the tf expressions for each of the critics '''
        with tf.name_scope("MetaCritic"):
            if self._critic_type == 'Gaussian':
                if precision_grid is None:
                    self._critic_list = [GaussianCritic(self._Z, tf.train.AdamOptimizer(learning_rate=self._learning_rate_critics),
                                                id=c_id, center=center, jitter=self._critics_jitter, normalizer=norm)
                                        for c_id, (center, norm) in enumerate(zip(center_grid, normalizers))]
                else:
                    self._critic_list = [GaussianCritic(self._Z, tf.train.AdamOptimizer(learning_rate=self._learning_rate_critics),
                                                id=c_id, center=center, precision=precision, jitter=self._critics_jitter, normalizer=norm)
                                        for c_id, (center, precision, norm) in enumerate(zip(center_grid, precision_grid, normalizers))]
            if self._critic_type == 'Uniform':
                self._critic_list = [BinCritic(self._Z, self._Leaf, leaf_id, tf.train.AdamOptimizer(learning_rate=self._learning_rate_critics), id=c_id, normalizer=norm) for c_id, (leaf_id, norm) in enumerate(zip(leaf_list, normalizers))]
            self._critic_weights = [tf.Variable(
                1. / len(self._critic_list), dtype=tf.float32, trainable=False, name="Critic_{}_Weight".format(c_id)) for c_id in range(len(self._critic_list))]
            [tf.summary.scalar("CriticWeights", cw)
             for cw in self._critic_weights]

    def _create_moment_list(self):
        ''' Creates the tf expressions for each moment corresponding to each critic. 
        Also creates the tf expression that stores the current moment values to the prev_moment variables.
        This is used in the two-sample-based unbiased gradient construction. '''
        with tf.name_scope("Moments"):
            self._moment_list = [tf.reduce_mean(tf.multiply(
                self._Y - self._modeler.output, critic.output, name="Moment_{}".format(c_id)), name="Avg_Moment_{}".format(c_id))
                for c_id, critic in enumerate(self._critic_list)]
            self._prev_moment_list = [tf.Variable(
                0, dtype=tf.float32, trainable=False, name="PrevMoment_{}".format(c_id)) for c_id in range(len(self._critic_list))]
            self._update_moments = [tf.assign(p, m) for p, m in zip(
                self._prev_moment_list, self._moment_list)]

    def _create_max_violation(self):
        ''' Creates the tf expression corresponding to the maximum moment violation '''
        with tf.name_scope("MaxViolation"):
            self._max_violation = tf.reduce_max(
                [tf.pow(m, 2) for m in self._moment_list], name="Max_Violation")
            tf.summary.scalar("MaxViolation", self._max_violation)

    def _create_gradient_step_modeler(self):
        ''' Creates the tf expression for the gradient of the modeler. '''
        with tf.name_scope("ModelerGradient"):
            l1_regularizer = tf.contrib.layers.l1_regularizer(
                scale=self._l1_reg_weight_modeler)
            l2_regularizer = tf.contrib.layers.l2_regularizer(
                scale=self._l2_reg_weight_modeler)
            l2_reg_gen = tf.contrib.layers.apply_regularization(
                l2_regularizer, self._modeler.weights)
            l1_reg_gen = tf.contrib.layers.apply_regularization(
                l1_regularizer, self._modeler.weights)
            reg_gen = l1_reg_gen + l2_reg_gen

            gvs_gen_acc = [0 for _ in self._critic_list]
            for it, (m, pm, mw) in enumerate(zip(self._moment_list, self._prev_moment_list, self._critic_weights)):
                gvs_gen = self._modeler.optimizer.compute_gradients(
                    m, var_list=self._modeler.trainable_vars)
                gvs_gen_acc = [tf.scalar_mul(
                    2 * mw * pm, g) + g_acc for (g, v), g_acc in zip(gvs_gen, gvs_gen_acc)]

            reg_gvs_gen = self._modeler.optimizer.compute_gradients(
                reg_gen, var_list=self._modeler.trainable_vars)
            mult_gvs_gen = [(tf.add(g_acc, rg), v) if rg is not None else (g_acc, v)
                            for g_acc, (rg, v) in zip(gvs_gen_acc, reg_gvs_gen)]

            self._gradient_step_modeler = self._modeler.optimizer.apply_gradients(
                mult_gvs_gen)

    def _create_gradient_step_critics(self):
        ''' Creates the tf expression for the gradient of each critic '''
        with tf.name_scope("CriticsGradient"):
            apply_grads_critics = []
            for it, (critic, m, pm) in enumerate(zip(self._critic_list, self._moment_list, self._prev_moment_list)):
                if not critic.trainable_vars:
                    continue
                with tf.name_scope("Critics{}Gradient".format(it)):
                    gvs_disc = critic.optimizer.compute_gradients(
                        m, var_list=critic.trainable_vars)
                    gvs_disc_diff = critic.optimizer.compute_gradients(
                        critic.output, var_list=critic.trainable_vars)
                    mult_gvs_disc = [(tf.scalar_mul(-2 * pm, g) - self._dissimilarity_eta * tf.scalar_mul(tf.reduce_mean(
                        critic.output - self._critic_list[it - 1].output), g_diff), v) for (g, v), (g_diff, vp) in zip(gvs_disc, gvs_disc_diff)]
                    apply_grads = critic.optimizer.apply_gradients(mult_gvs_disc)
                    with tf.control_dependencies([apply_grads]):
                        clip_precision = tf.assign(critic.precision, tf.clip_by_value(
                            critic.precision, critic.precision_l, critic.precision_u))
                        clip_center = tf.assign(critic.center, tf.clip_by_value(
                            critic.center, critic.center_l, critic.center_u))
                        clip_weights = tf.assign(critic.weights, tf.clip_by_value(critic.weights, -10, 10))
                apply_grads_critics.append(apply_grads)
                apply_grads_critics.append(clip_precision)
                apply_grads_critics.append(clip_center)
                apply_grads_critics.append(clip_weights)

        self._gradient_step_critics = apply_grads_critics

    def _create_gradient_step_meta_critic(self):
        ''' Creates the tf expression for the Hedge update of the meta-critic '''
        with tf.name_scope("MetaCriticHedge"):
            update_weights = [tf.assign(w, tf.scalar_mul(w, tf.exp(self._eta_hedge * tf.clip_by_value(tf.pow(m, 2), 0, self._loss_clip_hedge))))
                              for w, m in zip(self._critic_weights, self._moment_list)]
            with tf.control_dependencies(update_weights):
                l1_norm_weights = tf.add_n(self._critic_weights)
                with tf.control_dependencies([l1_norm_weights]):
                    normalize_weights = [tf.assign(w, tf.divide(
                        w, l1_norm_weights)) for w in self._critic_weights]

        self._gradient_step_meta_critic = update_weights + \
            [l1_norm_weights] + normalize_weights

    def create_graph(self, normalizers=None, leaf_list=None, center_grid=None, precision_grid=None):
        ''' Constructs all the tf expressions related to the GMM Game '''
        self._create_modeler()
        if self._critic_type == 'Gaussian':
            self._create_critics(normalizers=normalizers, center_grid=center_grid, precision_grid=precision_grid)
        else:
            self._create_critics(normalizers=normalizers, leaf_list=leaf_list)
        self._create_moment_list()
        self._create_max_violation()
        self._create_gradient_step_modeler()
        self._create_gradient_step_critics()
        self._create_gradient_step_meta_critic()

    def update_critics(self, leaf_list=None, center_grid=None, precision_grid=None):
        ''' Updates the tf expressions related to the critics of the GMM Game,
        as well as all expressions that are impacted by such an update. '''
        if self._critic_type == 'Gaussian':
            self._create_critics(normalizers=normalizers, center_grid=center_grid, precision_grid=precision_grid)
        else:
            self._create_critics(normalizers=normalizers, leaf_list=leaf_list)        
        self._create_moment_list()
        self._create_max_violation()
        self._create_gradient_step_modeler()
        self._create_gradient_step_critics()
        self._create_gradient_step_meta_critic()

    @property
    def modeler(self):
        return self._modeler
    @property
    def critics(self):
        return self._critic_list
    @property
    def critic_weights(self):
        return self._critic_weights
    @property
    def moment_list(self):
        return self._moment_list
    @property
    def update_prev_moments(self):
        return self._update_moments
    @property
    def gradient_step_modeler(self):
        return self._gradient_step_modeler
    @property
    def gradient_step_critics(self):
        return self._gradient_step_critics
    @property
    def gradient_step_meta_critic(self):
        return self._gradient_step_meta_critic
    @property
    def max_violation(self):
        return self._max_violation
