import numpy as np
import tensorflow as tf
from datetime import datetime
from .utils import LoopIterator, log_function, scope_variables
from .gmm_game_graph import GMMGameGraph

DEBUG=False

class DeepGMM:

    def __init__(self, n_critics=50, batch_size_modeler=200, num_steps=30, store_step=10,
                 display_step=20, check_loss_step=10, train_ratio=(1, 1), hedge_step=1,
                 eta_hedge=0.16, loss_clip_hedge=2, bootstrap_hedge=True,
                 learning_rate_modeler=0.01, learning_rate_critics=0.01, critics_jitter=False, critics_precision=None,
                 cluster_type='forest', critic_type='Gaussian', min_cluster_size=50, num_trees=5,
                 l1_reg_weight_modeler=0., l2_reg_weight_modeler=0.,
                 dnn_layers=[1000, 1000, 1000], dnn_poly_degree=1,
                 dissimilarity_eta=0.0, log_summary=True, summary_dir='./graphs',
                 random_seed=None):
        ''' Initialize parameters
        Parameters
        n_critics: number of critic functions
        batch_size_modeler: batch size for modeler gradient step
        num_steps: training steps
        store_step: at how many steps to store the function for calculating avg function
        display_step: at how many steps to print some info
        check_loss_step: at how many steps to check the loss for calculating the best function
        train_ratio: ratio of (modeler, critics) updates
        hedge_step: at how many steps to update the meta-critic with hedge
        eta_hedge: step size of hedge
        loss_clip_hedge: clipping of the moments so that hedge doesn't blow up
        bootstrap_hedge: whether to draw bootstrap subsamples for Hedge update
        learning_rate_modeler: step size for the modeler gradient descent
        learning_rate_critics: step size for the critics gradient descents
        critics_jitter: whether to perform gradient descent on the parameters of the critics
        critics_precision: the radius of the critics in number of samples
        cluster_type: ('forest' | 'kmeans' | 'random_points') which method to use to select the center of the different critics
        critic_type: ('Gaussian' | 'Uniform') whether to put a gaussian or a uniform on the sample points of the cluster
        min_cluster_size: how many points to include in each cluster of points 
        num_trees: only for the forest cluster type, how many trees to build
        l1_reg_weight_modeler: l1 regularization of modeler parameters
        l2_reg_weight_modeler: l2 regularization of modeler parameters
        dnn_layers: (list of int) sizes of fully connected layers
        dnn_poly_degree: how many polynomial features to create as input to the dnn
        dissimilarity_eta: coefficient in front of dissimilarity loss for flexible critics
        log_summary: whether to log the summary using tensorboard
        summary_dir: where to store the summary
        '''
        self._n_critics = n_critics
        self._batch_size_modeler = batch_size_modeler
        self._num_steps = num_steps
        self._display_step = display_step
        self._store_step = store_step
        self._check_loss_step = check_loss_step
        self._train_ratio = train_ratio
        self._hedge_step = hedge_step
        self._eta_hedge = eta_hedge
        self._bootstrap_hedge = bootstrap_hedge
        self._loss_clip_hedge = loss_clip_hedge
        self._learning_rate_modeler = learning_rate_modeler
        self._learning_rate_critics = learning_rate_critics
        self._critics_jitter = critics_jitter
        self._critics_precision = critics_precision
        self._cluster_type = cluster_type
        self._critic_type = critic_type
        self._min_cluster_size = min_cluster_size
        self._num_trees = num_trees
        self._l1_reg_weight_modeler = l1_reg_weight_modeler
        self._l2_reg_weight_modeler = l2_reg_weight_modeler
        self._dnn_layers = dnn_layers
        self._dnn_poly_degree = dnn_poly_degree
        self._dissimilarity_eta = dissimilarity_eta
        self._log_summary = log_summary
        self._summary_dir = summary_dir
        self._checkpoints = []
        self._random_seed = random_seed

    def _data_clusterings(self, data_z, data_p, data_y):
        ''' Returns the centers and precisions of an epsilon cover of gaussians.
        Currently the centers are just an epsilon grid and the precisions 1/(3*epsilon),
        i.e. the standard deviation is 3 times the distance between two grid points.
        Later this is exactly the function that will be implementing the tree style splitting.
        and returning a more tailored to the data epsilon cover.'''
        if self._cluster_type == 'forest':
            from sklearn.ensemble import RandomForestRegressor
            dtree = RandomForestRegressor(n_estimators=self._num_trees, max_leaf_nodes=self._n_critics, min_samples_leaf=self._min_cluster_size)
            dtree.fit(data_z, data_p)
            cluster_labels = dtree.apply(data_z)
            #dtree.fit(data_z, data_y)
            #cluster_labels = np.concatenate((cluster_labels, dtree.apply(data_z)), axis=1)
            cluster_ids = [np.unique(cluster_labels[:, c]) for c in range(cluster_labels.shape[1])]
        elif self._cluster_type == 'kmeans':
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self._n_critics).fit(data_z)
            cluster_labels = kmeans.labels_.reshape(-1, 1)
            cluster_ids = [np.unique(cluster_labels)]
        elif self._cluster_type == 'random_points':
            center_ids = np.random.choice(np.arange(data_z.shape[0]), size=self._n_critics, replace=False)
            cluster_labels = np.zeros((data_z.shape[0], self._n_critics))
            cluster_ids = np.ones((self._n_critics, 1))
            for it, center in enumerate(center_ids):
                distances = np.linalg.norm(data_z - data_z[center], axis=1)
                cluster_members = np.argsort(distances)[:self._min_cluster_size]
                cluster_labels[cluster_members, it] = 1
        else:
            raise Exception("Unknown option {}".format(self._cluster_type))

        #z_min = np.percentile(data_z, 0) - self._epsilon
        #z_max = np.percentile(data_z, 100) + self._epsilon
        #center_grid = np.arange(z_min, z_max, self._epsilon)
        #precision_grid = np.ones(center_grid.shape[0]) / (3 * self._epsilon)
        return cluster_labels, cluster_ids

    def fit(self, data_z, data_p, data_y):
        ''' Fits the treatment response model.
        Parameters
        data_z: (n x d np array) of instruments
        data_p: (n x p np array) of treatments
        data_y: (n x 1 np array) of outcomes
        '''

        num_instruments = data_z.shape[1]
        num_treatments = data_p.shape[1]
        num_outcomes = data_y.shape[1]
        self.num_treatments = num_treatments

        # Data iterators for critics/modeler and for meta-critic
        data_it = LoopIterator(
            np.arange(data_z.shape[0]), self._batch_size_modeler, random=True)
        data_it_hedge = LoopIterator(
            np.arange(data_z.shape[0]), data_z.shape[0], random=self._bootstrap_hedge)

        # Creat a test grid for calculating loss at intervals
        test_min = np.percentile(data_p, 5)
        test_max = np.percentile(data_p, 95)
        self.test_grid = np.linspace(test_min, test_max, 100)

        # Create the clusterings of the data that define the critics
        cluster_labels, cluster_ids = self._data_clusterings(data_z, data_p, data_y)
        if self._critic_type == 'Gaussian':
            # We put a symmetric gaussian encompassing all the data points of each cluster of each clustering
            center_grid = []
            precision_grid = []
            normalizers = []
            for tree in range(cluster_labels.shape[1]):
                for leaf in cluster_ids[tree]:
                    center = np.mean(data_z[cluster_labels[:, tree].flatten()==leaf, :], axis=0)
                    distance = np.linalg.norm(data_z - center, axis=1) / data_z.shape[1]
                    precision = 1./(np.sqrt(2)*(np.sort(distance)[self._min_cluster_size]))
                    center_grid.append(center)
                    precision_grid.append(precision)
                    normalizers.append((precision**num_instruments) * np.sum(np.exp(- (precision * distance)**2 )) / (np.power(2. * np.pi, num_instruments / 2.)))
            normalizers = np.ones(len(center_grid))  #np.array(normalizers) 
            center_grid = np.array(center_grid)
            precision_grid = np.array(precision_grid)
            if self._critics_precision is not None:
                precision_grid = self._critics_precision*np.ones(precision_grid.shape)
            #print(np.sort(center_grid[:, 0].flatten()))
            #print(precision_grid[np.argsort(center_grid[:, 0].flatten())])
        else:
            # We put a uniform kernel only on the data points of each cluster of each clustering
            normalizers = []
            center_grid = []
            leaf_id_list = []
            for tree in range(cluster_labels.shape[1]):
                for leaf in cluster_ids[tree]:
                    center_grid.append(np.mean(data_z[cluster_labels[:, tree].flatten()==leaf, :], axis=0)) # used only for tensorflow summary
                    normalizers.append(np.sum(cluster_labels[:, tree].flatten()==leaf))
                    leaf_id_list.append((tree, leaf))
            center_grid = np.array(center_grid)
            #print(np.sort(center_grid[:, 0].flatten()))
            #print(np.array(normalizers)[np.argsort(center_grid[:, 0].flatten())])
            normalizers = np.ones(len(center_grid)) #np.array(normalizers) 
            leaf_id_list = np.array(leaf_id_list)

        # tf Graph input
        if self._random_seed is not None:
            tf.set_random_seed(self._random_seed)

        self.Z = tf.placeholder("float", [None, num_instruments], name="instrument")
        self.P = tf.placeholder("float", [None, num_treatments], name="treatment")
        self.Y = tf.placeholder("float", [None, num_outcomes], name="outcome")
        self.Leaf = tf.placeholder("float", [None, cluster_labels.shape[1]], name="leaf_id")
        self.drop_prob = tf.placeholder_with_default(
            1.0, shape=(), name="drop_prob")

        self.gmm_graph = GMMGameGraph(self.Z, self.P, self.Y, self.Leaf, self.drop_prob,
                                 eta_hedge=self._eta_hedge,
                                 loss_clip_hedge=self._loss_clip_hedge,
                                 learning_rate_modeler=self._learning_rate_modeler,
                                 learning_rate_critics=self._learning_rate_critics, critics_jitter=self._critics_jitter, critic_type=self._critic_type,
                                 l1_reg_weight_modeler=self._l1_reg_weight_modeler,
                                 l2_reg_weight_modeler=self._l2_reg_weight_modeler,
                                 dnn_layers=self._dnn_layers, dnn_poly_degree=self._dnn_poly_degree,
                                 dissimilarity_eta=self._dissimilarity_eta)
        if self._critic_type == 'Gaussian':
            self.gmm_graph.create_graph(normalizers=normalizers, center_grid=center_grid, precision_grid=precision_grid)
        else:
            self.gmm_graph.create_graph(normalizers=normalizers, leaf_list=leaf_id_list)

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()
        if num_treatments == 1:
            self.avg_fn = []
            self.final_fn = []
            self.best_fn = []
        else:
            saver = tf.train.Saver(scope_variables("Modeler"), max_to_keep=self._num_steps)
            #print(scope_variables("Modeler"))
        avg_store_steps = list(np.random.choice(np.arange(int(0.2 * self._num_steps), self._num_steps), int(0.4 * self._num_steps), replace=False))
        #print(avg_store_steps)
        # Start training
        loss = np.inf
        with tf.Session() as sess:
            if self._log_summary:
                merged = tf.summary.merge_all()
                writer = tf.summary.FileWriter(self._summary_dir, sess.graph)

            # Run the initializer
            sess.run(init)
            d1 = d2 = d3 = d4 = d5 = d6 = 0.
            for step in range(1, self._num_steps + 1):

                t1 = datetime.now()

                # Modeler
                for inner_step in range(self._train_ratio[0]):
                    inds = data_it.get_next()
                    y1, p1, z1, leaf1 = data_y[inds], data_p[inds], data_z[inds], cluster_labels[inds]
                    inds = data_it.get_next()
                    y2, p2, z2, leaf2 = data_y[inds], data_p[inds], data_z[inds], cluster_labels[inds]
                    sess.run(self.gmm_graph.update_prev_moments, feed_dict={
                             self.Z: z1, self.P: p1, self.Y: y1, self.Leaf: leaf1, self.drop_prob: .9})
                    sess.run(self.gmm_graph.gradient_step_modeler, feed_dict={
                             self.Z: z2, self.P: p2, self.Y: y2, self.Leaf: leaf2, self.drop_prob: .9})

                t2 = datetime.now()
                d1 += (t2 - t1).seconds + (t2 - t1).microseconds * 1E-6

                if DEBUG:
                    new_loss = sess.run(self.gmm_graph.max_violation, feed_dict={
                                        self.Z: data_z, self.P: data_p, self.Y: data_y, self.Leaf: cluster_labels})
                    print("After modeler: Step " + str(step) + ", Moment violation= " +
                          "{:.10f}".format(new_loss))
                    print([sess.run([crt.precision, crt.weights, crt._normalized_translation, crt.center, crt.output[0]], feed_dict={
                             self.Z: data_z, self.P: data_p, self.Y: data_y, self.Leaf: cluster_labels}) for crt in self.gmm_graph.critics])
                    print(sess.run([cw.value() for cw in self.gmm_graph.critic_weights]))
                    

                # Critics
                for inner_step in range(self._train_ratio[1]):
                    inds = data_it.get_next()
                    y1, p1, z1, leaf1 = data_y[inds], data_p[inds], data_z[inds], cluster_labels[inds]
                    inds = data_it.get_next()
                    y2, p2, z2, leaf2 = data_y[inds], data_p[inds], data_z[inds], cluster_labels[inds]
                    sess.run(self.gmm_graph.update_prev_moments, feed_dict={
                             self.Z: z1, self.P: p1, self.Y: y1, self.Leaf: leaf1, self.drop_prob: .9})
                    sess.run(self.gmm_graph.gradient_step_critics, feed_dict={
                             self.Z: z2, self.P: p2, self.Y: y2, self.Leaf: leaf2, self.drop_prob: .9})

                if DEBUG:
                    new_loss = sess.run(self.gmm_graph.max_violation, feed_dict={
                                        self.Z: data_z, self.P: data_p, self.Y: data_y, self.Leaf: cluster_labels})
                    print("After Critic Step " + str(step) + ", Moment violation= " +
                          "{:.10f}".format(new_loss))
                    print([sess.run([crt.precision, crt.weights, crt._normalized_translation, crt.center, crt.output[0]], feed_dict={
                             self.Z: data_z, self.P: data_p, self.Y: data_y, self.Leaf: cluster_labels}) for crt in self.gmm_graph.critics])
                    print([sess.run(cw.value()) for cw in self.gmm_graph.critic_weights])
                t3 = datetime.now()
                d2 += (t3 - t2).seconds + (t3 - t2).microseconds * 1E-6

                # Meta-Critic
                if step % self._hedge_step == 0:
                    inds = data_it_hedge.get_next()
                    y1, p1, z1, leaf1 = data_y[inds], data_p[inds], data_z[inds], cluster_labels[inds]
                    sess.run(self.gmm_graph.gradient_step_meta_critic, feed_dict={
                             self.Z: z1, self.P: p1, self.Y: y1, self.Leaf: leaf1})

                if DEBUG:
                    new_loss = sess.run(self.gmm_graph.max_violation, feed_dict={
                                        self.Z: data_z, self.P: data_p, self.Y: data_y, self.Leaf: cluster_labels})
                    print("After Meta-Critic Step " + str(step) + ", Moment violation= " +
                          "{:.10f}".format(new_loss))
                    print([sess.run([crt.precision, crt.weights, crt._normalized_translation, crt.center, crt.output[0]], feed_dict={
                             self.Z: data_z, self.P: data_p, self.Y: data_y, self.Leaf: cluster_labels}) for crt in self.gmm_graph.critics])
                    print([sess.run(cw.value()) for cw in self.gmm_graph.critic_weights])

                t4 = datetime.now()
                d3 += (t4 - t3).seconds + (t4 - t3).microseconds * 1E-6

                if step % self._check_loss_step == 0 or step == 1 or step == self._num_steps:
                    # Calculate batch loss and accuracy
                    new_loss = sess.run(self.gmm_graph.max_violation, feed_dict={
                                        self.Z: data_z, self.P: data_p, self.Y: data_y, self.Leaf: cluster_labels})
                    if new_loss <= loss:
                        if num_treatments == 1:
                            self.best_fn = sess.run(self.gmm_graph.modeler.output, feed_dict={self.P:self.test_grid.reshape(-1,1)}).flatten()
                        else:
                            saver.save(sess, "./tmp/model_best.ckpt")
                            loss = new_loss

                t5 = datetime.now()
                d4 += (t5 - t4).seconds + (t5 - t4).microseconds * 1E-6
                
                if self._log_summary and step % self._store_step == 0:
                    summary = sess.run(merged, feed_dict={
                                       self.Z: data_z, self.P: data_p, self.Y: data_y, self.Leaf: cluster_labels})
                    writer.add_summary(summary, step)
                    log_function(writer, 'CriticWeights', center_grid, np.array([sess.run(cw.value()) for cw in self.gmm_graph.critic_weights]), step, agg='sum')
                    #log_function(writer, 'CriticPrecisions', center_grid, np.array([sess.run(cr.precision.value()) for cr in self.gmm_graph.critics]), step, agg='mean')
                
                t6 = datetime.now()
                d5 += (t6 - t5).seconds + (t6 - t5).microseconds * 1E-6
                

                if step in avg_store_steps: #step > .2 * self._num_steps:
                    if num_treatments == 1:
                        self.avg_fn.append(sess.run(self.gmm_graph.modeler.output, feed_dict={self.P:self.test_grid.reshape(-1,1)}).flatten())
                    else:
                        saver.save(sess, "./tmp/model_{}.ckpt".format(step))
                    self._checkpoints.append(step)

                t7 = datetime.now()
                d6 += (t7 - t6).seconds + (t7 - t6).microseconds * 1E-6

                if step % self._display_step == 0:
                    new_loss = sess.run(self.gmm_graph.max_violation, feed_dict={
                                        self.Z: data_z, self.P: data_p, self.Y: data_y, self.Leaf: cluster_labels})
                    print("Final Step " + str(step) + ", Moment violation= " +
                          "{:.10f}".format(new_loss))
                    print("Modeler train time: {:.2f}".format(d1))
                    print("Critic train time: {:.2f}".format(d2))
                    print("Meta-critic train time: {:.2f}".format(d3))
                    print("Best loss checking time: {:.2f}".format(d4))
                    print("Summary storing time: {:.2f}".format(d5))
                    print("Average model calculation time: {:.2f}".format(d6))

            print("Optimization Finished!")
            if num_treatments == 1:
                self.final_fn = sess.run(self.gmm_graph.modeler.output, feed_dict={self.P:self.test_grid.reshape(-1,1)}).flatten()
            else:
                saver.save(sess, "./tmp/model_final.ckpt")
            
            sess.close()

        if self._log_summary:
            writer.close()

    def predict(self, data_p, model='avg'):
        ''' Predicts outcome for each treatment vector.
        Parameters
        data_p: (n x p np array) of treatments
        model: (str one of avg | best | final) which version of the neural net model to use to predict
    
        Returns
        y_pred: (n x 1 np array) of counterfacual outcome predictions for each treatment
        '''
        
        if self.num_treatments == 1:
            if model == 'avg':
                output = []
                for it, ckp in enumerate(self._checkpoints):
                    output.append(np.interp(data_p.flatten(), self.test_grid, self.avg_fn[it]))
                return np.mean(np.array(output), axis=0).flatten()
            elif model == 'best':
                return np.interp(data_p.flatten(), self.test_grid, self.best_fn)
            else:
                return np.interp(data_p.flatten(), self.test_grid, self.final_fn)
                

        saver = tf.train.Saver(scope_variables("Modeler"))
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init) 
            if model == 'avg':
                output = []
                for it, ckp in enumerate(self._checkpoints):
                    saver.restore(sess, "./tmp/model_{}.ckpt".format(ckp))
                    output.append(sess.run(self.gmm_graph.modeler.output, feed_dict={self.P: data_p}))
                return np.mean(np.array(output), axis=0).flatten()
            elif model == 'best':
                saver.restore(sess, "./tmp/model_best.ckpt")
                return sess.run(self.gmm_graph.modeler.output, feed_dict={self.P: data_p}).flatten()
            else:
                saver.restore(sess, "./tmp/model_final.ckpt")
                return sess.run(self.gmm_graph.modeler.output, feed_dict={self.P: data_p}).flatten()




def test():
    ''' Test basic functionality of the DeepGMM Class '''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def get_data(n_samples, n_instruments, iv_strength, tau_fn):
        # Construct dataset
        confounder = np.random.normal(0, 1, size=(n_samples, 1))
        z = np.random.normal(0, 1, size=(n_samples, n_instruments))
        p = np.dot(z, iv_strength) + confounder + \
            np.random.normal(0, .1, size=(n_samples, 1))
        y = tau_fn(p) + 2 * confounder + \
            np.random.normal(0, .1, size=(n_samples, 1))
        return z, p, y

    # DGP parameters
    num_instruments = 1
    num_treatments = 1
    num_outcomes = 1
    tau_fn = lambda x: np.abs(x) #-1.5 * x + .9 * (x**2) #np.sin(x) #1. * (x<0) + 2.5 * (x>=0) #np.abs(x)  # 1. * (x<0) + 3. * (x>=0) #-1.5 * x + .9 * (x**2)  #-1.5 * x + .9 * (x**2) #np.abs(x) #-1.5 * x + .9 * (x**2) + x**3 #-1.5 * x + .9 * (x**2) + x**3 # np.sin(x) #-1.5 * x + .9 * (x**2) + x**3 #np.sin(x) #-1.5 * x + .9 * (x**2) + x**3 #np.sin(x) #np.abs(x) #np.sin(x) #2/(1+np.exp(-2*x)) #2/(1+np.exp(-2*x)) #1.5 * x - .9 * (x**2) #2/(1+np.exp(-2*x))#-1.5 * x + .9 * (x**2)
    iv_strength = np.array([1]).reshape(-1, 1)
    degree_benchmarks = 3
    n_samples = 4000
    hidden_layers = [1000, 1000, 1000]
    # Generate data
    data_z, data_p, data_y = get_data(
        n_samples, num_instruments, iv_strength, tau_fn)

    # Initialize class and fit
    dgmm = DeepGMM(n_critics=50, num_steps=100, learning_rate_modeler=0.01,
                    learning_rate_critics=0.1, critics_jitter=True,
                    eta_hedge=0.16, bootstrap_hedge=False,
                    l1_reg_weight_modeler=0.0, l2_reg_weight_modeler=0.0,
                    dnn_layers=hidden_layers, dnn_poly_degree=1, 
                    log_summary=True, summary_dir='./graphs_test')
    dgmm.fit(data_z, data_p, data_y)

    # Create a set of test points
    test_min = np.percentile(data_p, 10)
    test_max = np.percentile(data_p, 90)
    test_grid = np.linspace(test_min, test_max, 100)

    # Predict on these test points
    best_fn = dgmm.predict(test_grid.reshape(-1, 1), model='best')
    final_fn = dgmm.predict(test_grid.reshape(-1, 1), model='final')
    avg_fn = dgmm.predict(test_grid.reshape(-1, 1), model='avg')

    # Plot fitted functions
    plt.figure(figsize=(10, 10))
    plt.plot(test_grid, avg_fn, label='AvgANN y=g(p)')
    plt.plot(test_grid, best_fn, label='BestANN y=g(p)')
    plt.plot(test_grid, final_fn, label='FinalANN y=g(p)')
    plt.plot(test_grid, tau_fn(test_grid), label='true y=g(p)')
    plt.xlabel('Treatment')
    plt.ylabel('Outcome')
    plt.legend()
    plt.savefig('deep_gmm_test.png')

if __name__ == "__main__":
    test()
