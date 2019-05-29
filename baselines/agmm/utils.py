import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.mlab as mlab
from scipy.interpolate import griddata
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def generate_random_pw_linear(lb=-2, ub=2, n_pieces=5):
    print("started")
    splits = np.random.choice(np.arange(lb, ub, 0.1), n_pieces-1, replace=False)
    splits.sort()
    slopes = np.random.uniform(-4, 4, size=n_pieces)
    start = []
    start.append(np.random.uniform(-1, 1))
    for t in range(n_pieces-1):
        start.append(start[t] + slopes[t] * (splits[t] - (lb if t==0 else splits[t-1])))
    return lambda x: [start[ind] + slopes[ind] * (x - (lb if ind==0 else splits[ind-1])) for ind in [np.searchsorted(splits, x)]][0]

def plot_3d(z_list, y):
    ax = plt.axes(projection='3d')
    grid_x, grid_y = np.mgrid[min(z_list[:, 0]):max(z_list[:,0]):10j, min(z_list[:,1]):max(z_list[:,1]):10j]
    grid_z = griddata((z_list[:,0], z_list[:,1]), y, (grid_x, grid_y), method='cubic')

    ax.plot_surface(grid_x, grid_y, grid_z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')

def loadmodel(session, saver, checkpoint_dir):
    session.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(session, os.path.join(checkpoint_dir, ckpt_name))
        return True
    else:
        return False


def save(session, saver, checkpoint_dir, step):
    dir = os.path.join(checkpoint_dir, "model")
    saver.save(session, dir, global_step=step)

def scope_variables(name):
    with tf.variable_scope(name):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
                       scope=tf.get_variable_scope().name)


def log_function(writer, tag, grid, grid_fn, step, agg='sum'):
    """Logs the histogram of a list/vector of values."""
    values = np.array(grid_fn)
    if len(grid.shape)==1:
        grid.reshape(-1, 1)   
    for d in range(grid.shape[1]):
        bin_edges = np.unique(grid[:, d])
        counts = np.zeros(bin_edges.shape[0])
        for it, edge in enumerate(bin_edges):
            if agg=='sum':
                counts[it] = np.sum(values[grid[:, d] == edge])
            else:
                counts[it] = np.mean(values[grid[:, d] == edge])

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(grid[:, d]))
        hist.max = float(np.max(grid[:, d]))

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag="{}_{}".format(tag, d), histo=hist)])
        writer.add_summary(summary, step)


class LoopIterator():
    def __init__(self, data, batch_size, random=True):
        self.batch_size = batch_size
        self.data = data
        self.iter = 0
        self.random = random

    def get_next(self):
        if self.random:
            return np.random.choice(self.data, size=self.batch_size)

        if self.data.shape[0] - self.iter < self.batch_size:
            indices = np.concatenate((np.arange(self.iter, self.data.shape[0]),
                                      np.arange(self.batch_size - self.data.shape[0] + self.iter)))
            self.iter = self.batch_size - self.data.shape[0] + self.iter
            #print("Starting a new epoch on the data!")
        else:
            indices = np.arange(self.iter, self.iter + self.batch_size)
            self.iter += self.batch_size
        return self.data[indices]

if __name__=="__main__":
    fn = generate_random_pw_linear()
    print(fn(0))
    print(np.array([fn(x) for x in np.arange(-3, 3, 0.1)]))
    plt.figure()
    plt.plot(np.array(np.arange(-3, 3, 0.1)), np.array([fn(x) for x in np.arange(-3, 3, 0.1)]))
    fn = generate_random_pw_linear()
    plt.plot(np.array(np.arange(-3, 3, 0.1)), np.array([fn(x) for x in np.arange(-3, 3, 0.1)]))
    plt.savefig('test_fn.png')