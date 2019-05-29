import random
import numpy as np
import os
import torch

class Dataset(object):
    def __init__(self, x, z, y, g, w):
        self.x = x
        self.z = z 
        self.y = y
        self.g = g
        self.w = w
        self.size = None
    
    def to_tensor(self):
        self.x = torch.as_tensor(self.x).double()
        self.z = torch.as_tensor(self.z).double()
        self.y = torch.as_tensor(self.y).double()
        self.g = torch.as_tensor(self.g).double()
        self.w = torch.as_tensor(self.w).double()
    
    def to_2d(self):
        n_data = self.y.shape[0]
        if len(self.x.shape) > 2:
            self.x = self.x.reshape(n_data, -1)
        if len(self.z.shape) > 2:
            self.z = self.z.reshape(n_data, -1)
            
    def info(self, verbose=False):
        for name, x in [("x", self.x), ("z", self.z), ("y", self.y), ("g", self.g), ("w", self.w)]:
            print("  " + name + ":", x.__class__.__name__,  "(" + str(x.dtype) + "): ", "x".join([str(d) for d in x.shape]))
            if verbose: 
                print("      min: %.2f" % x.min(), ", max: %.2f" % x.max())

    def as_tuple(self):
        return self.x, self.z, self.y, self.g, self.w
    
    def as_dict(self, prefix = ""):
        d = {"x": self.x, "z": self.z, "y": self.y, "g": self.g, "w": self.w}
        return {prefix + k: v for k, v in d.items()}
    
    def to_numpy(self):
        self.x = self.x.data.numpy()
        self.z = self.z.data.numpy()
        self.y = self.y.data.numpy()
        self.g = self.g.data.numpy()
        self.w = self.w.data.numpy()
        
    def to_cuda(self):
        self.x = self.x.cuda()
        self.z = self.z.cuda()
        self.y = self.y.cuda()
        self.g = self.g.cuda()
        self.w = self.w.cuda()


class AbstractScenario(object):
    def __init__(self, filename=None):
        self.splits = {"test":None, "train":None, "dev":None}
        
        self.setup_args = None
        self.initialized = False
        if filename:
            self.from_file(filename)
            
    def to_cuda(self):
        for split in self.splits.values():
            split.to_cuda()
    
    def to_tensor(self):
        for split in self.splits.values():
            split.to_tensor()
            
    def to_numpy(self):
        for split in self.splits.values():
            split.to_numpy()
    
    def to_2d(self):
        """
        flatten x and z to 2D 
        """
        for split in self.splits.values():
            split.to_2d()
            
    def setup(self, num_train, num_dev=0, num_test=0, **args):
        """
        draw data internally, without actually returning anything
        """
        for split, num_data in (("train", num_train),
                                    ("dev", num_dev),
                                    ("test", num_test)):
            if num_data > 0:
                self.splits[split] = Dataset(*self.generate_data(num_data, **args))
                
        self.setup_args = args
        self.initialized = True

    def to_file(self, filename):
        all_splits = {"splits": list()}
        for split, dataset in self.splits.items():
            if dataset is not None:
                all_splits.update(dataset.as_dict(split + "_"))
                all_splits["splits"].append(split)
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        np.savez(filename, **all_splits)
        
    def from_file(self, filename):
        data = np.load(filename)
        for split in data["splits"].tolist():
            self.splits[split] = Dataset(*(data[split + "_" + var] for var in ["x", "z", "y", "g", "w"]))
        self.initialized = True     

    def info(self):
        for split, dataset in self.splits.items():
            print(split)
            dataset.info(verbose = (split=="train"))

    def generate_data(self, num_data, **args):
        raise NotImplementedError()

    def true_g_function(self, x):
        raise NotImplementedError()

    def get_setup_args(self):
        if self.initialized is False:
            raise LookupError("trying to access setup args before calling 'setup'")
        return self.setup_args

    def get_data(self, split):
        if self.initialized is False:
            raise LookupError("trying to access data before calling 'setup'")
        elif self.splits[split] is None:
            raise ValueError("no training data to get")
        return self.splits[split].as_tuple()
    
    def get_train_data(self):
        return self.get_data("train")
        
    def get_dataset(self, split):
        if self.initialized is False:
            raise LookupError("trying to access data before calling 'setup'")
        elif self.splits[split] is None:
            raise ValueError("no training data to get")
        return self.splits[split]
        
    def get_dev_data(self):
        return self.get_data("dev")

    def get_test_data(self):
        return self.get_data("test")
    
    def iterate_data(self, split, batch_size):
        """
        iterator over training data, using given batch size
        each iteration returns batch as tuple (x, z, y, g, w)
        """
        if self.initialized is False:
            raise LookupError("trying to access data before calling 'setup'")
        elif self.splits[split] is None:
            raise ValueError("no " + split + " data to iterate over")
        x, z, y, g, w = self.splits[split].as_tuple()
        n = len(y)
        idx = self._get_random_index_order(n, batch_size)
        num_batches = len(idx) // batch_size
        for batch_i in range(num_batches):
            yield self._get_batch(batch_i, batch_size, x, z, y, g, w, idx)
            
    def iterate_train_data(self, batch_size):
        return iterate_data(self, "train", batch_size)

    def iterate_dev_data(self, batch_size):
        return iterate_data(self, "dev", batch_size)

    def iterate_test_data(self, batch_size):
        return iterate_data(self, "test", batch_size)

    @staticmethod
    def _get_batch(batch_num, batch_size, x, z, y, g, w, index_order):
        l = batch_num * batch_size
        u = (batch_num + 1) * batch_size
        idx = index_order[l:u]
        return x[idx], z[idx], y[idx], g[idx], w[idx]

    @staticmethod
    def _get_random_index_order(num_data, batch_size):
        idx = list(range(num_data))
        idx.extend(random.sample(idx, num_data % batch_size))
        random.shuffle(idx)
        return idx


