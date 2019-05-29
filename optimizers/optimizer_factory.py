

class OptimizerFactory(object):
    def __init__(self, optimizer_class, **optimizer_kwargs):
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs

    def __str__(self):
        out = self.optimizer_class.__name__ + "::"
        for k, v in self.optimizer_kwargs.items():
            out += ":%r=%r" % (k, v)
        return out

    def __call__(self, model):
        return self.optimizer_class(model.parameters(), **self.optimizer_kwargs)
