__author__ = 'awbennett'

class AbstractMethod(object):
    def __init__(self):
        pass

    def fit(self, x_train, z_train, y_train, x_dev, z_dev, y_dev):
        raise NotImplementedError()

    def predict(self, x_test):
        raise NotImplementedError()
