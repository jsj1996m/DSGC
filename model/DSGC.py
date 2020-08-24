"""

"""

from time import time
import numpy as np
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.models import Model
from keras import callbacks
from sklearn.cluster import KMeans
from model.autoEncoder import autoEncoder
from tools.metrics import cos_sim
from tools import matFunction, metrics

S_batch = K.variable(K.ones([256,256]))

class EmbeddingLayer(Layer):

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(EmbeddingLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight((self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    # def get_config(self):
    #     config = {'n_clusters': self.n_clusters}
    #     base_config = super(EmbeddingLayer, self).get_config()
    #     return dict(list(base_config.items()) + list(config.items()))


class DSGC(object):
    def __init__(self,
                 dims,
                 n_clusters=10,
                 alpha=1.0,threshold=0.85,
                 init='glorot_uniform'):

        super(DSGC, self).__init__()

        self.dims = dims
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1
        self.threshold = threshold
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.autoEncoder, self.encoder = autoEncoder(self.dims, init=init)

        # prepare DSGC model
        embeddingLayer = EmbeddingLayer(self.n_clusters, name='embedding')(self.encoder.output)
        self.model = Model(inputs=self.encoder.input, outputs=embeddingLayer)

    def pretrain(self, x, y=None, optimizer='adam', epochs=200, batchSize=256, save_dir='results/temp'):
        print('...Pretraining...')
        self.autoEncoder.compile(optimizer=optimizer, loss='mse')

        csv_logger = callbacks.CSVLogger(save_dir + '/pretrain_log.csv')
        cb = [csv_logger]
        if y is not None:
            class PrintACC(callbacks.Callback):
                def __init__(self, x, y):
                    self.x = x
                    self.y = y
                    super(PrintACC, self).__init__()

                def on_epoch_end(self, epoch, logs=None):
                    if int(epochs/10) != 0 and epoch % int(epochs/10) != 0:
                        return
                    feature_model = Model(self.model.input,
                                          self.model.get_layer(
                                              'encoder_%d' % (int(len(self.model.layers) / 2) - 1)).output)
                    features = feature_model.predict(self.x)
                    km = KMeans(n_clusters=len(np.unique(self.y)), n_init=20, n_jobs=4)
                    y_pred = km.fit_predict(features)
                    # print()
                    print(' ' * 8 + '|==>  acc: %.4f,  nmi: %.4f  <==|'
                          % (metrics.acc(self.y, y_pred), metrics.nmi(self.y, y_pred)))

            cb.append(PrintACC(x, y))

        # begin pretraining
        t0 = time()
        self.autoEncoder.fit(x, x, batch_size=batchSize, epochs=epochs, callbacks=cb)
        print('Pretraining time: %ds' % round(time() - t0))
        self.autoEncoder.save_weights(save_dir + '/ae_weights.h5')
        print('Pretrained weights are saved to %s/ae_weights.h5' % save_dir)
        self.pretrained = True

    def load_weights(self, weights):
        self.model.load_weights(weights)

    def extract_features(self, x):
        return self.encoder.predict(x)

    def predict(self, x):
        q = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(self, optimizer='sgd', loss='kld'):
        self.model.compile(optimizer=optimizer, loss=loss)

    def fit(self, x, y=None, maxiter=2e4, batchSize=256, tol=1e-3,
            updateInterval=140, save_dir='./results/temp'):

        print('Update interval', updateInterval)
        save_interval = int(x.shape[0] / batchSize) * 5  # 5 epochs
        print('Save interval', save_interval)

        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        encodeX = self.encoder.predict(x)
        print("starting matlab...")
        S = np.array(matFunction.getCANS(encodeX, 4))
        S = np.sign(S)
        print("end matlab")
        print("rebuilding S")
        for i in range(len(S)):
            j = i
            if i % int(len(S) / 10) == 0:
                print("calc Sim in " + str(i / int(len(S) / 100)) + "%")
            while (j < len(S)):
                if (cos_sim(encodeX[i], encodeX[j]) > self.threshold):
                    S[i][j] = 1
                    S[j][i] = 1

                j += 1



        y_pred = kmeans.fit_predict(encodeX)
        #--
        y_pred_last = np.copy(y_pred)
        self.model.get_layer(name='embedding').set_weights([kmeans.cluster_centers_])

        import csv
        logfile = open(save_dir + '/DSGC_log.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'loss'])
        logwriter.writeheader()

        loss = 0
        index = 0
        indexArray = np.arange(x.shape[0])
        for ite in range(int(maxiter)):
            if ite % updateInterval == 0:
                q = self.model.predict(x, verbose=0)
                p = self.target_distribution(q)

                y_pred = q.argmax(1)
                if y is not None:
                    acc = np.round(metrics.acc(y, y_pred), 5)
                    nmi = np.round(metrics.nmi(y, y_pred), 5)
                    ari = np.round(metrics.ari(y, y_pred), 5)
                    loss = np.round(loss, 5)
                    logdict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, loss=loss)
                    logwriter.writerow(logdict)
                    print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc, nmi, ari), ' ; loss=', loss)
                # check stop criterion
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                if ite > 0 and delta_label < tol and delta_label > 0:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    logfile.close()
                    break

            idx = indexArray[index * batchSize: min((index+1) * batchSize, x.shape[0])]
            S_temp = K.cast_to_floatx(S[idx][:, idx])
            one = np.eye(batchSize, dtype=K.floatx())
            if len(S_temp) == len(one):
                S_temp -= one
            else:
                index = index + 1 if (index + 1) * batchSize <= x.shape[0] else 0
                ite += 1
                continue
            # # print(S_batch)
            K.set_value(S_batch,S_temp)
            # self.compile(optimizer="adam", loss='kld')

            loss = self.model.train_on_batch(x=x[idx], y=p[idx])
            index = index + 1 if (index + 1) * batchSize <= x.shape[0] else 0

            # save intermediate model
            if ite % save_interval == 0:
                print('saving model to:', save_dir + '/DSGCModel_' + str(ite) + '.h5')
                self.model.save_weights(save_dir + '/DSGCModel_' + str(ite) + '.h5')

            ite += 1

        # save the trained model
        logfile.close()
        print('saving model to:', save_dir + '/DSGCModelFinal.h5')
        self.model.save_weights(save_dir + '/DSGCModelFinal.h5')

        return y_pred


def loss_func(S,batchSize):

    def pairwise_loss_func(y_true, predicted):
        X2 = K.reshape(K.transpose(K.repeat(K.transpose(K.batch_dot(predicted, K.transpose(predicted))), batchSize)), [batchSize, batchSize])
        theta = X2 - 2 * K.dot(predicted, K.transpose(predicted)) + K.transpose(X2)
        sim = S
        theta = theta / K.mean(theta)
        notSim = K.ones([batchSize,batchSize]) - sim - K.eye(batchSize)

        # kld
        kld_y_true = K.clip(y_true, K.epsilon(), 1)
        kld_y_pred = K.clip(predicted, K.epsilon(), 1)
        kldLoss = K.sum(kld_y_true * K.log(kld_y_true / kld_y_pred), axis=-1)


        loss = -K.mean(notSim * theta - K.log(1 + K.exp(theta)))
        return   kldLoss +  loss
    return pairwise_loss_func

    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='mnist',
                        choices=['mnist', 'fmnist', 'usps', 'reuters10k', 'stl'])
    parser.add_argument('--batchSize', default=256, type=int)
    parser.add_argument('--maxiter', default=2e4, type=int)
    parser.add_argument('--threshold ', default=0.85, type=float)
    parser.add_argument('--pretrainEpochs', default=None, type=int)
    parser.add_argument('--updateInterval', default=None, type=int)
    parser.add_argument('--tol', default=0.000000001, type=float)
    parser.add_argument('--ae_weights', default=None)
    parser.add_argument('--save_dir', default='results')
    args = parser.parse_args()
    print(args)
    import os
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load dataset
    from tools.datasets import load_data
    x, y = load_data(args.dataset)
    n_clusters = len(np.unique(y))

    S_batch = K.variable(K.ones([256, 256]))

    init = 'glorot_uniform'
    pretrainOptimizer = 'adam'
    # setting parameters
    if args.dataset == 'mnist' or args.dataset == 'fmnist':
        updateInterval = 140
        pretrainEpochs = 300
        init = VarianceScaling(scale=1. / 3., mode='fan_in',
                               distribution='uniform')  # [-limit, limit], limit=sqrt(1./fan_in)
        pretrainOptimizer = SGD(lr=1, momentum=0.9)
    elif args.dataset == 'reuters10k':
        updateInterval = 30
        pretrainEpochs = 50
        init = VarianceScaling(scale=1. / 3., mode='fan_in',
                               distribution='uniform')  # [-limit, limit], limit=sqrt(1./fan_in)
        pretrainOptimizer = SGD(lr=1, momentum=0.9)
    elif args.dataset == 'usps':
        updateInterval = 30
        pretrainEpochs = 50
    elif args.dataset == 'stl':
        updateInterval = 30
        pretrainEpochs = 10

    if args.updateInterval is not None:
        updateInterval = args.updateInterval
    if args.pretrainEpochs is not None:
        pretrainEpochs = args.pretrainEpochs

    # prepare the DSGC model
    DSGC = DSGC(dims=[x.shape[-1], 500, 500, 2000, 10], n_clusters=n_clusters, init=init)

    if args.ae_weights is None:
        DSGC.pretrain(x=x, y=y, optimizer=pretrainOptimizer,
                      epochs=pretrainEpochs, batchSize=args.batchSize,
                      save_dir=args.save_dir)
    else:
        DSGC.autoEncoder.load_weights(args.ae_weights)

    DSGC.model.summary()
    t0 = time()
    DSGC.compile(optimizer=SGD(0.01, 0.9), loss=loss_func(S_batch, 256))
    y_pred = DSGC.fit(x, y=y, tol=args.tol, maxiter=args.maxiter, batchSize=args.batchSize,
                      updateInterval=updateInterval, save_dir=args.save_dir)


import numpy as np
from model.DSGC import DSGC,loss_func
from keras.initializers import VarianceScaling
from keras.optimizers import SGD
import keras.backend as K



if __name__ == "__main__":
    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='mnist',
                        choices=['mnist', 'fmnist', 'usps', 'reuters10k', 'stl'])
    parser.add_argument('--batchSize', default=256, type=int)
    parser.add_argument('--maxiter', default=2e4, type=int)
    parser.add_argument('--threshold ', default=0.85, type=float)
    parser.add_argument('--pretrainEpochs', default=None, type=int)
    parser.add_argument('--updateInterval', default=None, type=int)
    parser.add_argument('--tol', default=0.000000001, type=float)
    parser.add_argument('--ae_weights', default=None)
    parser.add_argument('--save_dir', default='../results')
    args = parser.parse_args()
    print(args)
    import os
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load dataset
    from tools.datasets import load_data
    x, y = load_data(args.dataset)
    n_clusters = len(np.unique(y))


    init = 'glorot_uniform'
    pretrainOptimizer = 'adam'
    # setting parameters
    if args.dataset == 'mnist' or args.dataset == 'fmnist':
        updateInterval = 140
        pretrainEpochs = 300
        init = VarianceScaling(scale=1. / 3., mode='fan_in',
                               distribution='uniform')  # [-limit, limit], limit=sqrt(1./fan_in)
        pretrainOptimizer = SGD(lr=1, momentum=0.9)
    elif args.dataset == 'reuters10k':
        updateInterval = 30
        pretrainEpochs = 50
        init = VarianceScaling(scale=1. / 3., mode='fan_in',
                               distribution='uniform')  # [-limit, limit], limit=sqrt(1./fan_in)
        pretrainOptimizer = SGD(lr=1, momentum=0.9)
    elif args.dataset == 'usps':
        updateInterval = 30
        pretrainEpochs = 50
    elif args.dataset == 'stl':
        updateInterval = 30
        pretrainEpochs = 10

    if args.updateInterval is not None:
        updateInterval = args.updateInterval
    if args.pretrainEpochs is not None:
        pretrainEpochs = args.pretrainEpochs

    # prepare the DSGC model
    DSGC = DSGC(dims=[x.shape[-1], 500, 500, 2000, 10], n_clusters=n_clusters, init=init)

    if args.ae_weights is None:
        DSGC.pretrain(x=x, y=y, optimizer=pretrainOptimizer,
                     epochs=pretrainEpochs, batchSize=args.batchSize,
                     save_dir=args.save_dir)
    else:
        DSGC.autoEncoder.load_weights(args.ae_weights)

    DSGC.model.summary()
    t0 = time()
    DSGC.compile(optimizer=SGD(0.01, 0.9), loss=loss_func(S_batch,256))
    y_pred = DSGC.fit(x, y=y, tol=args.tol, maxiter=args.maxiter, batchSize=args.batchSize,
                     updateInterval=updateInterval, save_dir=args.save_dir)

