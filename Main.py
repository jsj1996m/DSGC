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
