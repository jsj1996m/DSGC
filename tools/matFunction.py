import matlab.engine
import matlab


def getCANS(x,n_cluster):
    engine = matlab.engine.start_matlab()
    return engine.getCANS(matlab.double(x.T.tolist()), n_cluster)