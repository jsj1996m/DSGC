# Deep Structured Graph Clustering Network



#### Dependencies

- python 3.6
- keras 2.2.4
- scikit-learn 0.19.2
- tensorflow  1.11.0
- Matlab R2017b


#### Datasets
In this project, we conduct experimented on six public datasets：
- MNIST
- USPS
- Fashion_MNIST
- STL-10
- CIFAR-10
- REUTERS10k

We provide 5 datasets (MNIST, USPS, Fashion_MNIST, STL-10, CIFAR-10) in this code, and you can get REUTERS10k form [here](https://trec.nist.gov/data/reuters/reuters.html).

#### Usage
1. Install the matlab engine for python

       cd "matlabroot/extern/engines/python"
       python setup.py install
You might need administrator privileges to execute these commands.
2. Train DSGC in parallel by:

       $ python DSGC.py --dataset mnist