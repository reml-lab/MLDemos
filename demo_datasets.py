import numpy as np
from sklearn import datasets as skdatasets

def make_blobs():
    N = 1000
    X = np.vstack((2 * np.random.randn(int(N / 2), 2) - 3, (2 * np.random.randn(int(N / 2), 2) + 3)))
    y = np.hstack((np.zeros((int(N / 2))), np.ones((int(N / 2)))))
    ind = np.random.permutation(range(N))
    X=X[ind, :]
    y=y[ind]
    return(X,y)

def make_circles():
    N=1000
    X, y = skdatasets.make_circles(n_samples=N, shuffle=True, noise=0.08, random_state=None, factor=0.4)

    # Rescale the data to [-9, 9] range.
    X[:, 0] = (X[:, 0] + 1) * 18 / 2 + (-9)
    X[:, 1] = (X[:, 1] + 1) * 18 / 2 + (-9)

    return(X,y)

def make_xor():
    # generate points for the four quadrants
    N=1000
    n_4 = int(N/4)
    x1 = np.random.normal(loc=[5, 5], scale=2, size=(n_4, 2))
    x2 = np.random.normal(loc=[-5, 5], scale=2, size=(n_4, 2))
    x3 = np.random.normal(loc=[-5, -5], scale=2, size=(n_4, 2))
    x4 = np.random.normal(loc=[5, -5], scale=2, size=(n_4, 2))

    # make quadrant 1 and 3 as -1 label, 2 and 4 as +1 label.
    X = np.vstack((x1, x3, x2, x4))
    y = np.hstack((np.zeros(n_4 * 2), np.ones(n_4 * 2)))

    ind = np.random.permutation(range(N))  # indices to shuffle the data.
    X = X[ind, :]
    y = y[ind]

    return(X,y)

def make_moons():
    N=1000
    X, y = skdatasets.make_moons(n_samples=N, shuffle=True, noise=0.1, random_state=None)

    # Rescale the data to [-9, 9] range.
    old_min, old_max = np.min(X[:, 0]), np.max(X[:, 0])
    old_range = old_max - old_min
    X[:, 0] = (X[:, 0] - old_min) * 18 / old_range + (-9)

    old_min, old_max = np.min(X[:, 1]), np.max(X[:, 1])
    old_range = old_max - old_min
    X[:, 1] = (X[:, 1] - old_min) * 18 / old_range + (-9)

    return X,y

def make_random():
    N = 1000
    X = 20*(np.random.rand(N,2)-0.5)
    y = np.random.randint(0,2,N)
    return(X,y)

