import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
noise_variance =  .05
def entropy_estimator_kl(activations, var=0.05):
    # Upper Bound
    # KL-based upper bound on entropy of mixture of Gaussians with covariance matrix var * I
    #  see Kolchinsky and Tracey, Estimating Mixture Entropy with Pairwise Distances, Entropy, 2017. Section 4.
    #  and Kolchinsky and Tracey, Nonlinear Information Bottleneck, 2017. Eq. 10
    N, dims = activations.shape
    dists = euclidean_distances(activations, activations, squared=True)

    dists2 = dists / (2*var)
    normconst = (dims/2.0)*np.log(2*np.pi*var)
    lprobs = np.log(np.sum(np.exp(-dists2), axis=1)) - np.log(N) - normconst
    h = -np.mean(lprobs, axis=0)
    return dims/2 + h
def entropy_estimator_bd(x, var= .05):
    # Lower bound of Marginal Entropy
    # Bhattacharyya-based lower bound on entropy of mixture of Gaussians with covariance matrix var * I
    #  see Kolchinsky and Tracey, Estimating Mixture Entropy with Pairwise Distances, Entropy, 2017. Section 4.
    N, dims = activations.shape
    val = entropy_estimator_kl(x,4*var)
    return val + np.log(0.25)*dims/2

def kde_condentropy(output, var= 0.05):
    # Return entropy of a multivariate Gaussian, in nats
    dims = output.shape[1]
    return (dims/2.0)*(np.log(2*np.pi*var) + 1)


if __name__ == "__main__":
    np.random.seed(19)
    n=1000
    d=10
    activations = np.random.random(size=(n,d))

    entropy_up = entropy_estimator_kl(activations)
    entropy_low = entropy_estimator_bd(activations)

    print(entropy_up)
    print(entropy_low)




