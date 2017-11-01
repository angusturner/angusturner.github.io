---
layout: post
title:  "Gaussian Mixture Models in PyTorch"
date:   2017-10-30 18:58:12 +1000
categories: ml pytorch
---

{% include mathjax.html %}

### Introduction

Mixture models allow rich probability distributions to be represented as a combination
of simpler "component" distributions. They can be applied to a variety of statistical
learning problems such as unsupervised clustering, classification and instance segmentation.
It is an interesting class of models to study because it introduces the concepts of
latent variables and density estimation in an intuitive way. It is also a precursor
to more sophisticated generative models such as VAEs and GANs.

###  Why PyTorch?

[PyTorch][pytorch] is the slick new deep learning framework from Facebook.
I have chosen it for this project for a few reasons:
- Powerful linear algebra functions as part of the `torch.Tensor` API.
- Device agnostic development. i.e. the same code runs on CPU and GPU.
- Automatic differentiation! Instead of EM, we could also fit the parameters
with backpropagation. More on this later.


### The Gaussian Mixture Model

A gaussian mixture model (GMM) with $$ K $$ components takes the form:

$$
  p(x) = \sum_{k=1}^{K}p(x|z=k)p(z=k)
$$

where $$ z $$ is a categorical latent variable indicating the component identity. For
brevity we will denote the prior $$ \pi_k := p(z=k) $$ . The likelihood term for
the kth component is the parameterised gaussian:

$$
  p(x|z=k)\sim\mathcal{N}(\mu_k, \Sigma_k)
$$

Our goal is to learn the means $$ \mu_k $$ , covariances $$ \Sigma_k $$
and priors $$ \pi_k $$ using an iterative procedure called expectation
maximisation (EM).

The basic EM algorithm has three steps:
1. Randomly initialise the parameters of the component distributions.
2. Estimate the probability of each data point under the component parameters.
3. Recalculate the parameters based on the estimated assignment probabilities. Repeat Step 2.

Convergence is reached when the total likelihood of the data under the model stops
decreasing.

### Synthetic Data

In order to quickly test my implementation, I created a synthetic dataset of
points sampled from three 2-dimensional gaussians, as follows:

{% highlight python %}
def sample(mu, var, nb_samples=500):
    """
    :param mu: torch.Tensor (features)
    :param var: torch.Tensor (features) (note: zero covariance)
    :return: torch.Tensor (nb_samples, features)
    """
    out = []
    for i in range(nb_samples):
        out += [
            torch.normal(mu, var.sqrt()).view(1, -1)
        ]
    return torch.cat(out, dim=0)
{% endhighlight %}


![Synthetic]({{ "/assets/images/synthetic.png" }})


### Initialising the Parameters

For the sake of simplicity, I just randomly select `K` points from my dataset to
act as initial means. I use a fixed initial variance and equal priors.

{% highlight python %}
def initialize(data, K, var=1):
  """
  :param data: design matrix (examples, features)
  :param K: number of gaussians
  :param var: initial variance
  """
  # choose k points from data to initialize means
  M = data.size(0) # nb. of training examples
  idxs = Variable(torch.from_numpy(np.random.choice(M, K, replace=False)))
  mu = data[idxs]

  # fixed variance
  N = data.size(1) # nb. of features
  var = Variable(torch.Tensor(K, N).fill_(var))

  # equal priors
  pi = Variable(torch.Tensor(K).fill_(1)) / K

  return mu, var, pi
{% endhighlight %}


### The Multivariate Gaussian

Step 2. of the EM algorithm requires us to compute the relative likelihood of
each data point under each component. The p.d.f of the
multivariate gaussian is

$$
   p(x;\mu, \sigma)=\frac{1}{\sqrt{2\pi|\Sigma|} }\exp\left(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\right)
$$


By only considering diagonal covariance matrices $$ I\sigma^2 = \Sigma $$ ,
we can deal with the variances directly. This greatly simplifies our computation.
Firstly, we can exploit the fact that the determinant is equal to the product of the trace.

$$
  |\Sigma| = \prod_{j=1}^{N}\sigma_{j}^{2}
$$

Instead of computing the matrix inverse we can simply invert the variances.

$$
  Tr(\Sigma^{-1}) = \left[\sigma_1^{-2}, ..., \sigma_N^{-2}\right]
$$

And lastly, the exponent simplifies to,

$$
  -\frac{1}{2}\left(\left(x-\mu\right)\odot\left(x-\mu\right)\right)^{T}\sigma^{-2}
$$

where $$ \odot $$ represents elementwise multiplication and $$ \sigma^{-2} $$
is our vector of inverse variances.

### Calculating Likelihoods

The calculation of this function can be vectorised in PyTorch, such that
all likelihoods are computed under each component in parallel. Note that PyTorch
overloads the Python `@` so that it calls `torch.bmm`, allowing us to parallelise
over the batch (component) dimension.

{% highlight python %}
def get_k_likelihoods(X, mu, var):
    """
    Compute the densities of each data point under the parameterised gaussians.

    :param X: design matrix (examples, features)
    :param mu: the component means (K, features)
    :param var: the component variances (K, features)
    :return: relative likelihoods (K, examples)
    """

    if var.data.eq(0).any():
        raise Exception('variances must be nonzero')

    # get the trace of the inverse covar. matrix
    covar_inv = 1. /  var # (K, features)

    # compute the coefficient
    det = (2 * np.pi * var).prod(dim=1) # (K)
    coeff = 1. / det.sqrt() # (K)

    # tile the design matrix `K` times on the batch dimension
    K = mu.size(0)
    X = X.unsqueeze(0).repeat(K, 1, 1)

    # calculate the exponent
    a = (X - mu.unsqueeze(1)) # (K, examples, features)
    exponent = a ** 2 @ covar_inv.unsqueeze(2)
    exponent = -0.5 * exponent

    # compute probability density
    P = coeff.view(K, 1, 1) * exponent.exp() # (K, examples, 1)

    return P.squeeze(2)
{% endhighlight %}


### Computing Posteriors

In order to recompute the parameters, we need to apply Bayes rule to likelihoods,
as follows:

$$
  p(z|x)=\frac{p(x|z)p(z)}{\sum_{k}^{K}p(x|z=k)p(z=k)}
$$

The resulting values are sometimes referred to as the "membership weights",
and represent the extent to which component the component identity $$ z $$ can
explain the observation $$ x $$.


{% highlight python %}
def get_posteriors(P, pi, eps=1e-6):
  """
  :param P: likelihoods p(x|z) under each gaussian (K, examples)
  :param pi: priors p(z) (must sum to 1) (K)
  :return: posterior p(z|x): (K, examples)
  """
  P_sum = torch.sum(P, dim=0, keepdim=True)
  return (P / (P_sum+eps)) * pi.unsqueeze(1)
{% endhighlight %}

### Parameter Update

Using the membership weights, the parameter update proceeds in three steps:
1. Set new mean for each component to a weighted average of the data points.
2. Set new covariance matrix as weighted combination of covariances for each data point.
3. Set new prior, as the normalised sum of the membership weights.

{% highlight python %}
def get_parameters(X, gamma, eps=1e-6):
  """
  :param X: design matrix (examples, features)
  :param gamma: the posterior probabilities p(z|x) (K, examples)
  :returns mu, var, pi: (K, features) , (K, features) , (K)
  """

  # compute `N_k` the proxy "number of points" assigned to each distribution.
  K = gamma.size(0)
  N_k = torch.sum(gamma, dim=1) + eps # (K)
  N_k = N_k.view(K, 1, 1)

  # tile X on the `K` dimension
  X = X.unsqueeze(0).repeat(K, 1, 1)

  # get the means by taking the weighted combination of points
  mu = gamma.unsqueeze(1) @ X # (K, 1, features)
  mu = mu / N_k

  # compute the diagonal covar. matrix, weighting contributions from each point
  A = X - mu
  var = gamma.unsqueeze(1) @ (A ** 2) # (K, 1, features)
  var = var / N_k

  # recompute the mixing probabilities
  m = X.size(1) # nb. of training examples
  pi = N_k / N_k.sum()

  return mu.squeeze(1), var.squeeze(1), pi.view(-1)
{% endhighlight %}

### Results

Apart from some simple training logic, that is the bulk of the algorithm! Fitting
the synthetic dataset I presented earlier:

![1]({{ "/assets/images/fig_1.png" }})

![1]({{ "/assets/images/fig_5.png" }})

![1]({{ "/assets/images/fig_8.png" }})

The Jupyter Notebook for this project is available on my Github.

[pytorch]: https://pytorch.org
[kmeans]: https://en.wikipedia.org/wiki/K-means_clustering
