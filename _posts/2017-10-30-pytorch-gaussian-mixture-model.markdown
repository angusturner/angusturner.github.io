---
layout: post
title:  "Gaussian Mixture Models in PyTorch"
date:   2017-10-30 18:58:12 +1000
categories: ml pytorch
---

{% include mathjax.html %}

### Introduction

Mixture models allow complex multimodal distributions to be described by
a weighted combination of simpler distributions. They can be applied to a variety
of statistical learning problems such as unsupervised clustering, classification
and instance segmentation. In this post I will be implementing the
"Expectation Maximisation" algorithm in PyTorch to fit a mixture of multivariate
Gaussians to some unlabelled data.

###  Why PyTorch?

[PyTorch][pytorch] is the slick new deep learning framework from Facebook.
I have chosen it for this project for a few reasons:
- Powerful linear algebra functions as part of the `torch.Tensor` API.
- Device agnostic development. i.e. the same code runs on CPU and GPU.
- Automatic differentiation! Instead of E.M, we could also fit the parameters
with backpropagation. More on this later.

### The Gaussian Mixture Model

If you have worked with [k-means clustering][kmeans] before mixture models will
feel very familiar. Our goal is to learn the parameters for a mixture of `K`
component distributions that maximises the likelihood of the data. Note that
unlike k-means, where we assign each data point to a cluster, each observation in a
mixture model has an associated probability under every component.

In the gaussian case each component is parameterised by a mean $$ \mu_k $$ and covariance
matrix $$ \Sigma_k $$. We also define a "mixing coefficient" $$ \pi_k $$ to represent
the prior probability that a point was drawn from a particular component distribution.

The basic EM algorithm has three steps:
1. Randomly initialise the parameters of the `K` component distributions.
2. Estimate the likelihood of each data point under the component parameters.
3. Recalculate the parameters based on the estimated data likelihood. Repeat Step 2.

Step 2. and 3. are performed iteratively until the model converges. Convergence
is reached when the total likelihood of the data under the model stops
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

One recommended approach to initialising GMM is to use the centroids
from a precomputed k-means model. For the sake of simplicity, I just randomly
select `K` points from my dataset to act as initial means. I use a fixed initial
variance and equal priors.

{% highlight python %}
def initialize(data, K, var=1):
  """
  :param data: design matrix (examples, features)
  :param K: number of gaussians
  :param var: initial variance
  """
  # choose k points from data to initialize means
  M = data.size(0) # nb. of training examples
  idxs = Variable(torch.from_numpy(
      np.random.choice(M, K, replace=False)))
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
each data point under each component. The probability distribution of the
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

where $$ \odot $$ represents the hadamard product and $$ \sigma^{-2} $$ is our vector of
inverse variances.



### Implementation

I have vectorised my implementation so that the likelihood of all data points
under each component can be computed in parallel. Note that PyTorch overloads
the Python `@` so that it calls `torch.bmm`. This performs a batch of matrix
multiplies on Tensors of the form `(batch, rows, cols)`. In this case the batch
dimension corresponds the number of components `K`.

For those new to PyTorch, it is also worth noting the use of the `.unsqueeze(dim)`
method to add an additional singleton dimension in front of the specified dimension.

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
    P = coeff.view(K, 1, 1) * exponent.exp()

    # remove final dimension and return
    return P.squeeze(2)
{% endhighlight %}


### Computing Posteriors

The values returned by the likelihood function represent....


[pytorch]: https://pytorch.org
[kmeans]: https://en.wikipedia.org/wiki/K-means_clustering
