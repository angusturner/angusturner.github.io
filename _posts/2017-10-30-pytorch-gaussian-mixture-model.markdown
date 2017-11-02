---
layout: post
title:  "Gaussian Mixture Models in PyTorch"
date:   2017-11-03 18:58:12 +1000
categories: generative_models
---

{% include mathjax.html %}

### Introduction

Mixture models allow rich probability distributions to be represented as a combination
of simpler "component" distributions. For example, consider the mixture of 1-dimensional
gaussians in the image below:

![mixture]({{ "/assets/images/mixture.png" }})

While the representational capacity of a single gaussian is obviously limited,
a mixture is capable of approximating any distribution with an accuracy proportional
to the number of components<sup>2</sup>.

In practice mixture models are used for a variety of statistical learning problems
such as classification, image segmentation and clustering. My own interest
stems from their role as part of an increasingly diverse family of
generative models. Generative models are those which explicitly
model the data generation process. Though it is not always the goal, this permits
new data points to be sampled from the same distribution as the training data. In this
context mixture models are an important precursor to VAEs and
GANs (topics for a later post).

In this blog I will offer a brief introduction to the gaussian mixture model and
implement it in PyTorch. The full code will be available on my [github].

###  Why PyTorch?

PyTorch is the slick new deep learning framework from Facebook. I have
been using it in my work at [Popgun][popgun] for roughly 7 months now, and it has been a joy
to work with. Although it is primarily geared towards deep gradient based models,
it basically excels at any task with highly parallelizable matrix operations.
One of the key benefits is the ability to write "device agnostic" code. i.e the same
code can be run on both CPU and GPU with minimal adjustment. This makes it well
suited to the GMM.

### The Gaussian Mixture Model

A gaussian mixture model with $$ K $$ components takes the form<sup>1</sup>:

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
3. Recalculate the parameters based on the estimated probabilities. Repeat Step 2.

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
def get_posteriors(P, eps=1e-6):
    """
    :param P: the relative likelihood of each data point under each gaussian (K, examples)
    :return: p(z|x): (K, examples)
    """
    P_sum = torch.sum(P, dim=0, keepdim=True)
    return (P / (P_sum+eps))
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

Apart from some simple training logic, that is the bulk of the algorithm! Here is
a visualisation of EM fitting three components to the synthetic data I generated
earlier:

<video loop autoplay preload='auto' height='740px' width='740px' poster="/assets/images/fig_8.png">
  <source src='/assets/video/output.webm' type='video/webm; codecs="vp9, vorbis"'>
</video>


### Thanks for Reading!

If you found this post interesting or informative, have questions
or would like to offer feedback or corrections feel free to get in touch at my email
or on twitter. Thanks!

### References

For a more rigorous treatment of the EM algorithm see [1].

1. Bishop, C. (2006). Pattern Recognition and Machine Learning. Ch9.
2. Bengio, Y., Goodfellow, I. (2016). Deep Learning.


[pytorch]: https://pytorch.org
[kmeans]: https://en.wikipedia.org/wiki/K-means_clustering
[github]: https://github.com/angusturner
[popgun]: http://popgun.ai/
