---
layout: post
title:  "Gaussian Mixture Models in PyTorch"
date:   2017-10-30 18:58:12 +1000
categories: ml pytorch
---

### Introduction

Mixture models allow complex multimodal distributions to be described by
a weighted combination of simpler distributions. They can be applied to a variety
of statistical learning problems such as unsupervised clustering, classification
and instance segmentation. In this post I will be implementing the
"Expectation Maximisation" algorithm in PyTorch to fit a mixture of multivariate
Gaussians to some unlabelled data.

###  Why PyTorch?

[PyTorch][pytorch] is the slick new Deep Learning framework from Facebook.
I have chosen it for this project for a few reasons:
- Powerful linear algebra functions as part of the `torch.Tensor` API.
- "Device agnostic" development, allowing code to be run on both CPU and GPU.
- Automatic differentiation! A step of EM is fully differentiable. In a later post
I hope to exploit this fact to explore EM in combination with a neural net.
For now however, the `autograd` package will just quietly do its thing in the background
with no significant overhead.

### The Gaussian Mixture Model

If you have worked with [k-means clustering][kmeans] before mixture models will
feel very familiar to you. Like k-means we start with the assumption
that our data is comprised of `K` subpopulations. Given a set of data points with
`N` features, our goal is to learn the parameters of the `K` mixture components that
maximises the likelihood of the observed data. Unlike k-means, we do not strictly
assign each point to a component. Instead, each point has an associated likelihood
or probability under each of the `K` component gaussians.

The `K` component gaussians are each parameterised by a mean `mu`, covariance
matrix `sigma` (we will assume a diagonal covariance matrix for simplicity),
and a mixing coefficient `pi`. The mixing coefficient can be thought of as
the prior probability that a point was drawn from a particular distribution. Each
of these parameters can be iteratively learned by the EM algorithm.

The basic EM algorithm has three steps:
1. Randomly initialise the parameters of the `K` component distributions.
2. Estimate the likelihood of each data point under the component parameters.
3. Recalculate the parameters based on the estimated likelihood. Repeat Step 2.

Step 2. and 3. are performed iteratively until the model converges. Convergence
is reached when the total likelihood of the data under the model stops
decreasing.

### A synthetic dataset.

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


### Initialising the parameters

One recommended approach to initialising GMM is to use the centroids
from a precomputed k-means model. For the sake of simplicity, I just randomly
select `K` points from my dataset to act as initial means. I use a fixed initial variance
and equal priors.

{% highlight python %}
def initialize(data, k, var=1):
  """
  :param data: torch.Variable(examples, features)
  :param k: number of gaussians
  :param var: initial variance
  """
  # choose k points from data to initialize means
  m = data.size(0)
  idxs = Variable(torch.from_numpy(
      np.random.choice(m, k, replace=False)))
  mu = data[idxs]

  # fixed variance
  var = Variable(torch.Tensor(k, d).fill_(var))

  # equal priors
  pi = Variable(torch.Tensor(k).fill_(1)) / k

  return mu, var, pi
{% endhighlight %}


### The Multivariate Gaussian

Step 2. of the EM algorithm requires us to compute the relative likelihood of
each data point under each component. The p.d.f of the multivariate gaussian is:



[pytorch]: https://pytorch.org
[kmeans]: https://en.wikipedia.org/wiki/K-means_clustering
