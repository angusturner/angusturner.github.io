---
layout: post
title:  "From VAEs to Diffusion Models"
date:   2021-06-29
categories: generative_models
comments: false
---

{% include mathjax.html %}
{% include styles.html %}

$$ \require{cancel} $$

### Introduction

Recently I have been studying a class of generative models known as diffusion probabilistic models. These models were proposed by Sohl-Dickstein 
et al. in 2015 [[1]](#citation-1), however they first caught my attention last year when Ho et al. released 
"Denoising Diffusion Probabilistic Models" [[2]](#citation-2). Building on [[1]](#citation-1), Ho et al. showed that a model 
trained with a stable variational objective could match or surpass GANs on image generation.

Since the release of DDPM there has been a wave of renewed interest in diffusion models. New works have extended their success to the domain
of audio modelling [[8]](#citation-8), text-to-speech [[9]]((#citation-9)), and multivariate time-series forecasting [[10]](#citation-10). Furthermore, 
as shown by Ho et al. [[2]](#citation-2), these models exhibit close connections with score-based models [[11]](#citation-11), and the two perspectives 
were recently unified under the framework of stochastic differential equations [[4]](#citation-4). 

As a generative model, diffusion models have a number of unique and interesting properties. For example, trained models are able to
perform inpainting [[1]](#citation-1) and zero-shot denoising [[8]](#citation-8) without being explicitly designed for these tasks. Furthermore,
the variational bound used in DDPM highlights further connections to variational autoencoders and neural compression.

In this blog I want to explore diffusion models as they are presented in Sohl-Dickstein et al. [[1]](#citation-1), and Ho et al. [[2]](#citation-2).
When I was working through these derivations, I found it useful to conceptualise these models in terms of their relationship to VAEs [[3]](#citation-3). 
For this reason, I want to work up from a simple VAE derivation, through to hierarchical extensions, and finally the deep generative model
presented in [[1]](#citation-1) and [[2]](#citation-2). I will conclude with some thoughts on the relationship between the two models,
and some possible ideas for further research.

I have also released a PyTorch implementation of my [code].

### Variational Autoencoders

I want to begin with a quick refresher of variational autoencoders [[3]](#citation-3). If this is all feels familiar to you, feel free to skip ahead!

Consider the following generative model,

$$
  p(x) = \int_{z}p_{\theta}(x|z)p_{\theta}(z)
$$  

In general, the form of the prior $$ p_{\theta}(z) $$ and likelihood $$ p_{\theta}(x|z) $$ will depend on the data we are trying to model.
For simplicity, assume $$ p(z) = \mathcal{N}(0, I) $$. For continous data we can set $$ p_{\theta}(x|z) $$ to a diagonal gaussian,
with entries $$ \mu_{\theta}(z) \text{, } \sigma_{\theta}(z) $$ given by the output of a neural network. 

<!-- If you have studied mixture distributions before (see my [previous post] for an
introduction) this equation will look quite familiar. We have just defined an infinite mixture of gaussians! 
For each value $$ z \sim \mathcal{N}(0, I) $$ there is a corresponding likelihood given by $$ \mathcal{N}(\mu_{\theta}(z), \sigma_{\theta}(z)) $$. -->

Typically the true distribution $$ p(x) $$ is unknown, and we would like to fit the model to some empirically observed subset $$ \hat{p}(x) $$.
We could estimate the model parameters $$ \theta $$ with MLE as follows,

$$
  \theta^{*} = \underset{\theta}{\text{argmax}} \text{ } \mathbb{E}_{x \sim \hat{p}(x) \text{, } z \sim p(z)}  \left[ \text{log } p_{\theta}(x|z; \theta) \right]
$$

Since we cannot compute this expectation in closed form we can resort to monte-carlo sampling, evaluating the expression for random 
samples $$ z \sim p(z) $$. However, it is well known that this approach does not scale to high dimensions of $$ z $$ [[5]](#citation-5).
Intuitively, the likelihood of a random latent code $$ z \sim p(z) $$ corresponding to any particular data point is extremely small
(especially in high dimensions!).

The VAE solves this problem by sampling $$ z $$ from a new distribution $$ q_{\phi}(z|x) $$, which is jointly optimised with the generative model. 
This focuses the optimisation on regions of high probability (i.e. latent codes that are likely to have generated $$ x $$).

This can be derived as follows:

$$
  p(x) = \int_{z} p_{\theta}(x|z)p(z) \\
  p(x) = \int q_{\phi}(z|x) \frac{p_{\theta}(x|z)p(z)}{q_{\phi}(z|x)}\\
  \text{log }p(x) = \text{log } \mathbb{E}_{z \sim q_{\phi}(z|x)} \left[ \frac{p_{\theta}(x|z)p(z)}{q_{\phi}(z|x)} \right]\\
  \text{log }p(x) \geq \mathbb{E}_{z \sim q_{\phi}(z|x)} \left[ \text{log } \frac{p_{\theta}(x|z)p(z)}{q_{\phi}(z|x)} \right]
$$

The final step follows from Jensen's inequality and the fact that log is a concave function. The right-hand side of this inequality
is the evidence lower-bound (ELBO), sometimes denoted $$ \mathcal{L}(\theta, \phi) $$. The ELBO provides a joint optimisation objective, 
which simultaneously updates the variational posterior $$ q_{\phi}(z|x) $$ and likelihood model $$ p_{\theta}(x|z) $$.

To optimise the ELBO we have to obtain gradients for $$ \phi $$ through a stochastic sampling operation $$ z \sim q_{\phi}(z|x) $$.
Kingma and Welling famously solved this with a "reparameterization trick", which defines the sampling process as a combination of
a deterministic data-dependent function and data-independent noise [[3]](#citation-3).

__Figure 1 - Graphical Model for VAE__
{: style="text-align: center;" }
<img class="image" src='{{ "/assets/images/vae.png" }}' height=200px>

So much has been written about VAEs that I am barely scratching the surface here. For more details see: [Further Reading].


### Hierarchical Variational Autoencoders

Having defined a VAE with a single stochastic layer, it is trivial to derive multi-layer extensions.
Consider a VAE with two latent variables $$ z_1 $$ and $$ z_2 $$. We begin by considering the joint 
distribution $$ p(x, z_1, z_2) $$ and marginalising out the latent variables:  

$$
  p(x) = \int_{z_1} \int_{z_2} p(x, z_1, z_2) dz_1, dz_2
$$

Once again, we can introduce a variational posterior as follows:

$$
  p(x) = \int \int q(z_1, z_2|x) \frac{p(x, z_1, z_2)}{q(z_1, z_2|x)}\\
  p(x) = \mathbb{E}_{z_1 ,z_2 \sim q(z_1, z_2|x)} \left[ \frac{p(x, z_1, z_2)}{q(z_1, z_2|x)} \right]
$$

Taking the log and applying Jensen's rule:

$$
  \text{log } p(x) \geq \mathbb{E}_{z_1 ,z_2 \sim q(z_1, z_2|x)} \left[ \text{log } \frac{p(x, z_1, z_2)}{q(z_1, z_2|x)} \right]
$$

Notice that I have written these expressions in terms of the joint distributions 
$$ p(x, z_1, z_2) $$ and $$ q(z_1, z_2|x) $$. I have done this to emphasise the fact that we are free
to factorise the inference and generative models as we see fit. In practice, some factorisations
are more suitable than others (as shown in [[7]](#citation-7)). For now lets consider the factorisation in Figure 2.

__Figure 2 - A Hierarchical VAE__
{: style="text-align: center;" }
<img class="image" src='{{ "/assets/images/hierarchical-vae.png" }}' height=200px>

Writing this out, we have the generative model,

$$
  p(x,z_1,z_2) = p(x|z_1)p(z_1|z_2)p(z_2)
$$

and the inference model,

$$
  q(z_1,z_2|x) = q(z_1|x)q(z_2|z_1)
$$

Substituting these factorizations into the ELBO, we get:

$$
  \mathcal{L}(\theta, \phi) = \mathbb{E}_{q(z_1, z_2|x)} 
  \left[ \text{log }p(x|z_1) - \text{log }q(z_1|x) + \text{log }p(z_1|z_2) - \text{log }q(z_2 | z_1) + \text{log }p(z_2)  \right]\\
$$

This can be alternatively written as a "reconstruction term", and the KL divergence between each inference layer and its corresponding prior:

$$
= \mathbb{E}_{q(z_1|z_2)} \left[ \text{log }p(x|z_1) \right] - D_{KL} \left( q(z_1|x) \mathrel{\Vert} p(z_1|x) \right)
  - D_{KL} \left( q(z_2|z_1) \mathrel{\Vert} p(z_2) \right)
$$


### Diffusion Probabilistic Models

Now consider the model in Figure 3. We have sequence of $$ T $$ variables, where $$ x_0 \sim p(x) $$ is
our observed data and $$ x_{1:T} $$ are latent variables.

__Figure 3 - Diffusion Probabilistic Model__
{: style="text-align: center;" }
<img class="image" src='{{ "/assets/images/diffusion.png" }}' height=200px>

The lower bound for this model should look quite familiar by now (negated, for consistency with Ho et al. [[2]](#citation-2)):

$$
  - \mathcal{L} = \mathbb{E}_{q} \left[ -\text{log }p(x_T) - \sum_{t \geq 1}^{T} \text{log } \frac{p_{\theta}(x_{t-1}|x_t)}{q(x_t|x_{t-1})} \right]
$$

In fact, we can think of diffusion models as a specific realisation of a hierarchical VAE. What sets
them apart is a unique inference model, which contains no learnable parameters and is constructed so that
the final latent distribution $$ q(x_T) $$ converges to a standard gaussian. This "forward process"
model is defined as follows:

$$
  q(x_t|x_{t-1}) = \mathcal{N}(x_T\mathrel{;} x_{t-1}\sqrt{1 - \beta_{t}}, \beta_{t}I)
$$

The variables $$ \beta_1 \dots \beta_{T} $$ define a fixed variance schedule, chosen such
that $$ q(x_T|x_0) \approx \mathcal{N}(0, I) $$. The forward process is illustrated below, 
transforming samples from the 2d swiss roll distribution into gaussian noise:

<style>
  .video {
    display: flex;
    margin-left: auto;
    margin-right: auto;
    /* width: 100%; */
    max-height: 500px;
    max-width: 500px;
  }
</style>

<video class="video" loop autoplay preload='auto' poster="/assets/images/fig_000.png">
  <source src='/assets/video/forward_process.mp4' type='video/mp4'>
</video>

A nice property of the forward process is that we can directly sample from any timestep:

$$
  \alpha_t := 1-\beta_t\\
  \bar{\alpha}_t := \prod_{s=1}^{t} \alpha_s \\
  q(x_t|x_0) = \mathcal{N}\left(\sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I \right)
$$

One consequence of this, is that we can draw random samples $$ t ~ \sim \{1 \dots T \} $$ as part of
training procedure, optimising a random set of conditionals $$ p_{\theta} ( x_{t-1} | x_t ) $$ 
with each minibatch. In other words, we do not have to do a complete pass of the forward and reverse processes
on each step of training.

Another consequence, is that we can make additional manipulations to the lower-bound to reduce the variance.
This proceeeds by observing the following result (from bayes rule): 

$$
  q(x_t|x_{t-1}) = \frac{q(x_{t-1}|x_t)q(x_t)}{q(x_{t-1})}
$$

The terms $$ q(x_t) $$ and $$ q(x_{t-1}) $$ are intractable to compute, as they would require 
marginalising over all data points. However, due to the markov propety of the forward process we have:

$$
  q(x_t|x_{t-1}) = q(x_t|x_{t-1},x_0)
$$

Therefore, by conditioning all terms on $$ x_0 $$ we arrive at the following expression.

$$
  q(x_t|x_{t-1}) = \frac{q(x_{t-1}|x_t,x_0)q(x_t|x_0)}{q(x_{t-1}|x_0)} \tag{1}
$$

We now have an equation with one unknown $$ q(x_{t-1}|x_t,x_0) $$. This is
referred to as the "forward process posterior", and we will return to it shortly. 
First, lets continue our work on the ELBO. 

Separating the contribution from the first term in the sum:

$$
  L = \mathbb{E}_{q} \left[ -\text{log }p(x_T) - \text{log } \frac{p_{\theta}(x_0|x_1)}{q(x_1|x_0)} -
  \sum_{t \gt 1}^{T} \text{log } \frac{p_{\theta}(x_{t-1}|x_t)}{q(x_t|x_{t-1})} \right] \tag{2}
$$

Substituting (1) into (2),

$$
  \mathbb{E}_{q} \left[ -\text{log }p(x_T) - \text{log } \frac{p_{\theta}(x_0|x_1)}{q(x_1|x_0)} -
  \sum_{t \gt 1}^{T} \left( \text{log } \frac{p_{\theta}(x_{t-1}|x_t)}{q(x_{t-1}|x_t,x_0)} + 
  \text{log } \frac{q(x_{t-1}|x_0)}{q(x_t|x_0)} \right) \right]
$$

Notice what happens what happens to the conditionals
$$ q(x_t|x_0) $$ when we expand the sum:

$$
  \cancel{\text{log }q(x_1|x_0)} - \cancel{\text{log }q(x_1|x_0)} + \dots + \text{log }q(x_T|x_0)
$$

These cancellations lead to the following negative bound:

$$
  L := \mathbb{E}_{q} \left[ 
    \underbrace{-\text{log }p(x_T) + \text{log }q(x_T|x_0)}_{L_T} - 
    \underbrace{\text{log } p_{\theta}(x_0|x_1)}_{L_0} - 
    \underbrace{\sum_{t \gt 1}^{T} \text{log } \frac{p_{\theta}(x_{t-1}|x_t)}{q(x_{t-1}|x_t,x_0)}}_{L_{t-1}}
  \right]
$$

$$ L_T $$ has no parameters, and simply measures the effectiveness of the forward process in producing a standard gaussian. 
The summands in $$ L_{t-1} $$ are KL divergences between gaussians, which can be calculated in closed-form (as opposed to
using high-variance monte-carlo estimates). Lastly, $$ L_0 $$ is the familiar reconstruction term.


### Denoising Diffusion Probabilistic Models

So far our derivation matches with the original Sohl-Dickstein et al. paper [[1]](#citation-1), with notation
borrowed from [[2]](#citation-2) for consistency. In DDPM, Ho et al. propose a specific parameterization of the
generative model, which simplifies the training and connects it to score based modelling.

We start by noting the form of the forward process posterior. This result can be derived from bayes rule, 
substituting the known gaussian conditionals into equation (1). For those interested, a detailed derivation 
is presented by Kong et al. [[8]](#citation-8) (Appendix A).

$$
  q(x_{t-1}|x_t,x_0) = \mathcal{N}(x_{t-1};\tilde{\mu}(x_t,x_0),\tilde{\beta}_t I)\\
  \tilde{\mu}_t(x_t,x_0) = \frac{ \sqrt{\bar{\alpha}_{t-1}} \beta_{t} }{1 - \bar{\alpha}_t} x_0 + 
  \frac{ \sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1}) }{1 - \bar{\alpha}_t} x_t \tag{3} \\
  \tilde{\beta}_t = \frac{ 1 - \bar{\alpha}_{t-1} }{1 - \bar{\alpha}_t } \beta_t
$$


The corresponding reverse process distributions are defined as:

$$
  p_{\theta}(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_{\theta}(x_t, t), \sigma_{t}^2 I)
$$

The variance $$ \sigma_{t}^2 $$ is a time-dependent constant (either $$ \beta_t $$ or $$ \tilde{\beta}_t) $$. The 
mean $$ \mu_{\theta}(x_t, t) $$ is a neural network, which takes as input $$ x_t $$. In order to share 
parameters between timesteps, the conditioning variable $$ t $$ is introduced in the form of positional 
embeddings [[12]](#citation-12).

With these definitions, the KL divergence terms in $$ L_{t-1} $$ reduce to the following
mean-square error:

$$
  L_{t-1} = \mathbb{E}_{t,x_t,x0} \left[ \frac{1}{2\sigma_t^2} \lVert \tilde{\mu}_t(x_t,x_0) - \mu_{\theta}(x_t, t) \rVert \right] + C \tag{4}
$$

This objective is suitable for optimisation, however Ho et al. continue expanding to arrive at an alternative 
denoising objective. Note that forward process samples $$ x_t $$ can be written as a function of
$$ x_0 $$ and some noise $$ \epsilon \sim \mathcal{N}(0, I) $$.

$$
  x_t(x_0, \epsilon) = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon \tag{5}
$$

Conversely, for a given sample $$ x_t(x_0, \epsilon) $$ we can express $$ x_0 $$
as,

$$
  x_0 = \frac{x_t(x_0, \epsilon) - \sqrt{1 - \bar{\alpha}_t} \epsilon}{\sqrt{\bar{\alpha}_t}} \tag{6}
$$

At this point, Ho et al. rewrite the forward process posterior mean (3) in terms of $$ x_t(x_0, \epsilon) $$,
by evaluating:

$$
  \tilde{\mu}_t \left( x_t(x_0, \epsilon), \frac{1}{\sqrt{\bar{\alpha}_t}} \left( x_t(x_0, \epsilon) - \sqrt{1 - \bar{\alpha}_t} \epsilon \right) \right)
$$

This expands to,

$$
  = 
  \frac{ \beta_{t} \sqrt{\bar{\alpha}_{t-1}} (x_t - \epsilon \sqrt{\bar{\alpha}_t}) }{\sqrt{\bar{\alpha}_t}(1 - \bar{\alpha}_t)}  + 
  \frac{ \sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1}) }{1 - \bar{\alpha}_t} x_t\\
  = \left( \frac{ \beta_{t} \sqrt{\bar{\alpha}_{t-1}} }{ \sqrt{\bar{\alpha}_t}(1 - \bar{\alpha}_t) } +
  \frac{ \sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1}) }{1 - \bar{\alpha}_t} \right) x_t -
  \frac{\beta_t \sqrt{1 - \bar{\alpha}_{t-1}}}{ \sqrt{\bar{\alpha}_t}(1 - \bar{\alpha}_t) } \epsilon
$$

At this point we have a considerable mess of algebraic terms, however it simplifies down
quite nicely. Recall the fact that $$ \bar{\alpha}_t = \bar{\alpha}_{t-1} \alpha_t $$, and 
$$ \beta_t = 1 - \alpha_t $$. We can simplify the coefficient of $$ x_t $$ by multiplying the top 
and bottom by $$ \sqrt{\alpha_t} $$:

$$
  = \frac{1}{\sqrt{\alpha_t}} \cdot \frac{\beta_t + \alpha_t(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}\\
  = \frac{1}{\sqrt{\alpha_t}} \cdot \frac{1 - \alpha_t + \alpha_t - \alpha_t \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}\\
  = \frac{1}{\sqrt{\alpha_t}}
$$

The coefficient of $$ \epsilon $$ simplifies to,

$$
  = - \frac{1}{\sqrt{\alpha_t}} \cdot \frac{1}{\sqrt{1 - \bar{\alpha}_t}}
$$

Now we can rewrite (4) as follows,

$$
  L_{t-1} - C = \mathbb{E}_{x_0,\epsilon,t} \left[ \frac{1}{2\sigma_t^2} \left\lVert \frac{1}{\sqrt{\alpha_t}} \cdot 
  \left( x_t(x_0, \epsilon) - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon \right) - 
  \mu_{\theta}(x_t(x_0, \epsilon), t) \right\rVert \right] \tag{7}
$$

From (7), we observe that the model $$ \mu_{\theta} $$ must predict,

$$
  \frac{1}{\sqrt{\alpha}} \cdot 
  \left( x_t(x_0, \epsilon) - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \right)
$$

In other words, we are providing the model with $$ x_t, t $$ and asking it to predict some affine transformation
of $$ x_t $$. This affine transform is a combination of time-dependent constants, and a single free parameter $$ \epsilon $$. 
This motivates an alternative parameterisation, where the neural net is only responsible for predicting the noise $$ \epsilon $$. 

$$
  \mu_{\theta}(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \cdot 
  \left( x_t(x_0, \epsilon) - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_{\theta}(x_t(x_0, \epsilon), t) \right) \tag{8}
$$

Substituting (8) into (7),

$$
  = \mathbb{E}_{x_0,\epsilon,t} \left[ \frac{1}{2\sigma_t^2} \left\lVert \frac{\beta_t}{ \sqrt{\alpha_t} \sqrt{1 - \bar{\alpha}_t} } \cdot 
  \left( \epsilon - \epsilon_{\theta}(x_t(x_0, \epsilon), t) \right) \right\rVert \right]\\
$$

$$
  = \mathbb{E}_{x_0,\epsilon,t} \left[ \frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1 - \bar{\alpha}_t)} \left\lVert 
   \epsilon - \epsilon_{\theta}(x_t(x_0, \epsilon), t) \right\rVert \right] \tag{9}
$$

Finally Ho et al. note that omitting the term $$ \frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1 - \bar{\alpha}_t)} $$
leads to weighted variant of the lower bound, which (aside from being simpler) leads to better empirical results.

$$
  = \mathbb{E}_{x_0,\epsilon,t} \left[ 
      \left\lVert \epsilon - \epsilon_{\theta}(x_t(x_0, \epsilon), t) \right\rVert 
    \right] \tag{10}
$$

### DDPM Interpretation

On its own the DDPM derivation can seem a bit daunting. Let's summarise:

- The training objective in Equation (10) corresponds to a single layer of the variational bound. 
- We sample a data point $$ x_0 \sim p(x) $$, a random timestep $$ t $$ and some noise $$ \epsilon \sim \mathcal{N}(0, I) $$ 
- Based on the properties of the forward process, we can compute $$ x_t(x_0, \epsilon) $$,
which represents a sample from $$ q(x_t|x_0) $$.
- The forward process posterior $$ q(x_{t-1}|x_t,x_0) $$ has a closed-form distribution,
which we compute based on knowing $$ x_0 $$ and $$ x_t $$.
- We want to minimise the KL divergence between each generative layer $$ p_{\theta}(x_{t-1}|x_t) $$
and the corresponding forward process posterior.
- By analysing the bound we see that minimising this divergence is equivalent to predicting
the noise $$ \epsilon $$ that is required to invert the forward process.

### DDPM as a kind of VAE

Okay, so while it technically optimises a variational bound, the DDPM model looks quite different
from the original VAE presented by Kingma. Some of you are probably thinking: "Is it really
fair to claim this is a 'kind of VAE'"? 

And you would have a fair point. I mean, DDPM has no learnable parameters for the inference model.
And it doesn't really resemble an autoencoder. However, I do think it is valuable to think about the
connections for a few reasons. First, VAEs have been studied extensively, whereas DDPM seems 
relatively underexplored. It is possible that insights from the VAE literature might
be translatable to DDPM and vice versa.

Furthermore, thinking about how DDPM fits within the current taxonomy of generative models
has left me with a lot of questions. Perhaps some of these already have answers in the literature,
and others could be a topic for future exploration.

1. Can we do a multi-scale DDPM model, which progressively factorises out latent dimensions by having a faster
forward process for certain subsets? For example, similar to the multi-scale architecture in RealNVP [[14]](#citation-14).
2. Can we fit a diffusion model in the latent space of an autoencoder, as a means to:
  - a) sample from a deterministic autoencoder, as was done with GMMs in [[13]](#citation-13).
  - b) learn regions of the latent space that correspond to certain known attributes [[15]](#citation-15)
3. Can we use more flexible distributions for the forward process? For example, it was noted in [[4]](#citation-4) that 
sampling $$ q(x_t|x_0) $$ requires solving "Kolmogorov’s forward equation" for the chosen
process. However, perhaps for some flexible $$ q_{\phi}(x_t|x_{t-1}) $$ we could also sample from a (jointly learned) 
approximation $$ q_{\phi}(x_t|x_0) $$ ? 

As another way to state 3., perhaps there is some middle ground between diffusion models and VAE. For example,
we could retain the DDPM property of training each layer independently with shared parameters, but also reintroduce 
the learned inference models that are distinct to VAEs. I don't feel qualified to say whether or 
not this is a good idea, however I hope that I will have time to test it out.

### Conclusion

If you've made it this far, I hope that you have found these notes helpful! There is so much more that can
be said about diffusion models, and in fact I still have a great deal to learn myself. For example, I have not even
touched on the connection to score-based models or SDEs. However, there is a great blog by [Yang Song] that goes into
detail about these connections. I would highly recommend it to those interested in learning more.

### References

{% include citation.html
    no="1"
    authors="Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, Surya Ganguli"
    title="Deep Unsupervised Learning using Nonequilibrium Thermodynamics"
    year="2015"
    link="https://arxiv.org/abs/1503.03585v8"
%}

{% include citation.html
    no="2"
    authors="Jonathan Ho, Ajay Jain, Pieter Abbeel"
    title="Denoising Diffusion Probabilistic Models"
    year="2020"
    link="https://arxiv.org/abs/2006.11239v2"
%}

{% include citation.html
    no="3"
    authors="Diederik P Kingma, Max Welling"
    title="Auto-Encoding Variational Bayes"
    year="2013"
    link="https://arxiv.org/abs/1312.6114"
%}

{% include citation.html
    no="4"
    authors="Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole"
    title="Score-Based Generative Modeling through Stochastic Differential Equations"
    year="2021"
    link="https://arxiv.org/abs/2011.13456v2"
%}

{% include citation.html
    no="5"
    authors="Adam Kosiorek"
    title="What is wrong with VAEs?"
    year="2018"
    link="https://akosiorek.github.io/ml/2018/03/14/what_is_wrong_with_vaes.html"
%}

{% include citation.html
    no="6"
    authors="Lilian Weng"
    title="From Autoencoder to Beta-VAE"
    year="2018"
    link="https://lilianweng.github.io/lil-log/2018/08/12/from-autoencoder-to-beta-vae.html"
%}

{% include citation.html
    no="7"
    authors="Jakub M. Tomczak, Max Welling"
    title="VAE with a VampPrior"
    year="2018"
    link="https://arxiv.org/abs/1705.07120v5"
%}

{% include citation.html
  no="8"
  authors="Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, Bryan Catanzaro"
  title="DiffWave: A Versatile Diffusion Model for Audio Synthesis"
  year="2021"
  link="https://arxiv.org/abs/2009.09761v3"
%}

{% include citation.html
  no="9"
  authors="Vadim Popov, Ivan Vovk, Vladimir Gogoryan, Tasnima Sadekova, Mikhail Kudinov"
  title="Grad-TTS: A Diffusion Probabilistic Model for Text-to-Speech"
  year="2021"
  link="https://arxiv.org/abs/2105.06337v1"
%}

{% include citation.html
  no="10"
  authors="Kashif Rasul, Calvin Seward, Ingmar Schuster, Roland Vollgraf"
  title="Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting"
  year="2021"
  link="https://arxiv.org/abs/2101.12072v2"
%}

{% include citation.html
  no="11"
  authors="Yang Song, Stefano Ermon"
  title="Generative Modeling by Estimating Gradients of the Data Distribution"
  year="2020"
  link="https://arxiv.org/abs/1907.05600v3"
%}

{% include citation.html
  no="12"
  authors="Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin"
  title="Attention is All You Need"
  year="2017"
  link="https://arxiv.org/abs/1706.03762v5"
%}

{% include citation.html
  no="13"
  authors="Partha Ghosh, Mehdi S. M. Sajjadi, Antonio Vergari, Michael Black"
  title="From Variational to Deterministic Autoencoders"
  year="2020"
  link="https://arxiv.org/abs/1903.12436v4"
%}

{% include citation.html
  no="14"
  authors="Laurent Dinh, Jascha Sohl-Dickstein, Samy Bengio"
  title="Density estimation using Real NVP"
  year="2017"
  link="https://arxiv.org/abs/1605.08803v3"
%}

{% include citation.html
  no="15"
  authors="Jesse Engel, Matthew Hoffman, Adam Roberts"
  title="Latent Constraints: Learning to Generate Conditionally from Unconditional Generative Models"
  year="2017"
  link="https://arxiv.org/abs/1711.05772v2"
%}


{% if page.comments %}
{% include disqus.html %}
{% endif %}

[pytorch]: https://pytorch.org
[github]: https://github.com/angusturner/
[code]: https://github.com/angusturner/diffuse
[twitter]: https://twitter.com/AngusTurner9
[previous post]: https://angusturner.github.io/generative_models/2017/11/03/pytorch-gaussian-mixture-model.html
[Yang Song]: https://yang-song.github.io/blog/2021/score/