---
layout: page
title: About Me
permalink: /about/
---

{% include styles.html %}

<style>
    .post-title {
        display: none;
    }
    .fine-print {
        font-size: 12px;
    }
</style>

### Profile

<img class="profile" src='{{ "/assets/images/profile_pic.png" }}'>

I am a machine learning engineer at [Splash], where I have been 
working on models of audio and symbolic music since 2017.

I have a background in web development, and completed my undergraduate degree in computer science at Queensland University of Technology.

I am super passionate about generative modelling, and related topics such as variational inference and neural compression. This blog is a
place for me to share some of my learnings, and link to other code and projects that I am proud of.

### Projects

{% include project.html
    desc="A blog post explaining some deep generative models of piano music, which I developed at Splash in 2017-2018.<sup>&#8224;</sup>"
    title="Deep Learning for Expressive Piano Performances"
    link="https://popgun-labs.github.io/ml-blog/generative_models/piano/symbolic_music/2020/02/01/beatnet.html"
%}
{% include project.html
    desc="Reusable components for deep generative modelling, and a lightweight experiment runner. Written in PyTorch.<sup>&#8224;</sup>"
    title="PopGen - Generative Modelling Toolkit"
    link="https://github.com/Popgun-Labs/PopGen"
%}
{% include project.html
    desc="A PyTorch implementation of the bandpass convolutions described in <a href='https://arxiv.org/abs/1811.09725'>Interpretable Convolutional Filters with SincNet</a>.<sup>&#8224;</sup>"
    title="SincNet Convolutions"
    link="https://github.com/Popgun-Labs/SincNetConv"
%}
{% include project.html
    desc="An exercise in learning rust. Implements a few algorithms from Andrew Ng's machine learning course. (Not maintained)."
    title="Rust Machine Learning Algorithms"
    link="https://github.com/angusturner/rustml"
%}
{% include project.html
    desc="Another exercise in learning rust. Implements breadth-first search, to solve a 2x2 Rubiks cube."
    title="2x2 Rubik's Cube Solver"
    link="rubiks-2x2-solver"
%}




<div class='fine-print'>
    <sup>&#8224;</sup>Work completed at <a href="https://www.splashhq.com/tools">Splash</a>.
</div>

[Splash]: https://www.splashhq.com/tools