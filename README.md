# Detecting Sockpuppetry on Wikipedia using Meta-learning

Code for running experiments for my masters thesis, which later became the paper [*Detecting Sockpuppetry on Wikipedia using Meta-learning*](https://aclanthology.org/2025.acl-long.1083/). Briefly, this code trains transformer encoders to produce text authorship embeddings, as well as additional fully connected neural nets to form a binary classification. It includes options to train these encoders on single tasks, or to perform meta-learning over a distribution of tasks for improved generalisation. A significantly better overview of the architecture (with diagrams) and description of the method can be found in the paper.

This code was specifically designed for use with the [*WikiSocks*](https://github.com/lraszewski/wiki-socks/) dataset I collected. As I was working on this project alone, I concede it has insufficient documentation, and requires a significant cleanup. I am hoping to do so in the near future to sit alongside my published dataset and paper.

`main.py` contains the core code for running my experiments. Remaining files include libraries to interact with the dataset and implement meta-learning algorithm logic, as well as scripts that provide some basic analyses of the data, or perform hyper-parameter tuning.

## ACL Poster

![Poster](poster.png)
