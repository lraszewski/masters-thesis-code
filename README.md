# Detecting Sockpuppetry on Wikipedia using Meta-learning

Code for my masters thesis, which later became the paper [*Detecting Sockpuppetry on Wikipedia using Meta-learning*](https://aclanthology.org/2025.acl-long.1083/).

This code was specifically designed for use with the [*WikiSocks*](https://github.com/lraszewski/wiki-socks/) dataset I collected. As I was working on this project alone, I concede it has insufficient documentation, and requires a significant cleanup. I am hoping to do so in the near future to sit alongside my published dataset and paper.

`main.py` contains the core code for running my experiments. Remaining files include libraries to interact with the dataset and implement meta-learning algorithm logic, as well as scripts that provide some basic analyses of the data, or perform hyper-parameter tuning.
