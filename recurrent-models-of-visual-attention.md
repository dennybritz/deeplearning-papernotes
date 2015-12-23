## [Recurrent Models of Visual Attention](http://arxiv.org/abs/1406.6247)

TLDR; The authors train a RNN that takes as input a glimpse (part of the image subsamples to same size) and outputs a new glimpse and action (prediction, agent move) at each step. Thus, the model adaptively selects which part of an image to "attend" to. By defining the number of glimpses and their reoslutions we can control the complexity of the model independently of image size, which is not true for CNNs. The model is not differentiable, but can be trained using Reinforcement Learning techniques. The authors evaluate the model on the MNIST dataset, a cluttered version of MNIST, and a dynamic video game environment.

#### Questions / Notes

- I think the the author's claim taht the model works independently of image size is only partly true, as larger images are likely to require more glimpses or bigger regions.
- Would be nice to see some large-scale benchmarks as MNIST is very simple tasks. However, the authors clearly identify this as future work.
- No mentions about training time. Is it even feasible to train this for large images (which probably require more glimpses)?

