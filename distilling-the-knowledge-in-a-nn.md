## [Distilling the Knowledge in a Neural Network](http://arxiv.org/abs/1503.02531)

TLDR; The authors show that we can distill the knowledge of a complex ensemble of models into a smaller model by letting the smaller model learn directly from the "soft targets" (softmax output with high temperature) of the ensemble. Intuitively, this works because the errors in probability assignment (e.g. assigning 0.1% to the wrong class) carry a lot of information about what the network learns. Learning directly from logits (unnormalized scores) as was done in a previous paper, is a special case of the distillation approach. The authors show how distillation works on the MNIST and an ASR data set.


#### Key Points

- Can use unlabeled data to transfer knowledge, but using the same training data seems to work well in practice.
- Use softmax with temperature, values from 1-10 seem to work well, depending on the problem.
- The MNIST networks learn to recognize digits without ever having seen base, solely based on the "errors" that the teacher network makes. (Bias needs to be adjusted)
- Training on soft targets with less data performs much better than training on hard targets with same amount of data.


#### Notes/Question

- Breaking up the complex models into specialists didn't really fit into this paper without distilling those experts into one model. Also would've liked to see training of only specialists (without general network) and then distill their knowledge.