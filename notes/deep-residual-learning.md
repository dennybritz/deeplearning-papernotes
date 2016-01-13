## [Deep Residual Learning for Image Recognition](http://arxiv.org/abs/1512.03385)

TLDR; The authors present Residual Nets, which achieve 3.57% error on the ImageNet test set and won the 1st place on the ILSVRC 2015 challenge. ResNets work by introducing "shortcut" connections across stacks of layers, allowing the optimizer to learn an easier residual function instead of the original mapping. This allows for efficient training of very deep nets without the introduction of additional parameters or training complexity. The authors present results on ImageNet and CIFAR-100 with nets as deep as 152 layers (and one ~1000 layer deep net).


#### Key Points

- Problem: Deeper networks experience a *degradation* problem. They don't overfit but nonetheless perform worse than shallower networks on both training and test data due to being more difficult to optimize. 
- Because Deep Nets can in theory learn an identity mapping for their additional layers they should strict outperform shallower nets. In practice however, optimizers have problems learning identity (or near-identity) mappings. Learning residual mappings is easier, mitigating this problem.
- Residual Mapping: If the desired mapping is H(x), let the layers learn F(x) = H(x) - x and add x back through a shortcut connection H(x) = F(x) + x. An identity mapping can then be learned easily by driving the learned mapping F(x) to 0.
- No additional parameters or computational complexity are introduced by residuals nets.
- Similar to Highway Networks, but gates are not data-dependent (no extra parameters) and are always open.
- Due the the nature of the residual formula, input and output must be of same size (just like Highway Networks). We can do size transformation by zero-padding or projections. Projections introduce additional parameters. Authors found that projections perform slightly better, but are "not worth" the large number of extra parameters.
- 18 and 34-layer VGG-like plain net gets 27.94 and 28.54 error respectively, not that higher error for deeper net. ResNet gets 27.88 and 25.03 respectively. Error greatly reduces for deeper net.
- Use Bottleneck architecture with 1x1 convolutions to change dimensions.
- Single ResNet outperforms previous start of the art ensembles. ResNet ensemble even better.


#### Notes/Questions

- Love the simplicity of this.
- I wonder how performance depends on the number of layers skipped by the shortcut connections. The authors only present results with 2 or 3 layers.
- "Stacked" or recursive residuals?
- In principle Highway Networks should be able to learn the same mappings quite easily. Is this an optimization problem? Do we just not have enough data. What if we made the gates less fine-grained and substituted sigmoid with something else?
- Can we apply this to RNNs, similar to LSTM/GRU? Seems good for learning long-range dependencies.