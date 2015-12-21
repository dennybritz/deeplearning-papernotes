## [Training Very Deep Networks](http://arxiv.org/abs/1507.06228)

TLDR; The authors propose "Highway Networks", which uses gates (inspired by LSTMs) to determine how much of a layer's activations to transform or just pass through. Highway Networks can be used with any kind of activation function, including recurrent and convnolutional units, and trained using plain SGD. The gating mechanism allows highway networks with tens or hundreds of layers to be trained efficiently. The authors show that highway networks with fewer parameters achieve results competitive with state-of-the art for the MNIST and CIFAR tasks. Gates outputs vary significantly with the input examples, demonstrating that the network not just learns a "fixed structure", but dynamically routes data based for specific examples examples.

Datasets used: MNIST, CIFAR-10, CIFAR-100


#### Key Takeaways

- Apply LSTM-like gating to networks layers. Transform gate T and carry gate C.
- The gating forces the layer inputs/outputs to be of the same size. We can use additional plain layers for dimensionality transformations.
- Bias weights of the transform gates should be initialized to negative values (-1, -2, -3, etc) to initially force the networks to pass through information and learn long-term dependencies.
- HWN does not learn a fixed structure (same gate outputs), but dynamic routing based on current input.
- In complex data sets each layer makes an important contritbution, which is shown by lesioning (setting to pass-through) individual layers.


#### Notes / Questions

- Seems like the authors did not use dropout in their experiments. I wonder how these play together. Is dropout less effective for highway networks because the gates already learn efficients paths?
- If we see that certain gates outputs have low variance across examples, can we "prune" the network into a fixed strucure to make it more efficient (for production deployments)?