## [Spatial Transformer Networks](http://arxiv.org/abs/1506.02025)

TLDR; The authors introduce a new spatial transformation module that can be inserted into any Neural Network. The module consists of a spatial transformation network that predicts transformation parameters, a grid generator that chooses a sampling grid from the input, and a sampler that produces the output. Possible learned transformations include things cropping, translation, rotation, scaling or attention. The module can be trained end-to-end using backpropagation. The authors evaluate evaluate the module on both CNNs and MLPs, achieving state on distorted MNIST data, street view numbers, and fine-grained bird classification.

#### Key Points:

- STMs can be inserted between any layers, typically after the input or extracted features. The transform is dynamic and happens based on the input data.
- The module is fast and doesn't adversely impact training speed.
- The actual transformation parameters (output of localization network) can be fed into higher layers.
- Attention can be seen as a special transformation that increases computational efficiency.
- Can also be applied to RNNs, but more investigation is needed.