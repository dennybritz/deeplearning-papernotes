## [ReNet: A Recurrent Neural Network Based Alternative to Convolutional Networks](http://arxiv.org/abs/1505.00393)

TLDR; The authors propose a novel architecture called ReNet, which replaces convolutional and max-pooling layers with RNNs that sweep over the image vertically and horizontally. These RNN layers are then stacked. The authors demonstrate that ReNet architecture is a viable alternative to CNNs. ReNet doesn't outperform CNNs in this paper, but further optimizations and hyperparameter tuning are likely going to lead to improved results in the future.

#### Key Points:

- Split images into patches, feed one patch per time step into RNN, vertically then horizontally. 4 RNNs per layer, 2 vertical and 2 horizontal, one per diretion.
- Because the RNNs sweep over the whole image they can see the context of the full image, as opposed to just a local context in the case of conv/pool layers.
- Smooth from end-end to end.
- In experiments, 2 256-dimensional ReNet layers, 2x2 patches, 4096-dimensional affine layers.
- Flipping and shifting for data augmentation.

#### Notes/Questions:

- What is the training time/complexity compared to a CNN? 
- Why split the image into patches at all? I wonder if the authors have experimented with various patch sizes, like defining patches that go over the full vertical height. 2x2 patches as used in the experiment seem quite small and like a waste of computational resources.