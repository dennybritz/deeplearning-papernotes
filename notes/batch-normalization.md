## [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](http://arxiv.org/abs/1502.03167)

TLDR; The authors introduce Batch Normalization, a technique to normalize unit activations to zero mean and unit variance within the network. The authors show that in Feedforward and Convolutional Networks, Batch Normalization leads to faster training and better accuracies. BN also acts as a regularizer, reducing the need for Dropout, etc. Using an ensemble of batch normalized networks the authors achieve state of the art on ILSVRC.


#### Key Points

- Network training is complicated because the input distributions to higher level change as the parameter in lower layers are changing: Internal Covariate Shift. Solution: Normalize within the network.
- BN: Normalize input to nonlinearity to have zero mean and unit variance. Then add two additional parameters (scaling and bias) per unit to preserve expressability of the network. Statistics are calculated per minibatch.
- Network parameters increase, but not by much: 2 parameter per unit that has batch normalization applied to it.
- Works well for fully connected and convolutional layers. Authors didn't try RNNs.
- Change to make when adding BN: Increase learning rate, remove/decrease dropout and l2 regularization, accelerate learning rate decay, shuffle training examples more thoroughly.